import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from torch.utils.data import Dataset, DataLoader
from .config import ModelConfig
from .factors import FeatureEngineer


class CryptoDataLoader:
    """
    Loads A-share market data from tushare-style PostgreSQL tables
    (`stock_basic` and `daily`) and serves PyTorch tensors.

    Key schema assumptions
    ----------------------
    stock_basic(ts_code, symbol, name, area, industry, market, list_date)
    daily(ts_code, trade_date, open, high, low, close, vol, amount)

    vol  = volume in lots (100 shares/lot), so volume = vol * 100
    """

    def __init__(self, mode: str = "train", limit: int = None):
        self.mode = mode
        self.limit = limit
        self.raw_data_cache = {}        # {ts_code: {open, high, low, close, volume}}
        self.target_returns = None      # np.ndarray [n_stocks, hist_len-1]
        self.feat_tensor = None         # torch.Tensor [n_stocks, hist_len-1, input_dim]
        self.price_tensor = None        # torch.Tensor [n_stocks, hist_len]
        self.engineer = FeatureEngineer()

    # ------------------------------------------------------------------ #
    #  public helpers
    # ------------------------------------------------------------------ #

    def load_data(self):
        """
        Full pipeline:

        1.  Fetch stock list and daily bars from PostgreSQL.
        2.  Build raw_data_cache with keys
            {open, high, low, close, volume}.
        3.  Compute target returns (log close-to-close return).
        4.  Build feature tensor via FeatureEngineer.compute_features.
        5.  Move tensors to ModelConfig.DEVICE.
        """
        self._fetch_from_db()
        self._build_target_returns()
        self._build_feature_tensor()
        self._to_device()
        print(f"[DataLoader] Loaded {len(self.raw_data_cache)} stocks -> "
              f"feat_tensor {tuple(self.feat_tensor.shape)}, "
              f"target_returns {tuple(self.target_returns.shape)}")

    def get_training_batch(self, batch_size: int = None):
        """
        Returns (feat_tensor, target_returns, price_tensor) for the
        single-asset training loop.  All tensors are on DEVICE.

        TODO: per-stock batches – currently returns the full tensors.
        """
        bs = batch_size or ModelConfig.BATCH_SIZE
        return self.feat_tensor, self.target_returns, self.price_tensor

    def get_dataloader(self, batch_size: int = None):
        """
        PyTorch DataLoader wrapping a minimal Dataset.

        Yields (features, target_return, price_seq, stock_idx).
        """
        bs = batch_size or ModelConfig.BATCH_SIZE

        class _DS(Dataset):
            def __init__(self, feats, targets, prices):
                self.f = feats
                self.t = targets
                self.p = prices

            def __len__(self):
                return self.f.shape[0]

            def __getitem__(self, idx):
                return self.f[idx], self.t[idx], self.p[idx], idx

        return DataLoader(_DS(self.feat_tensor, self.target_returns, self.price_tensor),
                         batch_size=bs, shuffle=False)

    def get_stock_count(self) -> int:
        return len(self.raw_data_cache)

    def get_trading_dates(self) -> list:
        """Return sorted trade dates (as strings) across all stocks."""
        date_set = set()
        for rec in self.raw_data_cache.values():
            date_set.update(rec["trade_dates"])
        return sorted(date_set)

    # ------------------------------------------------------------------ #
    #  internals – DB
    # ------------------------------------------------------------------ #

    def _fetch_from_db(self):
        """
        Query stock_basic + daily, reshape into raw_data_cache.

        raw_data_cache[ts_code] = {
            "open":   np.ndarray,   # float32
            "high":   np.ndarray,
            "low":    np.ndarray,
            "close":  np.ndarray,
            "volume": np.ndarray,   # = vol * 100 (lots -> shares)
            "trade_dates": list[str]
        }
        """
        conn = psycopg2.connect(ModelConfig.DB_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 1. stock list ---------------------------------------------------
        cur.execute("""
            SELECT ts_code
            FROM   stock_basic
            WHERE  market IN ('主板','创业板','科创板')
            ORDER  BY ts_code
        """)
        codes = [r["ts_code"] for r in cur.fetchall()]
        if self.limit:
            codes = codes[:self.limit]
        print(f"[DataLoader] Querying {len(codes)} A-share stocks …")

        # 2. daily bars ---------------------------------------------------
        # Use ANY($1) with a list; for large universes batch the query.
        cur.execute("""
            SELECT ts_code, trade_date, open, high, low, close, vol, amount
            FROM   daily
            WHERE  ts_code = ANY(%s)
            ORDER  BY ts_code, trade_date
        """, (codes,))
        rows = cur.fetchall()
        conn.close()

        # 3. reshape ------------------------------------------------------
        for row in rows:
            code = row["ts_code"]
            if code not in self.raw_data_cache:
                self.raw_data_cache[code] = {
                    "open":  [], "high": [], "low":  [],
                    "close": [], "volume": [], "trade_dates": [],
                }
            rec = self.raw_data_cache[code]
            # trade_date may be int (20240101) or str — normalise to "YYYYMMDD"
            td = row["trade_date"]
            rec["trade_dates"].append(str(td))
            # tushare open/high/low/close are already float; vol is in lots
            rec["open"].append(float(row["open"]))
            rec["high"].append(float(row["high"]))
            rec["low"].append(float(row["low"]))
            rec["close"].append(float(row["close"]))
            rec["volume"].append(float(row["vol"]) * 100.0)

        # convert to arrays ------------------------------------------------
        for code, rec in self.raw_data_cache.items():
            for k in ("open", "high", "low", "close", "volume"):
                rec[k] = np.array(rec[k], dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  internals – targets & features
    # ------------------------------------------------------------------ #

    def _build_target_returns(self):
        """
        Close-to-close log return:

            r_t = log(close[t+1] / close[t])

        Handles stocks with different history lengths by truncating to the
        shortest series minus 1 (so every stock has at least one return).
        """
        stocks = sorted(self.raw_data_cache.keys())
        lengths = [len(self.raw_data_cache[c]["close"]) for c in stocks]
        min_len = min(lengths)
        # need at least 2 bars to compute one return
        assert min_len >= 2, f"Shortest price series has only {min_len} bars"

        returns = np.zeros((len(stocks), min_len - 1), dtype=np.float32)
        for i, code in enumerate(stocks):
            close = self.raw_data_cache[code]["close"][:min_len]
            log_close = np.log(np.clip(close, 1e-8, None))
            returns[i] = log_close[1:] - log_close[:-1]

        self.target_returns = torch.from_numpy(returns)

    def _build_feature_tensor(self):
        """
        Stack raw_data_cache into a 3-D array and delegate to
        FeatureEngineer.compute_features().

        Input shape to FeatureEngineer: (n_stocks, hist_len, 5)
        where the 5 columns are [open, high, low, close, volume].
        All stocks are truncated to the same ``hist_len`` (= min series
        length) so the array is rectangular.

        FeatureEngineer returns (n_stocks, hist_len, input_dim); we drop
        the last timestep so feat_tensor aligns with target_returns which
        has length hist_len-1.
        """
        stocks = sorted(self.raw_data_cache.keys())
        lengths = [len(self.raw_data_cache[c]["close"]) for c in stocks]
        hist_len = min(lengths)   # truncated length

        # (n_stocks, hist_len, 5)
        raw = np.zeros((len(stocks), hist_len, 5), dtype=np.float32)
        for i, code in enumerate(stocks):
            rec = self.raw_data_cache[code]
            raw[i, :, 0] = rec["open"][:hist_len]
            raw[i, :, 1] = rec["high"][:hist_len]
            raw[i, :, 2] = rec["low"][:hist_len]
            raw[i, :, 3] = rec["close"][:hist_len]
            raw[i, :, 4] = rec["volume"][:hist_len]

        # FeatureEngineer returns (n_stocks, hist_len, input_dim)
        feat_all = self.engineer.compute_features(raw)

        # drop last timestep → (n_stocks, hist_len-1, input_dim)
        self.feat_tensor = torch.from_numpy(feat_all[:, :-1, :]).float()

        # price_tensor: truncated close series for each stock
        close_mat = np.zeros((len(stocks), hist_len), dtype=np.float32)
        for i, code in enumerate(stocks):
            close_mat[i, :] = self.raw_data_cache[code]["close"][:hist_len]
        self.price_tensor = torch.from_numpy(close_mat).float()

    def _to_device(self):
        self.feat_tensor   = self.feat_tensor.to(ModelConfig.DEVICE)
        self.target_returns = self.target_returns.to(ModelConfig.DEVICE)
        self.price_tensor   = self.price_tensor.to(ModelConfig.DEVICE)
