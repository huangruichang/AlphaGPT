"""
Download A-share stock basic info and daily OHLCV data via akshare or tushare,
then write into PostgreSQL tables defined in ashare_db.py.

Tushare is used as the primary data source (more stable than akshare).
Akshare is kept as a fallback.

Usage:
    python -m data_pipeline.ashare_download          # default: HS300 + ZZ500
    python -m data_pipeline.ashare_download --all     # top 500 by market cap
"""

import argparse
import datetime as dt
import os
import time
from typing import List
from typing import Optional

import akshare as ak
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.ashare_db import SessionLocal, StockBasic, Daily, engine
from sqlalchemy.dialects.postgresql import insert as pg_insert

# ------------------------------------------------------------------
# Tushare initialisation (primary data source)
# ------------------------------------------------------------------
# Read token from environment variable TUSHARE_TOKEN (set in .env)
_TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
_tushare_pro: Optional[object] = None
_tushare_available: bool = False

if _TUSHARE_TOKEN:
    try:
        import tushare as ts
        ts.set_token(_TUSHARE_TOKEN)
        _tushare_pro = ts.pro_api()
        _tushare_available = True
        print("[init] Tushare initialised successfully.")
    except Exception as e:
        print(f"[init] Tushare initialisation failed: {e}. Will use akshare as fallback.")
else:
    print("[init] TUSHARE_TOKEN not set in environment. Will use akshare only.")


def _tushare_enabled() -> bool:
    """Return True if tushare pro API is available."""
    return _tushare_available and _tushare_pro is not None


# ------------------------------------------------------------------
# 1. Stock pool helpers
# ------------------------------------------------------------------

def get_hs300_codes() -> List[str]:
    """Return ts_code list of current HS300 constituents."""
    df = ak.index_stock_cons_csindex(symbol="000300")
    return df["成分券代码"].apply(_to_ts_code).tolist()


def get_zz500_codes() -> List[str]:
    """Return ts_code list of current ZZ500 constituents."""
    df = ak.index_stock_cons_csindex(symbol="000905")
    return df["成分券代码"].apply(_to_ts_code).tolist()


def get_top500_by_mcap() -> List[str]:
    """
    Return ts_code list of top A-share stocks by market cap.
    
    Since ak.stock_zh_a_spot_em() is unreliable (connection issues),
    we use index constituents as a proxy for top stocks:
    - ZZ800 (HS300 + ZZ500) = 800 stocks
    - SSE50 = 50 stocks  
    - STAR50 = 50 stocks
    This gives us ~800-900 stocks covering most of the market cap.
    """
    import time
    
    print("  Fetching stock list from major indices (ZZ800 + SSE50 + STAR50) ...")
    
    all_codes = set()
    
    # Helper to fetch index with retry
    def fetch_index(symbol: str, name: str) -> set:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = ak.index_stock_cons_csindex(symbol=symbol)
                codes = set(df["成分券代码"].tolist())
                print(f"    {name}: {len(codes)} stocks")
                return codes
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    {name}: retry {attempt + 1}/{max_retries} after error: {str(e)[:50]}")
                    time.sleep(2)
                else:
                    print(f"    {name}: failed after {max_retries} retries: {str(e)[:50]}")
                    return set()
    
    # Fetch from major indices
    # ZZ800 = HS300 + ZZ500 (covers top 800 by market cap)
    all_codes.update(fetch_index("000906", "ZZ800"))
    
    # SSE50 (Shanghai Stock Exchange 50) - large cap
    all_codes.update(fetch_index("000016", "SSE50"))
    
    # STAR50 (Science and Technology Innovation Board 50) - top STAR stocks
    all_codes.update(fetch_index("000688", "STAR50"))
    
    if len(all_codes) == 0:
        print("  Warning: All index APIs failed, falling back to all A-share stocks ...")
        try:
            df_all = ak.stock_info_a_code_name()
            all_codes = set(df_all["code"].tolist())
            print(f"    Got {len(all_codes)} stocks from stock_info_a_code_name()")
        except Exception as e:
            print(f"  Error: Failed to fetch stock list: {e}")
            raise
    
    # Convert to ts_code format
    result = [_to_ts_code(code) for code in all_codes]
    print(f"  Total unique stocks: {len(result)}")
    
    return result


def get_stock_pool(use_index: bool = True) -> List[str]:
    """
    Parameters
    ----------
    use_index : bool
        True  -> HS300 ∪ ZZ500 (default, ~600 stocks)
        False -> Top stocks by market cap via index constituents (ZZ800 + SSE50 + STAR50, ~800-900 stocks)
    """
    if use_index:
        codes = set(get_hs300_codes())
        codes.update(get_zz500_codes())
        return list(codes)
    return get_top500_by_mcap()


# ------------------------------------------------------------------
# 2. Download helpers
# ------------------------------------------------------------------

def download_stock_basic_tushare(ts_codes: List[str]) -> pd.DataFrame:
    """
    Fetch stock_basic info from tushare pro API.

    Tushare stock_basic returns:
        ts_code, symbol, name, area, industry, market, list_date, list_status
    """
    if not _tushare_enabled():
        raise RuntimeError("Tushare is not available. Set TUSHARE_TOKEN in .env")

    print("  Fetching stock basic info from tushare ...")
    # Tushare limits: max 5000 rows per call; we may need multiple calls
    # But stock_basic with list_status='L' returns all listed stocks at once
    df = _tushare_pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,list_status",
    )
    df["ts_code"] = df["ts_code"].str.strip()
    df_pool = df[df["ts_code"].isin(ts_codes)].copy()

    # Ensure all required columns exist
    for col in ["symbol", "name", "industry", "area", "list_status", "market", "list_date"]:
        if col not in df_pool.columns:
            df_pool[col] = None

    print(f"  Got {len(df_pool)} stocks from tushare")
    return df_pool


def download_stock_basic(ts_codes: List[str]) -> pd.DataFrame:
    """
    Fetch stock_basic info.
    Try tushare first (more stable), fall back to akshare if tushare fails.
    """
    if _tushare_enabled():
        try:
            return download_stock_basic_tushare(ts_codes)
        except Exception as e:
            print(f"  [WARN] Tushare download_stock_basic failed: {e}. Falling back to akshare ...")

    # Fallback to akshare
    print("  Fetching stock code → name mapping (akshare) ...")
    df_all = ak.stock_info_a_code_name()
    df_all["ts_code"] = df_all["code"].apply(_to_ts_code)
    df_pool = df_all[df_all["ts_code"].isin(ts_codes)].copy()
    df_pool = df_pool.rename(columns={"code": "symbol", "name": "name"})

    # Try to get industry/area info using available akshare APIs
    # Note: stock_a_industry_name_em() does not exist in akshare v1.18+
    # We'll leave industry/area empty for now
    print("  Note: industry/area info not fetched (API unavailable in current akshare version)")
    df_pool["industry"] = None
    df_pool["area"] = None

    df_pool["list_status"] = "L"
    df_pool["market"] = ""
    df_pool["list_date"] = None

    # Ensure all required columns exist
    for col in ["symbol", "name", "industry", "area", "list_status", "market", "list_date"]:
        if col not in df_pool.columns:
            df_pool[col] = None

    return df_pool


def _retry_akshare(func, retries: int = 8, delay: float = 5.0):
    """
    Call func(); on exception retry with exponential backoff + random jitter.

    Parameters
    ----------
    retries : int
        Max retry attempts (default 8).
    delay : float
        Initial delay in seconds (default 5.0).
        Each subsequent retry doubles the delay (exponential backoff).
    """
    import time, random
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            if i == retries - 1:
                raise
            # Exponential backoff: delay * 2^i, with random jitter ±25%
            backoff = delay * (2 ** i)
            jitter = backoff * random.uniform(-0.25, 0.25)
            sleep_time = max(delay, backoff + jitter)
            print(f"    retry {i+1}/{retries} after {sleep_time:.1f}s: {e}")
            time.sleep(sleep_time)


def download_daily_tushare(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily OHLCV for one stock via tushare pro API.

    Tushare daily returns:
        ts_code, trade_date, open, high, low, close, pre_close,
        change, pct_chg, vol, amount

    Parameters
    ----------
    ts_code : str  e.g. '600519.SH'
    start_date : str  'YYYYMMDD'
    end_date : str  'YYYYMMDD'
    """
    if not _tushare_enabled():
        raise RuntimeError("Tushare is not available. Set TUSHARE_TOKEN in .env")

    df = _tushare_pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        return pd.DataFrame()

    # Tushare returns trade_date as 'YYYYMMDD' string; convert to date
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.date

    # Ensure numeric types
    numeric_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg",
                    "vol", "amount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Columns expected by the daily table; fill missing with None
    keep = [
        "ts_code", "trade_date",
        "open", "high", "low", "close",
        "pre_close", "change", "pct_chg",
        "vol", "amount",
        "turnover_rate", "volume_ratio",
        "pe", "pb",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = None

    return df[keep]


def download_daily(ts_code: str, start_date: str, end_date: str,
                   between_delay: float = 1.5) -> pd.DataFrame:
    """
    Download daily OHLCV for one stock.
    Try tushare first (more stable), fall back to akshare if tushare fails.

    Parameters
    ----------
    between_delay : float
        Seconds to sleep after each API call to avoid rate limiting (default 1.5).
        Only used for akshare fallback.
    """
    if _tushare_enabled():
        try:
            df = download_daily_tushare(ts_code, start_date, end_date)
            if not df.empty:
                return df
            # Empty result from tushare, try akshare
        except Exception as e:
            print(f"  [WARN] Tushare download_daily failed for {ts_code}: {e}. "
                  f"Falling back to akshare ...")

    # Fallback to akshare
    symbol = ts_code[:6]
    try:
        df = _retry_akshare(
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="",
            )
        )
    except Exception as e:
        print(f"  [ERROR] {ts_code} (symbol={symbol}): "
              f"stock_zh_a_hist() failed after all retries. "
              f"start={start_date}, end={end_date}, error: {e}")
        return pd.DataFrame()
    finally:
        # Always sleep after API call to avoid overwhelming the server
        time.sleep(between_delay)

    if df.empty:
        return pd.DataFrame()

    # Rename only the columns that actually exist
    col_map = {
        "日期": "trade_date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "涨跌额": "change",
        "涨跌幅": "pct_chg",
        "成交量": "vol",
        "成交额": "amount",
        "换手率": "turnover_rate",
    }
    df = df.rename(columns=col_map)

    # Derive pre_close: shift close by 1 (previous row)
    df["pre_close"] = df["close"].shift(1)

    # Ensure numeric types
    numeric_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg",
                    "vol", "amount", "turnover_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ts_code"] = ts_code
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    # Columns expected by the daily table; fill missing with None
    keep = [
        "ts_code", "trade_date",
        "open", "high", "low", "close",
        "pre_close", "change", "pct_chg",
        "vol", "amount",
        "turnover_rate", "volume_ratio",
        "pe", "pb",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = None

    return df[keep]


# ------------------------------------------------------------------
# 3. Upsert helpers
# ------------------------------------------------------------------

def upsert_stock_basic(df: pd.DataFrame):
    """Bulk upsert into stock_basic table."""
    session = SessionLocal()
    try:
        for _, row in df.iterrows():
            stmt = pg_insert(StockBasic.__table__).values(
                ts_code=row.get("ts_code"),
                symbol=row.get("code", row.get("ts_code", "")[:6]),
                name=row.get("name", ""),
                area=row.get("area"),
                industry=row.get("industry"),
                list_date=row.get("list_date"),
                market=row.get("market"),
                list_status=row.get("list_status", "L"),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["ts_code"],
                set_={
                    "name": stmt.excluded.name,
                    "area": stmt.excluded.area,
                    "industry": stmt.excluded.industry,
                    "market": stmt.excluded.market,
                    "list_status": stmt.excluded.list_status,
                },
            )
            session.execute(stmt)
        session.commit()
        print(f"  Upserted {len(df)} rows into stock_basic")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _clean_nan(record: dict) -> dict:
    """Replace pandas NaN / NaT with None so PostgreSQL accepts the row."""
    for k, v in record.items():
        if v is None:
            continue
        try:
            if pd.isna(v):
                record[k] = None
        except (ValueError, TypeError):
            pass
    return record


def upsert_daily(df: pd.DataFrame):
    """Bulk upsert into daily table."""
    session = SessionLocal()
    try:
        for _, row in df.iterrows():
            values = _clean_nan(row.to_dict())
            stmt = pg_insert(Daily.__table__).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["ts_code", "trade_date"],
                set_={
                    col: getattr(stmt.excluded, col)
                    for col in [
                        "open", "high", "low", "close",
                        "pre_close", "change", "pct_chg",
                        "vol", "amount",
                        "turnover_rate", "volume_ratio",
                        "pe", "pb",
                    ]
                },
            )
            session.execute(stmt)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------

def main(use_index: bool = True, days_back: int = 730):
    """
    Parameters
    ----------
    use_index : bool
        True  -> download HS300 + ZZ500 constituents
        False -> download top 500 by market cap
    days_back : int
        How many days of daily data to fetch (default 2 years).
    """
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days_back)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    print("Initialising database schema …")
    from data_pipeline.ashare_db import Base
    Base.metadata.create_all(bind=engine)

    print("Fetching stock pool …")
    ts_codes = get_stock_pool(use_index=use_index)
    print(f"  Stock pool size: {len(ts_codes)}")

    # --- Stock basic ---
    print("Downloading stock basic info …")
    df_basic = download_stock_basic(ts_codes)
    upsert_stock_basic(df_basic)

    # --- Daily OHLCV ---
    print(f"Downloading daily data ({start_str} → {end_str}) …")
    print(f"  Total stocks to download: {len(ts_codes)}")
    print(f"  Rate limit: ~1.5s between requests, pause 15s every 10 stocks")
    batch_size = 10
    pause_seconds = 15
    failed_stocks = []
    for i, ts_code in enumerate(ts_codes, 1):
        print(f"  [{i}/{len(ts_codes)}] {ts_code} …")
        df_daily = download_daily(ts_code, start_str, end_str)
        if df_daily.empty:
            failed_stocks.append(ts_code)
            continue
        upsert_daily(df_daily)
        # Pause every batch_size stocks to avoid rate limiting
        if i % batch_size == 0 and i < len(ts_codes):
            print(f"  [...] Pausing {pause_seconds}s after {i} stocks "
                  f"({len(ts_codes) - i} remaining) …")
            time.sleep(pause_seconds)

    print(f"Daily data download complete. "
          f"Failed: {len(failed_stocks)}/{len(ts_codes)}")
    if failed_stocks:
        print(f"  Failed stocks: {failed_stocks[:20]}"
              f"{'...' if len(failed_stocks) > 20 else ''}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_ts_code(code: str) -> str:
    """Convert pure numeric code '600519' to tushare-style '600519.SH'."""
    code = str(code).zfill(6)
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


def _parse_date(s: str):
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download A-share data to PostgreSQL")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use top-500-by-market-cap instead of HS300+ZZ500",
    )
    parser.add_argument("--days", type=int, default=730, help="Days of history (default 730)")
    args = parser.parse_args()

    main(use_index=not args.all, days_back=args.days)
