"""
Download A-share stock basic info and daily OHLCV data via akshare,
then write into PostgreSQL tables defined in ashare_db.py.

Usage:
    python -m data_pipeline.ashare_download          # default: HS300 + ZZ500
    python -m data_pipeline.ashare_download --all     # top 500 by market cap
"""

import argparse
import datetime as dt
from typing import List

import akshare as ak
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.ashare_db import SessionLocal, StockBasic, Daily, engine
from sqlalchemy.dialects.postgresql import insert as pg_insert


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
    Return ts_code list of A-share stocks ranked by market cap
    (fallback when index constituent APIs are unavailable).
    """
    today = dt.date.today().strftime("%Y%m%d")
    df = ak.stock_zh_a_spot_em()
    df = df.sort_values("总市值", ascending=False).head(500)
    return df["代码"].apply(_to_ts_code).tolist()


def get_stock_pool(use_index: bool = True) -> List[str]:
    """
    Parameters
    ----------
    use_index : bool
        True  -> HS300 ∪ ZZ500
        False -> top 500 by market cap
    """
    if use_index:
        codes = set(get_hs300_codes())
        codes.update(get_zz500_codes())
        return list(codes)
    return get_top500_by_mcap()


# ------------------------------------------------------------------
# 2. Download helpers
# ------------------------------------------------------------------

def download_stock_basic(ts_codes: List[str]) -> pd.DataFrame:
    """
    Fetch stock_basic info from akshare bulk APIs.

    Uses:
      - stock_info_a_code_name()       : code → name mapping
      - stock_individual_info_em(code) : per-stock industry/area/list_date
        (called in batch; for large pools consider stock_a_lg_indicator() as a
        faster alternative if available in your akshare version)
    """
    # Bulk code → name
    df_all = ak.stock_info_a_code_name()
    df_all["ts_code"] = df_all["code"].apply(_to_ts_code)
    df_pool = df_all[df_all["ts_code"].isin(ts_codes)].copy()
    df_pool = df_pool.rename(columns={"code": "symbol", "name": "name"})

    # Bulk industry / area via stock_a_industry_name_em (covers entire market)
    try:
        df_ind = ak.stock_a_industry_name_em()
        df_ind["ts_code"] = df_ind["code"].apply(_to_ts_code)
        df_pool = df_pool.merge(
            df_ind[["ts_code", "industry", "area"]],
            on="ts_code",
            how="left",
        )
    except Exception:
        # fallback: leave industry/area empty
        pass

    df_pool["list_status"] = "L"
    df_pool["market"] = df_pool.get("market", "")
    return df_pool


def _retry_akshare(func, retries: int = 3, delay: float = 2.0):
    """Call func(); on ConnectionError retry with sleep."""
    import time
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"    retry {i+1}/{retries} after: {e}")
            time.sleep(delay)


def download_daily(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download daily OHLCV for one stock.
    ts_code format: '600519.SH' -> pass '600519' to akshare.

    akshare stock_zh_a_hist returns (v1.18+):
        日期, 股票代码, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    pe / pb / 量比 are NOT in this call — left as NULL.
    pre_close is derived from the previous row's close.
    """
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
        print(f"  [WARN] {ts_code}: daily download failed after retries: {e}")
        return pd.DataFrame()

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
    for i, ts_code in enumerate(ts_codes, 1):
        print(f"  [{i}/{len(ts_codes)}] {ts_code} …")
        df_daily = download_daily(ts_code, start_str, end_str)
        if df_daily.empty:
            continue
        upsert_daily(df_daily)

    print("Done.")


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
