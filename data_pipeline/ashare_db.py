"""
A-share database schema using SQLAlchemy (sync engine).

Tables:
  - stock_basic : static stock metadata (HS300 / ZZ500 / top-500 selection)
  - daily       : daily OHLCV + turnover + valuation for each stock
"""

import os
from datetime import date
from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Date,
    Float,
    Integer,
    PrimaryKeyConstraint,
    Index,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ashare")
DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

Base = declarative_base()


class StockBasic(Base):
    __tablename__ = "stock_basic"

    ts_code = Column(String(10), primary_key=True)   # e.g. 600519.SH
    symbol = Column(String(6))                        # e.g. 600519
    name = Column(String(50))
    area = Column(String(50))
    industry = Column(String(50))
    list_date = Column(Date)
    market = Column(String(10))     # main / gem / star …
    list_status = Column(String(1))  # L = listed, D = delisted, P = suspended

    __table_args__ = (
        Index("idx_stock_basic_symbol", "symbol"),
        Index("idx_stock_basic_name", "name"),
    )


class Daily(Base):
    __tablename__ = "daily"

    ts_code = Column(String(10), nullable=False)
    trade_date = Column(Date, nullable=False)

    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    pre_close = Column(Float)
    change = Column(Float)
    pct_chg = Column(Float)

    vol = Column(Float)          # volume in lots (手)
    amount = Column(Float)       # turnover in CNY

    turnover_rate = Column(Float)
    volume_ratio = Column(Float)

    pe = Column(Float)
    pb = Column(Float)

    __table_args__ = (
        PrimaryKeyConstraint("ts_code", "trade_date"),
        Index("idx_daily_trade_date", "trade_date"),
    )


# ------------------------------------------------------------------
# Engine / session helpers
# ------------------------------------------------------------------

engine = create_engine(DB_DSN, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables if they do not exist."""
    Base.metadata.create_all(engine)
    print("Tables created (if not exist): stock_basic, daily")


def drop_db():
    """Drop all tables (use with caution)."""
    Base.metadata.drop_all(engine)
    print("All tables dropped")


if __name__ == "__main__":
    init_db()
