# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import Column, Float, Index, Integer, MetaData, String, Table, Text, create_engine, event, text
from sqlalchemy.engine import Connection, Engine


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_sqlite_url(db_path: Path) -> str:
    db_path = db_path.expanduser().resolve()
    ensure_parent_dir(db_path)
    return f"sqlite:///{db_path.as_posix()}"


def normalize_db_url(db: str) -> str:
    """
    Accept either:
    - filesystem path to SQLite db (e.g. data/stock.db)
    - SQLAlchemy URL (e.g. mysql+pymysql://user:pass@host:3306/astock?charset=utf8mb4)
    """
    s = str(db).strip()
    if "://" in s:
        return s
    return to_sqlite_url(Path(s))


def _connect_args_for_url(url: str) -> dict:
    if url.startswith("sqlite:///"):
        return {"check_same_thread": False}
    return {}


def make_engine(
    db: str,
    *,
    pool_size: int = 10,
    max_overflow: int = 20,
) -> Engine:
    url = normalize_db_url(db)
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=pool_size,
        max_overflow=max_overflow,
        connect_args=_connect_args_for_url(url),
    )
    if engine.dialect.name == "sqlite":
        # Match previous sqlite3.connect pragmas (WAL speeds up readers).
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.close()

    return engine


@dataclass(frozen=True)
class DB:
    engine: Engine

    @classmethod
    def from_url(cls, db: str, *, pool_size: int = 10, max_overflow: int = 20) -> "DB":
        return cls(engine=make_engine(db, pool_size=pool_size, max_overflow=max_overflow))

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name


_META = MetaData()

stock_info = Table(
    "stock_info",
    _META,
    Column("stock_code", String(16), primary_key=True),
    Column("name", String(255)),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)

stock_daily = Table(
    "stock_daily",
    _META,
    Column("stock_code", String(16), primary_key=True),
    Column("date", String(10), primary_key=True),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("volume", Float),
    Column("amount", Float),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)

stock_indicators = Table(
    "stock_indicators",
    _META,
    Column("stock_code", String(16), primary_key=True),
    Column("date", String(10), primary_key=True),
    Column("ma20", Float),
    Column("ma60", Float),
    Column("ma120", Float),
    Column("rsi14", Float),
    Column("macd_diff", Float),
    Column("macd_dea", Float),
    Column("macd_hist", Float),
    Column("atr14", Float),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)

stock_levels = Table(
    "stock_levels",
    _META,
    Column("stock_code", String(16), primary_key=True),
    Column("calc_date", String(10), primary_key=True),
    Column("support_level", Float),
    Column("resistance_level", Float),
    Column("support_type", String(32)),
    Column("resistance_type", String(32)),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)

stock_scores = Table(
    "stock_scores",
    _META,
    Column("stock_code", String(16), primary_key=True),
    Column("score_date", String(10), primary_key=True),
    Column("total_score", Float),
    Column("trend_score", Float),
    Column("pullback_score", Float),
    Column("volume_price_score", Float),
    Column("rsi_score", Float),
    Column("macd_score", Float),
    Column("market_cap_score", Float),
    Column("tags", String(255)),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)

stock_scores_v3 = Table(
    "stock_scores_v3",
    _META,
    Column("stock_code", String(16), primary_key=True),
    Column("score_date", String(10), primary_key=True),
    Column("total_score", Float),
    Column("trend_score", Float),
    Column("pullback_score", Float),
    Column("volume_price_score", Float),
    Column("rsi_score", Float),
    Column("macd_score", Float),
    Column("base_structure_score", Float),
    Column("space_score", Float),
    Column("market_cap_score", Float),
    Column("status_tags", Text),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)

stock_future_perf = Table(
    "stock_future_perf",
    _META,
    Column("stock_code", String(16), primary_key=True),
    Column("signal_date", String(10), primary_key=True),
    Column("ne", Integer, primary_key=True),
    Column("max_price", Float),
    Column("min_price", Float),
    Column("final_price", Float),
    Column("max_return", Float),
    Column("min_return", Float),
    Column("final_return", Float),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)

Index("idx_stock_daily_date", stock_daily.c.date)  
Index("idx_stock_scores_date", stock_scores.c.score_date)
Index("idx_stock_scores_v3_date", stock_scores_v3.c.score_date)
# Speeds up Top-N queries like: WHERE score_date=? ORDER BY total_score DESC LIMIT N
Index("idx_stock_scores_v3_date_score", stock_scores_v3.c.score_date, stock_scores_v3.c.total_score)
Index("idx_stock_future_perf_signal_date", stock_future_perf.c.signal_date)


def init_db(engine: Engine) -> None:
    _META.create_all(engine)
    if engine.dialect.name == "mysql":
        _init_mysql_views(engine)
        _ensure_mysql_indexes(engine)


def _ensure_mysql_indexes(engine: Engine) -> None:
    """
    SQLAlchemy's create_all() won't add new indexes to an existing table.
    Keep a tiny idempotent "migration" here for performance-critical indexes.
    """
    with engine.begin() as conn:
        exists = conn.execute(
            text(
                """
                SELECT 1
                FROM information_schema.statistics
                WHERE table_schema = DATABASE()
                  AND table_name = 'stock_scores_v3'
                  AND index_name = 'idx_stock_scores_v3_date_score'
                LIMIT 1
                """
            )
        ).fetchone()
        if not exists:
            conn.execute(
                text("CREATE INDEX idx_stock_scores_v3_date_score ON stock_scores_v3 (score_date, total_score)")
            )


def _init_mysql_views(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE OR REPLACE VIEW vw_stock_pool_v3_latest AS
                SELECT
                    s.stock_code AS stock_code,
                    d.close AS close,
                    l.support_level AS support_level,
                    l.resistance_level AS resistance_level,
                    s.total_score AS total_score,
                    s.status_tags AS status_tags,
                    s.score_date AS score_date
                FROM stock_scores_v3 s
                LEFT JOIN stock_daily d
                    ON d.stock_code = s.stock_code AND d.date = s.score_date
                LEFT JOIN stock_levels l
                    ON l.stock_code = s.stock_code AND l.calc_date = s.score_date
                WHERE s.score_date = (SELECT MAX(score_date) FROM stock_scores_v3)
                ORDER BY s.total_score DESC
                """
            )
        )


def get_max_date(conn: Connection, table: str, stock_code: str, date_col: str) -> Optional[str]:
    row = conn.execute(
        text(f"SELECT MAX({date_col}) AS d FROM {table} WHERE stock_code = :stock_code"),
        {"stock_code": stock_code},
    ).fetchone()
    if not row:
        return None
    # row may be tuple or RowMapping depending on SQLAlchemy version
    val = row[0]
    return str(val) if val else None


def _upsert_sql(conn: Connection, table: str, cols: list[str], key_cols: list[str]) -> str:
    placeholders = ", ".join([f":{c}" for c in cols])
    col_list = ", ".join(cols)

    dialect = conn.engine.dialect.name
    if dialect == "mysql":
        update_cols = [c for c in cols if c not in key_cols]
        updates = ", ".join([f"{c}=VALUES({c})" for c in update_cols])
        return f"INSERT INTO {table}({col_list}) VALUES({placeholders}) ON DUPLICATE KEY UPDATE {updates}"

    # sqlite
    return f"INSERT OR REPLACE INTO {table}({col_list}) VALUES({placeholders})"


def upsert_stock_info(conn: Connection, stock_code: str, name: str) -> None:
    sql = _upsert_sql(conn, "stock_info", ["stock_code", "name"], ["stock_code"])
    conn.execute(text(sql), {"stock_code": stock_code, "name": name})


def upsert_daily(conn: Connection, stock_code: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0

    cols = ["stock_code", "date", "open", "high", "low", "close", "volume", "amount"]
    sql = _upsert_sql(conn, "stock_daily", cols, ["stock_code", "date"])
    rows = [
        {
            "stock_code": stock_code,
            "date": r.date,
            "open": float(r.open) if pd.notna(r.open) else None,
            "high": float(r.high) if pd.notna(r.high) else None,
            "low": float(r.low) if pd.notna(r.low) else None,
            "close": float(r.close) if pd.notna(r.close) else None,
            "volume": float(r.volume) if pd.notna(r.volume) else None,
            "amount": float(r.amount) if pd.notna(r.amount) else None,
        }
        for r in df.itertuples(index=False)
    ]
    conn.execute(text(sql), rows)
    return len(rows)


def load_daily(conn: Connection, stock_code: str, limit: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        text(
            """
            SELECT date, open, high, low, close, volume, amount
            FROM stock_daily
            WHERE stock_code = :stock_code
            ORDER BY date DESC
            LIMIT :limit
            """
        ),
        conn,
        params={"stock_code": stock_code, "limit": int(limit)},
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])
    return df.sort_values("date").reset_index(drop=True)


def load_daily_until(conn: Connection, stock_code: str, end_date: str, limit: int) -> pd.DataFrame:
    df = pd.read_sql_query(
        text(
            """
            SELECT date, open, high, low, close, volume, amount
            FROM stock_daily
            WHERE stock_code = :stock_code AND date <= :end_date
            ORDER BY date DESC
            LIMIT :limit
            """
        ),
        conn,
        params={"stock_code": stock_code, "end_date": end_date, "limit": int(limit)},
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])
    return df.sort_values("date").reset_index(drop=True)


def upsert_indicators(conn: Connection, stock_code: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0

    cols = [
        "stock_code",
        "date",
        "ma20",
        "ma60",
        "ma120",
        "rsi14",
        "macd_diff",
        "macd_dea",
        "macd_hist",
        "atr14",
    ]
    sql = _upsert_sql(conn, "stock_indicators", cols, ["stock_code", "date"])
    rows = [
        {
            "stock_code": stock_code,
            "date": r.date,
            "ma20": float(r.ma20) if pd.notna(r.ma20) else None,
            "ma60": float(r.ma60) if pd.notna(r.ma60) else None,
            "ma120": float(r.ma120) if pd.notna(r.ma120) else None,
            "rsi14": float(r.rsi14) if pd.notna(r.rsi14) else None,
            "macd_diff": float(r.macd_diff) if pd.notna(r.macd_diff) else None,
            "macd_dea": float(r.macd_dea) if pd.notna(r.macd_dea) else None,
            "macd_hist": float(r.macd_hist) if pd.notna(r.macd_hist) else None,
            "atr14": float(r.atr14) if pd.notna(r.atr14) else None,
        }
        for r in df.itertuples(index=False)
    ]
    conn.execute(text(sql), rows)
    return len(rows)


def upsert_levels(
    conn: Connection,
    stock_code: str,
    calc_date: str,
    support_level: Optional[float],
    resistance_level: Optional[float],
    support_type: str,
    resistance_type: str,
) -> None:
    cols = [
        "stock_code",
        "calc_date",
        "support_level",
        "resistance_level",
        "support_type",
        "resistance_type",
    ]
    sql = _upsert_sql(conn, "stock_levels", cols, ["stock_code", "calc_date"])
    conn.execute(
        text(sql),
        {
            "stock_code": stock_code,
            "calc_date": calc_date,
            "support_level": float(support_level) if support_level is not None else None,
            "resistance_level": float(resistance_level) if resistance_level is not None else None,
            "support_type": support_type,
            "resistance_type": resistance_type,
        },
    )


def upsert_score(
    conn: Connection,
    stock_code: str,
    score_date: str,
    total_score: float,
    trend_score: float,
    pullback_score: float,
    volume_price_score: float,
    rsi_score: float,
    macd_score: float,
    market_cap_score: float,
    tags: str,
) -> None:
    cols = [
        "stock_code",
        "score_date",
        "total_score",
        "trend_score",
        "pullback_score",
        "volume_price_score",
        "rsi_score",
        "macd_score",
        "market_cap_score",
        "tags",
    ]
    sql = _upsert_sql(conn, "stock_scores", cols, ["stock_code", "score_date"])
    conn.execute(
        text(sql),
        {
            "stock_code": stock_code,
            "score_date": score_date,
            "total_score": float(total_score),
            "trend_score": float(trend_score),
            "pullback_score": float(pullback_score),
            "volume_price_score": float(volume_price_score),
            "rsi_score": float(rsi_score),
            "macd_score": float(macd_score),
            "market_cap_score": float(market_cap_score),
            "tags": tags,
        },
    )


def upsert_score_v3(
    conn: Connection,
    stock_code: str,
    score_date: str,
    total_score: float,
    trend_score: float,
    pullback_score: float,
    volume_price_score: float,
    rsi_score: float,
    macd_score: float,
    base_structure_score: float,
    space_score: float,
    market_cap_score: float,
    status_tags: str,
) -> None:
    cols = [
        "stock_code",
        "score_date",
        "total_score",
        "trend_score",
        "pullback_score",
        "volume_price_score",
        "rsi_score",
        "macd_score",
        "base_structure_score",
        "space_score",
        "market_cap_score",
        "status_tags",
    ]
    sql = _upsert_sql(conn, "stock_scores_v3", cols, ["stock_code", "score_date"])
    conn.execute(
        text(sql),
        {
            "stock_code": stock_code,
            "score_date": score_date,
            "total_score": float(total_score),
            "trend_score": float(trend_score),
            "pullback_score": float(pullback_score),
            "volume_price_score": float(volume_price_score),
            "rsi_score": float(rsi_score),
            "macd_score": float(macd_score),
            "base_structure_score": float(base_structure_score),
            "space_score": float(space_score),
            "market_cap_score": float(market_cap_score),
            "status_tags": status_tags,
        },
    )


def upsert_future_perf(
    conn: Connection,
    stock_code: str,
    signal_date: str,
    ne: int,
    max_price: float,
    min_price: float,
    final_price: float,
    max_return: float,
    min_return: float,
    final_return: float,
) -> None:
    cols = [
        "stock_code",
        "signal_date",
        "ne",
        "max_price",
        "min_price",
        "final_price",
        "max_return",
        "min_return",
        "final_return",
    ]
    sql = _upsert_sql(conn, "stock_future_perf", cols, ["stock_code", "signal_date", "ne"])
    conn.execute(
        text(sql),
        {
            "stock_code": stock_code,
            "signal_date": signal_date,
            "ne": int(ne),
            "max_price": float(max_price),
            "min_price": float(min_price),
            "final_price": float(final_price),
            "max_return": float(max_return),
            "min_return": float(min_return),
            "final_return": float(final_return),
        },
    )
