#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
getData2.py

Fetch A-share daily K-line data using non-Eastmoney AkShare sources
(Tencent QQ for daily history, SSE/SZSE list for codes) and write to DB.
CLI parameters remain the same as getData.py.
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import akshare as ak
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError


DEFAULT_DB = "data/stock.db"


@dataclass
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    charset: str = "utf8mb4"


@dataclass
class AppConfig:
    db_url: Optional[str] = None
    mysql: MySQLConfig = field(default_factory=MySQLConfig)


def load_config(path: Path) -> AppConfig:
    import configparser

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    db_url = parser.get("database", "db_url", fallback=None)
    db_url = db_url.strip() if db_url else None
    mysql = MySQLConfig(
        host=parser.get("mysql", "host", fallback="127.0.0.1").strip() or "127.0.0.1",
        port=parser.getint("mysql", "port", fallback=3306),
        user=parser.get("mysql", "user", fallback="").strip(),
        password=parser.get("mysql", "password", fallback=""),
        database=parser.get("mysql", "database", fallback="").strip(),
        charset=parser.get("mysql", "charset", fallback="utf8mb4").strip() or "utf8mb4",
    )
    return AppConfig(db_url=db_url, mysql=mysql)


def build_mysql_url(cfg: MySQLConfig) -> Optional[str]:
    if not (cfg.user and cfg.database):
        return None
    from urllib.parse import quote_plus

    user = quote_plus(cfg.user)
    password = quote_plus(cfg.password or "")
    auth = f"{user}:{password}" if password else user
    return f"mysql+pymysql://{auth}@{cfg.host}:{int(cfg.port)}/{cfg.database}?charset={cfg.charset}"


def resolve_db_target(args: argparse.Namespace) -> str:
    if getattr(args, "db_url", None):
        return str(args.db_url)
    import os

    env = os.getenv("ASTOCK_DB_URL")
    if env and env.strip():
        return env.strip()
    if getattr(args, "db", None):
        return str(args.db)
    cfg_path = getattr(args, "config", None)
    cfg_file = Path(cfg_path) if cfg_path else Path("config.ini")
    if cfg_file.exists():
        cfg = load_config(cfg_file)
        if cfg.db_url:
            return cfg.db_url
        url = build_mysql_url(cfg.mysql)
        if url:
            return url
    return DEFAULT_DB


def make_engine(db_target: str, workers: int) -> Engine:
    pool_size = max(5, min(64, workers * 2))
    max_overflow = max(10, min(128, workers * 2))
    connect_args = {}
    if str(db_target).startswith("sqlite:///") or db_target.endswith(".db"):
        connect_args["check_same_thread"] = False
        if "://" not in str(db_target):
            db_path = Path(db_target).expanduser().resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_target = f"sqlite:///{db_path.as_posix()}"
    engine = create_engine(
        db_target,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=pool_size,
        max_overflow=max_overflow,
        connect_args=connect_args,
    )
    if engine.dialect.name == "sqlite":
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.close()
    return engine


def ensure_core_tables(engine: Engine) -> None:
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS stock_info (
            stock_code VARCHAR(16) PRIMARY KEY,
            name VARCHAR(255),
            float_cap_billion DOUBLE NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS stock_daily (
            stock_code VARCHAR(16) NOT NULL,
            date VARCHAR(10) NOT NULL,
            open DOUBLE NULL,
            high DOUBLE NULL,
            low DOUBLE NULL,
            close DOUBLE NULL,
            volume DOUBLE NULL,
            amount DOUBLE NULL,
            PRIMARY KEY (stock_code, date)
        )
        """,
    ]
    with engine.begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))
    _ensure_stock_info_float_cap(engine)


def _ensure_stock_info_float_cap(engine: Engine) -> None:
    stmt = "ALTER TABLE stock_info ADD COLUMN float_cap_billion DOUBLE NULL"
    if engine.dialect.name == "sqlite":
        stmt = "ALTER TABLE stock_info ADD COLUMN float_cap_billion DOUBLE"
    try:
        with engine.begin() as conn:
            conn.execute(text(stmt))
            logging.info("[getData2] stock_info add float_cap_billion")
    except SQLAlchemyError as exc:
        msg = str(getattr(exc, "orig", exc)).lower()
        if "duplicate" in msg or "exists" in msg:
            return
        raise


def _parse_float_cap(raw: object) -> Optional[float]:
    if raw in (None, "", "nan", "NaN"):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    import math

    if math.isnan(val) or math.isinf(val) or val <= 0:
        return None
    if val > 1e6:
        val = val / 1e8
    return round(val, 4)


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_set = set(columns)
    for name in candidates:
        if name in col_set:
            return name
    return None


def normalize_code(raw: object) -> str:
    code = str(raw).strip()
    if not code or code.lower() in {"nan", "none"}:
        return ""
    if code.isdigit() and len(code) < 6:
        code = code.zfill(6)
    return code


def to_tx_symbol(code: str) -> Optional[str]:
    if not code:
        return None
    code = code.strip()
    lower = code.lower()
    if lower.startswith(("sh", "sz")) and len(lower) == 8:
        return lower
    if not code.isdigit():
        return None
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("0", "2", "3")):
        return f"sz{code}"
    return None


def _fetch_df_with_retry(func, label: str, retries: int = 3, base_sleep: float = 1.0) -> Optional[pd.DataFrame]:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            df = func()
            if df is not None and not df.empty:
                return df
            logging.warning("[getData2] %s list empty (attempt %d/%d)", label, attempt, retries)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logging.warning("[getData2] %s list failed (attempt %d/%d): %s", label, attempt, retries, exc)
        if attempt < retries:
            time.sleep(base_sleep * attempt)
    if last_exc:
        logging.debug("[getData2] %s last error: %s", label, last_exc)
    return None


def _extract_code_name_rows(df: pd.DataFrame) -> List[Tuple[str, str, Optional[float]]]:
    code_col = _first_existing(df.columns, ["code", "A\u80a1\u4ee3\u7801", "\u80a1\u7968\u4ee3\u7801", "\u8bc1\u5238\u4ee3\u7801", "item"])
    name_col = _first_existing(df.columns, ["name", "A\u80a1\u7b80\u79f0", "\u80a1\u7968\u7b80\u79f0", "\u8bc1\u5238\u7b80\u79f0", "value", "\u540d\u79f0"])
    if not code_col:
        raise RuntimeError(f"Unexpected stock list columns: {list(df.columns)}")

    cols = list(df.columns)
    code_idx = cols.index(code_col)
    name_idx = cols.index(name_col) if name_col else None

    out: List[Tuple[str, str, Optional[float]]] = []
    for row in df.itertuples(index=False, name=None):
        code = normalize_code(row[code_idx])
        if not code:
            continue
        name = str(row[name_idx]).strip() if name_idx is not None else ""
        out.append((code, name, None))
    return out


def fetch_stock_list() -> List[Tuple[str, str, Optional[float]]]:
    sources: List[Tuple[str, object, int]] = []
    if hasattr(ak, "stock_info_sh_name_code"):
        sources.append(("SSE", ak.stock_info_sh_name_code, 3))
    if hasattr(ak, "stock_info_sz_name_code"):
        sources.append(("SZSE", ak.stock_info_sz_name_code, 3))

    # Tencent daily interface does not cover BSE codes; keep disabled by default to reduce failures.
    include_bj = False
    if include_bj and hasattr(ak, "stock_info_bj_name_code"):
        sources.append(("BSE", ak.stock_info_bj_name_code, 1))

    out: List[Tuple[str, str, Optional[float]]] = []
    seen: set[str] = set()
    for label, func, retries in sources:
        df = _fetch_df_with_retry(func, label, retries=retries, base_sleep=1.0)
        if df is None or df.empty:
            continue
        try:
            rows = _extract_code_name_rows(df)
        except Exception as exc:  # noqa: BLE001
            logging.warning("[getData2] %s list parse failed: %s", label, exc)
            continue
        for code, name, cap in rows:
            if code in seen:
                continue
            seen.add(code)
            out.append((code, name, cap))

    if out:
        return out

    # Fallback: aggregated list (may hit BSE).
    if hasattr(ak, "stock_info_a_code_name"):
        df = _fetch_df_with_retry(ak.stock_info_a_code_name, "A-ALL", retries=2, base_sleep=1.5)
        if df is not None and not df.empty:
            try:
                return _extract_code_name_rows(df)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"AkShare stock list parse failed: {exc}") from exc

    raise RuntimeError("AkShare stock list failed: no sources available")


HIST_ALIASES = {
    "date": ["日期", "date", "Date"],
    "open": ["开盘价", "开盘", "open"],
    "high": ["最高价", "最高", "high"],
    "low": ["最低价", "最低", "low"],
    "close": ["收盘价", "收盘", "close"],
    "volume": ["成交量", "volume"],
    "amount": ["成交金额", "成交额", "amount"],
}

REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]


def _rename_hist_columns(raw: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for target, candidates in HIST_ALIASES.items():
        for col in candidates:
            if col in raw.columns:
                rename[col] = target
                break
    df = raw.rename(columns=rename)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def fetch_daily(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    symbol = to_tx_symbol(stock_code)
    if not symbol:
        logging.debug("[getData2] skip unsupported code: %s", stock_code)
        return pd.DataFrame()
    try:
        raw = ak.stock_zh_a_hist_tx(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("AkShare TX hist failed %s: %s", stock_code, exc)
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = _rename_hist_columns(raw)
    df = raw[REQUIRED_COLS].copy()
    for col in REQUIRED_COLS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    return df


def get_latest_date(engine: Engine, stock_code: str) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT MAX(date) FROM stock_daily WHERE stock_code = :c"),
            {"c": stock_code},
        ).fetchone()
    if not row or not row[0]:
        return None
    return str(row[0])


def upsert_stock_info(engine: Engine, rows: Iterable[Tuple[str, str, Optional[float]]]) -> None:
    stmt = text(
        """
        INSERT INTO stock_info(stock_code, name, float_cap_billion)
        VALUES(:code, :name, :cap)
        ON CONFLICT(stock_code) DO UPDATE SET
          name=excluded.name,
          float_cap_billion=COALESCE(excluded.float_cap_billion, float_cap_billion)
        """
        if engine.dialect.name == "sqlite"
        else """
        INSERT INTO stock_info(stock_code, name, float_cap_billion)
        VALUES(:code, :name, :cap)
        ON DUPLICATE KEY UPDATE
          name=VALUES(name),
          float_cap_billion=COALESCE(VALUES(float_cap_billion), float_cap_billion)
        """
    )
    batch = [{"code": code, "name": name, "cap": cap} for code, name, cap in rows]
    if not batch:
        return
    with engine.begin() as conn:
        conn.execute(stmt, batch)


def upsert_daily(engine: Engine, stock_code: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    stmt = text(
        """
        INSERT INTO stock_daily(stock_code, date, open, high, low, close, volume, amount)
        VALUES(:stock_code, :date, :open, :high, :low, :close, :volume, :amount)
        ON CONFLICT(stock_code, date) DO UPDATE SET
          open=excluded.open,
          high=excluded.high,
          low=excluded.low,
          close=excluded.close,
          volume=excluded.volume,
          amount=excluded.amount
        """
        if engine.dialect.name == "sqlite"
        else """
        INSERT INTO stock_daily(stock_code, date, open, high, low, close, volume, amount)
        VALUES(:stock_code, :date, :open, :high, :low, :close, :volume, :amount)
        ON DUPLICATE KEY UPDATE
          open=VALUES(open),
          high=VALUES(high),
          low=VALUES(low),
          close=VALUES(close),
          volume=VALUES(volume),
          amount=VALUES(amount)
        """
    )
    records = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "stock_code": stock_code,
                "date": str(row.date),
                "open": float(row.open) if pd.notna(row.open) else None,
                "high": float(row.high) if pd.notna(row.high) else None,
                "low": float(row.low) if pd.notna(row.low) else None,
                "close": float(row.close) if pd.notna(row.close) else None,
                "volume": float(row.volume) if pd.notna(row.volume) else None,
                "amount": float(row.amount) if pd.notna(row.amount) else None,
            }
        )
    with engine.begin() as conn:
        conn.execute(stmt, records)
    return len(records)


def parse_date(s: str, *, default: Optional[str] = None) -> str:
    v = (s or default or "").strip()
    if not v:
        raise ValueError("date is required")
    if len(v) == 8 and v.isdigit():
        return f"{v[0:4]}{v[4:6]}{v[6:8]}"
    try:
        return dt.datetime.strptime(v, "%Y-%m-%d").strftime("%Y%m%d")
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date: {s}") from exc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch A-share daily data (163 source) into DB")
    parser.add_argument("--config", default=None, help="config.ini path")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL")
    parser.add_argument("--db", default=None, help="SQLite file path")
    parser.add_argument("--start-date", default="20000101", help="YYYYMMDD")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y%m%d"), help="YYYYMMDD")
    parser.add_argument("--workers", type=int, default=16, help="worker threads")
    parser.add_argument("--limit-stocks", type=int, default=None, help="debug: only first N stocks")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    if not hasattr(ak, "stock_zh_a_hist_tx"):
        raise SystemExit("AkShare missing stock_zh_a_hist_tx; please upgrade akshare or switch data source.")

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if start > end:
        raise SystemExit("start-date must be <= end-date")

    db_target = resolve_db_target(args)
    workers = max(1, int(args.workers))
    engine = make_engine(db_target, workers)
    ensure_core_tables(engine)

    stocks = fetch_stock_list()
    if args.limit_stocks:
        stocks = stocks[: int(args.limit_stocks)]
    upsert_stock_info(engine, stocks)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    progress = tqdm(total=len(stocks), desc="K update", unit="stock")

    def worker(code: str, name: str) -> None:
        latest = get_latest_date(engine, code)
        fetch_start = start
        if latest:
            next_day = (pd.to_datetime(latest) + pd.Timedelta(days=1)).strftime("%Y%m%d")
            fetch_start = max(fetch_start, next_day)
        if fetch_start > end:
            return
        df = fetch_daily(code, fetch_start, end)
        if latest:
            df = df[df["date"] > latest]
        inserted = upsert_daily(engine, code, df)
        if inserted:
            logging.info("%s(%s) +%d rows", code, name, inserted)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, code, name): code for code, name, _cap in stocks}
        for fut in as_completed(futs):
            _code = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                logging.exception("update failed %s: %s", _code, exc)
            finally:
                progress.update(1)
    progress.close()
    logging.info("K update finished: stocks=%d start=%s end=%s", len(stocks), start, end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
