#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
getDataBaoStock.py

唯一职责：从 BaoStock 获取 A 股 K 线数据并写入数据库。
- 支持 config.ini / 环境变量 / CLI 指定 DB
- 多线程按股票拉取，带 tqdm 进度条
- start-date / end-date 控制抓取窗口
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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import baostock as bs
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError


DEFAULT_DB = "data/stock.db"
MINUTE_TABLE = "stock_minute"
BAOSTOCK_API_LOCK = threading.Lock()
BAOSTOCK_API_MIN_INTERVAL_SEC = 0.0
_BAOSTOCK_LAST_CALL_MONO = 0.0


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


def ensure_minute_table(engine: Engine) -> None:
    ddl = f"""
        CREATE TABLE IF NOT EXISTS {MINUTE_TABLE} (
            stock_code VARCHAR(16) NOT NULL,
            date VARCHAR(10) NOT NULL,
            time VARCHAR(8) NOT NULL,
            open DOUBLE NULL,
            high DOUBLE NULL,
            low DOUBLE NULL,
            close DOUBLE NULL,
            volume DOUBLE NULL,
            amount DOUBLE NULL,
            frequency VARCHAR(8) NOT NULL,
            PRIMARY KEY (stock_code, date, time, frequency)
        )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def _ensure_stock_info_float_cap(engine: Engine) -> None:
    stmt = "ALTER TABLE stock_info ADD COLUMN float_cap_billion DOUBLE NULL"
    if engine.dialect.name == "sqlite":
        stmt = "ALTER TABLE stock_info ADD COLUMN float_cap_billion DOUBLE"
    try:
        with engine.begin() as conn:
            conn.execute(text(stmt))
            logging.info("[getData] stock_info 新增 float_cap_billion 列")
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


def _is_stock_type(raw: object) -> bool:
    if raw is None:
        return False
    text = str(raw).strip().lower()
    if not text or text in {"nan", "none"}:
        return False
    try:
        return int(float(text)) == 1
    except (TypeError, ValueError):
        return text in {"1", "stock"}


def _strip_bs_prefix(raw: object) -> Tuple[str, Optional[str]]:
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"}:
        return "", None
    if "." in text:
        exch, code = text.split(".", 1)
        exch = exch.lower().strip()
        code = code.strip()
        if code.isdigit() and len(code) < 6:
            code = code.zfill(6)
        return code, exch
    code = text
    if code.isdigit() and len(code) < 6:
        code = code.zfill(6)
    if code.startswith(("6", "9")):
        return code, "sh"
    if code.startswith(("0", "2", "3")):
        return code, "sz"
    return code, None


def _to_bs_symbol(raw: object) -> Optional[str]:
    code, exch = _strip_bs_prefix(raw)
    if not code or not exch:
        return None
    if exch not in {"sh", "sz"}:
        return None
    return f"{exch}.{code}"


def _ymd_dash(value: str) -> str:
    v = str(value).strip()
    if len(v) == 8 and v.isdigit():
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    return v


def _normalize_minute_time(value: object) -> Optional[str]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    hh = mm = ss = ""
    if len(digits) >= 14:
        hh, mm, ss = digits[8:10], digits[10:12], digits[12:14]
    elif len(digits) == 6:
        hh, mm, ss = digits[0:2], digits[2:4], digits[4:6]
    elif len(digits) == 4:
        hh, mm, ss = digits[0:2], digits[2:4], "00"
    elif len(text) >= 8 and text[2:3] == ":" and text[5:6] == ":":
        hh, mm, ss = text[0:2], text[3:5], text[6:8]
    if not (hh.isdigit() and mm.isdigit() and ss.isdigit()):
        return None
    if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59 and 0 <= int(ss) <= 59):
        return None
    return f"{hh}:{mm}:{ss}"


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "volume", "amount"]
    )


def _empty_minute_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "time", "open", "high", "low", "close", "volume", "amount"]
    )


def _throttle_baostock_call_locked() -> None:
    global _BAOSTOCK_LAST_CALL_MONO
    interval = float(BAOSTOCK_API_MIN_INTERVAL_SEC)
    if interval <= 0:
        return
    now = time.monotonic()
    wait = interval - (now - _BAOSTOCK_LAST_CALL_MONO)
    if wait > 0:
        time.sleep(wait)
    _BAOSTOCK_LAST_CALL_MONO = time.monotonic()


def _bs_rows_to_df(rs, label: str) -> pd.DataFrame:
    if rs is None:
        raise RuntimeError(f"BaoStock {label} failed: empty result")
    if getattr(rs, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock {label} failed: {getattr(rs, 'error_msg', '')}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if getattr(rs, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock {label} failed: {getattr(rs, 'error_msg', '')}")
    return pd.DataFrame(rows, columns=getattr(rs, "fields", None))


def fetch_stock_list() -> List[Tuple[str, str, Optional[float]]]:
    try:
        with BAOSTOCK_API_LOCK:
            _throttle_baostock_call_locked()
            rs = bs.query_stock_basic()
            df = _bs_rows_to_df(rs, "stock list")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"BaoStock stock list failed: {exc}") from exc
    if df is None or df.empty:
        raise RuntimeError("BaoStock returned empty stock list")
    code_col = "code" if "code" in df.columns else None
    name_col = "code_name" if "code_name" in df.columns else None
    type_col = "type" if "type" in df.columns else None
    if not code_col:
        raise RuntimeError(f"BaoStock stock list missing code column: {list(df.columns)}")
    out: List[Tuple[str, str, Optional[float]]] = []
    for _, row in df.iterrows():
        if type_col and not _is_stock_type(row[type_col]):
            continue
        code, exch = _strip_bs_prefix(row[code_col])
        if not code or exch not in {"sh", "sz"}:
            continue
        name = str(row[name_col]).strip() if name_col else ""
        out.append((code, name, None))
    if not out:
        raise RuntimeError("BaoStock returned empty A-share list")
    return out


def fetch_daily(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    symbol = _to_bs_symbol(stock_code)
    if not symbol:
        logging.debug("BaoStock skip unsupported code: %s", stock_code)
        return _empty_daily_frame()
    bs_start = _ymd_dash(start_date)
    bs_end = _ymd_dash(end_date)
    try:
        with BAOSTOCK_API_LOCK:
            _throttle_baostock_call_locked()
            rs = bs.query_history_k_data_plus(
                symbol,
                "date,open,high,low,close,volume,amount",
                start_date=bs_start,
                end_date=bs_end,
                frequency="d",
                adjustflag="2",
            )
            if rs.error_code != "0":
                logging.warning("BaoStock hist failed %s: %s", stock_code, rs.error_msg)
                return _empty_daily_frame()
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != "0":
                logging.warning("BaoStock hist failed %s: %s", stock_code, rs.error_msg)
                return _empty_daily_frame()
    except Exception as exc:  # noqa: BLE001
        logging.warning("BaoStock hist failed %s: %s", stock_code, exc)
        return _empty_daily_frame()
    if not rows:
        return _empty_daily_frame()
    raw = pd.DataFrame(rows, columns=rs.fields)
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    for col in cols:
        if col not in raw.columns:
            raw[col] = pd.NA
    df = raw[cols].copy()
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    return df


def _sleep_retry(attempt: int, base: float) -> None:
    if base <= 0:
        return
    delay = min(10.0, base * (2 ** (attempt - 1)))
    time.sleep(delay)


def fetch_minute(
    stock_code: str,
    start_date: str,
    end_date: str,
    frequency: str,
    retries: int = 3,
    retry_sleep: float = 1.0,
) -> pd.DataFrame:
    symbol = _to_bs_symbol(stock_code)
    if not symbol:
        logging.debug("BaoStock skip unsupported code: %s", stock_code)
        return _empty_minute_frame()
    bs_start = _ymd_dash(start_date)
    bs_end = _ymd_dash(end_date)
    max_retries = max(0, int(retries))
    rows: List[List[str]] = []
    fields: List[str] = []
    for attempt in range(max_retries + 1):
        err_msg: Optional[str] = None
        try:
            with BAOSTOCK_API_LOCK:
                _throttle_baostock_call_locked()
                rs = bs.query_history_k_data_plus(
                    symbol,
                    "date,time,open,high,low,close,volume,amount",
                    start_date=bs_start,
                    end_date=bs_end,
                    frequency=frequency,
                    adjustflag="2",
                )
                if rs.error_code != "0":
                    err_msg = str(rs.error_msg or rs.error_code)
                else:
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rs.error_code != "0":
                        err_msg = str(rs.error_msg or rs.error_code)
                    else:
                        fields = list(getattr(rs, "fields", []) or [])
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
        if err_msg:
            if attempt < max_retries:
                logging.warning(
                    "BaoStock minute failed %s: %s (retry %d/%d)",
                    stock_code,
                    err_msg,
                    attempt + 1,
                    max_retries,
                )
                _sleep_retry(attempt + 1, retry_sleep)
                continue
            logging.warning("BaoStock minute failed %s: %s", stock_code, err_msg)
            return _empty_minute_frame()
        break
    if not rows:
        return _empty_minute_frame()
    raw = pd.DataFrame(rows, columns=fields)
    cols = ["date", "time", "open", "high", "low", "close", "volume", "amount"]
    for col in cols:
        if col not in raw.columns:
            raw[col] = pd.NA
    df = raw[cols].copy()
    if "time" not in df.columns or df["time"].isna().all():
        if "date" in df.columns:
            parts = df["date"].astype(str).str.split(" ", n=1, expand=True)
            if parts.shape[1] >= 2:
                df["date"] = parts[0]
                df["time"] = parts[1]
    for col in cols[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["time"] = df["time"].apply(_normalize_minute_time)
    df = df.dropna(subset=["date", "time"])
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


def get_latest_dates(engine: Engine) -> dict[str, str]:
    sql = text("SELECT stock_code, MAX(date) AS latest FROM stock_daily GROUP BY stock_code")
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    return {str(code): str(latest) for code, latest in rows if code and latest}


def get_latest_minute_date(engine: Engine, stock_code: str, frequency: str) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                f"SELECT MAX(date) FROM {MINUTE_TABLE} WHERE stock_code = :c AND frequency = :f"
            ),
            {"c": stock_code, "f": frequency},
        ).fetchone()
    if not row or not row[0]:
        return None
    return str(row[0])


def get_latest_minute_dates(engine: Engine, frequency: str) -> dict[str, str]:
    sql = text(
        f"""
        SELECT stock_code, MAX(date) AS latest
        FROM {MINUTE_TABLE}
        WHERE frequency = :f
        GROUP BY stock_code
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"f": frequency}).fetchall()
    return {str(code): str(latest) for code, latest in rows if code and latest}


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


def upsert_minute(engine: Engine, stock_code: str, df: pd.DataFrame, frequency: str) -> int:
    if df is None or df.empty:
        return 0
    stmt = text(
        f"""
        INSERT INTO {MINUTE_TABLE}(stock_code, date, time, open, high, low, close, volume, amount, frequency)
        VALUES(:stock_code, :date, :time, :open, :high, :low, :close, :volume, :amount, :frequency)
        ON CONFLICT(stock_code, date, time, frequency) DO UPDATE SET
          open=excluded.open,
          high=excluded.high,
          low=excluded.low,
          close=excluded.close,
          volume=excluded.volume,
          amount=excluded.amount
        """
        if engine.dialect.name == "sqlite"
        else f"""
        INSERT INTO {MINUTE_TABLE}(stock_code, date, time, open, high, low, close, volume, amount, frequency)
        VALUES(:stock_code, :date, :time, :open, :high, :low, :close, :volume, :amount, :frequency)
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
                "time": str(row.time),
                "open": float(row.open) if pd.notna(row.open) else None,
                "high": float(row.high) if pd.notna(row.high) else None,
                "low": float(row.low) if pd.notna(row.low) else None,
                "close": float(row.close) if pd.notna(row.close) else None,
                "volume": float(row.volume) if pd.notna(row.volume) else None,
                "amount": float(row.amount) if pd.notna(row.amount) else None,
                "frequency": str(frequency),
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


def _shift_yyyymmdd(date_str: str, days: int) -> str:
    base = parse_date(date_str)
    d = dt.datetime.strptime(base, "%Y%m%d").date()
    return (d + dt.timedelta(days=int(days))).strftime("%Y%m%d")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 A 股 K 线数据写入数据库")
    parser.add_argument("--config", default=None, help="config.ini 路径")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL")
    parser.add_argument("--db", default=None, help="SQLite 文件路径")
    parser.add_argument("--start-date", default="20000101", help="YYYYMMDD")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y%m%d"), help="YYYYMMDD")
    parser.add_argument("--frequency", default="d", help="K线频率：d/5/15/30/60（分钟）")
    parser.add_argument("--minute-lookback-days", type=int, default=None, help="分钟线回测范围（仅frequency!=d）")
    parser.add_argument("--minute-backfill-days", type=int, default=3, help="分钟线增量回补窗口天数（仅frequency!=d）")
    parser.add_argument("--minute-retries", type=int, default=3, help="分钟线拉取失败重试次数（仅frequency!=d）")
    parser.add_argument("--minute-retry-sleep", type=float, default=1.0, help="分钟线重试基础等待秒（指数退避，仅frequency!=d）")
    parser.add_argument("--workers", type=int, default=16, help="线程数")
    parser.add_argument("--api-min-interval", type=float, default=0.0, help="BaoStock requests minimum interval seconds")
    parser.add_argument("--shard-total", type=int, default=1, help="Total shards for multi-process split")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index, 0-based")
    parser.add_argument("--limit-stocks", type=int, default=None, help="调试用，仅处理前 N 只股票")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    today_yyyymmdd = dt.date.today().strftime("%Y%m%d")
    historical_closed_range = end < today_yyyymmdd
    if start > end:
        raise SystemExit("start-date must be <= end-date")

    db_target = resolve_db_target(args)
    workers = max(1, int(args.workers))
    engine = make_engine(db_target, workers)
    frequency = str(args.frequency or "d").strip()
    allowed_freq = {"d", "5", "15", "30", "60"}
    if frequency not in allowed_freq:
        raise SystemExit("frequency 仅支持 d/5/15/30/60（BaoStock 不支持 1 分钟）")
    minute_lookback_days = args.minute_lookback_days
    minute_backfill_days = args.minute_backfill_days
    if minute_backfill_days is None:
        minute_backfill_days = 3
    minute_backfill_days = max(0, int(minute_backfill_days))
    minute_retries = args.minute_retries
    if minute_retries is None:
        minute_retries = 3
    minute_retries = max(0, int(minute_retries))
    minute_retry_sleep = args.minute_retry_sleep
    if minute_retry_sleep is None:
        minute_retry_sleep = 1.0
    minute_retry_sleep = max(0.0, float(minute_retry_sleep))
    api_min_interval = max(0.0, float(args.api_min_interval or 0.0))
    shard_total = max(1, int(args.shard_total or 1))
    shard_index = int(args.shard_index or 0)
    if shard_index < 0 or shard_index >= shard_total:
        raise SystemExit(f"shard-index must be in [0, {shard_total - 1}]")
    global BAOSTOCK_API_MIN_INTERVAL_SEC
    BAOSTOCK_API_MIN_INTERVAL_SEC = api_min_interval
    if frequency != "d" and minute_lookback_days is not None:
        minute_lookback_days = int(minute_lookback_days)
        if minute_lookback_days > 0:
            start = max(start, _shift_yyyymmdd(end, -(minute_lookback_days - 1)))
    ensure_core_tables(engine)
    if frequency != "d":
        ensure_minute_table(engine)
        if workers > 8:
            logging.warning(
                "minute frequency uses a serialized BaoStock socket; workers=%d has little gain, consider --workers 1-4",
                workers,
            )

    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"BaoStock login failed: {lg.error_msg}")

    stocks = fetch_stock_list()
    if shard_total > 1:
        stocks = [s for i, s in enumerate(stocks) if i % shard_total == shard_index]
        logging.info("shard mode: %d/%d, stocks=%d", shard_index, shard_total, len(stocks))
    if args.limit_stocks:
        stocks = stocks[: int(args.limit_stocks)]
    upsert_stock_info(engine, stocks)
    latest_daily_map: dict[str, str] = {}
    latest_minute_map: dict[str, str] = {}
    if frequency == "d":
        latest_daily_map = get_latest_dates(engine)
    else:
        latest_minute_map = get_latest_minute_dates(engine, frequency)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    progress = tqdm(total=len(stocks), desc="K线更新", unit="stock")

    def worker(code: str, name: str) -> None:
        if frequency == "d":
            latest = latest_daily_map.get(code)
            fetch_start = start
            if latest:
                next_day = (pd.to_datetime(latest) + pd.Timedelta(days=1)).strftime("%Y%m%d")
                fetch_start = max(fetch_start, next_day)
            if fetch_start > end:
                return
            df = fetch_daily(code, fetch_start, end)
            if df.empty or "date" not in df.columns:
                return
            if latest:
                df = df[df["date"] > latest]
                if df.empty:
                    return
            inserted = upsert_daily(engine, code, df)
            if inserted:
                logging.info("%s(%s) +%d 行", code, name, inserted)
        else:
            latest = latest_minute_map.get(code)
            if latest and historical_closed_range and parse_date(latest) >= end:
                return
            fetch_start = start
            if latest:
                if minute_backfill_days > 0:
                    backfill_start = _shift_yyyymmdd(latest, -(minute_backfill_days - 1))
                    fetch_start = max(fetch_start, backfill_start)
                else:
                    next_day = _shift_yyyymmdd(latest, 1)
                    fetch_start = max(fetch_start, next_day)
            if fetch_start > end:
                return
            df = fetch_minute(code, fetch_start, end, frequency, minute_retries, minute_retry_sleep)
            if df.empty or "date" not in df.columns:
                return
            inserted = upsert_minute(engine, code, df, frequency)
            if inserted:
                logging.info("%s(%s) +%d 行(min-%s)", code, name, inserted, frequency)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, code, name): code for code, name, _cap in stocks}
        for fut in as_completed(futs):
            _code = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                logging.exception("更新 %s 失败: %s", _code, exc)
            finally:
                progress.update(1)
    progress.close()
    bs.logout()
    logging.info(
        "K 线更新完成：freq=%s stocks=%d start=%s end=%s lookback=%s backfill=%s",
        frequency,
        len(stocks),
        start,
        end,
        minute_lookback_days if frequency != "d" else "-",
        minute_backfill_days if frequency != "d" else "-",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
