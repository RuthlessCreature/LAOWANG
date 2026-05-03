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
import os
import socket
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import baostock as bs
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError


DEFAULT_DB = "data/stock.db"
MINUTE_TABLE = "stock_minute"
INGEST_WATERMARK_TABLE = "stock_ingest_watermark"
DEFAULT_UPSERT_CHUNK_SIZE = 5000
DEFAULT_BAOSTOCK_PROXY = "auto"
DEFAULT_BAOSTOCK_CONNECT_TIMEOUT = 8.0
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
    ensure_ingest_watermark_table(engine)


def ensure_ingest_watermark_table(engine: Engine) -> None:
    ddl = f"""
        CREATE TABLE IF NOT EXISTS {INGEST_WATERMARK_TABLE} (
            stock_code VARCHAR(16) NOT NULL,
            frequency VARCHAR(8) NOT NULL,
            latest_date VARCHAR(10) NOT NULL,
            updated_at VARCHAR(19) NULL,
            PRIMARY KEY (stock_code, frequency)
        )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def ensure_minute_table(engine: Engine) -> None:
    if engine.dialect.name == "mysql":
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {MINUTE_TABLE} (
            stock_code CHAR(6) NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            open DOUBLE NULL,
            high DOUBLE NULL,
            low DOUBLE NULL,
            close DOUBLE NULL,
            volume DOUBLE NULL,
            amount DOUBLE NULL,
            frequency TINYINT UNSIGNED NOT NULL,
            PRIMARY KEY (stock_code, frequency, date, time),
            KEY idx_stock_minute_freq_date_time (frequency, date, time)
        )
        """
    else:
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
    ensure_ingest_watermark_table(engine)


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


def _is_baostock_login_error(message: object) -> bool:
    text_value = str(message or "").lower()
    return any(
        token in text_value
        for token in ("\u7528\u6237\u672a\u767b\u5f55", "\u672a\u767b\u5f55", "no login", "not login")
    )


def _is_baostock_connection_error(message: object) -> bool:
    text_value = str(message or "").lower()
    return any(
        token in text_value
        for token in ("timed out", "timeout", "socket", "connection", "\u8fdc\u7a0b\u4e3b\u673a")
    )


def _baostock_relogin_locked() -> None:
    try:
        bs.logout()
    except Exception:
        pass
    lg = bs.login()
    if getattr(lg, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock relogin failed: {getattr(lg, 'error_msg', '')}")
    logging.info("[getData] BaoStock relogin success")


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
    rows = []
    fields = []
    for attempt in range(2):
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
                    if attempt == 0 and _is_baostock_login_error(rs.error_msg):
                        logging.warning("BaoStock hist login expired %s: %s; relogin", stock_code, rs.error_msg)
                        _baostock_relogin_locked()
                        continue
                    logging.warning("BaoStock hist failed %s: %s", stock_code, rs.error_msg)
                    return _empty_daily_frame()
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rs.error_code != "0":
                    if attempt == 0 and _is_baostock_login_error(rs.error_msg):
                        logging.warning("BaoStock hist login expired %s: %s; relogin", stock_code, rs.error_msg)
                        _baostock_relogin_locked()
                        continue
                    logging.warning("BaoStock hist failed %s: %s", stock_code, rs.error_msg)
                    return _empty_daily_frame()
                fields = list(getattr(rs, "fields", []) or [])
        except Exception as exc:  # noqa: BLE001
            if attempt == 0 and _is_baostock_connection_error(exc):
                try:
                    with BAOSTOCK_API_LOCK:
                        _baostock_relogin_locked()
                    logging.warning("BaoStock hist connection reset %s: %s; relogin", stock_code, exc)
                    continue
                except Exception as relogin_exc:  # noqa: BLE001
                    logging.warning("BaoStock hist relogin failed %s: %s", stock_code, relogin_exc)
            logging.warning("BaoStock hist failed %s: %s", stock_code, exc)
            return _empty_daily_frame()
        break
    if not rows:
        return _empty_daily_frame()
    raw = pd.DataFrame(rows, columns=fields)
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


def _connect_direct(host: str, port: int, timeout: float) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(float(timeout))
    try:
        sock.connect((host, int(port)))
        sock.settimeout(max(15.0, min(60.0, float(timeout) * 3.0)))
        return sock
    except Exception:
        sock.close()
        raise


def _connect_http_proxy(proxy_url: str, host: str, port: int, timeout: float) -> socket.socket:
    parsed = urlparse(proxy_url if "://" in proxy_url else f"http://{proxy_url}")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported BaoStock proxy scheme: {parsed.scheme}")
    proxy_host = parsed.hostname
    proxy_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not proxy_host:
        raise ValueError(f"Invalid BaoStock proxy: {proxy_url}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(float(timeout))
    try:
        sock.connect((proxy_host, int(proxy_port)))
        target = f"{host}:{int(port)}"
        request = f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode("ascii")
        sock.sendall(request)
        response = b""
        while b"\r\n\r\n" not in response and len(response) < 8192:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
        first_line = response.split(b"\r\n", 1)[0].decode("latin1", errors="replace")
        if " 200 " not in f" {first_line} " and not first_line.endswith(" 200"):
            raise OSError(f"BaoStock proxy CONNECT failed: {first_line}")
        sock.settimeout(max(15.0, min(60.0, float(timeout) * 3.0)))
        return sock
    except Exception:
        sock.close()
        raise


def _baostock_auto_proxy_candidates() -> List[str]:
    candidates: List[str] = []
    for key in ("BAOSTOCK_PROXY", "TG_BOT_PROXY"):
        raw = os.getenv(key)
        if raw and raw.strip() and raw.strip().lower() not in {"none", "off", "false", "0"}:
            candidates.append(raw.strip())
    candidates.append("http://127.0.0.1:7890")
    out: List[str] = []
    for item in candidates:
        if item not in out:
            out.append(item)
    return out


def configure_baostock_transport(proxy: str, timeout: float) -> str:
    import baostock.common.contants as cons
    import baostock.common.context as context
    import baostock.util.socketutil as sockutil

    host = str(getattr(cons, "BAOSTOCK_SERVER_IP", "www.baostock.com"))
    port = int(getattr(cons, "BAOSTOCK_SERVER_PORT", 10030))
    timeout = max(1.0, float(timeout or DEFAULT_BAOSTOCK_CONNECT_TIMEOUT))
    proxy_value = str(proxy or DEFAULT_BAOSTOCK_PROXY).strip()
    proxy_lower = proxy_value.lower()

    attempts: List[Tuple[str, Optional[str]]] = []
    if proxy_lower in {"", "none", "off", "false", "0", "direct"}:
        attempts.append(("direct", None))
    elif proxy_lower == "auto":
        attempts.append(("direct", None))
        attempts.extend(("proxy", candidate) for candidate in _baostock_auto_proxy_candidates())
    else:
        attempts.append(("proxy", proxy_value))

    last_error: Optional[Exception] = None
    selected: Optional[Tuple[str, Optional[str]]] = None
    for mode, value in attempts:
        try:
            if mode == "direct":
                test_sock = _connect_direct(host, port, timeout)
            else:
                test_sock = _connect_http_proxy(str(value), host, port, timeout)
            test_sock.close()
            selected = (mode, value)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            label = "direct" if mode == "direct" else f"proxy {value}"
            logging.warning("[getData] BaoStock %s connect failed: %s", label, exc)

    if selected is None:
        detail = f"{type(last_error).__name__}: {last_error}" if last_error else "no route"
        raise SystemExit(f"BaoStock connect failed: {host}:{port} ({detail})")

    mode, value = selected

    def patched_connect(self):  # noqa: ANN001
        try:
            if mode == "direct":
                default_socket = _connect_direct(host, port, timeout)
            else:
                default_socket = _connect_http_proxy(str(value), host, port, timeout)
        except Exception as exc:  # noqa: BLE001
            setattr(context, "default_socket", None)
            raise RuntimeError(f"BaoStock socket connect failed via {mode}: {exc}") from exc
        setattr(context, "default_socket", default_socket)

    sockutil.SocketUtil.connect = patched_connect  # type: ignore[assignment]
    if hasattr(context, "default_socket") and getattr(context, "default_socket") is not None:
        try:
            getattr(context, "default_socket").close()
        except Exception:
            pass
        setattr(context, "default_socket", None)
    route = "direct" if mode == "direct" else f"proxy {value}"
    logging.info("[getData] BaoStock route: %s -> %s:%d", route, host, port)
    return route


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
                    if _is_baostock_login_error(err_msg):
                        _baostock_relogin_locked()
                else:
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rs.error_code != "0":
                        err_msg = str(rs.error_msg or rs.error_code)
                        if _is_baostock_login_error(err_msg):
                            _baostock_relogin_locked()
                    else:
                        fields = list(getattr(rs, "fields", []) or [])
        except Exception as exc:  # noqa: BLE001
            err_msg = str(exc)
            if _is_baostock_connection_error(err_msg):
                try:
                    with BAOSTOCK_API_LOCK:
                        _baostock_relogin_locked()
                except Exception as relogin_exc:  # noqa: BLE001
                    err_msg = f"{err_msg}; relogin failed: {relogin_exc}"
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


def _dedupe_codes(stock_codes: Optional[Iterable[str]]) -> List[str]:
    if not stock_codes:
        return []
    return [str(code) for code in dict.fromkeys(str(c).strip() for c in stock_codes) if code]


def _latest_date_text(value: object) -> Optional[str]:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        pass
    text_value = str(value).strip()
    return text_value or None


def get_latest_dates(engine: Engine, stock_codes: Optional[Iterable[str]] = None) -> dict[str, str]:
    codes = _dedupe_codes(stock_codes)
    if codes:
        sql = text(
            """
            SELECT stock_code, MAX(date) AS latest
            FROM stock_daily
            WHERE stock_code IN :codes
            GROUP BY stock_code
            """
        ).bindparams(bindparam("codes", expanding=True))
        params = {"codes": codes}
    else:
        sql = text("SELECT stock_code, MAX(date) AS latest FROM stock_daily GROUP BY stock_code")
        params = {}
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return {
        str(code): latest_text
        for code, latest in rows
        if code and (latest_text := _latest_date_text(latest))
    }


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


def get_minute_frequencies(engine: Engine) -> List[str]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"SELECT DISTINCT frequency FROM {MINUTE_TABLE}")).fetchall()
    except SQLAlchemyError:
        return []
    return sorted({str(row[0]) for row in rows if row and row[0] is not None})


def get_latest_minute_dates(
    engine: Engine,
    frequency: str,
    stock_codes: Optional[Iterable[str]] = None,
) -> dict[str, str]:
    codes = _dedupe_codes(stock_codes)
    params: dict[str, object] = {"f": str(frequency)}
    minute_freqs = get_minute_frequencies(engine)
    single_frequency_fast_path = len(minute_freqs) == 1 and minute_freqs[0] == str(frequency)
    where_parts: List[str] = []
    if not single_frequency_fast_path:
        where_parts.append("frequency = :f")
    if codes:
        where_parts.append("stock_code IN :codes")
        params["codes"] = codes
    where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    sql = text(
        f"""
        SELECT stock_code, MAX(date) AS latest
        FROM {MINUTE_TABLE}
        {where_sql}
        GROUP BY stock_code
        """
    )
    if codes:
        sql = sql.bindparams(bindparam("codes", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return {
        str(code): latest_text
        for code, latest in rows
        if code and (latest_text := _latest_date_text(latest))
    }


def get_watermark_dates(
    engine: Engine,
    frequency: str,
    stock_codes: Optional[Iterable[str]] = None,
) -> dict[str, str]:
    codes = _dedupe_codes(stock_codes)
    ensure_ingest_watermark_table(engine)
    params: dict[str, object] = {"f": str(frequency)}
    where = "frequency = :f"
    sql = text(
        f"""
        SELECT stock_code, latest_date
        FROM {INGEST_WATERMARK_TABLE}
        WHERE {where}
        """
    )
    if codes:
        sql = text(
            f"""
            SELECT stock_code, latest_date
            FROM {INGEST_WATERMARK_TABLE}
            WHERE {where} AND stock_code IN :codes
            """
        ).bindparams(bindparam("codes", expanding=True))
        params["codes"] = codes
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return {
        str(code): latest_text
        for code, latest in rows
        if code and (latest_text := _latest_date_text(latest))
    }


def upsert_watermarks(
    engine: Engine,
    frequency: str,
    rows: Iterable[Tuple[str, str]],
    *,
    chunk_size: int = DEFAULT_UPSERT_CHUNK_SIZE,
) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch = [
        {"stock_code": str(code), "frequency": str(frequency), "latest_date": str(latest), "updated_at": now}
        for code, latest in rows
        if code and latest
    ]
    if not batch:
        return
    stmt = text(
        f"""
        INSERT INTO {INGEST_WATERMARK_TABLE}(stock_code, frequency, latest_date, updated_at)
        VALUES(:stock_code, :frequency, :latest_date, :updated_at)
        ON CONFLICT(stock_code, frequency) DO UPDATE SET
          latest_date=CASE
            WHEN excluded.latest_date > latest_date THEN excluded.latest_date
            ELSE latest_date
          END,
          updated_at=excluded.updated_at
        """
        if engine.dialect.name == "sqlite"
        else f"""
        INSERT INTO {INGEST_WATERMARK_TABLE}(stock_code, frequency, latest_date, updated_at)
        VALUES(:stock_code, :frequency, :latest_date, :updated_at)
        ON DUPLICATE KEY UPDATE
          latest_date=GREATEST(latest_date, VALUES(latest_date)),
          updated_at=VALUES(updated_at)
        """
    )
    with engine.begin() as conn:
        _execute_in_chunks(conn, stmt, batch, chunk_size)


def load_latest_date_map(
    engine: Engine,
    frequency: str,
    stock_codes: Iterable[str],
    *,
    chunk_size: int = DEFAULT_UPSERT_CHUNK_SIZE,
) -> dict[str, str]:
    codes = _dedupe_codes(stock_codes)
    if not codes:
        return {}
    latest = get_watermark_dates(engine, frequency, codes)
    missing = [code for code in codes if code not in latest]
    if missing:
        if str(frequency) == "d":
            fallback = get_latest_dates(engine, missing)
        else:
            fallback = get_latest_minute_dates(engine, str(frequency), missing)
        if fallback:
            upsert_watermarks(engine, str(frequency), fallback.items(), chunk_size=chunk_size)
            latest.update(fallback)
    return latest


def _execute_in_chunks(conn, stmt, records: List[dict], chunk_size: int) -> None:  # noqa: ANN001
    size = max(1, int(chunk_size or DEFAULT_UPSERT_CHUNK_SIZE))
    for start in range(0, len(records), size):
        conn.execute(stmt, records[start : start + size])


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


def upsert_daily(
    engine: Engine,
    stock_code: str,
    df: pd.DataFrame,
    *,
    chunk_size: int = DEFAULT_UPSERT_CHUNK_SIZE,
) -> int:
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
        _execute_in_chunks(conn, stmt, records, chunk_size)
    return len(records)


def upsert_minute(
    engine: Engine,
    stock_code: str,
    df: pd.DataFrame,
    frequency: str,
    *,
    chunk_size: int = DEFAULT_UPSERT_CHUNK_SIZE,
) -> int:
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
        _execute_in_chunks(conn, stmt, records, chunk_size)
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


def _append_cli_arg(cli: List[str], flag: str, value: object) -> None:
    if value is not None:
        cli.extend([flag, str(value)])


def _build_shard_cli(args: argparse.Namespace, shard_total: int, shard_index: int) -> List[str]:
    cli: List[str] = []
    _append_cli_arg(cli, "--config", getattr(args, "config", None))
    _append_cli_arg(cli, "--db-url", getattr(args, "db_url", None))
    _append_cli_arg(cli, "--db", getattr(args, "db", None))
    _append_cli_arg(cli, "--start-date", getattr(args, "start_date", None))
    _append_cli_arg(cli, "--end-date", getattr(args, "end_date", None))
    _append_cli_arg(cli, "--frequency", getattr(args, "frequency", None))
    _append_cli_arg(cli, "--minute-lookback-days", getattr(args, "minute_lookback_days", None))
    _append_cli_arg(cli, "--minute-backfill-days", getattr(args, "minute_backfill_days", None))
    _append_cli_arg(cli, "--minute-retries", getattr(args, "minute_retries", None))
    _append_cli_arg(cli, "--minute-retry-sleep", getattr(args, "minute_retry_sleep", None))
    _append_cli_arg(cli, "--workers", getattr(args, "workers", None))
    _append_cli_arg(cli, "--api-min-interval", getattr(args, "api_min_interval", None))
    _append_cli_arg(cli, "--upsert-chunk-size", getattr(args, "upsert_chunk_size", None))
    _append_cli_arg(cli, "--baostock-proxy", getattr(args, "baostock_proxy", None))
    _append_cli_arg(cli, "--baostock-connect-timeout", getattr(args, "baostock_connect_timeout", None))
    _append_cli_arg(cli, "--limit-stocks", getattr(args, "limit_stocks", None))
    _append_cli_arg(cli, "--log-level", getattr(args, "log_level", None))
    cli.extend(["--shard-total", str(shard_total), "--shard-index", str(shard_index), "--process-shards", "1"])
    return cli


def run_process_shards(args: argparse.Namespace) -> int:
    shard_count = max(1, int(getattr(args, "process_shards", 1) or 1))
    if shard_count <= 1:
        return 0
    script = Path(__file__).resolve()
    procs: List[subprocess.Popen] = []
    logging.info("[getData] spawning %d BaoStock shard processes", shard_count)
    for idx in range(shard_count):
        cmd = [sys.executable, str(script), *_build_shard_cli(args, shard_count, idx)]
        procs.append(subprocess.Popen(cmd, cwd=str(script.parent)))
    exit_codes = [proc.wait() for proc in procs]
    failed = [code for code in exit_codes if code != 0]
    if failed:
        logging.error("[getData] shard processes failed: %s", exit_codes)
        return max(failed)
    return 0


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
    parser.add_argument("--process-shards", type=int, default=1, help="Spawn N independent BaoStock shard processes")
    parser.add_argument("--api-min-interval", type=float, default=0.0, help="BaoStock requests minimum interval seconds")
    parser.add_argument("--upsert-chunk-size", type=int, default=DEFAULT_UPSERT_CHUNK_SIZE, help="DB executemany rows per chunk")
    parser.add_argument("--baostock-proxy", default=DEFAULT_BAOSTOCK_PROXY, help="BaoStock TCP proxy: auto/direct/none/http://127.0.0.1:7890")
    parser.add_argument("--baostock-connect-timeout", type=float, default=DEFAULT_BAOSTOCK_CONNECT_TIMEOUT, help="BaoStock TCP connect timeout seconds")
    parser.add_argument("--shard-total", type=int, default=1, help="Total shards for multi-process split")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index, 0-based")
    parser.add_argument("--limit-stocks", type=int, default=None, help="调试用，仅处理前 N 只股票")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    if max(1, int(args.process_shards or 1)) > 1:
        return run_process_shards(args)

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    today_yyyymmdd = dt.date.today().strftime("%Y%m%d")
    historical_closed_range = end < today_yyyymmdd
    if start > end:
        raise SystemExit("start-date must be <= end-date")

    db_target = resolve_db_target(args)
    workers = max(1, int(args.workers))
    upsert_chunk_size = max(1, int(args.upsert_chunk_size or DEFAULT_UPSERT_CHUNK_SIZE))
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

    configure_baostock_transport(args.baostock_proxy, args.baostock_connect_timeout)
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
    stock_codes = [code for code, _name, _cap in stocks]
    latest_daily_map: dict[str, str] = {}
    latest_minute_map: dict[str, str] = {}
    if frequency == "d":
        latest_daily_map = load_latest_date_map(engine, "d", stock_codes, chunk_size=upsert_chunk_size)
    else:
        latest_minute_map = load_latest_date_map(engine, frequency, stock_codes, chunk_size=upsert_chunk_size)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    progress = tqdm(total=len(stocks), desc="K线更新", unit="stock")

    def worker(code: str, name: str) -> Optional[Tuple[str, str, pd.DataFrame]]:
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
            return code, name, df
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
            return code, name, df

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, code, name): code for code, name, _cap in stocks}
        for fut in as_completed(futs):
            _code = futs[fut]
            try:
                result = fut.result()
                if result is not None:
                    code, name, df = result
                    if frequency == "d":
                        inserted = upsert_daily(engine, code, df, chunk_size=upsert_chunk_size)
                        if inserted:
                            latest_date = _latest_date_text(df["date"].max())
                            if latest_date:
                                upsert_watermarks(engine, "d", [(code, latest_date)], chunk_size=upsert_chunk_size)
                            logging.info("%s(%s) +%d rows", code, name, inserted)
                    else:
                        inserted = upsert_minute(engine, code, df, frequency, chunk_size=upsert_chunk_size)
                        if inserted:
                            latest_date = _latest_date_text(df["date"].max())
                            if latest_date:
                                upsert_watermarks(engine, frequency, [(code, latest_date)], chunk_size=upsert_chunk_size)
                            logging.info("%s(%s) +%d rows(min-%s)", code, name, inserted, frequency)
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
