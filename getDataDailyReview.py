#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
getDataDailyReview.py

Fetch daily review source pools and write them into DB:
1) Xuangubao: limit_up / limit_down / broken pools (into daily_review_akshare_pool for compatibility)
2) Xuangubao: limit_up pool detail (daily_review_xgb_limit_up)
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import datetime as dt
import json
import logging
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine


DEFAULT_DB = "data/stock.db"
XGB_POOL_DETAIL_URL = "https://flash-api.xuangubao.cn/api/pool/detail?pool_name={pool_name}&date={date}"
REVIEW_POOL_TYPES: Tuple[str, ...] = ("limit_up", "limit_down", "broken")
DEFAULT_TIMEOUT = 10


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
    connect_args: Dict[str, Any] = {}
    if "://" not in db_target and db_target.endswith(".db"):
        db_path = Path(db_target).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_target = f"sqlite:///{db_path.as_posix()}"
    if db_target.startswith("sqlite:///"):
        connect_args["check_same_thread"] = False
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


def parse_date_arg(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        raise ValueError("date is required")
    if len(v) == 8 and v.isdigit():
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    return dt.datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")


def ensure_daily_review_tables(engine: Engine) -> None:
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS daily_review_akshare_pool (
            trade_date VARCHAR(10) NOT NULL,
            pool_type VARCHAR(16) NOT NULL,
            stock_code VARCHAR(16) NOT NULL,
            stock_name VARCHAR(255) NULL,
            reason VARCHAR(255) NULL,
            board_count INT NULL,
            is_st INT NULL,
            raw_json TEXT NULL,
            updated_at VARCHAR(19) NULL,
            PRIMARY KEY (trade_date, pool_type, stock_code)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS daily_review_xgb_limit_up (
            trade_date VARCHAR(10) NOT NULL,
            stock_code VARCHAR(16) NOT NULL,
            stock_name VARCHAR(255) NULL,
            limit_up_days INT NULL,
            change_percent DOUBLE NULL,
            reason VARCHAR(255) NULL,
            plates TEXT NULL,
            is_st INT NULL,
            raw_json TEXT NULL,
            updated_at VARCHAR(19) NULL,
            PRIMARY KEY (trade_date, stock_code)
        )
        """,
    ]
    with engine.begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))


def list_trade_dates(engine: Engine, start_date: str, end_date: str) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT date
                FROM stock_daily
                WHERE date BETWEEN :s AND :e
                ORDER BY date
                """
            ),
            {"s": start_date, "e": end_date},
        ).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def _normalize_code(raw: object) -> str:
    text_val = str(raw or "").strip()
    if not text_val or text_val.lower() in {"nan", "none"}:
        return ""
    text_val = text_val.replace(".", "").replace("-", "").replace("_", "").upper()
    for prefix in ("SH", "SZ", "BJ"):
        if text_val.startswith(prefix):
            text_val = text_val[len(prefix) :]
            break
    digits = "".join(ch for ch in text_val if ch.isdigit())
    if not digits:
        return ""
    if len(digits) < 6:
        digits = digits.zfill(6)
    return digits


def _is_st_name(stock_name: str) -> bool:
    name = str(stock_name or "").strip()
    if not name:
        return False
    return ("ST" in name.upper()) or ("退" in name)


def _parse_board_count(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        pass
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _safe_json(raw: Any) -> str:
    try:
        return json.dumps(raw, ensure_ascii=False)
    except Exception:
        return "{}"


def _fetch_xgb_pool_items(trade_date: str, pool_name: str, timeout: int) -> Tuple[bool, List[Dict[str, Any]]]:
    url = XGB_POOL_DETAIL_URL.format(pool_name=pool_name, date=trade_date)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logging.warning("[daily-review-data] xgb fetch failed %s (%s): %s", trade_date, pool_name, exc)
        return False, []

    if not isinstance(payload, dict):
        return False, []
    if payload.get("code") != 20000:
        logging.warning(
            "[daily-review-data] xgb bad response code on %s (%s): %s",
            trade_date,
            pool_name,
            payload.get("code"),
        )
        return False, []

    data = payload.get("data")
    if not isinstance(data, list):
        return True, []
    items = [item for item in data if isinstance(item, dict)]
    return True, items


def _extract_reason(item: Dict[str, Any]) -> Optional[str]:
    surge_reason = item.get("surge_reason")
    if isinstance(surge_reason, dict):
        for key in ("stock_reason", "reason_type_desc", "reason_type", "reason"):
            value = str(surge_reason.get(key) or "").strip()
            if value:
                return value
    for key in ("stock_reason", "reason", "reason_type_desc", "reason_type"):
        value = item.get(key)
        if isinstance(value, str):
            txt = value.strip()
            if txt:
                return txt
    return None


def _extract_board_count(item: Dict[str, Any], pool_type: str) -> Optional[int]:
    candidates: List[Any] = []
    if pool_type == "limit_up":
        candidates.extend([item.get("limit_up_days"), item.get("m_days_n_boards_boards")])
    elif pool_type == "limit_down":
        candidates.extend([item.get("limit_down_days"), item.get("m_days_n_boards_boards")])
    elif pool_type == "broken":
        candidates.extend([item.get("limit_up_days"), item.get("m_days_n_boards_boards")])

    for value in candidates:
        count = _parse_board_count(value)
        if count is not None and count > 0:
            return count
    return None


def fetch_review_pool_items(
    trade_date: str,
    timeout: int,
) -> Tuple[Dict[str, bool], Dict[str, List[Dict[str, Any]]]]:
    status: Dict[str, bool] = {}
    items: Dict[str, List[Dict[str, Any]]] = {}
    for pool_type in REVIEW_POOL_TYPES:
        ok, rows = _fetch_xgb_pool_items(trade_date, pool_type, timeout=max(1, int(timeout)))
        status[pool_type] = ok
        items[pool_type] = rows
    return status, items


def build_review_pool_rows(trade_date: str, pool_items: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for pool_type in REVIEW_POOL_TYPES:
        for item in pool_items.get(pool_type, []):
            code = _normalize_code(item.get("symbol") or item.get("code") or item.get("stock_code"))
            if not code:
                continue
            stock_name = str(item.get("stock_chi_name") or item.get("stock_name") or item.get("name") or "").strip()
            is_st = 1 if _is_st_name(stock_name) else 0
            reason = _extract_reason(item)
            out_rows.append(
                {
                    "trade_date": trade_date,
                    "pool_type": pool_type,
                    "stock_code": code,
                    "stock_name": stock_name,
                    "reason": reason if reason else None,
                    "board_count": _extract_board_count(item, pool_type),
                    "is_st": is_st,
                    "raw_json": _safe_json(item),
                }
            )
    return out_rows


def build_xgb_limit_up_rows(trade_date: str, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for item in items:
        code = _normalize_code(item.get("symbol") or item.get("code") or item.get("stock_code"))
        if not code:
            continue
        stock_name = str(item.get("stock_chi_name") or item.get("stock_name") or item.get("name") or "").strip()
        is_st = 1 if _is_st_name(stock_name) else 0
        try:
            limit_up_days = int(item.get("limit_up_days") or 0)
        except Exception:
            limit_up_days = 0
        try:
            change_percent = float(item.get("change_percent") or 0.0)
        except Exception:
            change_percent = 0.0
        reason = _extract_reason(item)
        surge_reason = item.get("surge_reason") if isinstance(item.get("surge_reason"), dict) else {}
        related = surge_reason.get("related_plates") if isinstance(surge_reason.get("related_plates"), list) else []
        plates: List[str] = []
        for plate in related:
            if not isinstance(plate, dict):
                continue
            plate_name = str(plate.get("plate_name") or "").strip()
            if plate_name:
                plates.append(plate_name)
        out_rows.append(
            {
                "trade_date": trade_date,
                "stock_code": code,
                "stock_name": stock_name,
                "limit_up_days": int(limit_up_days),
                "change_percent": float(change_percent),
                "reason": reason,
                "plates": _safe_json(plates),
                "is_st": is_st,
                "raw_json": _safe_json(item),
            }
        )
    return out_rows


def _chunked(rows: Sequence[Dict[str, Any]], size: int = 500) -> Iterable[List[Dict[str, Any]]]:
    buff: List[Dict[str, Any]] = []
    for row in rows:
        buff.append(row)
        if len(buff) >= size:
            yield buff
            buff = []
    if buff:
        yield buff


def upsert_akshare_rows(engine: Engine, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO daily_review_akshare_pool
            (trade_date, pool_type, stock_code, stock_name, reason, board_count, is_st, raw_json, updated_at)
        VALUES
            (:trade_date, :pool_type, :stock_code, :stock_name, :reason, :board_count, :is_st, :raw_json, :updated_at)
        ON CONFLICT(trade_date, pool_type, stock_code) DO UPDATE SET
            stock_name=excluded.stock_name,
            reason=excluded.reason,
            board_count=excluded.board_count,
            is_st=excluded.is_st,
            raw_json=excluded.raw_json,
            updated_at=excluded.updated_at
        """
        if engine.dialect.name == "sqlite"
        else """
        INSERT INTO daily_review_akshare_pool
            (trade_date, pool_type, stock_code, stock_name, reason, board_count, is_st, raw_json, updated_at)
        VALUES
            (:trade_date, :pool_type, :stock_code, :stock_name, :reason, :board_count, :is_st, :raw_json, :updated_at)
        ON DUPLICATE KEY UPDATE
            stock_name=VALUES(stock_name),
            reason=VALUES(reason),
            board_count=VALUES(board_count),
            is_st=VALUES(is_st),
            raw_json=VALUES(raw_json),
            updated_at=VALUES(updated_at)
        """
    )
    with engine.begin() as conn:
        for chunk in _chunked(rows):
            conn.execute(stmt, chunk)


def upsert_xgb_rows(engine: Engine, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO daily_review_xgb_limit_up
            (trade_date, stock_code, stock_name, limit_up_days, change_percent, reason, plates, is_st, raw_json, updated_at)
        VALUES
            (:trade_date, :stock_code, :stock_name, :limit_up_days, :change_percent, :reason, :plates, :is_st, :raw_json, :updated_at)
        ON CONFLICT(trade_date, stock_code) DO UPDATE SET
            stock_name=excluded.stock_name,
            limit_up_days=excluded.limit_up_days,
            change_percent=excluded.change_percent,
            reason=excluded.reason,
            plates=excluded.plates,
            is_st=excluded.is_st,
            raw_json=excluded.raw_json,
            updated_at=excluded.updated_at
        """
        if engine.dialect.name == "sqlite"
        else """
        INSERT INTO daily_review_xgb_limit_up
            (trade_date, stock_code, stock_name, limit_up_days, change_percent, reason, plates, is_st, raw_json, updated_at)
        VALUES
            (:trade_date, :stock_code, :stock_name, :limit_up_days, :change_percent, :reason, :plates, :is_st, :raw_json, :updated_at)
        ON DUPLICATE KEY UPDATE
            stock_name=VALUES(stock_name),
            limit_up_days=VALUES(limit_up_days),
            change_percent=VALUES(change_percent),
            reason=VALUES(reason),
            plates=VALUES(plates),
            is_st=VALUES(is_st),
            raw_json=VALUES(raw_json),
            updated_at=VALUES(updated_at)
        """
    )
    with engine.begin() as conn:
        for chunk in _chunked(rows):
            conn.execute(stmt, chunk)


def replace_for_trade_date(
    engine: Engine,
    trade_date: str,
    *,
    review_success: bool,
    review_rows: Sequence[Dict[str, Any]],
    xgb_success: bool,
    xgb_rows: Sequence[Dict[str, Any]],
) -> None:
    now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if review_success:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM daily_review_akshare_pool WHERE trade_date = :d"),
                {"d": trade_date},
            )
        payload = [dict(row, updated_at=now_text) for row in review_rows]
        upsert_akshare_rows(engine, payload)

    if xgb_success:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM daily_review_xgb_limit_up WHERE trade_date = :d"),
                {"d": trade_date},
            )
        payload = [dict(row, updated_at=now_text) for row in xgb_rows]
        upsert_xgb_rows(engine, payload)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Xuangubao daily review pools and persist to DB.")
    parser.add_argument("--config", default=None, help="config.ini path")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL")
    parser.add_argument("--db", default=None, help="SQLite DB file path")
    parser.add_argument("--start-date", default="2025-01-01", help="YYYYMMDD / YYYY-MM-DD")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y-%m-%d"), help="YYYYMMDD / YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=1, help="parallel workers by trade date")
    parser.add_argument("--xgb-timeout", type=int, default=DEFAULT_TIMEOUT, help="XGB API timeout seconds")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    start_date = parse_date_arg(args.start_date)
    end_date = parse_date_arg(args.end_date)
    if start_date > end_date:
        raise SystemExit("start-date must be <= end-date")

    workers = max(1, int(args.workers or 1))
    engine = make_engine(resolve_db_target(args), workers)
    ensure_daily_review_tables(engine)

    trade_dates = list_trade_dates(engine, start_date, end_date)
    if not trade_dates:
        logging.warning("[daily-review-data] no trade dates in %s ~ %s", start_date, end_date)
        return 0

    logging.info("[daily-review-data] dates=%d range=%s~%s", len(trade_dates), trade_dates[0], trade_dates[-1])

    def worker(trade_date: str) -> Tuple[str, bool, int, bool, int, Dict[str, bool]]:
        statuses, pool_items = fetch_review_pool_items(trade_date, timeout=max(1, int(args.xgb_timeout)))
        review_rows = build_review_pool_rows(trade_date, pool_items)
        review_success = all(bool(statuses.get(t)) for t in REVIEW_POOL_TYPES)
        limit_up_success = bool(statuses.get("limit_up"))
        xgb_rows = build_xgb_limit_up_rows(trade_date, pool_items.get("limit_up") or []) if limit_up_success else []
        replace_for_trade_date(
            engine,
            trade_date,
            review_success=review_success,
            review_rows=review_rows,
            xgb_success=limit_up_success,
            xgb_rows=xgb_rows,
        )
        return trade_date, review_success, len(review_rows), limit_up_success, len(xgb_rows), statuses

    from tqdm import tqdm

    prog = tqdm(total=len(trade_dates), desc="dailyReview data", unit="day")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, d): d for d in trade_dates}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                trade_date, review_success, review_n, xgb_success, xgb_n, statuses = fut.result()
                pool_status = ",".join(
                    f"{pool}:{'ok' if statuses.get(pool) else 'skip'}" for pool in REVIEW_POOL_TYPES
                )
                logging.info(
                    "[daily-review-data] %s pools=%s review=%s(%d) xgb_limit_up=%s(%d)",
                    trade_date,
                    pool_status,
                    "ok" if review_success else "skip",
                    review_n,
                    "ok" if xgb_success else "skip",
                    xgb_n,
                )
            except Exception as exc:  # noqa: BLE001
                logging.exception("[daily-review-data] failed on %s: %s", d, exc)
            finally:
                prog.update(1)
    prog.close()

    logging.info("[daily-review-data] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
