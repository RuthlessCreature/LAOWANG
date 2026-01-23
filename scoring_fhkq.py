#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scoring_fhkq.py

功能：根据数据库中的 K 线数据，按交易日计算 FHKQ 连板开板/反抽信号，并写入 model_fhkq。
- 支持 start-date / end-date
- 单文件实现，按交易日串行、候选股并行，带进度条
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine


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
    pool_size = max(5, min(32, workers * 2))
    max_overflow = max(10, min(64, workers * 2))
    connect_args = {}
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
    v = (value or "").strip()
    if len(v) == 8 and v.isdigit():
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    try:
        return dt.datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date: {value}") from exc


def list_trade_dates(engine: Engine, start_date: str, end_date: str) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT date FROM stock_daily WHERE date BETWEEN :s AND :e ORDER BY date"),
            {"s": start_date, "e": end_date},
        ).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def prev_trade_date(engine: Engine, trade_date: str) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(date) FROM stock_daily WHERE date < :d"), {"d": trade_date}).fetchone()
    if not row or not row[0]:
        return None
    return str(row[0])


# ------------------------- FHKQ 评分逻辑 -------------------------


def _round_half_up_2(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.floor(arr * 100.0 + 0.5) / 100.0
    return pd.Series(out, index=values.index)


def _infer_limit_pct(stock_code: str, stock_name: str, is_st_flag: bool) -> float:
    if is_st_flag:
        return 0.05
    code = str(stock_code).strip()
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def _is_st_name(stock_name: str) -> bool:
    name = str(stock_name or "").strip()
    if not name:
        return False
    if "退" in name:
        return True
    return "ST" in name.upper()


def _score_structure(consecutive_limit_down: int) -> int:
    if consecutive_limit_down == 2:
        return 10
    if consecutive_limit_down == 3:
        return 20
    if consecutive_limit_down == 4:
        return 30
    if consecutive_limit_down >= 5:
        return 15
    return 0


def _score_volume_ratio(volume_ratio: float) -> int:
    if not np.isfinite(volume_ratio):
        return 0
    x = float(volume_ratio)
    if x < 0.5:
        return 0
    if x < 1.0:
        return 10
    if x <= 2.0:
        return 20
    return 15


def _score_amount_ratio(amount_ratio: float) -> int:
    if not np.isfinite(amount_ratio):
        return 0
    x = float(amount_ratio)
    if x < 0.5:
        return 0
    if x < 1.5:
        return 5
    return 10


def _fhkq_level(score: float) -> str:
    s = float(score)
    if s >= 80:
        return "A"
    if s >= 60:
        return "B"
    if s >= 40:
        return "C"
    return "D"


def _calc_fhkq_for_one_stock(df_daily_one: pd.DataFrame, trade_date: str, stock_code: str, stock_name: str) -> Optional[Dict[str, object]]:
    if df_daily_one is None or df_daily_one.empty:
        return None
    d = df_daily_one.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    d = d[d["date"] <= trade_date].reset_index(drop=True)
    if d.empty or str(d["date"].iloc[-1]) != trade_date:
        return None
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    is_st_flag = False
    if "is_st" in d.columns:
        try:
            is_st_flag = int(pd.to_numeric(d["is_st"], errors="coerce").fillna(0).iloc[-1]) == 1
        except Exception:  # noqa: BLE001
            is_st_flag = False
    if is_st_flag or _is_st_name(stock_name):
        return None
    limit_pct = _infer_limit_pct(stock_code, stock_name, is_st_flag)
    prev_close = d["close"].shift(1)
    d["limit_down"] = _round_half_up_2(prev_close * (1.0 - float(limit_pct)))
    eps = 1e-3
    is_limit_down = (d["close"] - d["limit_down"]).abs() <= eps
    is_limit_down = is_limit_down.fillna(False)
    if not bool(is_limit_down.iloc[-1]):
        return None
    consecutive = 0
    for v in is_limit_down.iloc[::-1].to_list():
        if bool(v):
            consecutive += 1
        else:
            break
    if consecutive < 2:
        return None
    if len(d) >= 5:
        vol5 = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0).tail(5)
        if bool(is_limit_down.tail(5).all()) and bool((vol5 == 0).all()):
            return None
    if len(d) >= 10:
        c0 = float(d["close"].iloc[-10]) if pd.notna(d["close"].iloc[-10]) else np.nan
        c1 = float(d["close"].iloc[-1]) if pd.notna(d["close"].iloc[-1]) else np.nan
        if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 and (c1 / c0 - 1.0) <= -0.60:
            return None
    vol_today = float(d["volume"].iloc[-1]) if pd.notna(d["volume"].iloc[-1]) else 0.0
    amt_today = float(d["amount"].iloc[-1]) if pd.notna(d["amount"].iloc[-1]) else 0.0
    vol_mean5 = float(pd.to_numeric(d["volume"], errors="coerce").fillna(0.0).tail(5).mean())
    amt_mean5 = float(pd.to_numeric(d["amount"], errors="coerce").fillna(0.0).tail(5).mean())
    volume_ratio = float(vol_today / vol_mean5) if vol_mean5 > 0 else 0.0
    amount_ratio = float(amt_today / amt_mean5) if amt_mean5 > 0 else 0.0
    ld_today = float(d["limit_down"].iloc[-1]) if pd.notna(d["limit_down"].iloc[-1]) else np.nan
    high_today = float(d["high"].iloc[-1]) if pd.notna(d["high"].iloc[-1]) else np.nan
    low_today = float(d["low"].iloc[-1]) if pd.notna(d["low"].iloc[-1]) else np.nan
    open_board_flag = 0
    if np.isfinite(ld_today):
        if np.isfinite(high_today) and high_today > ld_today + eps:
            open_board_flag = 1
        elif np.isfinite(low_today) and low_today < ld_today - eps:
            open_board_flag = 1
    liquidity_exhaust = int(consecutive >= 3 and volume_ratio >= 1.0 and open_board_flag == 1)
    structure_score = _score_structure(consecutive)
    volume_score = _score_volume_ratio(volume_ratio)
    amount_score = _score_amount_ratio(amount_ratio)
    open_board_score = 20 if open_board_flag == 1 else 0
    exhaust_score = 20 if liquidity_exhaust == 1 else 0
    score = float(structure_score + volume_score + amount_score + open_board_score + exhaust_score)
    if consecutive > 6:
        penalty = min(20.0, float(consecutive - 6) * 5.0)
        score = max(0.0, score - penalty)
    score = float(max(0.0, min(100.0, score)))
    level = _fhkq_level(score)
    last_limit_down = 0
    if len(d) >= 2:
        last_limit_down = int(bool(is_limit_down.iloc[-2]))
    return {
        "trade_date": trade_date,
        "stock_code": str(stock_code),
        "stock_name": str(stock_name or ""),
        "consecutive_limit_down": int(consecutive),
        "last_limit_down": int(last_limit_down),
        "volume_ratio": float(volume_ratio),
        "amount_ratio": float(amount_ratio),
        "open_board_flag": int(open_board_flag),
        "liquidity_exhaust": int(liquidity_exhaust),
        "fhkq_score": int(round(score)),
        "fhkq_level": str(level),
    }


# ------------------------- DB 辅助 -------------------------


def upsert_rows(engine: Engine, table: str, cols: Sequence[str], rows: Sequence[Dict[str, object]], key_cols: Sequence[str]) -> None:
    if not rows:
        return
    placeholders = ", ".join([f":{c}" for c in cols])
    col_list = ", ".join(cols)
    if engine.dialect.name == "sqlite":
        stmt = f"INSERT OR REPLACE INTO {table}({col_list}) VALUES({placeholders})"
    else:
        updates = ", ".join([f"{c}=VALUES({c})" for c in cols if c not in key_cols])
        stmt = f"INSERT INTO {table}({col_list}) VALUES({placeholders}) ON DUPLICATE KEY UPDATE {updates}"
    with engine.begin() as conn:
        conn.execute(text(stmt), rows)


def fetch_limit_down_candidates(engine: Engine, trade_date: str, prev_date: str) -> List[Tuple[str, str, float, float]]:
    sql = """
        SELECT d.stock_code,
               COALESCE(i.name, '') AS stock_name,
               d.close AS close,
               p.close AS prev_close
        FROM stock_daily d
        INNER JOIN stock_daily p
            ON p.stock_code = d.stock_code AND p.date = :prev
        LEFT JOIN stock_info i
            ON i.stock_code = d.stock_code
        WHERE d.date = :d
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"d": trade_date, "prev": prev_date}).fetchall()
    return [(str(r[0]), str(r[1] or ""), float(r[2]), float(r[3])) for r in rows if r and r[0] is not None]


def load_history_for_codes(engine: Engine, codes: Sequence[str], end_date: str, limit: int = 120) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    with engine.connect() as conn:
        for code in codes:
            rows = conn.execute(
                text(
                    """
                    SELECT date, open, high, low, close, volume, amount
                    FROM stock_daily
                    WHERE stock_code = :c AND date <= :d
                    ORDER BY date DESC
                    LIMIT :lim
                    """
                ),
                {"c": code, "d": end_date, "lim": int(limit)},
            ).fetchall()
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"]).sort_values("date")
            out[code] = df
    return out


# ------------------------- 主流程 -------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FHKQ 连板博弈评分（单文件版）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    start = parse_date_arg(args.start_date)
    end = parse_date_arg(args.end_date)
    if start > end:
        raise SystemExit("start-date 必须 <= end-date")

    db_target = resolve_db_target(args)
    workers = max(1, int(args.workers))
    engine = make_engine(db_target, workers)

    trade_dates = list_trade_dates(engine, start, end)
    if not trade_dates:
        logging.warning("区间内没有交易日数据：%s~%s", start, end)
        return 0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    outer_progress = tqdm(total=len(trade_dates), desc="FHKQ按日", unit="day")

    for trade_date in trade_dates:
        prev = prev_trade_date(engine, trade_date)
        if not prev:
            outer_progress.update(1)
            continue
        candidates = fetch_limit_down_candidates(engine, trade_date, prev)
        filtered: List[Tuple[str, str]] = []
        eps = 1e-3
        for code, name, close, prev_close in candidates:
            if not (np.isfinite(close) and np.isfinite(prev_close) and prev_close > 0):
                continue
            limit_pct = _infer_limit_pct(code, name, False)
            limit_down = float(_round_half_up_2(pd.Series([prev_close * (1.0 - limit_pct)])).iloc[0])
            if abs(close - limit_down) <= eps and not _is_st_name(name):
                filtered.append((code, name))
        if not filtered:
            delete_by = text("DELETE FROM model_fhkq WHERE trade_date = :d")
            with engine.begin() as conn:
                conn.execute(delete_by, {"d": trade_date})
            outer_progress.update(1)
            continue
        histories = load_history_for_codes(engine, [c for c, _ in filtered], trade_date, limit=150)
        results: List[Dict[str, object]] = []
        inner_progress = tqdm(total=len(filtered), desc=f"{trade_date}", unit="stock", leave=False)

        def worker(code: str, name: str) -> Optional[Dict[str, object]]:
            df = histories.get(code)
            if df is None or df.empty:
                return None
            return _calc_fhkq_for_one_stock(df, trade_date=trade_date, stock_code=code, stock_name=name)

        with ThreadPoolExecutor(max_workers=min(workers, len(filtered))) as ex:
            futs = {ex.submit(worker, code, name): (code, name) for code, name in filtered}
            for fut in as_completed(futs):
                inner_progress.update(1)
                try:
                    row = fut.result()
                    if row:
                        results.append(row)
                except Exception as exc:  # noqa: BLE001
                    logging.exception("FHKQ 计算 %s 失败: %s", futs[fut][0], exc)
        inner_progress.close()
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM model_fhkq WHERE trade_date = :d"), {"d": trade_date})
        if results:
            upsert_rows(
                engine,
                "model_fhkq",
                [
                    "trade_date",
                    "stock_code",
                    "stock_name",
                    "consecutive_limit_down",
                    "last_limit_down",
                    "volume_ratio",
                    "amount_ratio",
                    "open_board_flag",
                    "liquidity_exhaust",
                    "fhkq_score",
                    "fhkq_level",
                ],
                results,
                ["trade_date", "stock_code"],
            )
        logging.info("FHKQ %s -> %d 条信号", trade_date, len(results))
        outer_progress.update(1)
    outer_progress.close()
    logging.info("FHKQ 完成：%d 个交易日", len(trade_dates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
