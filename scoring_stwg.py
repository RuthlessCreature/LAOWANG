#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scoring_stwg.py

功能：读取数据库中的 K 线数据，根据 docs/scoring_stwg.md 描述的“缩头乌龟”模型规则批量计算评分。
- 输出单股逐日评分表：stock_scores_stwg
- 根据得分构建每日股票池：model_stwg_pool
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
    pool_size = max(5, min(64, workers * 2))
    max_overflow = max(10, min(128, workers * 2))
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


@dataclass(frozen=True)
class Settings:
    min_history_rows: int = 150
    stageA_min_window: int = 60
    stageA_max_window: int = 120
    amplitude_window: int = 18
    volume_fast_window: int = 5
    volume_slow_window: int = 20
    amplitude_score_full: float = 0.06
    amplitude_score_mid: float = 0.10
    volume_ratio_full: float = 0.6
    volume_ratio_mid: float = 0.8
    platform_recent_window: int = 20
    platform_reference_window: int = 45
    breakout_region_window: int = 60
    breakout_tolerance: float = 0.003
    ma60_trend_window: int = 20
    space_lookback: int = 240


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def enrich_history(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    close = df["close"]
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    df["ma120"] = close.rolling(120).mean()
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["high_max_60"] = df["high"].rolling(60).max()
    df["high_max_120"] = df["high"].rolling(120).max()
    df["high_max_240"] = df["high"].rolling(240).max()
    df["low_min_20"] = df["low"].rolling(20).min()
    df["low_min_60"] = df["low"].rolling(60).min()
    df["rsi14"] = calc_rsi(close, 14)
    return df


def _recent_value(series: pd.Series, idx: int = -1) -> Optional[float]:
    if series.empty:
        return None
    try:
        val = float(series.iloc[idx])
    except (IndexError, ValueError, TypeError):
        return None
    if np.isnan(val):
        return None
    return val


def risk_filters(df_slice: pd.DataFrame, settings: Settings) -> bool:
    if df_slice.empty:
        return True
    close = _recent_value(df_slice["close"])
    ma20 = _recent_value(df_slice["ma20"])
    ma60 = _recent_value(df_slice["ma60"])
    if close is None or ma20 is None or ma60 is None:
        return True
    cond_downtrend = ma20 < ma60
    cond_ma_slopes = False
    if len(df_slice) > 10:
        ma20_prev = _recent_value(df_slice["ma20"], idx=-5)
        ma60_prev = _recent_value(df_slice["ma60"], idx=-settings.ma60_trend_window)
        if ma20_prev is not None and ma60_prev is not None:
            cond_ma_slopes = (ma20 - ma20_prev) <= 0 and (ma60 - ma60_prev) <= 0
    cond_below_ma60 = False
    if len(df_slice) >= 3:
        recent_close = df_slice["close"].tail(3)
        recent_ma60 = df_slice["ma60"].tail(3)
        cond_below_ma60 = bool((recent_close < recent_ma60).all())
    drop_today = False
    if len(df_slice) >= 2:
        prev_close = _recent_value(df_slice["close"], idx=-2)
        if prev_close:
            drop_today = (close / prev_close) - 1 <= -0.08
    drop_three = False
    if len(df_slice) >= 4:
        prev3 = _recent_value(df_slice["close"], idx=-4)
        if prev3:
            drop_three = (close / prev3) - 1 <= -0.12
    vol = _recent_value(df_slice["volume"])
    vol_ma20 = _recent_value(df_slice["vol_ma20"])
    violent_down = False
    if vol is not None and vol_ma20 not in (None, 0) and len(df_slice) >= 2:
        prev_close = _recent_value(df_slice["close"], idx=-2) or 0.0
        violent_down = vol >= 1.8 * vol_ma20 and close < prev_close
    high120 = _recent_value(df_slice["high_max_120"])
    overextension = False
    if high120 and high120 > 0:
        overextension = (high120 - close) / high120 < 0.12
    return cond_downtrend or cond_ma_slopes or cond_below_ma60 or drop_today or drop_three or violent_down or overextension


def score_trend_base(df_slice: pd.DataFrame, settings: Settings) -> float:
    ma20 = _recent_value(df_slice["ma20"])
    ma60 = _recent_value(df_slice["ma60"])
    if ma20 is None or ma60 is None or ma20 <= ma60:
        return 0.0
    if len(df_slice) > settings.ma60_trend_window:
        ma60_prev = _recent_value(df_slice["ma60"], idx=-settings.ma60_trend_window)
    else:
        ma60_prev = None
    if ma60_prev is not None and ma60 > ma60_prev:
        return 10.0
    return 6.0


def score_stage_a(df_slice: pd.DataFrame, settings: Settings) -> Tuple[float, Optional[float]]:
    if df_slice.empty:
        return 0.0, None
    window = min(len(df_slice), settings.stageA_max_window)
    if window < settings.stageA_min_window:
        return 0.0, None
    segment = df_slice.tail(window)
    half = window // 2
    first = segment.head(half)
    second = segment.tail(window - half)
    if first.empty or second.empty:
        return 0.0, None
    close_first = first["close"].mean()
    close_second = second["close"].mean()
    low_first = first["low"].min()
    low_second = second["low"].min()
    vol_first = first["volume"].mean()
    vol_second = second["volume"].mean()
    if close_first is None or close_second is None or close_first <= 0 or vol_first <= 0:
        return 0.0, None
    price_gain = (close_second - close_first) / close_first
    low_gain = (low_second - low_first) / max(low_first, 1e-9) if low_first else 0.0
    vol_gain = (vol_second - vol_first) / vol_first
    region_high = float(second["high"].max()) if not second["high"].isna().all() else None
    clear = price_gain >= 0.15 and low_gain >= 0.03 and vol_gain >= 0.08
    exist = price_gain >= 0.08 and low_gain >= 0 and vol_gain >= 0.03
    if clear:
        return 10.0, region_high
    if exist:
        return 6.0, region_high
    return 0.0, region_high


def score_stage_b(df_slice: pd.DataFrame, settings: Settings) -> Tuple[float, float, float]:
    if len(df_slice) < settings.amplitude_window:
        return 0.0, 0.0, 0.0
    recent = df_slice.tail(settings.amplitude_window)
    high = recent["high"].max()
    low = recent["low"].min()
    if low is None or low <= 0:
        amp_score = 0.0
    else:
        amplitude = (high - low) / low
        if amplitude <= settings.amplitude_score_full:
            amp_score = 10.0
        elif amplitude <= settings.amplitude_score_mid:
            amp_score = 6.0
        else:
            amp_score = 0.0
    vol_ma5 = _recent_value(df_slice["vol_ma5"])
    vol_ma20 = _recent_value(df_slice["vol_ma20"])
    if vol_ma5 is None or vol_ma20 in (None, 0):
        vol_score = 0.0
    else:
        ratio = vol_ma5 / vol_ma20
        if ratio <= settings.volume_ratio_full:
            vol_score = 10.0
        elif ratio <= settings.volume_ratio_mid:
            vol_score = 6.0
        else:
            vol_score = 0.0
    return (amp_score + vol_score) / 2.0, amp_score, vol_score


def score_platform_support(df_slice: pd.DataFrame, settings: Settings) -> float:
    close = _recent_value(df_slice["close"])
    ma10 = _recent_value(df_slice["ma10"])
    ma20 = _recent_value(df_slice["ma20"])
    if close is None or (ma10 is None and ma20 is None):
        return 0.0
    close_ok = (ma10 is not None and close >= ma10) or (ma20 is not None and close >= ma20)
    if not close_ok:
        return 0.0
    slope10 = None
    if len(df_slice) > 5 and ma10 is not None:
        prev_ma10 = _recent_value(df_slice["ma10"], idx=-5)
        if prev_ma10 is not None:
            slope10 = ma10 - prev_ma10
    slope20 = None
    if len(df_slice) > 5 and ma20 is not None:
        prev_ma20 = _recent_value(df_slice["ma20"], idx=-5)
        if prev_ma20 is not None:
            slope20 = ma20 - prev_ma20
    recent_low = df_slice["low"].tail(settings.platform_recent_window).min()
    ref_slice = df_slice.head(max(len(df_slice) - settings.platform_recent_window, 1)).tail(settings.platform_reference_window)
    ref_low = ref_slice["low"].min() if not ref_slice.empty else recent_low
    if recent_low is None or ref_low is None or ref_low <= 0:
        return 0.0
    strong = (
        (slope10 is None or slope10 >= 0)
        and (slope20 is None or slope20 >= 0)
        and recent_low >= ref_low * 0.98
    )
    if strong:
        return 10.0
    mild = (
        (slope10 is None or slope10 >= -0.005 * close)
        and (slope20 is None or slope20 >= -0.01 * close)
        and recent_low >= ref_low * 0.95
    )
    if mild:
        return 6.0
    return 0.0


def score_breakout(df_slice: pd.DataFrame, region_high: Optional[float], settings: Settings) -> Tuple[float, bool, float]:
    if region_high is None or region_high <= 0:
        return 0.0, False, 0.0
    close = _recent_value(df_slice["close"])
    volume = _recent_value(df_slice["volume"])
    vol_ma20 = _recent_value(df_slice["vol_ma20"])
    if close is None or volume is None or vol_ma20 in (None, 0):
        return 0.0, False, 0.0
    price_ok = close >= region_high * (1 - settings.breakout_tolerance)
    ratio = volume / vol_ma20
    if not price_ok or ratio < 1.2:
        return 0.0, False, ratio
    if ratio >= 1.8:
        return 10.0, True, ratio
    return 6.0, True, ratio


def score_rsi_state(df_slice: pd.DataFrame) -> float:
    rsi = _recent_value(df_slice["rsi14"])
    if rsi is None:
        return 0.0
    if 45 <= rsi <= 60:
        return 10.0
    if 60 < rsi < 65:
        return 6.0
    if rsi < 45:
        return 3.0
    if rsi >= 70:
        return 0.0
    return 6.0


def score_space(df_slice: pd.DataFrame, breakout: bool, settings: Settings) -> float:
    if not breakout or len(df_slice) <= settings.amplitude_window:
        return 0.0
    close = _recent_value(df_slice["close"])
    if close is None or close <= 0:
        return 0.0
    history = df_slice.iloc[: -settings.amplitude_window]
    if history.empty:
        return 0.0
    resistance = history["high"].tail(settings.space_lookback).max()
    if resistance is None or np.isnan(resistance) or resistance <= close:
        return 0.0
    space = (resistance - close) / close
    if space >= 0.20:
        return 10.0
    if space >= 0.12:
        return 6.0
    return 0.0


def build_status_tags(
    *,
    stage_a: float,
    stage_b: float,
    amp_score: float,
    vol_score: float,
    platform: float,
    breakout: bool,
    breakout_score: float,
    space_score: float,
) -> List[str]:
    tags: List[str] = []
    if stage_a >= 6:
        tags.append("STAGE_A_OK")
    if stage_b >= 6 or amp_score >= 6:
        tags.append("STAGE_B_COMPRESSED")
    if vol_score >= 6:
        tags.append("VOLUME_DRY_UP")
    if platform > 0:
        tags.append("AT_PLATFORM")
    if breakout:
        tags.append("BREAKOUT_R")
        if breakout_score > 0:
            tags.append("VOLUME_EXPANSION")
    if space_score > 0:
        tags.append("SPACE_OK")
    return tags


def calc_score_components(df_slice: pd.DataFrame, settings: Settings) -> Dict[str, object]:
    if risk_filters(df_slice, settings):
        return {
            "total_score": 0.0,
            "trend_base_score": 0.0,
            "stageA_structure_score": 0.0,
            "stageB_compression_score": 0.0,
            "platform_support_score": 0.0,
            "breakout_confirmation_score": 0.0,
            "rsi_state_score": 0.0,
            "space_after_breakout_score": 0.0,
            "status_tags": ["RISK_FILTERED"],
        }
    trend = score_trend_base(df_slice, settings)
    stage_a, region_high = score_stage_a(df_slice, settings)
    stage_b, amp_score, vol_score = score_stage_b(df_slice, settings)
    platform = score_platform_support(df_slice, settings)
    breakout, breakout_flag, _ = score_breakout(df_slice, region_high, settings)
    rsi_score = score_rsi_state(df_slice)
    space_score = score_space(df_slice, breakout_flag, settings)
    total = (
        (trend / 10.0) * 15.0
        + (stage_a / 10.0) * 15.0
        + (stage_b / 10.0) * 25.0
        + (platform / 10.0) * 10.0
        + (breakout / 10.0) * 20.0
        + (rsi_score / 10.0) * 5.0
        + (space_score / 10.0) * 10.0
    )
    tags = build_status_tags(
        stage_a=stage_a,
        stage_b=stage_b,
        amp_score=amp_score,
        vol_score=vol_score,
        platform=platform,
        breakout=breakout_flag,
        breakout_score=breakout,
        space_score=space_score,
    )
    if not tags:
        tags = ["STAGE_B_COMPRESSED"] if stage_b >= 6 else []
    return {
        "total_score": float(total),
        "trend_base_score": float(trend),
        "stageA_structure_score": float(stage_a),
        "stageB_compression_score": float(stage_b),
        "platform_support_score": float(platform),
        "breakout_confirmation_score": float(breakout),
        "rsi_state_score": float(rsi_score),
        "space_after_breakout_score": float(space_score),
        "status_tags": tags,
    }


def fetch_trade_dates(engine: Engine, start_date: str, end_date: str) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT date FROM stock_daily WHERE date BETWEEN :s AND :e ORDER BY date"),
            {"s": start_date, "e": end_date},
        ).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def list_stock_codes(engine: Engine) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT stock_code FROM stock_daily")).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def load_history(engine: Engine, stock_code: str, end_date: str, min_rows: int) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT date, open, high, low, close, volume, amount
                FROM stock_daily
                WHERE stock_code = :c AND date <= :e
                ORDER BY date
                """
            ),
            {"c": stock_code, "e": end_date},
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if len(df) < min_rows:
        return pd.DataFrame()
    return enrich_history(df)


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


def delete_by_trade_date(engine: Engine, table: str, trade_date: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table} WHERE trade_date = :d"), {"d": trade_date})


def parse_date_arg(value: str, *, default: Optional[str] = None) -> str:
    v = (value or default or "").strip()
    if len(v) == 8 and v.isdigit():
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    try:
        return dt.datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date: {value}") from exc


def compute_scores_for_stock(
    *,
    engine: Engine,
    stock_code: str,
    target_dates: Sequence[str],
    settings: Settings,
) -> List[Dict[str, object]]:
    df_hist = load_history(engine, stock_code, target_dates[-1], min_rows=settings.min_history_rows)
    if df_hist.empty:
        return []
    hist_dates = set(df_hist["date"].tolist())
    score_rows: List[Dict[str, object]] = []
    for trade_date in target_dates:
        if trade_date not in hist_dates:
            continue
        df_slice = df_hist[df_hist["date"] <= trade_date]
        if len(df_slice) < settings.min_history_rows:
            continue
        comp = calc_score_components(df_slice, settings)
        score_rows.append(
            {
                "stock_code": stock_code,
                "score_date": trade_date,
                "total_score": comp["total_score"],
                "trend_base_score": comp["trend_base_score"],
                "stageA_structure_score": comp["stageA_structure_score"],
                "stageB_compression_score": comp["stageB_compression_score"],
                "platform_support_score": comp["platform_support_score"],
                "breakout_confirmation_score": comp["breakout_confirmation_score"],
                "rsi_state_score": comp["rsi_state_score"],
                "space_after_breakout_score": comp["space_after_breakout_score"],
                "status_tags": json.dumps(comp["status_tags"], ensure_ascii=False),
            }
        )
    return score_rows


def build_pool(engine: Engine, trade_date: str, top_n: int, min_score: float) -> None:
    delete_by_trade_date(engine, "model_stwg_pool", trade_date)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT s.stock_code,
                       COALESCE(i.name, '') AS stock_name,
                       d.close AS close,
                       s.total_score,
                       s.stageB_compression_score,
                       s.breakout_confirmation_score,
                       s.status_tags
                FROM stock_scores_stwg s
                LEFT JOIN stock_info i ON i.stock_code = s.stock_code
                LEFT JOIN stock_daily d ON d.stock_code = s.stock_code AND d.date = s.score_date
                WHERE s.score_date = :d AND s.total_score >= :min_score
                ORDER BY s.total_score DESC, s.stageB_compression_score DESC
                LIMIT :lim
                """
            ),
            {"d": trade_date, "min_score": float(min_score), "lim": max(top_n * 3, top_n)},
        ).fetchall()
    rows = rows[:top_n]
    payload: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        payload.append(
            {
                "trade_date": trade_date,
                "rank_no": idx,
                "stock_code": row[0],
                "stock_name": row[1],
                "close": row[2],
                "total_score": row[3],
                "stageB_compression_score": row[4],
                "breakout_confirmation_score": row[5],
                "status_tags": row[6],
            }
        )
    if payload:
        upsert_rows(
            engine,
            "model_stwg_pool",
            ["trade_date", "rank_no", "stock_code", "stock_name", "close", "total_score", "stageB_compression_score", "breakout_confirmation_score", "status_tags"],
            payload,
            ["trade_date", "stock_code"],
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="缩头乌龟（STWG）评分脚本")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--top", type=int, default=150)
    parser.add_argument("--min-score", type=float, default=55.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    start = parse_date_arg(args.start_date)
    end = parse_date_arg(args.end_date)
    if start > end:
        raise SystemExit("start-date must be <= end-date")

    db_target = resolve_db_target(args)
    workers = max(1, int(args.workers))
    engine = make_engine(db_target, workers)

    trade_dates = fetch_trade_dates(engine, start, end)
    if not trade_dates:
        logging.warning("指定区间内没有交易日数据：%s ~ %s", start, end)
        return 0

    stock_codes = list_stock_codes(engine)
    if not stock_codes:
        logging.warning("stock_daily 为空，无法计算")
        return 0

    settings = Settings()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    score_rows_all: List[Dict[str, object]] = []
    progress = tqdm(total=len(stock_codes), desc="STWG评分", unit="stock")

    def worker(code: str) -> List[Dict[str, object]]:
        return compute_scores_for_stock(engine=engine, stock_code=code, target_dates=trade_dates, settings=settings)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, code): code for code in stock_codes}
        for fut in as_completed(futs):
            try:
                scores = fut.result()
                score_rows_all.extend(scores)
            except Exception as exc:  # noqa: BLE001
                logging.exception("计算 %s 失败: %s", futs[fut], exc)
            finally:
                progress.update(1)
    progress.close()

    logging.info("写入 stock_scores_stwg: %d 行", len(score_rows_all))
    upsert_rows(
        engine,
        "stock_scores_stwg",
        [
            "stock_code",
            "score_date",
            "total_score",
            "trend_base_score",
            "stageA_structure_score",
            "stageB_compression_score",
            "platform_support_score",
            "breakout_confirmation_score",
            "rsi_state_score",
            "space_after_breakout_score",
            "status_tags",
        ],
        score_rows_all,
        ["stock_code", "score_date"],
    )

    pool_progress = tqdm(total=len(trade_dates), desc="生成STWG股票池", unit="day")
    for trade_date in trade_dates:
        try:
            build_pool(engine, trade_date, top_n=int(args.top), min_score=float(args.min_score))
        except Exception as exc:  # noqa: BLE001
            logging.exception("生成 %s 股票池失败: %s", trade_date, exc)
        finally:
            pool_progress.update(1)
    pool_progress.close()

    logging.info("STWG 完成：dates=%d stocks=%d", len(trade_dates), len(stock_codes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
