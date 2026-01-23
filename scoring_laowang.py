#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scoring_laowang.py

功能：读取数据库中的 K 线数据，批量计算 LAOWANG 评分并写入
  - stock_scores_v3（逐股逐日评分）
  - stock_levels（支撑/阻力）
  - model_laowang_pool（按交易日的 TopN 股票池）

特性：
  - 单文件实现，含指标/支撑/评分算法
  - start-date / end-date 控制计算窗口
  - 多线程按股票计算，tqdm 进度条
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

# ------------------------- 配置解析 / 连接 -------------------------

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


# ------------------------- 算法参数 / 函数 -------------------------


@dataclass(frozen=True)
class Settings:
    indicator_lookback: int = 360
    level_lookback: int = 120
    breakout_lookback: int = 60
    at_support_pct: float = 0.05
    near_resistance_pct: float = 0.12
    ma_support_max_distance_pct: float = 0.06


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calc_indicators(df_daily: pd.DataFrame) -> pd.DataFrame:
    out = df_daily[["date", "high", "low", "close"]].copy()
    close = out["close"]

    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["ma120"] = close.rolling(120).mean()
    out["rsi14"] = calc_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = ema12 - ema26
    dea = diff.ewm(span=9, adjust=False).mean()
    hist = (diff - dea) * 2

    out["macd_diff"] = diff
    out["macd_dea"] = dea
    out["macd_hist"] = hist
    out["atr14"] = calc_atr(out["high"], out["low"], close, 14)

    out = out.drop(columns=["high", "low", "close"])
    return out


def volume_profile_levels(df: pd.DataFrame, bins: int = 20) -> Tuple[Optional[float], Optional[float]]:
    if df.empty or df["close"].isna().all():
        return None, None
    low = float(df["low"].min())
    high = float(df["high"].max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None, None
    typical = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy(dtype=float)
    volume = df["volume"].fillna(0).to_numpy(dtype=float)
    edges = np.linspace(low, high, bins + 1)
    idx = np.digitize(typical, edges) - 1
    idx = idx.clip(0, bins - 1).astype(int)
    vol_by_bin = np.bincount(idx, weights=volume, minlength=bins).astype(float)
    mids = (edges[:-1] + edges[1:]) / 2
    current_close = float(df["close"].iloc[-1])
    support = None
    mask_support = mids < current_close
    if mask_support.any():
        best = np.argmax(vol_by_bin[mask_support])
        support = float(mids[mask_support][best])
    resistance = None
    mask_res = mids > current_close
    if mask_res.any():
        best = np.argmax(vol_by_bin[mask_res])
        resistance = float(mids[mask_res][best])
    return support, resistance


def recent_pivot_low(df: pd.DataFrame, window: int = 5) -> Optional[float]:
    if df.empty:
        return None
    low = df["low"].astype(float)
    rolling_min = low.rolling(window, center=True).min()
    pivots = df[low == rolling_min]
    if not pivots.empty:
        return float(pivots["low"].iloc[-1])
    return float(low.min())


def recent_pivot_high(df: pd.DataFrame, window: int = 5) -> Optional[float]:
    if df.empty:
        return None
    high = df["high"].astype(float)
    rolling_max = high.rolling(window, center=True).max()
    pivots = df[high == rolling_max]
    if not pivots.empty:
        return float(pivots["high"].iloc[-1])
    return float(high.max())


def calc_levels(df_daily: pd.DataFrame, df_ind: pd.DataFrame, settings: Settings) -> Tuple[Optional[float], Optional[float], str, str]:
    if df_daily.empty or df_ind.empty:
        return None, None, "", ""
    close = float(df_daily["close"].iloc[-1])
    ma60 = df_ind["ma60"].iloc[-1]
    ma120 = df_ind["ma120"].iloc[-1]
    ma_candidates = []
    if pd.notna(ma60) and float(ma60) < close:
        ma_candidates.append(("MA60", float(ma60)))
    if pd.notna(ma120) and float(ma120) < close:
        ma_candidates.append(("MA120", float(ma120)))
    nearest_ma_type = ""
    nearest_ma = None
    if ma_candidates:
        nearest_ma_type, nearest_ma = min(ma_candidates, key=lambda x: abs(close - x[1]))
    pivot_low = recent_pivot_low(df_daily)
    pivot_high = recent_pivot_high(df_daily)
    vp_support, vp_resistance = volume_profile_levels(df_daily)
    support_level = None
    support_type = ""
    if nearest_ma is not None:
        dist = (close - nearest_ma) / close
        if dist <= settings.ma_support_max_distance_pct:
            support_level = nearest_ma
            support_type = nearest_ma_type
    if support_level is None and pivot_low is not None:
        support_level = float(pivot_low)
        support_type = "前低"
    if vp_support is not None:
        if support_level is None:
            support_level = float(vp_support)
            support_type = "成交密集区"
        else:
            if abs(float(support_level) - float(vp_support)) / close <= 0.03:
                support_level = max(float(support_level), float(vp_support))
                support_type = f"{support_type}+成交密集区"
    resistance_level = None
    resistance_type = ""
    resistance_candidates = []
    if pivot_high is not None and float(pivot_high) > close:
        resistance_candidates.append(("前高", float(pivot_high)))
    if vp_resistance is not None and float(vp_resistance) > close:
        resistance_candidates.append(("成交密集区", float(vp_resistance)))
    if resistance_candidates:
        resistance_type, resistance_level = min(resistance_candidates, key=lambda x: abs(x[1] - close))
    return support_level, resistance_level, support_type, resistance_type


# ----- scoring (来自原 laowang scoring 逻辑，略作裁剪，market_cap_score 设为 0) -----


@dataclass(frozen=True)
class RiskFlags:
    crash_filtered: bool
    high_pos_filtered: bool
    distance_from_recent_high: Optional[float]


def crash_filter(df_daily: pd.DataFrame, df_ind: pd.DataFrame) -> bool:
    if df_daily is None or df_daily.empty or len(df_daily) < 25:
        return False
    d = df_daily.tail(120).reset_index(drop=True)
    closes = pd.to_numeric(d["close"], errors="coerce")
    vols = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
    if closes.isna().any():
        return False
    ret = closes.pct_change()
    if (ret.tail(20) <= -0.09).any():
        return True
    if len(closes) >= 4:
        c3 = closes / closes.shift(3) - 1.0
        if (c3.tail(20) <= -0.15).any():
            return True
    if df_ind is not None and not df_ind.empty and "ma20" in df_ind.columns:
        ind = df_ind.tail(len(d)).reset_index(drop=True)
        ma20 = pd.to_numeric(ind.get("ma20"), errors="coerce")
        avg20 = vols.rolling(20).mean()
        panic = (vols > avg20 * 1.8) & (closes < ma20)
        if panic.tail(20).any():
            return True
    return False


def distance_from_recent_high(df_daily: pd.DataFrame, lookback: int = 120) -> Optional[float]:
    if df_daily is None or df_daily.empty:
        return None
    seg = df_daily.tail(int(lookback))
    if seg.empty:
        return None
    high_max = pd.to_numeric(seg["high"], errors="coerce").max()
    close = pd.to_numeric(seg["close"], errors="coerce").iloc[-1]
    if not np.isfinite(high_max) or not np.isfinite(close) or high_max <= 0:
        return None
    return float((high_max - close) / high_max)


def high_position_filter(dist: Optional[float]) -> bool:
    if dist is None or not np.isfinite(dist):
        return False
    return float(dist) < 0.15


def latest_sma(df_daily: pd.DataFrame, window: int) -> Optional[float]:
    if df_daily is None or df_daily.empty or window <= 0:
        return None
    close = pd.to_numeric(df_daily["close"], errors="coerce").tail(int(window))
    if len(close) < int(window) or close.isna().any():
        return None
    v = float(close.mean())
    return v if np.isfinite(v) else None


def score_trend(df_daily: pd.DataFrame, df_ind: pd.DataFrame) -> float:
    ma20 = df_ind["ma20"].iloc[-1]
    ma60 = df_ind["ma60"].iloc[-1]
    ma120 = df_ind["ma120"].iloc[-1]
    close = float(df_daily["close"].iloc[-1])
    if not all(pd.notna([ma20, ma60, ma120])) or close <= 0:
        return 0.0
    ma5 = latest_sma(df_daily, 5)
    ma10 = latest_sma(df_daily, 10)
    if ma5 is not None or ma10 is not None:
        above_fast_ma = (ma5 is not None and close > float(ma5)) or (ma10 is not None and close > float(ma10))
        if not above_fast_ma:
            return 0.0
    ma20 = float(ma20)
    ma60 = float(ma60)
    ma120 = float(ma120)
    if ma20 > ma60 > ma120 and close > ma60:
        ma60_series = pd.to_numeric(df_ind["ma60"], errors="coerce")
        if len(ma60_series) >= 15 and pd.notna(ma60_series.iloc[-15]):
            slope = float(ma60_series.iloc[-1]) - float(ma60_series.iloc[-15])
            if slope > 0:
                return 10.0
        return 6.0
    mas = np.array([ma20, ma60, ma120], dtype=float)
    if np.isfinite(mas).all() and (mas.max() - mas.min()) / close <= 0.02:
        return 3.0
    return 0.0


def score_pullback(df_daily: pd.DataFrame, df_ind: pd.DataFrame, support_level: Optional[float], settings: Settings) -> float:
    close = float(df_daily["close"].iloc[-1])
    prev_close = float(df_daily["close"].iloc[-2]) if len(df_daily) >= 2 else close
    ma60 = df_ind["ma60"].iloc[-1] if "ma60" in df_ind.columns else None
    if support_level is not None and np.isfinite(support_level):
        if close < float(support_level):
            return 0.0
        near_support = (close - float(support_level)) / close <= settings.at_support_pct
    else:
        near_support = False
    near_ma60 = False
    if ma60 is not None and pd.notna(ma60) and np.isfinite(float(ma60)) and close > 0:
        near_ma60 = abs(close - float(ma60)) / close <= settings.at_support_pct
    stopped_falling = close >= prev_close
    if (near_support or near_ma60) and stopped_falling:
        return 10.0
    if near_support or near_ma60:
        return 6.0
    return 0.0


def score_volume_price(df_daily: pd.DataFrame, breakout_lookback: int = 60) -> float:
    if df_daily is None or df_daily.empty or len(df_daily) < 40:
        return 5.0
    close = pd.to_numeric(df_daily["close"], errors="coerce")
    volume = pd.to_numeric(df_daily["volume"], errors="coerce").fillna(0.0)
    avg20 = volume.rolling(20).mean()
    if len(close) >= 2 and float(close.iloc[-1]) < float(close.iloc[-2]) and float(volume.iloc[-1]) >= 1.5 * float(avg20.iloc[-1] or 0):
        return 0.0
    rolling_high_prev = pd.to_numeric(df_daily["high"], errors="coerce").rolling(breakout_lookback).max().shift(1)
    breakout_idx = (close >= rolling_high_prev) & (volume >= 1.3 * avg20)
    if breakout_idx.tail(breakout_lookback).any():
        pullback_shrink = (close.diff() < 0) & (volume <= 0.9 * avg20)
        if pullback_shrink.tail(5).any():
            return 10.0
        return 5.0
    return 5.0


def score_rsi(rsi14: Optional[float]) -> float:
    if rsi14 is None or not np.isfinite(rsi14):
        return 0.0
    r = float(rsi14)
    if 45 <= r <= 60:
        return 10.0
    if 35 <= r < 45:
        return 6.0
    if r > 70:
        return 0.0
    if 60 < r <= 70:
        return 5.0
    if 30 <= r < 35:
        return 3.0
    return 0.0


def score_macd(df_ind: pd.DataFrame) -> float:
    if df_ind is None or df_ind.empty or len(df_ind) < 2:
        return 0.0
    latest = df_ind.iloc[-1]
    prev = df_ind.iloc[-2]
    diff = latest.get("macd_diff")
    dea = latest.get("macd_dea")
    prev_diff = prev.get("macd_diff")
    prev_dea = prev.get("macd_dea")
    if not all(pd.notna([diff, dea, prev_diff, prev_dea])):
        return 0.0
    diff = float(diff)
    dea = float(dea)
    prev_diff = float(prev_diff)
    prev_dea = float(prev_dea)
    crossed_up = prev_diff <= prev_dea and diff > dea
    crossed_down = prev_diff >= prev_dea and diff < dea
    if crossed_down:
        return 0.0
    if crossed_up:
        return 10.0 if diff > 0 else 5.0
    return 5.0 if diff > dea else 0.0


def score_base_structure(df_daily: pd.DataFrame, df_ind: pd.DataFrame, settings: Settings) -> Tuple[float, float, float]:
    if df_daily is None or df_daily.empty or df_ind is None or df_ind.empty:
        return 0.0, 0.0, 0.0
    seg = df_daily.tail(120).reset_index(drop=True)
    if len(seg) < 60:
        return 0.0, 0.0, 0.0
    low_min = pd.to_numeric(seg["low"], errors="coerce").min()
    high_max = pd.to_numeric(seg["high"], errors="coerce").max()
    if not np.isfinite(low_min) or not np.isfinite(high_max) or low_min <= 0:
        return 0.0, 0.0, 0.0
    amplitude = float((high_max - low_min) / low_min)
    platform = amplitude <= 0.20
    ma60 = pd.to_numeric(df_ind["ma60"], errors="coerce")
    atr14 = pd.to_numeric(df_ind["atr14"], errors="coerce")
    if len(ma60) >= 20 and pd.notna(ma60.iloc[-20]) and pd.notna(ma60.iloc[-1]):
        platform = platform and float(ma60.iloc[-1]) >= float(ma60.iloc[-20]) * 0.99
    if len(atr14) >= 20 and pd.notna(atr14.iloc[-20]) and pd.notna(atr14.iloc[-1]):
        platform = platform and float(atr14.iloc[-1]) <= float(atr14.iloc[-20])
    close = pd.to_numeric(df_daily["close"], errors="coerce")
    volume = pd.to_numeric(df_daily["volume"], errors="coerce").fillna(0.0)
    avg20 = volume.rolling(20).mean()
    rolling_high_prev = pd.to_numeric(df_daily["high"], errors="coerce").rolling(settings.breakout_lookback).max().shift(1)
    breakout = bool(
        pd.notna(rolling_high_prev.iloc[-1])
        and pd.notna(close.iloc[-1])
        and float(close.iloc[-1]) >= float(rolling_high_prev.iloc[-1])
        and float(volume.iloc[-1]) >= 1.3 * float(avg20.iloc[-1] or 0)
    )
    platform_score = 10.0 if (platform and breakout) else (6.0 if platform else 0.0)
    tail = df_daily.tail(20).reset_index(drop=True)
    c = pd.to_numeric(tail["close"], errors="coerce")
    v = pd.to_numeric(tail["volume"], errors="coerce").fillna(0.0)
    dc = c.diff()
    up = v[dc > 0]
    down = v[dc < 0]
    if len(up) == 0 and len(down) == 0:
        ratio = 1.0
    elif len(down) == 0:
        ratio = 2.0
    elif float(down.mean() or 0) <= 0:
        ratio = 2.0
    else:
        ratio = float((up.mean() or 0) / float(down.mean()))
    if ratio >= 1.8:
        turnover_score = 10.0
    elif 1.2 <= ratio < 1.8:
        turnover_score = 6.0
    else:
        turnover_score = 0.0
    base_structure_score = (platform_score + turnover_score) / 2.0
    return float(base_structure_score), float(platform_score), float(turnover_score)


def score_space(close: float, atr14: Optional[float], resistance_level: Optional[float], settings: Settings) -> Tuple[float, Optional[float], Optional[float]]:
    if close <= 0 or atr14 is None or not np.isfinite(atr14):
        return 0.0, None, None
    expected_return = float((float(atr14) * 20.0) / close)
    if not np.isfinite(expected_return):
        return 0.0, None, None
    if resistance_level is None or not np.isfinite(resistance_level) or float(resistance_level) <= close:
        resistance_distance = None
    else:
        resistance_distance = float((float(resistance_level) - close) / close)
    min_space_pct = float(settings.near_resistance_pct) * 2.0
    if resistance_distance is None:
        score = 5.0 if expected_return >= 0.18 else 0.0
        return score, expected_return, None
    if resistance_distance < min_space_pct:
        return 0.0, expected_return, float(resistance_distance)
    reachable_space = float(min(expected_return, resistance_distance))
    if reachable_space >= 0.18:
        score = 10.0
    elif reachable_space >= min_space_pct:
        score = 5.0
    else:
        score = 0.0
    return score, expected_return, float(resistance_distance)


def build_status_tags(
    *,
    close: float,
    support_level: Optional[float],
    resistance_level: Optional[float],
    trend_score: float,
    pullback_score: float,
    base_structure_score: float,
    space_score: float,
    risk_filtered: bool,
    settings: Settings,
) -> List[str]:
    tags: List[str] = []
    if risk_filtered:
        tags.append("RISK_FILTERED")
        return tags
    if trend_score >= 6:
        tags.append("TREND_UP")
    if base_structure_score >= 6:
        tags.append("LOW_BASE")
    if pullback_score >= 6:
        tags.append("PULLBACK")
    if support_level is not None and np.isfinite(support_level) and close > 0:
        if (close - float(support_level)) / close <= settings.at_support_pct:
            tags.append("AT_SUPPORT")
    if space_score >= 5:
        tags.append("SPACE_OK")
    if resistance_level is not None and np.isfinite(resistance_level) and close > 0:
        if float(resistance_level) > close and (float(resistance_level) - close) / close <= settings.near_resistance_pct:
            tags.append("NEAR_RESISTANCE")
    return tags


def calc_score_v3(
    *,
    df_daily: pd.DataFrame,
    df_ind: pd.DataFrame,
    support_level: Optional[float],
    resistance_level: Optional[float],
    settings: Settings,
) -> Tuple[float, float, float, float, float, float, float, float, float, str, RiskFlags]:
    close = float(df_daily["close"].iloc[-1])
    dist = distance_from_recent_high(df_daily, lookback=120)
    crash = crash_filter(df_daily, df_ind)
    high_pos = high_position_filter(dist)
    risk_flags = RiskFlags(crash_filtered=crash, high_pos_filtered=high_pos, distance_from_recent_high=dist)
    risk_filtered = crash or high_pos
    if risk_filtered:
        tags_json = json.dumps(["RISK_FILTERED"], ensure_ascii=False)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tags_json, risk_flags
    trend_score = score_trend(df_daily, df_ind)
    pullback_score = score_pullback(df_daily, df_ind, support_level, settings)
    volume_price_score = score_volume_price(df_daily, settings.breakout_lookback)
    rsi_score = score_rsi(float(df_ind["rsi14"].iloc[-1]) if pd.notna(df_ind["rsi14"].iloc[-1]) else None)
    macd_score = score_macd(df_ind)
    base_structure_score, *_ = score_base_structure(df_daily, df_ind, settings)
    space_score, *_ = score_space(
        close=close,
        atr14=float(df_ind["atr14"].iloc[-1]) if pd.notna(df_ind["atr14"].iloc[-1]) else None,
        resistance_level=resistance_level,
        settings=settings,
    )
    total = (
        (trend_score / 10.0) * 20.0
        + (pullback_score / 10.0) * 15.0
        + (volume_price_score / 10.0) * 10.0
        + (rsi_score / 10.0) * 10.0
        + (macd_score / 10.0) * 5.0
        + (base_structure_score / 10.0) * 15.0
        + (space_score / 10.0) * 15.0
        + 0.0  # market cap 简化为 0
    )
    tags = build_status_tags(
        close=close,
        support_level=support_level,
        resistance_level=resistance_level,
        trend_score=trend_score,
        pullback_score=pullback_score,
        base_structure_score=base_structure_score,
        space_score=space_score,
        risk_filtered=False,
        settings=settings,
    )
    tags_json = json.dumps(tags, ensure_ascii=False)
    return (
        float(total),
        float(trend_score),
        float(pullback_score),
        float(volume_price_score),
        float(rsi_score),
        float(macd_score),
        float(base_structure_score),
        float(space_score),
        0.0,
        tags_json,
        risk_flags,
    )


# ------------------------- DB 操作工具 -------------------------


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
    return df


def chunked(iterable: Sequence[Dict[str, object]], size: int = 1000) -> Iterable[List[Dict[str, object]]]:
    buff: List[Dict[str, object]] = []
    for item in iterable:
        buff.append(item)
        if len(buff) >= size:
            yield buff
            buff = []
    if buff:
        yield buff


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
        for chunk in chunked(rows):
            conn.execute(text(stmt), chunk)


def delete_by_trade_date(engine: Engine, table: str, trade_date: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table} WHERE trade_date = :d"), {"d": trade_date})


# ------------------------- 主流程 -------------------------


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
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    df_hist = load_history(engine, stock_code, target_dates[-1], min_rows=150)
    if df_hist.empty:
        return [], []
    df_ind = calc_indicators(df_hist)
    hist_dates = set(df_hist["date"].tolist())
    score_rows: List[Dict[str, object]] = []
    level_rows: List[Dict[str, object]] = []
    for trade_date in target_dates:
        if trade_date not in hist_dates:
            continue
        mask = df_hist["date"] <= trade_date
        df_slice = df_hist[mask]
        if len(df_slice) < 150:
            continue
        ind_slice = df_ind[df_ind["date"] <= trade_date]
        if ind_slice.empty:
            continue
        df_level_hist = df_slice.tail(settings.level_lookback).reset_index(drop=True)
        df_level_ind = ind_slice.tail(len(df_level_hist)).reset_index(drop=True)
        support, resistance, support_type, resistance_type = calc_levels(df_level_hist, df_level_ind, settings)
        (
            total,
            s_trend,
            s_pullback,
            s_vp,
            s_rsi,
            s_macd,
            s_base,
            s_space,
            s_mcap,
            status_tags,
            risk_flags,
        ) = calc_score_v3(df_daily=df_slice, df_ind=ind_slice, support_level=support, resistance_level=resistance, settings=settings)
        if risk_flags.crash_filtered or risk_flags.high_pos_filtered:
            continue
        score_rows.append(
            {
                "stock_code": stock_code,
                "score_date": trade_date,
                "total_score": total,
                "trend_score": s_trend,
                "pullback_score": s_pullback,
                "volume_price_score": s_vp,
                "rsi_score": s_rsi,
                "macd_score": s_macd,
                "base_structure_score": s_base,
                "space_score": s_space,
                "market_cap_score": s_mcap,
                "status_tags": status_tags,
            }
        )
        level_rows.append(
            {
                "stock_code": stock_code,
                "calc_date": trade_date,
                "support_level": support,
                "resistance_level": resistance,
                "support_type": support_type,
                "resistance_type": resistance_type,
            }
        )
    return score_rows, level_rows


def build_pool(engine: Engine, trade_date: str, top_n: int, min_score: float) -> None:
    delete_by_trade_date(engine, "model_laowang_pool", trade_date)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT s.stock_code,
                       COALESCE(i.name, '') AS stock_name,
                       d.close AS close,
                       l.support_level,
                       l.resistance_level,
                       s.total_score,
                       s.status_tags
                FROM stock_scores_v3 s
                LEFT JOIN stock_info i ON i.stock_code = s.stock_code
                LEFT JOIN stock_daily d ON d.stock_code = s.stock_code AND d.date = s.score_date
                LEFT JOIN stock_levels l ON l.stock_code = s.stock_code AND l.calc_date = s.score_date
                WHERE s.score_date = :d AND s.total_score >= :min_score
                ORDER BY s.total_score DESC
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
                "support_level": row[3],
                "resistance_level": row[4],
                "total_score": row[5],
                "status_tags": row[6],
            }
        )
    upsert_rows(engine, "model_laowang_pool", list(payload[0].keys()) if payload else ["trade_date", "stock_code"], payload, ["trade_date", "stock_code"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LAOWANG 评分脚本（单文件版）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--min-score", type=float, default=60.0)
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
    level_rows_all: List[Dict[str, object]] = []

    progress = tqdm(total=len(stock_codes), desc="LAOWANG评分", unit="stock")

    def worker(code: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        return compute_scores_for_stock(engine=engine, stock_code=code, target_dates=trade_dates, settings=settings)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, code): code for code in stock_codes}
        for fut in as_completed(futs):
            try:
                scores, levels = fut.result()
                score_rows_all.extend(scores)
                level_rows_all.extend(levels)
            except Exception as exc:  # noqa: BLE001
                logging.exception("计算 %s 失败: %s", futs[fut], exc)
            finally:
                progress.update(1)
    progress.close()

    logging.info("写入 stock_scores_v3: %d 行", len(score_rows_all))
    upsert_rows(
        engine,
        "stock_scores_v3",
        [
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
        ],
        score_rows_all,
        ["stock_code", "score_date"],
    )

    logging.info("写入 stock_levels: %d 行", len(level_rows_all))
    upsert_rows(
        engine,
        "stock_levels",
        ["stock_code", "calc_date", "support_level", "resistance_level", "support_type", "resistance_type"],
        level_rows_all,
        ["stock_code", "calc_date"],
    )

    pool_progress = tqdm(total=len(trade_dates), desc="生成股票池", unit="day")
    for trade_date in trade_dates:
        try:
            build_pool(engine, trade_date, top_n=int(args.top), min_score=float(args.min_score))
        except Exception as exc:  # noqa: BLE001
            logging.exception("生成 %s 股票池失败: %s", trade_date, exc)
        finally:
            pool_progress.update(1)
    pool_progress.close()

    logging.info("LAOWANG 完成：dates=%d stocks=%d", len(trade_dates), len(stock_codes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
