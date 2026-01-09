# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .settings import Settings


@dataclass(frozen=True)
class RiskFlags:
    crash_filtered: bool
    high_pos_filtered: bool
    distance_from_recent_high: Optional[float]


def crash_filter(df_daily: pd.DataFrame, df_ind: pd.DataFrame) -> bool:
    """
    Crash filter (v3):
    Recent 20 trading days, any of:
    1) single day drop >= 9%
    2) 3-day cumulative drop >= 15%
    3) panic selloff: volume > 20d avg * 1.8 and close < MA20
    """
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

    # Panic selloff: volume spike + close below MA20
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
    # Veto if current price is too close to recent high (<15% distance)
    if dist is None or not np.isfinite(dist):
        return False
    return float(dist) < 0.15


def latest_sma(df_daily: pd.DataFrame, window: int) -> Optional[float]:
    if df_daily is None or df_daily.empty:
        return None
    if window <= 0:
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

    # TREND_UP basic requirement: close above MA5 or MA10.
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


def score_pullback(
    df_daily: pd.DataFrame,
    df_ind: pd.DataFrame,
    support_level: Optional[float],
    settings: Settings,
) -> float:
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

    # Heavy selloff -> veto
    if len(close) >= 2 and float(close.iloc[-1]) < float(close.iloc[-2]) and float(volume.iloc[-1]) >= 1.5 * float(
        avg20.iloc[-1] or 0
    ):
        return 0.0

    rolling_high_prev = pd.to_numeric(df_daily["high"], errors="coerce").rolling(breakout_lookback).max().shift(1)
    breakout_idx = (close >= rolling_high_prev) & (volume >= 1.3 * avg20)

    if breakout_idx.tail(breakout_lookback).any():
        # Any pullback shrink volume in last 5 days
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
    """
    Returns: (base_structure_score, platform_score, turnover_score)
    """
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

    # Turnover completion factor: avg(up volume) / avg(down volume) on last 20 days.
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


def score_space(
    close: float,
    atr14: Optional[float],
    resistance_level: Optional[float],
    settings: Settings,
) -> Tuple[float, Optional[float], Optional[float]]:
    if close <= 0 or atr14 is None or not np.isfinite(atr14):
        return 0.0, None, None

    expected_return = float((float(atr14) * 20.0) / close)
    if not np.isfinite(expected_return):
        return 0.0, None, None

    if resistance_level is None or not np.isfinite(resistance_level) or float(resistance_level) <= close:
        resistance_distance = None
    else:
        resistance_distance = float((float(resistance_level) - close) / close)

    # Space score should be constrained by overhead resistance.
    # If resistance is too close, there is essentially no operating room even if ATR is high.
    min_space_pct = float(settings.near_resistance_pct) * 2.0

    if resistance_distance is None:
        # No usable resistance -> be conservative and only score by volatility potential.
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
) -> list[str]:
    tags: list[str] = []
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
    market_cap_score: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, str, RiskFlags]:
    """
    Returns:
    total_score,
    trend_score,
    pullback_score,
    volume_price_score,
    rsi_score,
    macd_score,
    base_structure_score,
    space_score,
    market_cap_score,
    status_tags(JSON string),
    risk_flags
    """
    close = float(df_daily["close"].iloc[-1])

    dist = distance_from_recent_high(df_daily, lookback=120)
    crash = crash_filter(df_daily, df_ind)
    high_pos = high_position_filter(dist)
    risk_flags = RiskFlags(crash_filtered=crash, high_pos_filtered=high_pos, distance_from_recent_high=dist)
    risk_filtered = crash or high_pos

    if risk_filtered:
        tags_json = json.dumps(["RISK_FILTERED"], ensure_ascii=False)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(market_cap_score), tags_json, risk_flags

    trend_score = score_trend(df_daily, df_ind)
    pullback_score = score_pullback(df_daily, df_ind, support_level, settings)
    volume_price_score = score_volume_price(df_daily, settings.breakout_lookback)
    rsi_score = score_rsi(float(df_ind["rsi14"].iloc[-1]) if pd.notna(df_ind["rsi14"].iloc[-1]) else None)
    macd_score = score_macd(df_ind)
    base_structure_score, _platform_score, _turnover_score = score_base_structure(df_daily, df_ind, settings)
    space_score, _expected_return, _resistance_distance = score_space(
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
        + (float(market_cap_score) / 10.0) * 10.0
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
        float(market_cap_score),
        tags_json,
        risk_flags,
    )
