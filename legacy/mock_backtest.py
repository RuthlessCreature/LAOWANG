# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

import relay_strategy_model as relay_model

EPS = 1e-9


@dataclass
class ModelVariant:
    name: str
    hidden: int
    epochs: int
    lr: float
    weight_decay: float
    target_gt: float
    seed: int


@dataclass
class SearchResult:
    variant: str
    alpha: int
    sell_rule: str
    top_k: int
    gap_min: Optional[float]
    gap_max: Optional[float]
    max_board: Optional[int]
    risk_profile: str
    max_broken_rate: Optional[float]
    min_red_rate: Optional[float]
    max_limit_down: Optional[int]
    max_pullback: Optional[float]
    alloc_pct: float
    train_acc: float
    threshold_metrics: Dict[float, Dict[str, float]]
    min_ret: float
    avg_ret: float
    objective: float


@dataclass
class ExecutionRules:
    source_file: str
    buy_delay: int
    block_limit_up_buy: bool
    fee_rate: float
    min_fee: float
    train_start_date: Optional[str]
    train_end_date: Optional[str]
    backtest_start_date: Optional[str]
    backtest_end_date: Optional[str]
    min_return_pct: Optional[float]
    max_drawdown_pct: Optional[float]


BASE_VARIANT_SPECS: List[Dict[str, Any]] = [
    {"name": "mlp_h48_e300_gt0", "hidden": 48, "epochs": 300, "lr": 0.030, "weight_decay": 1e-4, "target_gt": 0.00, "seed": 31},
    {"name": "mlp_h56_e320_gt1", "hidden": 56, "epochs": 320, "lr": 0.030, "weight_decay": 1e-4, "target_gt": 0.01, "seed": 32},
    {"name": "mlp_h64_e360_gt2", "hidden": 64, "epochs": 360, "lr": 0.028, "weight_decay": 1e-4, "target_gt": 0.02, "seed": 33},
    {"name": "mlp_h72_e380_gt15", "hidden": 72, "epochs": 380, "lr": 0.026, "weight_decay": 1e-4, "target_gt": 0.015, "seed": 34},
]

EXPANDED_VARIANT_SPECS: List[Dict[str, Any]] = [
    {"name": "mlp_h40_e280_gt5", "hidden": 40, "epochs": 280, "lr": 0.031, "weight_decay": 1e-4, "target_gt": 0.005, "seed": 42},
    {"name": "mlp_h96_e460_gt15", "hidden": 96, "epochs": 460, "lr": 0.023, "weight_decay": 1.5e-4, "target_gt": 0.015, "seed": 44},
]

AGGRESSIVE_VARIANT_SPECS: List[Dict[str, Any]] = [
    {"name": "mlp_h88_e500_gt1_wd2e4", "hidden": 88, "epochs": 500, "lr": 0.021, "weight_decay": 2e-4, "target_gt": 0.01, "seed": 52},
    {"name": "mlp_h128_e560_gt25", "hidden": 128, "epochs": 560, "lr": 0.020, "weight_decay": 2.2e-4, "target_gt": 0.025, "seed": 53},
]


def list_variant_specs(profile: str = "base") -> List[Dict[str, Any]]:
    key = str(profile or "base").strip().lower()
    specs: List[Dict[str, Any]] = list(BASE_VARIANT_SPECS)
    if key in {"expanded", "aggressive"}:
        specs.extend(EXPANDED_VARIANT_SPECS)
    if key == "aggressive":
        specs.extend(AGGRESSIVE_VARIANT_SPECS)
    return [dict(s) for s in specs]


def variant_spec_map(profile: str = "aggressive") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for spec in list_variant_specs(profile):
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        out[name] = dict(spec)
    return out


def resolve_variant_spec(variant_name: str) -> Optional[Dict[str, Any]]:
    return variant_spec_map("aggressive").get(str(variant_name or "").strip())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterative retrain + mock backtests.")
    p.add_argument("--config", default="config.ini")
    p.add_argument("--db-url", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--rules-file", default="rules.txt")
    p.add_argument("--start-date", default=None, help="Backtest start date (default from rules file or 2025-01-01).")
    p.add_argument("--end-date", default=None, help="Backtest end date (default from rules file or latest trade date).")
    p.add_argument("--train-start-date", default=None, help="Train start date (default from rules file or 2020-01-01).")
    p.add_argument("--train-end-date", default=None, help="Train end date (default from rules file or backtest-start-1d).")
    p.add_argument("--thresholds", default="0.5,0.75,0.99")
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--retain-ratio", type=float, default=0.80)
    p.add_argument("--min-return-floor", type=float, default=4.0)
    p.add_argument(
        "--variant-profile",
        choices=["base", "expanded", "aggressive"],
        default="base",
        help="Model architecture search profile: base keeps old behavior; expanded/aggressive broaden hidden-size/depth search.",
    )
    return p.parse_args(argv)


def _to_iso(d: str) -> str:
    s = str(d or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def _to_iso_optional(d: Optional[str]) -> Optional[str]:
    s = str(d or "").strip()
    if not s:
        return None
    try:
        return _to_iso(s)
    except Exception:
        return None


def _fmt_pct(v: float) -> str:
    if not math.isfinite(v):
        return "N/A"
    return f"{v * 100.0:.2f}%"


def _fmt_opt_pct(v: Optional[float]) -> str:
    if v is None:
        return "None"
    return f"{v * 100.0:.2f}%"


def _fmt_opt_int(v: Optional[int]) -> str:
    if v is None:
        return "None"
    return str(int(v))


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _safe_date(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def _threshold_tag(th: float) -> str:
    return f"{float(th):.2f}".rstrip("0").rstrip(".")


def _round_half_up_2(v: float) -> float:
    return float(math.floor(v * 100.0 + 0.5) / 100.0)


def _feature_cols() -> List[str]:
    cols = [
        "board_count",
        "one_word",
        "opened_limit",
        "ret1",
        "ret2",
        "ret3",
        "amount_chg1",
        "amp",
        "close_open_ret",
        "pullback",
        "lu_recent3",
        "broken_recent3",
        "float_cap_billion",
        "limit_up_count",
        "limit_down_count",
        "broken_rate",
        "max_board",
        "red_rate",
        "broken_red_rate",
        "amount_change5",
    ]
    lag_bases = [
        "limit_up_count",
        "limit_down_count",
        "broken_rate",
        "max_board",
        "red_rate",
        "broken_red_rate",
        "amount_change5",
    ]
    for c in lag_bases:
        for lag in (1, 2, 3):
            cols.append(f"{c}_lag{lag}")
    return cols


def _parse_fee_rate(text: str) -> float:
    m = re.search(r"手续费[^\d]*([\d.]+)\s*%", text)
    if not m:
        return 0.00025
    try:
        return float(m.group(1)) / 100.0
    except Exception:
        return 0.00025


def _parse_min_fee(text: str) -> float:
    m = re.search(r"不满\s*([\d.]+)\s*元", text)
    if not m:
        return 5.0
    try:
        return float(m.group(1))
    except Exception:
        return 5.0


def _parse_buy_delay(text: str) -> int:
    m = re.search(r"T\+(\d+)", text)
    if not m:
        return 1
    try:
        return max(1, int(m.group(1)))
    except Exception:
        return 1


def _parse_rule_date_range(text: str, keyword: str) -> Tuple[Optional[str], Optional[str]]:
    pattern = (
        rf"{re.escape(keyword)}[^\n\r\d]*"
        r"(\d{4}(?:-?\d{2}){2})\s*[-~—至]+\s*(\d{4}(?:-?\d{2}){2})"
    )
    m = re.search(pattern, text)
    if not m:
        return None, None
    start_raw = m.group(1).replace("-", "")
    end_raw = m.group(2).replace("-", "")
    return _to_iso_optional(start_raw), _to_iso_optional(end_raw)


def _parse_min_return_pct(text: str) -> Optional[float]:
    m = re.search(r"收益率[^\n\r\d]*(?:不低于|不少于|>=|＞=|大于等于)\s*([\d.]+)\s*%", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_max_drawdown_pct(text: str) -> Optional[float]:
    patterns = [
        r"最大回撤[^\n\r\d]*(?:不高于|不超过|<=|＜=|小于等于)\s*([\d.]+)\s*%",
        # Legacy rules may write '不低于' by mistake; still treat as drawdown upper bound.
        r"最大回撤[^\n\r\d]*(?:不低于|不少于)\s*([\d.]+)\s*%",
    ]
    for ptn in patterns:
        m = re.search(ptn, text)
        if not m:
            continue
        try:
            return float(m.group(1))
        except Exception:
            continue
    return None


def _load_execution_rules(path: str) -> ExecutionRules:
    p = Path(path)
    if not p.exists():
        return ExecutionRules(
            source_file=path,
            buy_delay=1,
            block_limit_up_buy=True,
            fee_rate=0.00025,
            min_fee=5.0,
            train_start_date=None,
            train_end_date=None,
            backtest_start_date=None,
            backtest_end_date=None,
            min_return_pct=None,
            max_drawdown_pct=None,
        )
    text = p.read_text(encoding="utf-8", errors="ignore")
    train_start, train_end = _parse_rule_date_range(text, "训练集")
    backtest_start, backtest_end = _parse_rule_date_range(text, "回测集")
    return ExecutionRules(
        source_file=str(p.as_posix()),
        buy_delay=_parse_buy_delay(text),
        block_limit_up_buy=("涨停" in text and "买不到" in text),
        fee_rate=_parse_fee_rate(text),
        min_fee=_parse_min_fee(text),
        train_start_date=train_start,
        train_end_date=train_end,
        backtest_start_date=backtest_start,
        backtest_end_date=backtest_end,
        min_return_pct=_parse_min_return_pct(text),
        max_drawdown_pct=_parse_max_drawdown_pct(text),
    )


def _load_trade_dates(engine, start_date: str, end_date: Optional[str] = None) -> List[str]:
    if end_date:
        sql = text("SELECT DISTINCT date FROM stock_daily WHERE date >= :s AND date <= :e ORDER BY date ASC")
        params = {"s": start_date, "e": end_date}
    else:
        sql = text("SELECT DISTINCT date FROM stock_daily WHERE date >= :s ORDER BY date ASC")
        params = {"s": start_date}
    with engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def _attach_future_fields(engine, df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    with engine.connect() as conn:
        d = pd.read_sql(
            text("SELECT stock_code, date, open, close FROM stock_daily WHERE date >= :s"),
            conn,
            params={"s": start_date},
        )
    out = df.copy()
    future_cols = [
        "next_date",
        "next_open",
        "next_close",
        "next2_date",
        "next2_close",
        "next3_date",
        "next3_close",
    ]
    out = out.drop(columns=[c for c in future_cols if c in out.columns], errors="ignore")
    if d.empty:
        for c in future_cols:
            out[c] = np.nan
        return out

    d["stock_code"] = d["stock_code"].astype(str)
    d["date"] = d["date"].astype(str)
    d["open"] = pd.to_numeric(d["open"], errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.sort_values(["stock_code", "date"])
    grp = d.groupby("stock_code", sort=False)
    d["next_date"] = grp["date"].shift(-1)
    d["next_open"] = grp["open"].shift(-1)
    d["next_close"] = grp["close"].shift(-1)
    d["next2_date"] = grp["date"].shift(-2)
    d["next2_close"] = grp["close"].shift(-2)
    d["next3_date"] = grp["date"].shift(-3)
    d["next3_close"] = grp["close"].shift(-3)

    keep = ["stock_code", "date"] + future_cols
    return out.merge(d[keep], on=["stock_code", "date"], how="left")


def _score_from_raw_prob(raw_prob: np.ndarray, alpha: int) -> np.ndarray:
    arr = np.asarray(raw_prob, dtype=float)
    if arr.size == 0:
        return arr
    fill_v = float(np.nanmean(arr)) if np.isfinite(arr).any() else 0.5
    arr = np.where(np.isfinite(arr), arr, fill_v)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    cdf = ranks / float(len(arr))
    score = 1.0 - np.power(1.0 - cdf, int(alpha))
    return np.clip(score, 0.0, 1.0)


def _build_ranked_by_date(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    if df.empty:
        return {}
    ordered = df.sort_values(["date", "score"], ascending=[True, False])
    out: Dict[str, List[Dict[str, Any]]] = {}
    for d, g in ordered.groupby("date", sort=True):
        rows: List[Dict[str, Any]] = []
        for row in g.itertuples(index=False):
            rows.append(
                {
                    "date": str(getattr(row, "date")),
                    "stock_code": str(getattr(row, "stock_code", "")),
                    "stock_name": str(getattr(row, "stock_name", "")),
                    "is_st": bool(getattr(row, "is_st", False)),
                    "score": _safe_float(getattr(row, "score")),
                    "raw_prob": _safe_float(getattr(row, "raw_prob")),
                    "close": _safe_float(getattr(row, "close")),
                    "next_date": _safe_date(getattr(row, "next_date", "")),
                    "next_open": _safe_float(getattr(row, "next_open")),
                    "next2_date": _safe_date(getattr(row, "next2_date", "")),
                    "next2_close": _safe_float(getattr(row, "next2_close")),
                    "next3_date": _safe_date(getattr(row, "next3_date", "")),
                    "next3_close": _safe_float(getattr(row, "next3_close")),
                    "board_count": int(_safe_float(getattr(row, "board_count"), 0.0)),
                    "ret1": _safe_float(getattr(row, "ret1")),
                    "amp": _safe_float(getattr(row, "amp")),
                    "pullback": _safe_float(getattr(row, "pullback")),
                    "close_open_ret": _safe_float(getattr(row, "close_open_ret")),
                    "broken_rate": _safe_float(getattr(row, "broken_rate")),
                    "red_rate": _safe_float(getattr(row, "red_rate")),
                    "limit_up_count": int(_safe_float(getattr(row, "limit_up_count"), 0.0)),
                    "limit_down_count": int(_safe_float(getattr(row, "limit_down_count"), 0.0)),
                    "amount_change5": _safe_float(getattr(row, "amount_change5")),
                    "tomorrow_stage": str(getattr(row, "tomorrow_stage", "N/A") or "N/A"),
                    # Some callers build ranked pools before market-stage merge.
                    "tomorrow_prob": _safe_float(getattr(row, "tomorrow_prob", float("nan"))),
                }
            )
        out[str(d)] = rows
    return out


def _limit_pct_for_code(stock_code: str, is_st: bool) -> float:
    code = str(stock_code or "")
    if bool(is_st):
        return 0.05
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def _limit_up_price(prev_close: float, stock_code: str, is_st: bool) -> float:
    if not (math.isfinite(prev_close) and prev_close > 0):
        return float("nan")
    pct = _limit_pct_for_code(stock_code=stock_code, is_st=is_st)
    return _round_half_up_2(prev_close * (1.0 + pct))


def _calc_fee(amount: float, fee_rate: float, min_fee: float) -> float:
    if not (math.isfinite(amount) and amount > 0):
        return 0.0
    return max(float(min_fee), float(amount) * float(fee_rate))


def _max_lots_for_cash(cash: float, buy_px: float, fee_rate: float, min_fee: float) -> int:
    if not (math.isfinite(cash) and cash > 0 and math.isfinite(buy_px) and buy_px > 0):
        return 0
    lots = int(cash // (buy_px * 100.0))
    while lots > 0:
        amt = lots * 100.0 * buy_px
        need = amt + _calc_fee(amt, fee_rate=fee_rate, min_fee=min_fee)
        if need <= cash + 1e-9:
            return lots
        lots -= 1
    return 0


def _can_buy_by_rules(
    row: Dict[str, Any],
    rules: ExecutionRules,
    gap_min: Optional[float],
    gap_max: Optional[float],
) -> Tuple[bool, str]:
    buy_date = _safe_date(row.get("next_date"))
    buy_px = _safe_float(row.get("next_open"))
    if not buy_date:
        return False, "missing_t1_trade_date"
    if not (math.isfinite(buy_px) and buy_px > 0):
        return False, "missing_t1_open"

    signal_close = _safe_float(row.get("close"))
    if (gap_min is not None) or (gap_max is not None):
        if not (math.isfinite(signal_close) and signal_close > 0):
            return False, "missing_signal_close"
        gap = buy_px / signal_close - 1.0
        if gap_min is not None and gap < float(gap_min) - EPS:
            return False, "gap_below_min"
        if gap_max is not None and gap > float(gap_max) + EPS:
            return False, "gap_above_max"

    if rules.block_limit_up_buy:
        limit_up = _limit_up_price(
            prev_close=signal_close,
            stock_code=str(row.get("stock_code", "")),
            is_st=bool(row.get("is_st", False)),
        )
        if math.isfinite(limit_up) and buy_px >= limit_up - EPS:
            return False, "limit_up_unfilled"
    return True, ""


def _passes_risk_filters(
    row: Dict[str, Any],
    *,
    max_broken_rate: Optional[float],
    min_red_rate: Optional[float],
    max_limit_down: Optional[int],
    max_pullback: Optional[float],
) -> Tuple[bool, str]:
    if max_broken_rate is not None:
        broken_rate = _safe_float(row.get("broken_rate"))
        if not math.isfinite(broken_rate) or broken_rate > float(max_broken_rate) + EPS:
            return False, "broken_rate_high"

    if min_red_rate is not None:
        red_rate = _safe_float(row.get("red_rate"))
        if not math.isfinite(red_rate) or red_rate < float(min_red_rate) - EPS:
            return False, "red_rate_low"

    if max_limit_down is not None:
        limit_down_count = int(_safe_float(row.get("limit_down_count"), 0.0))
        if limit_down_count > int(max_limit_down):
            return False, "limit_down_high"

    if max_pullback is not None:
        pullback = _safe_float(row.get("pullback"))
        if not math.isfinite(pullback) or pullback > float(max_pullback) + EPS:
            return False, "pullback_high"
    return True, ""


def _decide_sell(row: Dict[str, Any], buy_price: float, sell_rule: str) -> Tuple[str, float, str]:
    t2 = _safe_float(row.get("next2_close"))
    t3 = _safe_float(row.get("next3_close"))
    d2 = _safe_date(row.get("next2_date"))
    d3 = _safe_date(row.get("next3_date"))
    if not (math.isfinite(buy_price) and buy_price > 0 and math.isfinite(t2) and t2 > 0 and d2):
        return "", float("nan"), "缺少T+2收盘价，无法卖出"

    r2 = t2 / buy_price - 1.0
    if sell_rule == "t2_close":
        return d2, t2, "固定规则：T+2收盘卖出（买入后第1个可卖日）"
    if sell_rule == "weak_hold_t3":
        if r2 <= -0.03 and math.isfinite(t3) and t3 > 0 and d3:
            return d3, t3, f"T+2收盘回撤{r2*100:.2f}%<=-3%，延长到T+3收盘卖出"
        return d2, t2, f"T+2收盘回撤{r2*100:.2f}%未触发延长，T+2收盘卖出"
    if sell_rule == "strong_hold_t3":
        if r2 >= 0.06 and math.isfinite(t3) and t3 > 0 and d3:
            return d3, t3, f"T+2收盘上涨{r2*100:.2f}%>=6%，延长到T+3收盘卖出"
        return d2, t2, f"T+2收盘上涨{r2*100:.2f}%未触发延长，T+2收盘卖出"
    if sell_rule == "profit_lock_t3":
        if 0.02 <= r2 <= 0.10 and math.isfinite(t3) and t3 > 0 and d3:
            return d3, t3, f"T+2收盘收益{r2*100:.2f}%落在2%~10%，延长到T+3收盘卖出"
        return d2, t2, f"T+2收盘收益{r2*100:.2f}%不在2%~10%，T+2收盘卖出"
    if sell_rule == "loss_rebound_t3":
        if r2 <= -0.05 and math.isfinite(t3) and t3 > 0 and d3:
            return d3, t3, f"T+2收盘回撤{r2*100:.2f}%<=-5%，延长到T+3收盘卖出"
        return d2, t2, f"T+2收盘回撤{r2*100:.2f}%>-5%，T+2收盘卖出"
    return d2, t2, "未知规则，降级为T+2收盘卖出"


def _select_candidate(
    rows: List[Dict[str, Any]],
    *,
    threshold: float,
    top_k: int,
    max_board: Optional[int],
    gap_min: Optional[float],
    gap_max: Optional[float],
    max_broken_rate: Optional[float],
    min_red_rate: Optional[float],
    max_limit_down: Optional[int],
    max_pullback: Optional[float],
    rules: ExecutionRules,
) -> Tuple[Optional[Dict[str, Any]], str, int]:
    if not rows:
        return None, "no_candidate", 0

    cap = min(len(rows), max(1, int(top_k)))
    cand = rows[:cap]
    has_above_threshold = False
    last_reason = "threshold_blocked"

    for idx, row in enumerate(cand, 1):
        if _safe_float(row.get("score")) < threshold:
            continue
        has_above_threshold = True

        board = int(_safe_float(row.get("board_count"), 0.0))
        if (max_board is not None) and (board > int(max_board)):
            last_reason = "board_blocked"
            continue

        ok_risk, _ = _passes_risk_filters(
            row=row,
            max_broken_rate=max_broken_rate,
            min_red_rate=min_red_rate,
            max_limit_down=max_limit_down,
            max_pullback=max_pullback,
        )
        if not ok_risk:
            last_reason = "risk_blocked"
            continue

        ok_buy, reason = _can_buy_by_rules(
            row=row,
            rules=rules,
            gap_min=gap_min,
            gap_max=gap_max,
        )
        if not ok_buy:
            if "limit_up" in reason:
                last_reason = "rule_blocked"
            elif "gap_" in reason:
                last_reason = "gap_blocked"
            else:
                last_reason = "bad_buy_quote"
            continue
        return row, "ok", idx

    if not has_above_threshold:
        return None, "threshold_blocked", 0
    return None, last_reason, 0


def _simulate_fast(
    *,
    trade_dates: List[str],
    ranked_by_date: Dict[str, List[Dict[str, Any]]],
    threshold: float,
    initial_capital: float,
    sell_rule: str,
    top_k: int,
    max_board: Optional[int],
    gap_min: Optional[float],
    gap_max: Optional[float],
    max_broken_rate: Optional[float],
    min_red_rate: Optional[float],
    max_limit_down: Optional[int],
    max_pullback: Optional[float],
    alloc_pct: float,
    rules: ExecutionRules,
) -> Dict[str, float]:
    date_idx = {d: i for i, d in enumerate(trade_dates)}
    cash = float(initial_capital)
    trades = 0
    total_fee = 0.0
    curve: List[float] = [cash]
    i = 0

    while i < len(trade_dates):
        d = trade_dates[i]
        row, _, _ = _select_candidate(
            rows=ranked_by_date.get(d, []),
            threshold=threshold,
            top_k=top_k,
            max_board=max_board,
            gap_min=gap_min,
            gap_max=gap_max,
            max_broken_rate=max_broken_rate,
            min_red_rate=min_red_rate,
            max_limit_down=max_limit_down,
            max_pullback=max_pullback,
            rules=rules,
        )
        if row is None:
            i += 1
            continue

        buy_date = _safe_date(row.get("next_date"))
        buy_px = _safe_float(row.get("next_open"))
        if buy_date not in date_idx or not (math.isfinite(buy_px) and buy_px > 0):
            i += 1
            continue
        buy_idx = date_idx[buy_date]
        if buy_idx <= i:
            i += 1
            continue

        lots = _max_lots_for_cash(
            cash=cash * max(0.0, min(1.0, float(alloc_pct))),
            buy_px=buy_px,
            fee_rate=rules.fee_rate,
            min_fee=rules.min_fee,
        )
        shares = lots * 100
        if shares < 100:
            i += 1
            continue

        sell_date, sell_px, _ = _decide_sell(row, buy_price=buy_px, sell_rule=sell_rule)
        if not sell_date or sell_date not in date_idx or not (math.isfinite(sell_px) and sell_px > 0):
            i += 1
            continue
        sell_idx = date_idx[sell_date]
        if sell_idx <= buy_idx:
            i += 1
            continue

        buy_amt = shares * buy_px
        buy_fee = _calc_fee(buy_amt, fee_rate=rules.fee_rate, min_fee=rules.min_fee)
        if buy_amt + buy_fee > cash + 1e-9:
            i += 1
            continue

        sell_amt = shares * sell_px
        sell_fee = _calc_fee(sell_amt, fee_rate=rules.fee_rate, min_fee=rules.min_fee)
        cash = cash - buy_amt - buy_fee + sell_amt - sell_fee
        total_fee += buy_fee + sell_fee
        trades += 1
        curve.append(cash)
        i = sell_idx + 1

    ret = cash / float(initial_capital) - 1.0
    peak = curve[0] if curve else float(initial_capital)
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)
    return {
        "final": cash,
        "ret": ret,
        "trades": float(trades),
        "fee": total_fee,
        "max_dd": max_dd,
    }
def _simulate_detailed(
    *,
    trade_dates: List[str],
    ranked_by_date: Dict[str, List[Dict[str, Any]]],
    threshold: float,
    initial_capital: float,
    sell_rule: str,
    variant_name: str,
    alpha: int,
    top_k: int,
    max_board: Optional[int],
    gap_min: Optional[float],
    gap_max: Optional[float],
    max_broken_rate: Optional[float],
    min_red_rate: Optional[float],
    max_limit_down: Optional[int],
    max_pullback: Optional[float],
    risk_profile: str,
    alloc_pct: float,
    rules: ExecutionRules,
) -> Dict[str, Any]:
    date_idx = {d: i for i, d in enumerate(trade_dates)}
    cash = float(initial_capital)
    total_fee = 0.0
    trades: List[Dict[str, Any]] = []
    skip_counts: Dict[str, int] = {
        "no_candidate": 0,
        "threshold_blocked": 0,
        "board_blocked": 0,
        "risk_blocked": 0,
        "gap_blocked": 0,
        "rule_blocked": 0,
        "bad_buy_quote": 0,
        "calendar_miss": 0,
        "insufficient_cash": 0,
        "bad_exit": 0,
    }

    i = 0
    while i < len(trade_dates):
        d = trade_dates[i]
        row, reason, rank = _select_candidate(
            rows=ranked_by_date.get(d, []),
            threshold=threshold,
            top_k=top_k,
            max_board=max_board,
            gap_min=gap_min,
            gap_max=gap_max,
            max_broken_rate=max_broken_rate,
            min_red_rate=min_red_rate,
            max_limit_down=max_limit_down,
            max_pullback=max_pullback,
            rules=rules,
        )
        if row is None:
            skip_counts[reason] = int(skip_counts.get(reason, 0)) + 1
            i += 1
            continue

        buy_date = _safe_date(row.get("next_date"))
        buy_px = _safe_float(row.get("next_open"))
        if not (math.isfinite(buy_px) and buy_px > 0 and buy_date):
            skip_counts["bad_buy_quote"] += 1
            i += 1
            continue
        if buy_date not in date_idx:
            skip_counts["calendar_miss"] += 1
            i += 1
            continue
        buy_idx = date_idx[buy_date]
        if buy_idx <= i:
            skip_counts["calendar_miss"] += 1
            i += 1
            continue

        lots = _max_lots_for_cash(
            cash=cash * max(0.0, min(1.0, float(alloc_pct))),
            buy_px=buy_px,
            fee_rate=rules.fee_rate,
            min_fee=rules.min_fee,
        )
        shares = lots * 100
        if shares < 100:
            skip_counts["insufficient_cash"] += 1
            i += 1
            continue

        sell_date, sell_px, sell_logic = _decide_sell(row, buy_price=buy_px, sell_rule=sell_rule)
        if not sell_date or sell_date not in date_idx or not (math.isfinite(sell_px) and sell_px > 0):
            skip_counts["bad_exit"] += 1
            i += 1
            continue
        sell_idx = date_idx[sell_date]
        if sell_idx <= buy_idx:
            skip_counts["bad_exit"] += 1
            i += 1
            continue

        buy_amt = shares * buy_px
        buy_fee = _calc_fee(buy_amt, fee_rate=rules.fee_rate, min_fee=rules.min_fee)
        if buy_amt + buy_fee > cash + 1e-9:
            skip_counts["insufficient_cash"] += 1
            i += 1
            continue
        sell_amt = shares * sell_px
        sell_fee = _calc_fee(sell_amt, fee_rate=rules.fee_rate, min_fee=rules.min_fee)

        cash_before = cash
        cash = cash - buy_amt - buy_fee + sell_amt - sell_fee
        total_fee += buy_fee + sell_fee
        pnl = cash - cash_before
        pnl_pct = pnl / (buy_amt + buy_fee) if (buy_amt + buy_fee) > 0 else 0.0

        pick_reason = (
            f"Top{top_k} pick rank={rank}; score={_fmt_pct(_safe_float(row.get('score')))}; "
            f"raw_prob={_fmt_pct(_safe_float(row.get('raw_prob')))}; "
            f"board={int(_safe_float(row.get('board_count'), 0.0))}; "
            f"ret1={_fmt_pct(_safe_float(row.get('ret1')))}."
        )
        buy_reason = (
            f"model={variant_name}, alpha={alpha}; score>=threshold; "
            f"gap_filter=[{_fmt_opt_pct(gap_min)}, {_fmt_opt_pct(gap_max)}], "
            f"max_board={_fmt_opt_int(max_board)}; "
            f"risk_profile={risk_profile} "
            f"(broken<={_fmt_opt_pct(max_broken_rate)}, red>={_fmt_opt_pct(min_red_rate)}, "
            f"limit_down<={_fmt_opt_int(max_limit_down)}, pullback<={_fmt_opt_pct(max_pullback)}); "
            f"capital_use={max(0.0, min(1.0, float(alloc_pct)))*100:.0f}%; "
            "T signal -> T+1 open buy, limit-up open is unfilled."
        )
        sell_reason = f"{sell_logic} (no intraday-high optimization)."

        trades.append(
            {
                "signal_date": d,
                "buy_date": buy_date,
                "buy_time": "09:30:00",
                "sell_date": sell_date,
                "sell_time": "15:00:00",
                "stock_code": row.get("stock_code", ""),
                "stock_name": row.get("stock_name", ""),
                "shares": shares,
                "buy_price": buy_px,
                "sell_price": sell_px,
                "buy_amount": buy_amt,
                "sell_amount": sell_amt,
                "buy_fee": buy_fee,
                "sell_fee": sell_fee,
                "total_fee": buy_fee + sell_fee,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "equity_after": cash,
                "pick_reason": pick_reason,
                "buy_reason": buy_reason,
                "sell_reason": sell_reason,
            }
        )
        i = sell_idx + 1

    trade_n = len(trades)
    win_n = sum(1 for t in trades if t["pnl"] > 0)
    loss_n = sum(1 for t in trades if t["pnl"] < 0)
    win_rate = (win_n / trade_n) if trade_n else 0.0
    avg_pnl = (sum(t["pnl_pct"] for t in trades) / trade_n) if trade_n else 0.0
    final_cap = cash
    total_ret = final_cap / float(initial_capital) - 1.0

    curve = [initial_capital]
    for t in trades:
        curve.append(float(t["equity_after"]))
    peak = curve[0] if curve else initial_capital
    max_dd = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)

    return {
        "threshold": threshold,
        "initial_capital": initial_capital,
        "final_capital": final_cap,
        "total_ret": total_ret,
        "total_fee": total_fee,
        "trade_n": trade_n,
        "win_n": win_n,
        "loss_n": loss_n,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "max_dd": max_dd,
        "trades": trades,
        "skip_counts": skip_counts,
    }


def _sort_key_for_threshold(sr: SearchResult, th: float) -> Tuple[float, float, float, float]:
    m = sr.threshold_metrics.get(float(th), {})
    ret = float(m.get("ret", -1e9))
    max_dd = float(m.get("max_dd", 1e9))
    trades = float(m.get("trades", 0.0))
    return (
        ret,
        -max_dd,
        trades,
        float(sr.min_ret),
    )


def _select_best_for_threshold(
    candidates: List[SearchResult],
    threshold: float,
    retain_ratio: float = 0.80,
    min_return_floor: float = 4.0,
    target_return_pct: Optional[float] = None,
    target_max_dd_pct: Optional[float] = None,
) -> SearchResult:
    if not candidates:
        raise ValueError("empty candidates")

    rows: List[Tuple[SearchResult, float, float, float]] = []
    best_ret = -1e9
    for sr in candidates:
        m = sr.threshold_metrics.get(float(threshold), {})
        ret = float(m.get("ret", -1e9))
        dd = float(m.get("max_dd", 1e9))
        trades = float(m.get("trades", 0.0))
        rows.append((sr, ret, dd, trades))
        if ret > best_ret:
            best_ret = ret

    # Rule-aware selection: if there are candidates satisfying explicit return/DD targets,
    # prefer the one with highest return, then lowest drawdown.
    if target_return_pct is not None or target_max_dd_pct is not None:
        feasible_rows: List[Tuple[SearchResult, float, float, float]] = []
        for item in rows:
            _, ret, dd, _ = item
            ok_ret = (target_return_pct is None) or (ret >= float(target_return_pct) - 1e-12)
            ok_dd = (target_max_dd_pct is None) or (dd <= float(target_max_dd_pct) + 1e-12)
            if ok_ret and ok_dd:
                feasible_rows.append(item)
        if feasible_rows:
            feasible_rows.sort(key=lambda x: (x[1], -x[2], x[3]), reverse=True)
            return feasible_rows[0][0]

    rr = max(0.01, min(1.0, float(retain_ratio)))
    abs_floor = float(min_return_floor)
    if best_ret >= 0:
        floor = max(best_ret * rr, abs_floor)
    else:
        floor = best_ret / rr

    filtered = [r for r in rows if r[1] >= floor]
    if not filtered and math.isfinite(abs_floor):
        filtered = [r for r in rows if r[1] >= abs_floor]
    if not filtered:
        filtered = rows

    filtered.sort(key=lambda x: (x[2], -x[1], -x[3]))
    return filtered[0][0]


def _write_operations_csv(path: Path, trades: Sequence[Dict[str, Any]], initial_capital: float) -> None:
    import csv

    headers = [
        "操作日期",
        "操作时间",
        "股票代码",
        "股票名称",
        "操作（买or卖）",
        "操作价格",
        "操作原因",
        "总资产",
    ]
    rows: List[List[str]] = []
    cash = float(initial_capital)
    for t in trades:
        buy_fee = float(t["buy_fee"])
        buy_asset = cash - buy_fee
        rows.append(
            [
                str(t["buy_date"]),
                str(t.get("buy_time", "09:30:00")),
                str(t["stock_code"]),
                str(t["stock_name"]),
                "buy",
                f"{float(t['buy_price']):.3f}",
                str(t.get("buy_reason", "")),
                f"{buy_asset:.2f}",
            ]
        )
        cash = float(t["equity_after"])
        rows.append(
            [
                str(t["sell_date"]),
                str(t.get("sell_time", "15:00:00")),
                str(t["stock_code"]),
                str(t["stock_name"]),
                "sell",
                f"{float(t['sell_price']):.3f}",
                str(t.get("sell_reason", "")),
                f"{cash:.2f}",
            ]
        )

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _render_report(
    *,
    threshold: float,
    start_date: str,
    end_date: str,
    variant: ModelVariant,
    selected: SearchResult,
    detail: Dict[str, Any],
    rules: ExecutionRules,
    search_top: List[SearchResult],
    retain_ratio: float,
    min_return_floor: float,
) -> str:
    lines: List[str] = [
        f"# Mock Backtest Report (threshold {threshold:.2f})",
        "",
        f"- Range: {start_date} ~ {end_date}",
        f"- Initial capital: {detail['initial_capital']:.2f}",
        (
            f"- Final strategy: model={variant.name} "
            f"(hidden={variant.hidden}, epochs={variant.epochs}, lr={variant.lr}, "
            f"target=T+1 open -> T+2 close > {variant.target_gt:.2%})"
        ),
        f"- Score mapping: score = 1 - (1 - CDF(raw_prob))^{selected.alpha}",
        f"- Sell rule: {selected.sell_rule}",
        (
            f"- Filters: TopK={selected.top_k}, gap=[{_fmt_opt_pct(selected.gap_min)}, "
            f"{_fmt_opt_pct(selected.gap_max)}], max_board={_fmt_opt_int(selected.max_board)}, "
            f"risk={selected.risk_profile}, broken<={_fmt_opt_pct(selected.max_broken_rate)}, "
            f"red>={_fmt_opt_pct(selected.min_red_rate)}, limit_down<={_fmt_opt_int(selected.max_limit_down)}, "
            f"pullback<={_fmt_opt_pct(selected.max_pullback)}, capital_use={selected.alloc_pct*100:.0f}%"
        ),
        f"- Model train acc: {_fmt_pct(selected.train_acc)}",
        "",
        "## Rules (rules.txt)",
        f"- Rules file: {rules.source_file}",
        f"- Buy: T signal then T+{rules.buy_delay} open (current run uses T+1)",
        f"- Limit-up open unfilled: {'yes' if rules.block_limit_up_buy else 'no'}",
        f"- Fee: {rules.fee_rate*100:.4f}% per side, min {rules.min_fee:.2f} CNY/order",
        "- Slippage: ignored",
        "- Execution: buy at T+1 open, sell at rule close, no intraday-high optimization",
        (
            f"- Selection policy: keep candidates above max({retain_ratio*100:.0f}% of best return, "
            f"{_fmt_pct(min_return_floor)}), then minimize drawdown"
        ),
        (
            f"- Rule targets: return>={rules.min_return_pct if rules.min_return_pct is not None else 'N/A'}%, "
            f"max_dd<={rules.max_drawdown_pct if rules.max_drawdown_pct is not None else 'N/A'}%; "
            "if feasible candidates exist, prioritize rule-feasible set first"
        ),
        "",
        "## Search Top 8 (current threshold)",
    ]
    for i, r in enumerate(search_top[:8], 1):
        cur = r.threshold_metrics.get(float(threshold), {})
        lines.append(
            f"- {i}. {r.variant}, alpha={r.alpha}, rule={r.sell_rule}, topk={r.top_k}, "
            f"gap=[{_fmt_opt_pct(r.gap_min)},{_fmt_opt_pct(r.gap_max)}], "
            f"board<={_fmt_opt_int(r.max_board)}, risk={r.risk_profile}, "
            f"alloc={r.alloc_pct*100:.0f}%, train_acc={_fmt_pct(r.train_acc)}, "
            f"R@{threshold:.2f}={_fmt_pct(cur.get('ret', float('nan')))}, "
            f"DD@{threshold:.2f}={_fmt_pct(cur.get('max_dd', float('nan')))}, "
            f"trades={int(cur.get('trades', 0.0) or 0)}, minR={_fmt_pct(r.min_ret)}"
        )

    lines.extend(
        [
            "",
            "## Summary",
            f"- Final capital: {detail['final_capital']:.2f}",
            f"- Total return: {_fmt_pct(detail['total_ret'])}",
            f"- Total fees: {detail['total_fee']:.2f}",
            f"- Trades: {detail['trade_n']}",
            f"- Win rate: {_fmt_pct(detail['win_rate'])} ({detail['win_n']}W/{detail['loss_n']}L)",
            f"- Avg trade return: {_fmt_pct(detail['avg_pnl'])}",
            f"- Max drawdown: {_fmt_pct(detail['max_dd'])}",
            "",
            "## Skip stats",
            f"- no_candidate: {detail['skip_counts'].get('no_candidate', 0)}",
            f"- threshold_blocked: {detail['skip_counts'].get('threshold_blocked', 0)}",
            f"- board_blocked: {detail['skip_counts'].get('board_blocked', 0)}",
            f"- risk_blocked: {detail['skip_counts'].get('risk_blocked', 0)}",
            f"- gap_blocked: {detail['skip_counts'].get('gap_blocked', 0)}",
            f"- rule_blocked: {detail['skip_counts'].get('rule_blocked', 0)}",
            f"- bad_buy_quote: {detail['skip_counts'].get('bad_buy_quote', 0)}",
            f"- calendar_miss: {detail['skip_counts'].get('calendar_miss', 0)}",
            f"- insufficient_cash: {detail['skip_counts'].get('insufficient_cash', 0)}",
            f"- bad_exit: {detail['skip_counts'].get('bad_exit', 0)}",
            "",
            "## Trades",
        ]
    )

    trades = detail["trades"]
    if not trades:
        lines.append("- No executed trades.")
        return "\n".join(lines) + "\n"

    for i, t in enumerate(trades, 1):
        lines.extend(
            [
                "",
                f"### {i}. {t['stock_code']} {t['stock_name']}",
                f"- Signal date: {t['signal_date']}",
                f"- Pick reason: {t['pick_reason']}",
                "- Buy:",
                f"  - Time: {t['buy_date']} {t['buy_time']}",
                f"  - Price: {t['buy_price']:.3f}",
                f"  - Shares: {t['shares']}",
                f"  - Amount: {t['buy_amount']:.2f}",
                f"  - Fee: {t['buy_fee']:.2f}",
                f"  - Reason: {t['buy_reason']}",
                "- Sell:",
                f"  - Time: {t['sell_date']} {t['sell_time']}",
                f"  - Price: {t['sell_price']:.3f}",
                f"  - Amount: {t['sell_amount']:.2f}",
                f"  - Fee: {t['sell_fee']:.2f}",
                f"  - Reason: {t['sell_reason']}",
                "- Result:",
                f"  - Total fee: {t['total_fee']:.2f}",
                f"  - PnL: {t['pnl']:.2f}",
                f"  - Return: {_fmt_pct(t['pnl_pct'])}",
                f"  - Equity after trade: {t['equity_after']:.2f}",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if str(x).strip()]
    thresholds = [max(0.50, min(0.99, t)) for t in thresholds]
    thresholds = sorted(set(thresholds))
    initial_capital = float(args.initial_capital)
    seed_offset = int(args.seed_offset)
    retain_ratio = max(0.01, min(1.0, float(args.retain_ratio)))
    min_return_floor = float(args.min_return_floor)
    variant_profile = str(args.variant_profile or "base")
    rules = _load_execution_rules(args.rules_file)

    backtest_start = (
        _to_iso_optional(args.start_date)
        or _to_iso_optional(rules.backtest_start_date)
        or "2025-01-01"
    )
    backtest_end = _to_iso_optional(args.end_date) or _to_iso_optional(rules.backtest_end_date)
    train_start = (
        _to_iso_optional(args.train_start_date)
        or _to_iso_optional(rules.train_start_date)
        or "2020-01-01"
    )
    train_end = _to_iso_optional(args.train_end_date) or _to_iso_optional(rules.train_end_date)
    if not train_end:
        try:
            train_end = (dt.datetime.strptime(backtest_start, "%Y-%m-%d").date() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            train_end = None
    if not train_end:
        raise SystemExit("Cannot resolve train-end date.")
    if train_start > train_end:
        raise SystemExit(f"train date range invalid: {train_start} ~ {train_end}")
    if backtest_end and backtest_start > backtest_end:
        raise SystemExit(f"backtest date range invalid: {backtest_start} ~ {backtest_end}")

    ns = argparse.Namespace(config=args.config, db_url=args.db_url, db=args.db)
    engine = relay_model.make_engine(relay_model.resolve_db_target(ns))
    trade_dates = _load_trade_dates(engine, backtest_start, backtest_end)
    if not trade_dates:
        tail = f"~{backtest_end}" if backtest_end else "+"
        raise SystemExit(f"No trade dates in backtest range: {backtest_start}{tail}.")
    end_date = trade_dates[-1]
    train_start_dt = dt.datetime.strptime(train_start, "%Y-%m-%d").date()
    load_start = (train_start_dt - dt.timedelta(days=60)).strftime("%Y-%m-%d")

    base_raw = relay_model.load_base_frame(engine, load_start, end_date)
    if base_raw.empty:
        raise SystemExit(f"No stock_daily data in range: {load_start}~{end_date}.")
    full = relay_model.compute_flags_and_stock_features(base_raw)
    market = relay_model.compute_market_features(full)
    full_pool, _, _ = relay_model.build_sample_frames(
        full_df=full,
        market_df=market,
        start_iso=train_start,
        end_iso=end_date,
        tp=0.02,
        sl=-0.01,
    )
    if full_pool.empty:
        raise SystemExit("No relay candidate rows after sample build.")

    full_pool["date"] = full_pool["date"].astype(str)
    full_pool["stock_code"] = full_pool["stock_code"].astype(str)
    full_pool = _attach_future_fields(engine, full_pool, train_start)
    full_pool["tomorrow_stage"] = "N/A"
    full_pool["tomorrow_prob"] = np.nan

    base = full_pool[(full_pool["date"] >= backtest_start) & (full_pool["date"] <= end_date)].copy()
    if base.empty:
        raise SystemExit(f"No candidate rows from backtest start {backtest_start}.")

    train_df = full_pool[(full_pool["date"] >= train_start) & (full_pool["date"] <= train_end)].copy()
    train_df = train_df[(train_df["next_open"].notna()) & (train_df["next2_close"].notna())].copy()
    train_df["next_open"] = pd.to_numeric(train_df["next_open"], errors="coerce")
    train_df["next2_close"] = pd.to_numeric(train_df["next2_close"], errors="coerce")
    train_df = train_df[(train_df["next_open"] > 0) & (train_df["next2_close"] > 0)].copy()
    train_df["t1_hold_ret"] = train_df["next2_close"] / train_df["next_open"] - 1.0
    if train_df.empty:
        raise SystemExit(f"No train samples in range {train_start}~{train_end} after T+1 buy / T+2 sell target.")

    feat_cols = _feature_cols()
    x_train_raw = train_df[feat_cols].to_numpy(dtype=float)
    x_all_raw = full_pool[feat_cols].to_numpy(dtype=float)
    x_train, x_all, _, _ = relay_model.fill_and_scale(x_train_raw, x_all_raw)

    variants = [
        ModelVariant(
            name=str(spec["name"]),
            hidden=int(spec["hidden"]),
            epochs=int(spec["epochs"]),
            lr=float(spec["lr"]),
            weight_decay=float(spec["weight_decay"]),
            target_gt=float(spec["target_gt"]),
            seed=int(spec["seed"]) + seed_offset,
        )
        for spec in list_variant_specs(variant_profile)
    ]
    print(
        f"[mock-backtest] train_range={train_start}~{train_end} "
        f"backtest_range={backtest_start}~{end_date}",
        flush=True,
    )
    if rules.min_return_pct is not None or rules.max_drawdown_pct is not None:
        print(
            "[mock-backtest] rule_targets="
            f"ret>={rules.min_return_pct if rules.min_return_pct is not None else 'N/A'}% "
            f"dd<={rules.max_drawdown_pct if rules.max_drawdown_pct is not None else 'N/A'}%",
            flush=True,
        )
    print(
        f"[mock-backtest] variant_profile={variant_profile} variants={len(variants)}",
        flush=True,
    )
    alphas = [2, 5, 10, 15]
    sell_rules = ["t2_close", "weak_hold_t3", "strong_hold_t3", "profit_lock_t3", "loss_rebound_t3"]
    top_ks = [1, 2, 3]
    gap_windows: List[Tuple[Optional[float], Optional[float]]] = [
        (None, None),
        (-0.005, 0.015),
        (-0.01, 0.02),
        (-0.005, 0.025),
        (-0.01, 0.03),
        (-0.02, 0.04),
    ]
    board_caps: List[Optional[int]] = [None, 1, 2, 3]
    risk_profiles: List[Tuple[str, Optional[float], Optional[float], Optional[int], Optional[float]]] = [
        ("off", None, None, None, None),
        ("balanced", 0.45, 0.24, 20, 0.09),
        ("strict", 0.40, 0.28, 15, 0.08),
        ("ultra", 0.35, 0.30, 12, 0.07),
        ("defensive", 0.30, 0.35, 8, 0.06),
    ]
    alloc_pcts = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]

    date_set = set(trade_dates)
    search_logs: List[SearchResult] = []
    variant_map: Dict[str, ModelVariant] = {v.name: v for v in variants}
    variant_base_prob: Dict[str, np.ndarray] = {}

    for mv in variants:
        print(f"[mock-backtest] train {mv.name} ...", flush=True)
        y = (train_df["t1_hold_ret"] > float(mv.target_gt)).astype(np.int8).to_numpy(np.int8)
        w1, b1, w2, b2 = relay_model.train_binary_mlp(
            x=x_train,
            y=y,
            hidden_size=int(mv.hidden),
            epochs=int(mv.epochs),
            learning_rate=float(mv.lr),
            weight_decay=float(mv.weight_decay),
            seed=int(mv.seed),
        )
        train_pred, _ = relay_model.predict_binary_mlp(x_train, w1, b1, w2, b2)
        _, all_prob = relay_model.predict_binary_mlp(x_all, w1, b1, w2, b2)
        train_acc = float((train_pred == y).mean()) if len(y) else 0.0
        print(f"[mock-backtest] train {mv.name} acc={train_acc*100:.2f}%", flush=True)

        prob_full = pd.Series(all_prob, index=full_pool.index)
        base_prob = prob_full.reindex(base.index).to_numpy(dtype=float)
        variant_base_prob[mv.name] = base_prob

        for alpha in alphas:
            score = _score_from_raw_prob(base_prob, alpha=alpha)
            scored = base.copy()
            scored["raw_prob"] = base_prob
            scored["score"] = score
            scored = scored[scored["date"].isin(date_set)].copy()
            ranked = _build_ranked_by_date(scored)

            for rule in sell_rules:
                for top_k in top_ks:
                    for gap_min, gap_max in gap_windows:
                        for max_board in board_caps:
                            for risk_name, max_broken_rate, min_red_rate, max_limit_down, max_pullback in risk_profiles:
                                for alloc_pct in alloc_pcts:
                                    metrics: Dict[float, Dict[str, float]] = {}
                                    rets: List[float] = []
                                    for th in thresholds:
                                        m = _simulate_fast(
                                            trade_dates=trade_dates,
                                            ranked_by_date=ranked,
                                            threshold=float(th),
                                            initial_capital=initial_capital,
                                            sell_rule=rule,
                                            top_k=top_k,
                                            max_board=max_board,
                                            gap_min=gap_min,
                                            gap_max=gap_max,
                                            max_broken_rate=max_broken_rate,
                                            min_red_rate=min_red_rate,
                                            max_limit_down=max_limit_down,
                                            max_pullback=max_pullback,
                                            alloc_pct=alloc_pct,
                                            rules=rules,
                                        )
                                        metrics[float(th)] = m
                                        rets.append(float(m["ret"]))

                                    min_ret = min(rets) if rets else float("-inf")
                                    avg_ret = (sum(rets) / len(rets)) if rets else float("-inf")
                                    objective = min_ret + 0.35 * avg_ret
                                    search_logs.append(
                                        SearchResult(
                                            variant=mv.name,
                                            alpha=alpha,
                                            sell_rule=rule,
                                            top_k=top_k,
                                            gap_min=gap_min,
                                            gap_max=gap_max,
                                            max_board=max_board,
                                            risk_profile=risk_name,
                                            max_broken_rate=max_broken_rate,
                                            min_red_rate=min_red_rate,
                                            max_limit_down=max_limit_down,
                                            max_pullback=max_pullback,
                                            alloc_pct=float(alloc_pct),
                                            train_acc=train_acc,
                                            threshold_metrics=metrics,
                                            min_ret=min_ret,
                                            avg_ret=avg_ret,
                                            objective=objective,
                                        )
                                    )

    if not search_logs:
        raise SystemExit("No strategy candidate generated.")

    for th in thresholds:
        ranked_for_th = sorted(search_logs, key=lambda r: _sort_key_for_threshold(r, th), reverse=True)
        chosen = _select_best_for_threshold(
            ranked_for_th,
            threshold=float(th),
            retain_ratio=retain_ratio,
            min_return_floor=min_return_floor,
            target_return_pct=rules.min_return_pct,
            target_max_dd_pct=rules.max_drawdown_pct,
        )
        mv = variant_map[chosen.variant]
        base_prob = variant_base_prob[chosen.variant]

        scored = base.copy()
        scored["raw_prob"] = base_prob
        scored["score"] = _score_from_raw_prob(base_prob, alpha=chosen.alpha)
        scored = scored[scored["date"].isin(date_set)].copy()
        ranked = _build_ranked_by_date(scored)

        detail = _simulate_detailed(
            trade_dates=trade_dates,
            ranked_by_date=ranked,
            threshold=float(th),
            initial_capital=initial_capital,
            sell_rule=chosen.sell_rule,
            variant_name=mv.name,
            alpha=int(chosen.alpha),
            top_k=int(chosen.top_k),
            max_board=chosen.max_board,
            gap_min=chosen.gap_min,
            gap_max=chosen.gap_max,
            max_broken_rate=chosen.max_broken_rate,
            min_red_rate=chosen.min_red_rate,
            max_limit_down=chosen.max_limit_down,
            max_pullback=chosen.max_pullback,
            risk_profile=chosen.risk_profile,
            alloc_pct=chosen.alloc_pct,
            rules=rules,
        )
        report = _render_report(
            threshold=float(th),
            start_date=backtest_start,
            end_date=end_date,
            variant=mv,
            selected=chosen,
            detail=detail,
            rules=rules,
            search_top=ranked_for_th[:8],
            retain_ratio=retain_ratio,
            min_return_floor=min_return_floor,
        )
        out = Path(f"mockReport_{_threshold_tag(float(th))}.md")
        out.write_text(report, encoding="utf-8")
        ops_csv = Path(f"mockOperations_{_threshold_tag(float(th))}.csv")
        _write_operations_csv(ops_csv, detail["trades"], initial_capital=initial_capital)

        rule_flags: List[str] = []
        if rules.min_return_pct is not None:
            ok_ret = detail["total_ret"] * 100.0 >= float(rules.min_return_pct)
            rule_flags.append(f"ret_target={'OK' if ok_ret else 'FAIL'}")
        if rules.max_drawdown_pct is not None:
            ok_dd = detail["max_dd"] * 100.0 <= float(rules.max_drawdown_pct)
            rule_flags.append(f"dd_target={'OK' if ok_dd else 'FAIL'}")
        rule_suffix = f" rules[{', '.join(rule_flags)}]" if rule_flags else ""

        print(
            f"[mock-backtest] th={th:.2f} final={detail['final_capital']:.2f} "
            f"ret={detail['total_ret']*100:.2f}% dd={detail['max_dd']*100:.2f}% "
            f"fee={detail['total_fee']:.2f} trades={detail['trade_n']} "
            f"cfg={chosen.variant}/a{chosen.alpha}/{chosen.sell_rule}/k{chosen.top_k}/"
            f"gap[{chosen.gap_min},{chosen.gap_max}]/board<={chosen.max_board}/"
            f"risk={chosen.risk_profile}/alloc={chosen.alloc_pct:.2f}{rule_suffix}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
