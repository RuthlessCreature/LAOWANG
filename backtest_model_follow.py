#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backtest model follow rules on real daily data."""

from __future__ import annotations

import argparse
import configparser
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine, text

from strategy_protocols import ordinary_model_plan_config as load_ordinary_model_plan_config


DEFAULT_DB = "data/stock.db"
DEFAULT_FEE_RATE = 0.00025
DEFAULT_MIN_FEE = 5.0
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_STAMP_TAX_RATE = 0.0005
DEFAULT_TRANSFER_FEE_RATE = 0.00001
DEFAULT_SLIPPAGE_RATE = 0.0005
DEFAULT_MAX_TOTAL_POSITION_PCT = 0.50
DEFAULT_MAX_NEW_POSITIONS_PER_DAY = 1


@dataclass
class Trade:
    model: str
    signal_date: str
    buy_date: str
    sell_date: str
    stock_code: str
    stock_name: str
    signal_score: float
    buy_price: float
    sell_price: float
    shares: int
    buy_fee: float
    sell_fee: float
    pnl: float
    pnl_pct: float
    hold_days: int
    exit_rule: str
    reason: str
    equity_after: float
    rule: str = ""


def _to_iso(v: str) -> str:
    s = str(v or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def _to_yyyymmdd(v: str) -> str:
    s = str(v or "").strip()
    if len(s) == 8 and s.isdigit():
        return s
    return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest daily model follow rules.")
    p.add_argument("--config", default="config.ini")
    p.add_argument("--db-url", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--start-date", default="2025-01-01")
    p.add_argument("--end-date", default="2026-05-01")
    p.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    p.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    p.add_argument("--min-fee", type=float, default=DEFAULT_MIN_FEE)
    p.add_argument("--stamp-tax-rate", type=float, default=DEFAULT_STAMP_TAX_RATE)
    p.add_argument("--transfer-fee-rate", type=float, default=DEFAULT_TRANSFER_FEE_RATE)
    p.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE)
    p.add_argument("--model", default="all", help="all/laowang/ywcx/stwg/fhkq/relay")
    p.add_argument("--signal-threshold-laowang", type=float, default=60.0)
    p.add_argument("--signal-threshold-ywcx", type=float, default=55.0)
    p.add_argument("--signal-threshold-stwg", type=float, default=55.0)
    p.add_argument("--signal-threshold-fhkq", type=float, default=60.0)
    p.add_argument("--relay-threshold", type=float, default=0.62)
    p.add_argument("--relay-top-k", type=int, default=3)
    p.add_argument("--relay-max-board", type=int, default=None)
    p.add_argument("--plan-threshold-laowang", type=float, default=65.0)
    p.add_argument("--plan-threshold-ywcx", type=float, default=65.0)
    p.add_argument("--plan-threshold-stwg", type=float, default=65.0)
    p.add_argument("--plan-threshold-fhkq", type=float, default=80.0)
    p.add_argument("--plan-threshold-relay", type=float, default=0.70)
    p.add_argument("--plan-relay-top-k", type=int, default=2)
    p.add_argument("--min-open-gap", type=float, default=None, help="Override ordinary-model min T+1 open gap, decimal form.")
    p.add_argument("--max-open-gap", type=float, default=None, help="Override ordinary-model max T+1 open gap, decimal form.")
    p.add_argument("--block-limit-up-open", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-total-position-pct", type=float, default=DEFAULT_MAX_TOTAL_POSITION_PCT)
    p.add_argument("--max-new-positions-per-day", type=int, default=DEFAULT_MAX_NEW_POSITIONS_PER_DAY)
    p.add_argument("--relay-hybrid-position-pct", type=float, default=0.04)
    p.add_argument("--relay-trigger-ret-pct", type=float, default=0.005)
    p.add_argument("--relay-max-entry-drawdown-pct", type=float, default=-0.03)
    p.add_argument("--relay-weak-open-stop-pct", type=float, default=-0.03)
    p.add_argument("--relay-hard-stop-close-pct", type=float, default=-0.035)
    p.add_argument("--relay-profit-extend-pct", type=float, default=0.05)
    p.add_argument("--output-dir", default="reports/model_follow_backtest")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def load_config(path: Path) -> Dict[str, Any]:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    db_url = parser.get("database", "db_url", fallback=None)
    if db_url:
        db_url = db_url.strip()
    return {
        "db_url": db_url,
        "mysql": {
            "host": parser.get("mysql", "host", fallback="127.0.0.1"),
            "port": parser.getint("mysql", "port", fallback=3306),
            "user": parser.get("mysql", "user", fallback=""),
            "password": parser.get("mysql", "password", fallback=""),
            "database": parser.get("mysql", "database", fallback=""),
            "charset": parser.get("mysql", "charset", fallback="utf8mb4"),
        },
    }


def build_mysql_url(cfg: Dict[str, Any]) -> Optional[str]:
    if not (cfg.get("user") and cfg.get("database")):
        return None
    from urllib.parse import quote_plus

    user = quote_plus(str(cfg["user"]))
    password = quote_plus(str(cfg.get("password") or ""))
    auth = f"{user}:{password}" if password else user
    return f"mysql+pymysql://{auth}@{cfg['host']}:{int(cfg['port'])}/{cfg['database']}?charset={cfg['charset']}"


def resolve_db_target(args: argparse.Namespace) -> str:
    if getattr(args, "db_url", None):
        return str(args.db_url)
    import os

    env = os.getenv("ASTOCK_DB_URL")
    if env and env.strip():
        return env.strip()
    if getattr(args, "db", None):
        return str(args.db)
    cfg_path = getattr(args, "config", None) or "config.ini"
    cfg_file = Path(cfg_path)
    if cfg_file.exists():
        cfg = load_config(cfg_file)
        if cfg.get("db_url"):
            return str(cfg["db_url"])
        mysql_url = build_mysql_url(cfg["mysql"])
        if mysql_url:
            return mysql_url
    return DEFAULT_DB


def make_engine(db_target: str):
    if "://" not in db_target and db_target.endswith(".db"):
        db_path = Path(db_target).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_target = f"sqlite:///{db_path.as_posix()}"
    connect_args: Dict[str, Any] = {}
    if db_target.startswith("sqlite:///"):
        connect_args["check_same_thread"] = False
    return create_engine(db_target, pool_pre_ping=True, pool_recycle=3600, connect_args=connect_args)


def round_half_up_2(v: float) -> float:
    return float(math.floor(float(v) * 100.0 + 0.5) / 100.0)


def infer_limit_pct(stock_code: str, stock_name: str) -> float:
    name = str(stock_name or "").upper()
    if "ST" in name or "退" in name:
        return 0.05
    code = str(stock_code or "")
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def calc_fee(amount: float, fee_rate: float, min_fee: float) -> float:
    if not (math.isfinite(amount) and amount > 0):
        return 0.0
    return max(float(min_fee), float(amount) * float(fee_rate))


def calc_side_cost(
    amount: float,
    *,
    fee_rate: float,
    min_fee: float,
    transfer_fee_rate: float,
    stamp_tax_rate: float = 0.0,
) -> float:
    if not (math.isfinite(amount) and amount > 0):
        return 0.0
    commission = calc_fee(amount, fee_rate, min_fee)
    transfer = float(amount) * max(0.0, float(transfer_fee_rate))
    stamp = float(amount) * max(0.0, float(stamp_tax_rate))
    return float(commission + transfer + stamp)


def max_lots_for_cash(cash: float, buy_px: float, fee_rate: float, min_fee: float, transfer_fee_rate: float = 0.0) -> int:
    if not (math.isfinite(cash) and cash > 0 and math.isfinite(buy_px) and buy_px > 0):
        return 0
    lots = int(cash // (buy_px * 100.0))
    while lots > 0:
        amt = lots * 100.0 * buy_px
        need = amt + calc_side_cost(
            amt,
            fee_rate=fee_rate,
            min_fee=min_fee,
            transfer_fee_rate=transfer_fee_rate,
        )
        if need <= cash + 1e-9:
            return lots
        lots -= 1
    return 0


def load_daily_frame(engine, start_date: str, end_date: str) -> pd.DataFrame:
    sql = text(
        """
        SELECT d.stock_code,
               d.date,
               d.open,
               d.high,
               d.low,
               d.close,
               d.volume,
               d.amount,
               COALESCE(i.name, '') AS stock_name
        FROM stock_daily d
        LEFT JOIN stock_info i
            ON i.stock_code = d.stock_code
        WHERE d.date BETWEEN :s AND :e
        ORDER BY d.stock_code, d.date
        """
    )
    df = pd.read_sql(sql, engine, params={"s": start_date, "e": end_date})
    if df.empty:
        return df
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str)
    df["date"] = df["date"].astype(str)
    return df


def add_ma(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    grp = out.groupby("stock_code", sort=False)
    for win in (5, 10, 20):
        out[f"ma{win}"] = grp["close"].transform(lambda s, w=win: s.rolling(w, min_periods=w).mean())
    out["amount_ma20"] = grp["amount"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    return out


def load_signals(
    engine,
    start_date: str,
    end_date: str,
    model: str,
    thresholds: Dict[str, float],
    relay_threshold: float,
    relay_top_k: int,
    relay_max_board: Optional[int],
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    if model in {"all", "laowang"}:
        q = text(
            """
            SELECT s.score_date AS signal_date,
                   s.stock_code,
                   COALESCE(i.name, '') AS stock_name,
                   d.close AS signal_close,
                   s.total_score AS signal_score,
                   s.status_tags AS status_tags,
                   'laowang' AS model
            FROM stock_scores_v3 s
            LEFT JOIN stock_info i ON i.stock_code = s.stock_code
            LEFT JOIN stock_daily d ON d.stock_code = s.stock_code AND d.date = s.score_date
            WHERE s.score_date BETWEEN :s AND :e
              AND s.total_score >= :th
              AND COALESCE(s.status_tags, '') NOT LIKE '%RISK_FILTERED%'
            ORDER BY s.score_date, s.total_score DESC, s.stock_code
            """
        )
        parts.append(pd.read_sql(q, engine, params={"s": start_date, "e": end_date, "th": float(thresholds["laowang"])}))
    if model in {"all", "ywcx"}:
        q = text(
            """
            SELECT s.score_date AS signal_date,
                   s.stock_code,
                   COALESCE(i.name, '') AS stock_name,
                   d.close AS signal_close,
                   s.total_score AS signal_score,
                   s.status_tags AS status_tags,
                   'ywcx' AS model
            FROM stock_scores_ywcx s
            LEFT JOIN stock_info i ON i.stock_code = s.stock_code
            LEFT JOIN stock_daily d ON d.stock_code = s.stock_code AND d.date = s.score_date
            WHERE s.score_date BETWEEN :s AND :e
              AND s.total_score >= :th
              AND COALESCE(s.status_tags, '') NOT LIKE '%RISK_FILTERED%'
            ORDER BY s.score_date, s.total_score DESC, s.stock_code
            """
        )
        parts.append(pd.read_sql(q, engine, params={"s": start_date, "e": end_date, "th": float(thresholds["ywcx"])}))
    if model in {"all", "stwg"}:
        q = text(
            """
            SELECT s.score_date AS signal_date,
                   s.stock_code,
                   COALESCE(i.name, '') AS stock_name,
                   d.close AS signal_close,
                   s.total_score AS signal_score,
                   s.status_tags AS status_tags,
                   'stwg' AS model
            FROM stock_scores_stwg s
            LEFT JOIN stock_info i ON i.stock_code = s.stock_code
            LEFT JOIN stock_daily d ON d.stock_code = s.stock_code AND d.date = s.score_date
            WHERE s.score_date BETWEEN :s AND :e
              AND s.total_score >= :th
              AND COALESCE(s.status_tags, '') NOT LIKE '%RISK_FILTERED%'
            ORDER BY s.score_date, s.total_score DESC, s.stock_code
            """
        )
        parts.append(pd.read_sql(q, engine, params={"s": start_date, "e": end_date, "th": float(thresholds["stwg"])}))
    if model in {"all", "fhkq"}:
        q = text(
            """
            SELECT m.trade_date AS signal_date,
                   m.stock_code,
                   COALESCE(i.name, '') AS stock_name,
                   d.close AS signal_close,
                   m.fhkq_score AS signal_score,
                   CAST(m.consecutive_limit_down AS CHAR) AS status_tags,
                   'fhkq' AS model
            FROM model_fhkq m
            LEFT JOIN stock_info i ON i.stock_code = m.stock_code
            LEFT JOIN stock_daily d ON d.stock_code = m.stock_code AND d.date = m.trade_date
            WHERE m.trade_date BETWEEN :s AND :e
              AND m.fhkq_score >= :th
            ORDER BY m.trade_date, m.fhkq_score DESC, m.stock_code
            """
        )
        parts.append(pd.read_sql(q, engine, params={"s": start_date, "e": end_date, "th": float(thresholds["fhkq"])}))
    if model in {"all", "relay"}:
        q = text(
            """
            SELECT p.trade_date AS signal_date,
                   p.stock_code,
                   COALESCE(i.name, '') AS stock_name,
                   d.close AS signal_close,
                   p.model_score AS signal_score,
                   p.model_prob AS model_prob,
                   p.rank_no AS rank_no,
                   p.default_threshold AS default_threshold,
                   p.top_k AS top_k,
                   p.max_board_filter AS max_board_filter,
                   p.gap_min AS gap_min,
                   p.gap_max AS gap_max,
                   p.max_broken_rate_filter AS max_broken_rate_filter,
                   p.min_red_rate_filter AS min_red_rate_filter,
                   p.max_limit_down_filter AS max_limit_down_filter,
                   p.max_pullback_filter AS max_pullback_filter,
                   p.broken_rate AS broken_rate,
                   p.red_rate AS red_rate,
                   p.limit_down_count AS limit_down_count,
                   p.pullback AS pullback,
                   p.max_board AS max_board,
                   p.one_word_flag AS one_word_flag,
                   p.risk_profile AS risk_profile,
                   'relay' AS model
            FROM model_relay_pool p
            LEFT JOIN stock_info i ON i.stock_code = p.stock_code
            LEFT JOIN stock_daily d ON d.stock_code = p.stock_code AND d.date = p.trade_date
            WHERE p.trade_date BETWEEN :s AND :e
              AND p.rank_no <= :topk
              AND p.model_score >= :th
              AND (:max_board IS NULL OR p.max_board <= :max_board)
            ORDER BY p.trade_date, p.model_score DESC, p.rank_no ASC, p.stock_code
            """
        )
        parts.append(
            pd.read_sql(
                q,
                engine,
                params={
                    "s": start_date,
                    "e": end_date,
                    "th": float(relay_threshold),
                    "topk": int(relay_top_k),
                    "max_board": relay_max_board,
                },
            )
        )
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        return df
    df["signal_date"] = df["signal_date"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    return df.sort_values(["signal_date", "model", "signal_score", "stock_code"], ascending=[True, True, False, True]).reset_index(drop=True)


def _signal_limit_up_blocked(signal_close: float, next_open: float, stock_code: str, stock_name: str) -> bool:
    if not (math.isfinite(signal_close) and signal_close > 0 and math.isfinite(next_open) and next_open > 0):
        return True
    pct = infer_limit_pct(stock_code, stock_name)
    limit_up = round_half_up_2(signal_close * (1.0 + pct))
    return next_open >= limit_up - 1e-3


def pick_entry_row(model: str, signal_rows: pd.DataFrame, code_hist: pd.DataFrame, signal_date: str, trade_date: str) -> Tuple[Optional[pd.Series], str]:
    rows = signal_rows[signal_rows["signal_date"] == signal_date]
    if rows.empty:
        return None, "no_signal"
    rows = rows.sort_values(["signal_score", "stock_code"], ascending=[False, True]).copy()
    for _, row in rows.iterrows():
        if str(row["stock_code"]) not in set(code_hist["stock_code"].astype(str)):
            continue
        day_rows = code_hist[code_hist["date"] == trade_date]
        if day_rows.empty:
            continue
        next_open = float(day_rows.iloc[0]["open"])
        signal_close = float(row["signal_close"]) if pd.notna(row["signal_close"]) else float("nan")
        if _signal_limit_up_blocked(signal_close, next_open, str(row["stock_code"]), str(row["stock_name"])):
            continue
        if model == "relay":
            gap_min = row.get("gap_min")
            gap_max = row.get("gap_max")
            if pd.notna(gap_min) or pd.notna(gap_max):
                gap = next_open / signal_close - 1.0 if math.isfinite(signal_close) and signal_close > 0 else np.nan
                if pd.notna(gap_min) and gap < float(gap_min) - 1e-3:
                    continue
                if pd.notna(gap_max) and gap > float(gap_max) + 1e-3:
                    continue
            if pd.notna(row.get("max_board_filter")) and pd.notna(row.get("max_board")):
                if int(row["max_board"]) > int(row["max_board_filter"]):
                    continue
            if pd.notna(row.get("max_broken_rate_filter")) and pd.notna(row.get("broken_rate")):
                if float(row["broken_rate"]) > float(row["max_broken_rate_filter"]) + 1e-3:
                    continue
            if pd.notna(row.get("min_red_rate_filter")) and pd.notna(row.get("red_rate")):
                if float(row["red_rate"]) < float(row["min_red_rate_filter"]) - 1e-3:
                    continue
            if pd.notna(row.get("max_limit_down_filter")) and pd.notna(row.get("limit_down_count")):
                if int(row["limit_down_count"]) > int(row["max_limit_down_filter"]):
                    continue
            if pd.notna(row.get("max_pullback_filter")) and pd.notna(row.get("pullback")):
                if float(row["pullback"]) > float(row["max_pullback_filter"]) + 1e-3:
                    continue
        return row, "ok"
    return None, "filtered"


def exit_fixed_hold(code_hist: pd.DataFrame, buy_idx: int, hold_days: int) -> Tuple[int, str]:
    sell_idx = min(len(code_hist) - 1, buy_idx + max(1, int(hold_days)))
    return sell_idx, f"fixed_hold_{hold_days}"


def exit_ma_break(code_hist: pd.DataFrame, buy_idx: int, ma_col: str, max_hold: int) -> Tuple[int, str]:
    max_hold_days = max(2, int(max_hold))
    end_idx = min(len(code_hist) - 1, buy_idx + max_hold_days - 1)
    start_idx = min(len(code_hist) - 1, buy_idx + 1)
    for idx in range(start_idx, end_idx + 1):
        row = code_hist.iloc[idx]
        ma = row.get(ma_col)
        close = row.get("close")
        if pd.notna(ma) and pd.notna(close) and float(close) < float(ma):
            return idx, f"{ma_col}_break"
    return end_idx, f"max_hold_{max_hold}"


def exit_relay_style(code_hist: pd.DataFrame, buy_idx: int) -> Tuple[int, str]:
    t2_idx = min(len(code_hist) - 1, buy_idx + 1)
    t3_idx = min(len(code_hist) - 1, buy_idx + 2)
    buy_px = float(code_hist.iloc[buy_idx]["open"])
    t2_close = float(code_hist.iloc[t2_idx]["close"])
    if t2_close / buy_px - 1.0 <= -0.05 and t3_idx > t2_idx:
        return t3_idx, "relay_loss_rebound_t3"
    return t2_idx, "relay_t2_close"


def backtest_model(
    model_name: str,
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    trade_dates: List[str],
    *,
    initial_capital: float,
    fee_rate: float,
    min_fee: float,
    stamp_tax_rate: float,
    transfer_fee_rate: float,
    slippage_rate: float,
    exit_mode: str,
) -> Tuple[List[Trade], Dict[str, Any]]:
    if signals.empty:
        return [], {
            "model": model_name,
            "signals": 0,
            "trades": 0,
            "final_capital": initial_capital,
            "return_pct": 0.0,
            "max_dd_pct": 0.0,
            "win_rate": 0.0,
            "avg_hold_days": 0.0,
        }

    trades: List[Trade] = []
    cash = float(initial_capital)
    peak = cash
    max_dd = 0.0
    i = 0
    signal_dates = sorted(signals["signal_date"].unique().tolist())
    daily_groups = {code: grp.reset_index(drop=True) for code, grp in daily.groupby("stock_code", sort=False)}
    trade_date_set = set(trade_dates)

    while i < len(signal_dates):
        sdate = signal_dates[i]
        if sdate not in trade_date_set:
            i += 1
            continue
        signal_day = signals[signals["signal_date"] == sdate].sort_values(["signal_score", "stock_code"], ascending=[False, True])
        picked_row = None
        pick_reason = ""
        buy_date = None
        buy_hist = None
        buy_idx = None
        for _, row in signal_day.iterrows():
            hist = daily_groups.get(str(row["stock_code"]))
            if hist is None or hist.empty:
                continue
            sig_idx = hist.index[hist["date"] == sdate]
            if len(sig_idx) == 0:
                continue
            sig_idx = int(sig_idx[0])
            if sig_idx + 1 >= len(hist):
                continue
            candidate_buy = hist.iloc[sig_idx + 1]
            if _signal_limit_up_blocked(
                float(row["signal_close"]) if pd.notna(row["signal_close"]) else float("nan"),
                float(candidate_buy["open"]),
                str(row["stock_code"]),
                str(row["stock_name"]),
            ):
                continue
            if model_name == "relay":
                gap_min = row.get("gap_min")
                gap_max = row.get("gap_max")
                if pd.notna(gap_min) or pd.notna(gap_max):
                    gap = candidate_buy["open"] / row["signal_close"] - 1.0 if pd.notna(row["signal_close"]) and row["signal_close"] > 0 else np.nan
                    if pd.notna(gap_min) and gap < float(gap_min) - 1e-3:
                        continue
                    if pd.notna(gap_max) and gap > float(gap_max) + 1e-3:
                        continue
                if pd.notna(row.get("max_board_filter")) and pd.notna(row.get("max_board")):
                    if int(row["max_board"]) > int(row["max_board_filter"]):
                        continue
                if pd.notna(row.get("max_broken_rate_filter")) and pd.notna(row.get("broken_rate")):
                    if float(row["broken_rate"]) > float(row["max_broken_rate_filter"]) + 1e-3:
                        continue
                if pd.notna(row.get("min_red_rate_filter")) and pd.notna(row.get("red_rate")):
                    if float(row["red_rate"]) < float(row["min_red_rate_filter"]) - 1e-3:
                        continue
                if pd.notna(row.get("max_limit_down_filter")) and pd.notna(row.get("limit_down_count")):
                    if int(row["limit_down_count"]) > int(row["max_limit_down_filter"]):
                        continue
                if pd.notna(row.get("max_pullback_filter")) and pd.notna(row.get("pullback")):
                    if float(row["pullback"]) > float(row["max_pullback_filter"]) + 1e-3:
                        continue
            picked_row = row
            pick_reason = "ok"
            buy_date = str(candidate_buy["date"])
            buy_hist = hist
            buy_idx = sig_idx + 1
            break

        if picked_row is None or buy_hist is None or buy_idx is None or buy_date is None:
            i += 1
            continue

        raw_buy_px = float(buy_hist.iloc[buy_idx]["open"])
        buy_px = raw_buy_px * (1.0 + max(0.0, float(slippage_rate)))
        lots = max_lots_for_cash(cash, buy_px, fee_rate, min_fee, transfer_fee_rate=transfer_fee_rate)
        shares = lots * 100
        if shares < 100:
            i += 1
            continue

        if exit_mode == "fixed_t2":
            sell_idx, exit_rule = exit_fixed_hold(buy_hist, buy_idx, 1)
        elif exit_mode == "fixed_t3":
            sell_idx, exit_rule = exit_fixed_hold(buy_hist, buy_idx, 2)
        elif exit_mode == "ma5_break":
            sell_idx, exit_rule = exit_ma_break(buy_hist, buy_idx, "ma5", 10)
        elif exit_mode == "ma10_break":
            sell_idx, exit_rule = exit_ma_break(buy_hist, buy_idx, "ma10", 12)
        elif exit_mode == "ma20_break":
            sell_idx, exit_rule = exit_ma_break(buy_hist, buy_idx, "ma20", 20)
        elif exit_mode == "relay_style":
            sell_idx, exit_rule = exit_relay_style(buy_hist, buy_idx)
        else:
            raise ValueError(f"unknown exit_mode: {exit_mode}")

        buy_amt = shares * buy_px
        buy_fee = calc_side_cost(
            buy_amt,
            fee_rate=fee_rate,
            min_fee=min_fee,
            transfer_fee_rate=transfer_fee_rate,
        )
        if buy_amt + buy_fee > cash + 1e-9:
            i += 1
            continue
        raw_sell_px = float(buy_hist.iloc[sell_idx]["close"])
        sell_px = raw_sell_px * (1.0 - max(0.0, float(slippage_rate)))
        sell_date = str(buy_hist.iloc[sell_idx]["date"])
        hold_days = int(sell_idx - buy_idx + 1)
        sell_amt = shares * sell_px
        sell_fee = calc_side_cost(
            sell_amt,
            fee_rate=fee_rate,
            min_fee=min_fee,
            transfer_fee_rate=transfer_fee_rate,
            stamp_tax_rate=stamp_tax_rate,
        )
        cash_before = cash
        cash_after_buy = cash - buy_amt - buy_fee
        for mark_idx in range(buy_idx, sell_idx + 1):
            close_px = float(buy_hist.iloc[mark_idx]["close"])
            mark_amt = shares * close_px
            mark_fee = calc_side_cost(
                mark_amt,
                fee_rate=fee_rate,
                min_fee=min_fee,
                transfer_fee_rate=transfer_fee_rate,
                stamp_tax_rate=stamp_tax_rate,
            )
            equity = cash_after_buy + mark_amt - mark_fee
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
        cash = cash_after_buy + sell_amt - sell_fee
        pnl = cash - cash_before
        pnl_pct = pnl / (buy_amt + buy_fee) if buy_amt + buy_fee > 0 else 0.0

        trades.append(
            Trade(
                model=model_name,
                signal_date=sdate,
                buy_date=buy_date,
                sell_date=sell_date,
                stock_code=str(picked_row["stock_code"]),
                stock_name=str(picked_row["stock_name"] or ""),
                signal_score=float(picked_row["signal_score"]),
                buy_price=float(buy_px),
                sell_price=float(sell_px),
                shares=int(shares),
                buy_fee=float(buy_fee),
                sell_fee=float(sell_fee),
                pnl=float(pnl),
                pnl_pct=float(pnl_pct),
                hold_days=hold_days,
                exit_rule=exit_rule,
                reason=pick_reason,
                equity_after=float(cash),
            )
        )
        while i < len(signal_dates) and signal_dates[i] <= sell_date:
            i += 1

    total_ret = cash / float(initial_capital) - 1.0
    win_rate = float(np.mean([t.pnl > 0 for t in trades])) if trades else 0.0
    avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else 0.0
    summary = {
        "model": model_name,
        "signals": int(signals["signal_date"].nunique()),
        "trades": len(trades),
        "final_capital": float(cash),
        "return_pct": float(total_ret * 100.0),
        "max_dd_pct": float(max_dd * 100.0),
        "win_rate": float(win_rate * 100.0),
        "avg_hold_days": float(avg_hold),
    }
    return trades, summary


def trade_to_dict(t: Trade) -> Dict[str, Any]:
    return {
        "model": t.model,
        "rule": t.rule,
        "signal_date": t.signal_date,
        "buy_date": t.buy_date,
        "sell_date": t.sell_date,
        "stock_code": t.stock_code,
        "stock_name": t.stock_name,
        "signal_score": t.signal_score,
        "buy_price": t.buy_price,
        "sell_price": t.sell_price,
        "shares": t.shares,
        "buy_fee": t.buy_fee,
        "sell_fee": t.sell_fee,
        "pnl": t.pnl,
        "pnl_pct": t.pnl_pct,
        "hold_days": t.hold_days,
        "exit_rule": t.exit_rule,
        "reason": t.reason,
        "equity_after": t.equity_after,
    }


TRADE_COLUMNS = list(trade_to_dict(Trade("", "", "", "", "", "", 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, "", "", 0.0)).keys())
MANUAL_EQUITY_COLUMNS = ["date", "equity", "cash", "open_positions", "opened_today", "closed_today"]


def _row_float(row: pd.Series, key: str, default: float = float("nan")) -> float:
    try:
        v = row.get(key)
        if pd.isna(v):
            return default
        out = float(v)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _has_all_tags(tags_text: Any, required: Sequence[str]) -> bool:
    text_v = str(tags_text or "")
    return all(tag in text_v for tag in required)


def _has_any_tag(tags_text: Any, blocked: Sequence[str]) -> bool:
    text_v = str(tags_text or "")
    return any(tag in text_v for tag in blocked)


def _find_next_bar(daily_groups: Dict[str, pd.DataFrame], stock_code: str, signal_date: str) -> Tuple[Optional[pd.Series], str]:
    hist = daily_groups.get(str(stock_code))
    if hist is None or hist.empty:
        return None, "no_history"
    idx = hist.index[hist["date"] == signal_date]
    if len(idx) == 0:
        return None, "no_signal_bar"
    sig_idx = int(idx[0])
    if sig_idx + 1 >= len(hist):
        return None, "no_next_bar"
    return hist.iloc[sig_idx + 1], "ok"


def _open_gap(signal_close: float, next_open: float) -> float:
    if not (math.isfinite(signal_close) and signal_close > 0 and math.isfinite(next_open) and next_open > 0):
        return float("nan")
    return float(next_open / signal_close - 1.0)


def _ordinary_model_plan_config(model: str) -> Dict[str, Any]:
    return dict(load_ordinary_model_plan_config(model))


def _with_arg_threshold(cfg: Dict[str, Any], args: argparse.Namespace, model: str) -> Dict[str, Any]:
    out = dict(cfg)
    key = f"plan_threshold_{model}"
    if hasattr(args, key):
        out["min_score"] = float(getattr(args, key))
    if getattr(args, "min_open_gap", None) is not None:
        out["min_gap"] = float(args.min_open_gap)
    if getattr(args, "max_open_gap", None) is not None:
        out["max_gap"] = float(args.max_open_gap)
    return out


def build_next_day_plan(signal_df: pd.DataFrame, daily: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    daily_groups = {code: grp.reset_index(drop=True) for code, grp in daily.groupby("stock_code", sort=False)}
    rows: List[Dict[str, Any]] = []
    if signal_df.empty:
        return pd.DataFrame()
    ordinary = signal_df[signal_df["model"].isin(["laowang", "ywcx", "stwg", "fhkq"])].copy()
    for _, sig in ordinary.iterrows():
        model = str(sig["model"])
        cfg = _with_arg_threshold(_ordinary_model_plan_config(model), args, model)
        next_bar, reason = _find_next_bar(daily_groups, str(sig["stock_code"]), str(sig["signal_date"]))
        signal_close = _row_float(sig, "signal_close")
        next_open = _row_float(next_bar, "open") if next_bar is not None else float("nan")
        gap = _open_gap(signal_close, next_open)
        amount_ma20 = _row_float(next_bar, "amount_ma20") if next_bar is not None else float("nan")
        blocked_limit = (
            _signal_limit_up_blocked(signal_close, next_open, str(sig["stock_code"]), str(sig["stock_name"]))
            if next_bar is not None
            else False
        )
        skip: List[str] = []
        score = _row_float(sig, "signal_score", 0.0)
        tags = sig.get("status_tags", "")
        if next_bar is None:
            skip.append(reason)
        if score < float(cfg["min_score"]):
            skip.append("score_below_plan_threshold")
        if cfg.get("required_tags") and not _has_all_tags(tags, cfg["required_tags"]):
            skip.append("missing_required_tags")
        if cfg.get("any_tags") and not _has_any_tag(tags, cfg["any_tags"]):
            skip.append("missing_trigger_tags")
        if cfg.get("blocked_tags") and _has_any_tag(tags, cfg["blocked_tags"]):
            skip.append("blocked_tags")
        if math.isfinite(gap):
            if gap < float(cfg["min_gap"]):
                skip.append("open_gap_too_low")
            if gap > float(cfg["max_gap"]):
                skip.append("open_gap_too_high")
        else:
            skip.append("missing_open_gap")
        if blocked_limit and bool(getattr(args, "block_limit_up_open", True)):
            skip.append("limit_up_open_unfilled")
        if math.isfinite(amount_ma20) and amount_ma20 < float(cfg["min_amount_ma20"]):
            skip.append("liquidity_low")
        if model == "ywcx":
            skip.append("model_paused_until_data_fixed")

        action = "buy" if not skip and float(cfg["planned_position_pct"]) > 0 else "skip"
        rows.append(
            {
                "signal_date": str(sig["signal_date"]),
                "buy_date": str(next_bar["date"]) if next_bar is not None else "",
                "model": model,
                "stock_code": str(sig["stock_code"]),
                "stock_name": str(sig.get("stock_name") or ""),
                "score": score,
                "signal_close": signal_close,
                "next_open": next_open,
                "open_gap_pct": gap * 100.0 if math.isfinite(gap) else np.nan,
                "amount_ma20": amount_ma20,
                "status_tags": str(tags or ""),
                "planned_action": action,
                "planned_position_pct": float(cfg["planned_position_pct"]),
                "hard_stop_pct": float(cfg["hard_stop_pct"]),
                "structure_stop": str(cfg["structure_stop"]),
                "max_hold_days": int(cfg["max_hold_days"]),
                "skip_reason": ";".join(skip),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_date", "model", "score"], ascending=[True, True, False])


def _relay_passes_risk(sig: pd.Series) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if _row_float(sig, "signal_score", 0.0) < 0.70:
        reasons.append("score_below_0.70")
    if _row_float(sig, "rank_no", 999.0) > 2:
        reasons.append("rank_above_top2")
    if _row_float(sig, "max_board", 999.0) > 3:
        reasons.append("max_board_gt_3")
    broken_rate = _row_float(sig, "broken_rate")
    red_rate = _row_float(sig, "red_rate")
    if math.isfinite(broken_rate) and broken_rate > 0.40:
        reasons.append("broken_rate_gt_0.40")
    if math.isfinite(red_rate) and red_rate < 0.28:
        reasons.append("red_rate_lt_0.28")
    if _row_float(sig, "limit_down_count", 999.0) > 15:
        reasons.append("limit_down_count_gt_15")
    pullback = _row_float(sig, "pullback")
    if math.isfinite(pullback) and pullback > 0.08:
        reasons.append("pullback_gt_0.08")
    return len(reasons) == 0, reasons


def build_relay_watchlist(signal_df: pd.DataFrame, daily: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    daily_groups = {code: grp.reset_index(drop=True) for code, grp in daily.groupby("stock_code", sort=False)}
    rows: List[Dict[str, Any]] = []
    if signal_df.empty:
        return pd.DataFrame()
    relay = signal_df[signal_df["model"] == "relay"].copy()
    if relay.empty:
        return pd.DataFrame()
    relay = relay[
        (pd.to_numeric(relay["signal_score"], errors="coerce") >= float(args.plan_threshold_relay))
        & (pd.to_numeric(relay["rank_no"], errors="coerce") <= int(args.plan_relay_top_k))
    ].copy()
    for _, sig in relay.iterrows():
        next_bar, reason = _find_next_bar(daily_groups, str(sig["stock_code"]), str(sig["signal_date"]))
        signal_close = _row_float(sig, "signal_close")
        next_open = _row_float(next_bar, "open") if next_bar is not None else float("nan")
        auction_gap = _open_gap(signal_close, next_open)
        amount_ma20 = _row_float(next_bar, "amount_ma20") if next_bar is not None else float("nan")
        daily_ok, daily_reasons = _relay_passes_risk(sig)
        auction_reasons: List[str] = []
        if next_bar is None:
            auction_reasons.append(reason)
        if not math.isfinite(auction_gap):
            auction_reasons.append("missing_auction_proxy")
        else:
            if auction_gap < -0.01:
                auction_reasons.append("auction_gap_lt_-1pct")
            if auction_gap > 0.04:
                auction_reasons.append("auction_gap_gt_4pct")
        if next_bar is not None and _signal_limit_up_blocked(signal_close, next_open, str(sig["stock_code"]), str(sig["stock_name"])):
            auction_reasons.append("limit_up_open_unfilled")
        if math.isfinite(amount_ma20) and amount_ma20 < 150_000_000.0:
            auction_reasons.append("liquidity_lt_150m")
        status = "watch_intraday" if daily_ok and not auction_reasons else "drop"
        rows.append(
            {
                "signal_date": str(sig["signal_date"]),
                "watch_date": str(next_bar["date"]) if next_bar is not None else "",
                "model": "relay",
                "stock_code": str(sig["stock_code"]),
                "stock_name": str(sig.get("stock_name") or ""),
                "model_score": _row_float(sig, "signal_score", 0.0),
                "model_prob": _row_float(sig, "model_prob"),
                "rank_no": int(_row_float(sig, "rank_no", 999.0)),
                "signal_close": signal_close,
                "auction_proxy_open": next_open,
                "auction_gap_pct": auction_gap * 100.0 if math.isfinite(auction_gap) else np.nan,
                "amount_ma20": amount_ma20,
                "broken_rate": _row_float(sig, "broken_rate"),
                "red_rate": _row_float(sig, "red_rate"),
                "limit_down_count": _row_float(sig, "limit_down_count"),
                "pullback": _row_float(sig, "pullback"),
                "max_board": _row_float(sig, "max_board"),
                "daily_candidate_status": "pass" if daily_ok else "drop",
                "auction_filter_status": "pass" if not auction_reasons else "drop",
                "planned_status": status,
                "planned_position_pct": 0.04 if status == "watch_intraday" else 0.0,
                "intraday_trigger": "wait_for_vwap_reclaim_or_reseal_or_open_price_rebreak",
                "invalidation_rule": "drop_if_first_15m_ret<=-3pct_or_cannot_reclaim_vwap",
                "exit_protocol": "t2_weak_open_sell_or_t2_close_exit; t3_clear_if_extended",
                "drop_reason": ";".join(daily_reasons + auction_reasons),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal_date", "model_score", "rank_no"], ascending=[True, False, True])


def _daily_limit_down_blocked(prev_close: float, close_px: float, stock_code: str, stock_name: str) -> bool:
    if not (math.isfinite(prev_close) and prev_close > 0 and math.isfinite(close_px) and close_px > 0):
        return False
    pct = infer_limit_pct(stock_code, stock_name)
    limit_down = round_half_up_2(prev_close * (1.0 - pct))
    return close_px <= limit_down + 1e-3


def _manual_exit_for_model(code_hist: pd.DataFrame, buy_idx: int, model: str, buy_px: float, stock_code: str, stock_name: str) -> Tuple[int, str]:
    cfg = _ordinary_model_plan_config(model)
    if model == "fhkq":
        sell_idx, rule = exit_fixed_hold(code_hist, buy_idx, 1)
        return sell_idx, "event_first_sell_day" if rule.startswith("fixed_hold") else rule

    max_hold = int(cfg["max_hold_days"])
    hard_stop_pct = float(cfg["hard_stop_pct"])
    end_idx = min(len(code_hist) - 1, buy_idx + max(1, max_hold))
    start_idx = min(len(code_hist) - 1, buy_idx + 1)
    if start_idx <= buy_idx:
        return buy_idx, "forced_data_end_no_t1_sell_bar"

    profit_armed = False
    for idx in range(start_idx, end_idx + 1):
        row = code_hist.iloc[idx]
        close_px = _row_float(row, "close")
        prev_close = _row_float(code_hist.iloc[idx - 1], "close") if idx > 0 else float("nan")
        if _daily_limit_down_blocked(prev_close, close_px, stock_code, stock_name) and idx < end_idx:
            continue
        ret = close_px / buy_px - 1.0 if math.isfinite(close_px) and buy_px > 0 else float("nan")
        if math.isfinite(ret) and ret <= hard_stop_pct:
            return idx, "hard_stop_close"

        ma5 = _row_float(row, "ma5")
        ma10 = _row_float(row, "ma10")
        ma20 = _row_float(row, "ma20")
        if model == "laowang":
            if math.isfinite(ret) and ret >= 0.12:
                profit_armed = True
            if profit_armed and math.isfinite(ma10) and close_px < ma10:
                return idx, "profit_trailing_ma10_break"
            if math.isfinite(ma20) and close_px < ma20:
                return idx, "ma20_break"
        elif model == "stwg":
            if math.isfinite(ret) and ret >= 0.10:
                profit_armed = True
            if profit_armed and math.isfinite(ma5) and close_px < ma5:
                return idx, "profit_trailing_ma5_break"
            if math.isfinite(ma10) and close_px < ma10:
                return idx, "ma10_break"
        elif model == "ywcx":
            if math.isfinite(ret) and ret >= 0.08:
                return idx, "profit_target_8pct"
            if math.isfinite(ma5) and close_px < ma5:
                return idx, "ma5_break"
    return end_idx, f"max_hold_{max_hold}"


def _open_position_value(active: List[Dict[str, Any]], date: str, fallback_price_key: str = "buy_price") -> float:
    total = 0.0
    for pos in active:
        hist = pos["hist"]
        rows = hist[hist["date"] == date]
        if not rows.empty:
            px = _row_float(rows.iloc[0], "close")
        else:
            px = float(pos.get(fallback_price_key, 0.0))
        if math.isfinite(px) and px > 0:
            total += float(pos["shares"]) * px
    return float(total)


def backtest_manual_open_v1(next_day_plan: pd.DataFrame, daily: pd.DataFrame, args: argparse.Namespace) -> Tuple[List[Trade], Dict[str, Any], pd.DataFrame]:
    rule_name = "manual_open_v1"
    empty_summary = {
        "model": "ordinary_manual",
        "signals": int(len(next_day_plan)) if next_day_plan is not None else 0,
        "planned_buys": 0,
        "trades": 0,
        "final_capital": float(args.initial_capital),
        "return_pct": 0.0,
        "max_dd_pct": 0.0,
        "win_rate": 0.0,
        "avg_hold_days": 0.0,
        "rule": rule_name,
    }
    if next_day_plan is None or next_day_plan.empty:
        return [], empty_summary, pd.DataFrame()

    plan = next_day_plan[next_day_plan["planned_action"] == "buy"].copy()
    empty_summary["planned_buys"] = int(len(plan))
    if plan.empty:
        return [], empty_summary, pd.DataFrame()

    daily_groups = {code: grp.reset_index(drop=True) for code, grp in daily.groupby("stock_code", sort=False)}
    dates = sorted(daily["date"].unique().tolist())
    plan_by_date = {d: grp.copy() for d, grp in plan.groupby("buy_date", sort=False)}
    model_priority = {"stwg": 0, "laowang": 1, "fhkq": 2, "ywcx": 3}

    cash = float(args.initial_capital)
    active: List[Dict[str, Any]] = []
    trades: List[Trade] = []
    equity_rows: List[Dict[str, Any]] = []
    peak = cash
    max_dd = 0.0
    max_new_per_day = max(1, int(args.max_new_positions_per_day))
    max_total_position_pct = min(1.0, max(0.0, float(args.max_total_position_pct)))

    for cur_date in dates:
        opened_today = 0
        if cur_date in plan_by_date and opened_today < max_new_per_day:
            candidates = plan_by_date[cur_date].copy()
            candidates["_priority"] = candidates["model"].map(model_priority).fillna(99).astype(int)
            candidates = candidates.sort_values(["_priority", "score", "stock_code"], ascending=[True, False, True])
            for _, row in candidates.iterrows():
                if opened_today >= max_new_per_day:
                    break
                model = str(row["model"])
                stock_code = str(row["stock_code"])
                hist = daily_groups.get(stock_code)
                if hist is None or hist.empty:
                    continue
                buy_rows = hist.index[hist["date"] == cur_date]
                if len(buy_rows) == 0:
                    continue
                buy_idx = int(buy_rows[0])
                if buy_idx + 1 >= len(hist):
                    continue
                raw_buy_px = _row_float(hist.iloc[buy_idx], "open")
                if not (math.isfinite(raw_buy_px) and raw_buy_px > 0):
                    continue
                buy_px = raw_buy_px * (1.0 + max(0.0, float(args.slippage_rate)))
                current_position_value = _open_position_value(active, cur_date)
                current_equity = cash + current_position_value
                remaining_risk_budget = max(0.0, current_equity * max_total_position_pct - current_position_value)
                target_cash = min(cash, remaining_risk_budget, current_equity * float(row["planned_position_pct"]))
                lots = max_lots_for_cash(
                    target_cash,
                    buy_px,
                    float(args.fee_rate),
                    float(args.min_fee),
                    transfer_fee_rate=float(args.transfer_fee_rate),
                )
                shares = lots * 100
                if shares < 100:
                    continue
                buy_amt = shares * buy_px
                buy_fee = calc_side_cost(
                    buy_amt,
                    fee_rate=float(args.fee_rate),
                    min_fee=float(args.min_fee),
                    transfer_fee_rate=float(args.transfer_fee_rate),
                )
                if buy_amt + buy_fee > cash + 1e-9:
                    continue
                sell_idx, exit_rule = _manual_exit_for_model(hist, buy_idx, model, buy_px, stock_code, str(row.get("stock_name") or ""))
                if sell_idx <= buy_idx:
                    continue
                cash -= buy_amt + buy_fee
                active.append(
                    {
                        "model": model,
                        "signal_date": str(row["signal_date"]),
                        "buy_date": cur_date,
                        "stock_code": stock_code,
                        "stock_name": str(row.get("stock_name") or ""),
                        "signal_score": float(row["score"]),
                        "buy_price": float(buy_px),
                        "shares": int(shares),
                        "buy_fee": float(buy_fee),
                        "buy_amt": float(buy_amt),
                        "hist": hist,
                        "buy_idx": int(buy_idx),
                        "sell_idx": int(sell_idx),
                        "exit_rule": exit_rule,
                    }
                )
                opened_today += 1

        still_active: List[Dict[str, Any]] = []
        closed_today: List[Trade] = []
        for pos in active:
            hist = pos["hist"]
            sell_idx = int(pos["sell_idx"])
            sell_date = str(hist.iloc[sell_idx]["date"])
            if sell_date != cur_date:
                still_active.append(pos)
                continue
            raw_sell_px = _row_float(hist.iloc[sell_idx], "close")
            if not (math.isfinite(raw_sell_px) and raw_sell_px > 0):
                still_active.append(pos)
                continue
            sell_px = raw_sell_px * (1.0 - max(0.0, float(args.slippage_rate)))
            sell_amt = int(pos["shares"]) * sell_px
            sell_fee = calc_side_cost(
                sell_amt,
                fee_rate=float(args.fee_rate),
                min_fee=float(args.min_fee),
                transfer_fee_rate=float(args.transfer_fee_rate),
                stamp_tax_rate=float(args.stamp_tax_rate),
            )
            cash += sell_amt - sell_fee
            pnl = sell_amt - sell_fee - float(pos["buy_amt"]) - float(pos["buy_fee"])
            pnl_pct = pnl / (float(pos["buy_amt"]) + float(pos["buy_fee"])) if float(pos["buy_amt"]) + float(pos["buy_fee"]) > 0 else 0.0
            closed_today.append(
                Trade(
                    model=str(pos["model"]),
                    signal_date=str(pos["signal_date"]),
                    buy_date=str(pos["buy_date"]),
                    sell_date=sell_date,
                    stock_code=str(pos["stock_code"]),
                    stock_name=str(pos["stock_name"]),
                    signal_score=float(pos["signal_score"]),
                    buy_price=float(pos["buy_price"]),
                    sell_price=float(sell_px),
                    shares=int(pos["shares"]),
                    buy_fee=float(pos["buy_fee"]),
                    sell_fee=float(sell_fee),
                    pnl=float(pnl),
                    pnl_pct=float(pnl_pct),
                    hold_days=int(sell_idx - int(pos["buy_idx"]) + 1),
                    exit_rule=str(pos["exit_rule"]),
                    reason=rule_name,
                    equity_after=0.0,
                    rule=rule_name,
                )
            )
        active = still_active
        equity = cash + _open_position_value(active, cur_date)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        for trade in closed_today:
            trade.equity_after = float(equity)
            trades.append(trade)
        equity_rows.append(
            {
                "date": cur_date,
                "equity": float(equity),
                "cash": float(cash),
                "open_positions": int(len(active)),
                "opened_today": int(opened_today),
                "closed_today": int(len(closed_today)),
            }
        )

    final_equity = equity_rows[-1]["equity"] if equity_rows else float(args.initial_capital)
    win_rate = float(np.mean([t.pnl > 0 for t in trades])) * 100.0 if trades else 0.0
    avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else 0.0
    summary = {
        "model": "ordinary_manual",
        "signals": int(len(next_day_plan)),
        "planned_buys": int(len(plan)),
        "trades": int(len(trades)),
        "final_capital": float(final_equity),
        "return_pct": float(final_equity / float(args.initial_capital) - 1.0) * 100.0,
        "max_dd_pct": float(max_dd * 100.0),
        "win_rate": float(win_rate),
        "avg_hold_days": float(avg_hold),
        "rule": rule_name,
    }
    return trades, summary, pd.DataFrame(equity_rows)


def _relay_hybrid_entry_proxy(watch_row: pd.Series, day_bar: pd.Series, args: argparse.Namespace) -> Tuple[bool, float, str, Dict[str, float]]:
    open_px = _row_float(day_bar, "open")
    high_px = _row_float(day_bar, "high")
    low_px = _row_float(day_bar, "low")
    close_px = _row_float(day_bar, "close")
    metrics = {
        "watch_open": open_px,
        "watch_high": high_px,
        "watch_low": low_px,
        "watch_close": close_px,
        "proxy_high_ret_from_open": high_px / open_px - 1.0 if open_px > 0 and math.isfinite(high_px) else float("nan"),
        "proxy_low_drawdown_from_open": low_px / open_px - 1.0 if open_px > 0 and math.isfinite(low_px) else float("nan"),
        "proxy_close_ret_from_open": close_px / open_px - 1.0 if open_px > 0 and math.isfinite(close_px) else float("nan"),
    }
    if not (math.isfinite(open_px) and open_px > 0 and math.isfinite(high_px) and high_px > 0 and math.isfinite(low_px) and low_px > 0):
        return False, float("nan"), "missing_daily_proxy_bar", metrics
    low_drawdown = metrics["proxy_low_drawdown_from_open"]
    if math.isfinite(low_drawdown) and low_drawdown <= float(args.relay_max_entry_drawdown_pct):
        return False, float("nan"), "proxy_entry_drawdown_breached", metrics
    raw_entry = open_px * (1.0 + max(0.0, float(args.relay_trigger_ret_pct)))
    if high_px < raw_entry:
        return False, float("nan"), "proxy_trigger_not_reached", metrics
    return True, raw_entry, "proxy_open_rebreak_trigger", metrics


def _relay_hybrid_exit(code_hist: pd.DataFrame, buy_idx: int, buy_px: float, args: argparse.Namespace) -> Tuple[int, str, str]:
    if buy_idx + 1 >= len(code_hist):
        return buy_idx, "data_end_no_t2", "close"
    t2_idx = buy_idx + 1
    t2 = code_hist.iloc[t2_idx]
    t2_open = _row_float(t2, "open")
    t2_close = _row_float(t2, "close")
    if math.isfinite(t2_open) and buy_px > 0 and t2_open / buy_px - 1.0 <= float(args.relay_weak_open_stop_pct):
        return t2_idx, "relay_t2_weak_open_stop", "open"
    if math.isfinite(t2_close) and buy_px > 0 and t2_close / buy_px - 1.0 <= float(args.relay_hard_stop_close_pct):
        return t2_idx, "relay_t2_hard_stop_close", "close"
    if math.isfinite(t2_close) and buy_px > 0 and t2_close / buy_px - 1.0 >= float(args.relay_profit_extend_pct) and t2_idx + 1 < len(code_hist):
        return t2_idx + 1, "relay_profit_extend_t3_clear", "close"
    return t2_idx, "relay_t2_close_exit", "close"


def _mark_active_value(active: List[Dict[str, Any]], date: str) -> float:
    total = 0.0
    for pos in active:
        hist = pos["hist"]
        rows = hist[hist["date"] == date]
        px = _row_float(rows.iloc[0], "close") if not rows.empty else float(pos.get("buy_price", 0.0))
        if math.isfinite(px) and px > 0:
            total += int(pos["shares"]) * px
    return float(total)


def backtest_relay_hybrid_v1(
    relay_watchlist: pd.DataFrame,
    daily: pd.DataFrame,
    args: argparse.Namespace,
    *,
    stock_minute_exists: bool,
) -> Tuple[List[Trade], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rule_name = "relay_hybrid_v1" if stock_minute_exists else "relay_hybrid_v1_proxy_daily"
    empty_summary = {
        "model": "relay_manual",
        "signals": int(len(relay_watchlist)) if relay_watchlist is not None else 0,
        "watch_intraday": 0,
        "planned_buys": 0,
        "trades": 0,
        "final_capital": float(args.initial_capital),
        "return_pct": 0.0,
        "max_dd_pct": 0.0,
        "win_rate": 0.0,
        "avg_hold_days": 0.0,
        "rule": rule_name,
    }
    decision_columns = [
        "signal_date",
        "watch_date",
        "stock_code",
        "stock_name",
        "model_score",
        "rank_no",
        "planned_status",
        "decision",
        "decision_reason",
        "entry_price_proxy",
        "data_mode",
        "proxy_high_ret_from_open",
        "proxy_low_drawdown_from_open",
        "proxy_close_ret_from_open",
    ]
    if relay_watchlist is None or relay_watchlist.empty:
        return [], empty_summary, pd.DataFrame(), pd.DataFrame(columns=decision_columns)

    daily_groups = {code: grp.reset_index(drop=True) for code, grp in daily.groupby("stock_code", sort=False)}
    watch = relay_watchlist.copy()
    data_mode = "minute" if stock_minute_exists else "daily_proxy"
    watch_intraday = watch[watch["planned_status"] == "watch_intraday"].copy()
    empty_summary["watch_intraday"] = int(len(watch_intraday))
    dates = sorted(daily["date"].unique().tolist())
    watch_by_date = {d: grp.copy() for d, grp in watch_intraday.groupby("watch_date", sort=False)}

    decisions: List[Dict[str, Any]] = []
    for _, row in watch[watch["planned_status"] != "watch_intraday"].iterrows():
        decisions.append(
            {
                "signal_date": str(row.get("signal_date", "")),
                "watch_date": str(row.get("watch_date", "")),
                "stock_code": str(row.get("stock_code", "")),
                "stock_name": str(row.get("stock_name", "")),
                "model_score": _row_float(row, "model_score", 0.0),
                "rank_no": int(_row_float(row, "rank_no", 999.0)),
                "planned_status": str(row.get("planned_status", "")),
                "decision": "drop_before_intraday",
                "decision_reason": str(row.get("drop_reason", "")),
                "entry_price_proxy": np.nan,
                "data_mode": data_mode,
                "proxy_high_ret_from_open": np.nan,
                "proxy_low_drawdown_from_open": np.nan,
                "proxy_close_ret_from_open": np.nan,
            }
        )

    cash = float(args.initial_capital)
    active: List[Dict[str, Any]] = []
    trades: List[Trade] = []
    equity_rows: List[Dict[str, Any]] = []
    peak = cash
    max_dd = 0.0
    max_new_per_day = max(1, int(args.max_new_positions_per_day))
    max_total_position_pct = min(1.0, max(0.0, float(args.max_total_position_pct)))

    def close_due_positions(cur_date: str, price_col: str) -> int:
        nonlocal cash, active, trades
        closed = 0
        still_active: List[Dict[str, Any]] = []
        for pos in active:
            hist = pos["hist"]
            sell_idx = int(pos["sell_idx"])
            sell_date = str(hist.iloc[sell_idx]["date"])
            if sell_date != cur_date or str(pos["sell_price_col"]) != price_col:
                still_active.append(pos)
                continue
            raw_sell_px = _row_float(hist.iloc[sell_idx], price_col)
            if not (math.isfinite(raw_sell_px) and raw_sell_px > 0):
                still_active.append(pos)
                continue
            sell_px = raw_sell_px * (1.0 - max(0.0, float(args.slippage_rate)))
            sell_amt = int(pos["shares"]) * sell_px
            sell_fee = calc_side_cost(
                sell_amt,
                fee_rate=float(args.fee_rate),
                min_fee=float(args.min_fee),
                transfer_fee_rate=float(args.transfer_fee_rate),
                stamp_tax_rate=float(args.stamp_tax_rate),
            )
            cash += sell_amt - sell_fee
            pnl = sell_amt - sell_fee - float(pos["buy_amt"]) - float(pos["buy_fee"])
            cost_basis = float(pos["buy_amt"]) + float(pos["buy_fee"])
            trades.append(
                Trade(
                    model="relay",
                    signal_date=str(pos["signal_date"]),
                    buy_date=str(pos["buy_date"]),
                    sell_date=sell_date,
                    stock_code=str(pos["stock_code"]),
                    stock_name=str(pos["stock_name"]),
                    signal_score=float(pos["signal_score"]),
                    buy_price=float(pos["buy_price"]),
                    sell_price=float(sell_px),
                    shares=int(pos["shares"]),
                    buy_fee=float(pos["buy_fee"]),
                    sell_fee=float(sell_fee),
                    pnl=float(pnl),
                    pnl_pct=float(pnl / cost_basis) if cost_basis > 0 else 0.0,
                    hold_days=int(sell_idx - int(pos["buy_idx"]) + 1),
                    exit_rule=str(pos["exit_rule"]),
                    reason=data_mode,
                    equity_after=0.0,
                    rule=rule_name,
                )
            )
            closed += 1
        active = still_active
        return closed

    for cur_date in dates:
        closed_open = close_due_positions(cur_date, "open")
        opened_today = 0
        if cur_date in watch_by_date:
            candidates = watch_by_date[cur_date].sort_values(["rank_no", "model_score", "stock_code"], ascending=[True, False, True])
            for _, row in candidates.iterrows():
                base_decision = {
                    "signal_date": str(row.get("signal_date", "")),
                    "watch_date": str(row.get("watch_date", "")),
                    "stock_code": str(row.get("stock_code", "")),
                    "stock_name": str(row.get("stock_name", "")),
                    "model_score": _row_float(row, "model_score", 0.0),
                    "rank_no": int(_row_float(row, "rank_no", 999.0)),
                    "planned_status": str(row.get("planned_status", "")),
                    "data_mode": data_mode,
                }
                if opened_today >= max_new_per_day:
                    decisions.append({**base_decision, "decision": "skip", "decision_reason": "daily_new_position_limit", "entry_price_proxy": np.nan})
                    continue
                stock_code = str(row["stock_code"])
                hist = daily_groups.get(stock_code)
                if hist is None or hist.empty:
                    decisions.append({**base_decision, "decision": "skip", "decision_reason": "no_history", "entry_price_proxy": np.nan})
                    continue
                buy_rows = hist.index[hist["date"] == cur_date]
                if len(buy_rows) == 0:
                    decisions.append({**base_decision, "decision": "skip", "decision_reason": "no_watch_bar", "entry_price_proxy": np.nan})
                    continue
                buy_idx = int(buy_rows[0])
                ok, raw_entry, reason, metrics = _relay_hybrid_entry_proxy(row, hist.iloc[buy_idx], args)
                if not ok:
                    decisions.append({**base_decision, "decision": "skip", "decision_reason": reason, "entry_price_proxy": raw_entry, **metrics})
                    continue
                if buy_idx + 1 >= len(hist):
                    decisions.append({**base_decision, "decision": "skip", "decision_reason": "no_t2_exit_bar", "entry_price_proxy": raw_entry, **metrics})
                    continue
                buy_px = raw_entry * (1.0 + max(0.0, float(args.slippage_rate)))
                current_position_value = _mark_active_value(active, cur_date)
                current_equity = cash + current_position_value
                remaining_risk_budget = max(0.0, current_equity * max_total_position_pct - current_position_value)
                target_cash = min(cash, remaining_risk_budget, current_equity * float(args.relay_hybrid_position_pct))
                lots = max_lots_for_cash(
                    target_cash,
                    buy_px,
                    float(args.fee_rate),
                    float(args.min_fee),
                    transfer_fee_rate=float(args.transfer_fee_rate),
                )
                shares = lots * 100
                if shares < 100:
                    decisions.append({**base_decision, "decision": "skip", "decision_reason": "insufficient_position_cash", "entry_price_proxy": raw_entry, **metrics})
                    continue
                buy_amt = shares * buy_px
                buy_fee = calc_side_cost(
                    buy_amt,
                    fee_rate=float(args.fee_rate),
                    min_fee=float(args.min_fee),
                    transfer_fee_rate=float(args.transfer_fee_rate),
                )
                if buy_amt + buy_fee > cash + 1e-9:
                    decisions.append({**base_decision, "decision": "skip", "decision_reason": "cash_not_enough_after_fee", "entry_price_proxy": raw_entry, **metrics})
                    continue
                sell_idx, exit_rule, sell_price_col = _relay_hybrid_exit(hist, buy_idx, buy_px, args)
                if sell_idx <= buy_idx:
                    decisions.append({**base_decision, "decision": "skip", "decision_reason": "invalid_exit_bar", "entry_price_proxy": raw_entry, **metrics})
                    continue
                cash -= buy_amt + buy_fee
                active.append(
                    {
                        "signal_date": str(row["signal_date"]),
                        "buy_date": cur_date,
                        "stock_code": stock_code,
                        "stock_name": str(row.get("stock_name") or ""),
                        "signal_score": float(row["model_score"]),
                        "buy_price": float(buy_px),
                        "shares": int(shares),
                        "buy_fee": float(buy_fee),
                        "buy_amt": float(buy_amt),
                        "hist": hist,
                        "buy_idx": int(buy_idx),
                        "sell_idx": int(sell_idx),
                        "sell_price_col": str(sell_price_col),
                        "exit_rule": str(exit_rule),
                    }
                )
                decisions.append({**base_decision, "decision": "buy_proxy", "decision_reason": reason, "entry_price_proxy": raw_entry, **metrics})
                opened_today += 1
        closed_close = close_due_positions(cur_date, "close")
        equity = cash + _mark_active_value(active, cur_date)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        for trade in trades:
            if trade.sell_date == cur_date and trade.equity_after == 0.0:
                trade.equity_after = float(equity)
        equity_rows.append(
            {
                "date": cur_date,
                "equity": float(equity),
                "cash": float(cash),
                "open_positions": int(len(active)),
                "opened_today": int(opened_today),
                "closed_today": int(closed_open + closed_close),
            }
        )

    final_equity = equity_rows[-1]["equity"] if equity_rows else float(args.initial_capital)
    win_rate = float(np.mean([t.pnl > 0 for t in trades])) * 100.0 if trades else 0.0
    avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else 0.0
    summary = {
        "model": "relay_manual",
        "signals": int(len(relay_watchlist)),
        "watch_intraday": int(len(watch_intraday)),
        "planned_buys": int(sum(1 for d in decisions if d.get("decision") == "buy_proxy")),
        "trades": int(len(trades)),
        "final_capital": float(final_equity),
        "return_pct": float(final_equity / float(args.initial_capital) - 1.0) * 100.0,
        "max_dd_pct": float(max_dd * 100.0),
        "win_rate": float(win_rate),
        "avg_hold_days": float(avg_hold),
        "rule": rule_name,
    }
    return trades, summary, pd.DataFrame(equity_rows), pd.DataFrame(decisions, columns=decision_columns)


def build_risk_tables(trades: List[Trade]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    risk_columns = [
        "model",
        "rule",
        "trades",
        "wins",
        "losses",
        "win_rate_pct",
        "total_pnl",
        "avg_pnl_pct",
        "max_single_win_pct",
        "max_single_loss_pct",
        "max_consecutive_losses",
    ]
    monthly_columns = ["model", "rule", "month", "trades", "wins", "losses", "pnl", "avg_pnl_pct"]
    if not trades:
        return pd.DataFrame(columns=risk_columns), pd.DataFrame(columns=monthly_columns)

    df = pd.DataFrame([trade_to_dict(t) for t in trades], columns=TRADE_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=risk_columns), pd.DataFrame(columns=monthly_columns)
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)
    df["win"] = df["pnl"] > 0
    df["month"] = df["sell_date"].astype(str).str.slice(0, 7)
    df["rule"] = df["rule"].fillna("").replace("", "unknown")

    risk_rows: List[Dict[str, Any]] = []
    for (model, rule), grp in df.sort_values(["sell_date", "stock_code"]).groupby(["model", "rule"], sort=True):
        losses = (~grp["win"]).astype(int).tolist()
        max_streak = 0
        cur_streak = 0
        for is_loss in losses:
            if is_loss:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 0
        risk_rows.append(
            {
                "model": model,
                "rule": rule,
                "trades": int(len(grp)),
                "wins": int(grp["win"].sum()),
                "losses": int((~grp["win"]).sum()),
                "win_rate_pct": float(grp["win"].mean() * 100.0) if len(grp) else 0.0,
                "total_pnl": float(grp["pnl"].sum()),
                "avg_pnl_pct": float(grp["pnl_pct"].mean() * 100.0) if len(grp) else 0.0,
                "max_single_win_pct": float(grp["pnl_pct"].max() * 100.0) if len(grp) else 0.0,
                "max_single_loss_pct": float(grp["pnl_pct"].min() * 100.0) if len(grp) else 0.0,
                "max_consecutive_losses": int(max_streak),
            }
        )

    monthly = (
        df.groupby(["model", "rule", "month"], sort=True)
        .agg(
            trades=("pnl", "size"),
            wins=("win", "sum"),
            pnl=("pnl", "sum"),
            avg_pnl_pct=("pnl_pct", "mean"),
        )
        .reset_index()
    )
    if monthly.empty:
        monthly = pd.DataFrame(columns=monthly_columns)
    else:
        monthly["wins"] = monthly["wins"].astype(int)
        monthly["losses"] = monthly["trades"].astype(int) - monthly["wins"].astype(int)
        monthly["avg_pnl_pct"] = monthly["avg_pnl_pct"].astype(float) * 100.0
        monthly = monthly[monthly_columns]
    return pd.DataFrame(risk_rows, columns=risk_columns), monthly


def write_outputs(
    out_dir: Path,
    summaries: List[Dict[str, Any]],
    all_trades: List[Trade],
    meta: Dict[str, Any],
    next_day_plan: Optional[pd.DataFrame] = None,
    relay_watchlist: Optional[pd.DataFrame] = None,
    manual_open_trades: Optional[List[Trade]] = None,
    manual_open_summary: Optional[Dict[str, Any]] = None,
    manual_open_equity: Optional[pd.DataFrame] = None,
    relay_hybrid_trades: Optional[List[Trade]] = None,
    relay_hybrid_summary: Optional[Dict[str, Any]] = None,
    relay_hybrid_equity: Optional[pd.DataFrame] = None,
    relay_hybrid_decisions: Optional[pd.DataFrame] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame([trade_to_dict(t) for t in all_trades], columns=TRADE_COLUMNS)
    trades_df.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    if next_day_plan is not None:
        next_day_plan.to_csv(out_dir / "next_day_plan.csv", index=False, encoding="utf-8-sig")
    if relay_watchlist is not None:
        relay_watchlist.to_csv(out_dir / "relay_watchlist.csv", index=False, encoding="utf-8-sig")
    if manual_open_trades is not None:
        pd.DataFrame([trade_to_dict(t) for t in manual_open_trades], columns=TRADE_COLUMNS).to_csv(
            out_dir / "manual_open_trades.csv", index=False, encoding="utf-8-sig"
        )
    if manual_open_summary is not None:
        (out_dir / "manual_open_summary.json").write_text(json.dumps(manual_open_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if manual_open_equity is not None:
        manual_open_equity.reindex(columns=MANUAL_EQUITY_COLUMNS).to_csv(
            out_dir / "manual_open_equity.csv", index=False, encoding="utf-8-sig"
        )
    if relay_hybrid_trades is not None:
        pd.DataFrame([trade_to_dict(t) for t in relay_hybrid_trades], columns=TRADE_COLUMNS).to_csv(
            out_dir / "relay_hybrid_trades.csv", index=False, encoding="utf-8-sig"
        )
    if relay_hybrid_summary is not None:
        (out_dir / "relay_hybrid_summary.json").write_text(json.dumps(relay_hybrid_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if relay_hybrid_equity is not None:
        relay_hybrid_equity.reindex(columns=MANUAL_EQUITY_COLUMNS).to_csv(
            out_dir / "relay_hybrid_equity.csv", index=False, encoding="utf-8-sig"
        )
    if relay_hybrid_decisions is not None:
        relay_hybrid_decisions.to_csv(out_dir / "relay_hybrid_decisions.csv", index=False, encoding="utf-8-sig")
    risk_trades = list(all_trades)
    if manual_open_trades:
        risk_trades.extend(manual_open_trades)
    if relay_hybrid_trades:
        risk_trades.extend(relay_hybrid_trades)
    risk_summary, monthly_returns = build_risk_tables(risk_trades)
    risk_summary.to_csv(out_dir / "risk_summary.csv", index=False, encoding="utf-8-sig")
    monthly_returns.to_csv(out_dir / "monthly_returns.csv", index=False, encoding="utf-8-sig")
    audit_parts: List[pd.DataFrame] = []
    if next_day_plan is not None and not next_day_plan.empty:
        ordinary_audit = next_day_plan.copy()
        ordinary_audit["audit_type"] = "ordinary_next_day_plan"
        ordinary_audit["audit_status"] = ordinary_audit["planned_action"]
        ordinary_audit["audit_reason"] = ordinary_audit["skip_reason"]
        audit_parts.append(
            ordinary_audit[
                [
                    "audit_type",
                    "audit_status",
                    "signal_date",
                    "buy_date",
                    "model",
                    "stock_code",
                    "stock_name",
                    "score",
                    "audit_reason",
                ]
            ].rename(columns={"buy_date": "action_date"})
        )
    if relay_watchlist is not None and not relay_watchlist.empty:
        relay_audit = relay_watchlist.copy()
        relay_audit["audit_type"] = "relay_watchlist"
        relay_audit["audit_status"] = relay_audit["planned_status"]
        relay_audit["audit_reason"] = relay_audit["drop_reason"]
        audit_parts.append(
            relay_audit[
                [
                    "audit_type",
                    "audit_status",
                    "signal_date",
                    "watch_date",
                    "model",
                    "stock_code",
                    "stock_name",
                    "model_score",
                    "audit_reason",
                ]
            ].rename(columns={"watch_date": "action_date", "model_score": "score"})
        )
    if audit_parts:
        pd.concat(audit_parts, ignore_index=True).to_csv(out_dir / "signal_audit.csv", index=False, encoding="utf-8-sig")
    payload = {"meta": meta, "summaries": summaries, "trade_count": len(all_trades)}
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 模型 follow 回测报告",
        "",
        f"- 回测区间：{meta['start_date']} ~ {meta['end_date']}",
        f"- 实际数据终点：{meta['data_end_date']}",
        f"- 初始资金：{meta['initial_capital']:.2f}",
        f"- 成本假设：佣金双边 {meta['fee_rate']*100:.4f}% / 笔最低 {meta['min_fee']:.2f}，卖出印花税 {meta['stamp_tax_rate']*100:.4f}%，过户/规费 {meta['transfer_fee_rate']*100:.4f}%，单边滑点 {meta['slippage_rate']*100:.4f}%",
        "",
        "## 结论先说",
        "- 这几个评分模型本身只负责打分，没有代码级的统一卖出条件。",
        "- 仓库里没有把 `5日线破位` 写成四个模型的默认卖出规则。",
        "- 下面是我按真实日线做的统一 follow 回测，分成了 `MA5破位` 和 `模型风格规则` 两种视角。",
        "- 每一行都是该模型+规则单独用 10 万本金跑；同一模型持仓未平时不再叠买下一条信号。",
        "- 最大回撤按持仓期间每日收盘净值估算，不只看平仓后的资金曲线。",
        "",
        "## Repo 确认点",
        f"- 本次运行数据库：`{meta['db_dialect']}`；当前配置默认走 MySQL URL，所以按本 repo 的现状本地跑真实数据需要一个可连接的 MySQL，除非显式改用 `--db data/stock.db`。",
        f"- 当前库 `stock_minute`：{'存在' if meta['stock_minute_exists'] else '不存在'}。分钟线表只有运行 `getDataBaoStock.py --frequency 5/15/30/60` 后才会创建。",
        "- 次日接力评分读取的是 `stock_daily`、`stock_info` 和日线复盘池特征；代码里没有把分时数据接进 `model_relay_pool` 的打分特征。",
        "- 本轮新增输出：普通模型 `next_day_plan.csv`，Relay 单独 `relay_watchlist.csv`，统一审计 `signal_audit.csv`；Relay 表只代表次日盯盘，不代表开盘直接买。",
        "- 性能上已有水位表、批量 upsert、线程/进程分片；本轮已把按日期/分数查询的二级索引和策略日志表补进 `init.py`。",
        "",
        "## 结果汇总",
        "| 模型 | 规则 | 信号日数 | 交易数 | 复合收益 | 最大回撤 | 胜率 | 平均持有 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {rule} | {signals} | {trades} | {ret:.2f}% | {dd:.2f}% | {win:.2f}% | {hold:.2f}天 |".format(
                model=row["model"],
                rule=row["rule"],
                signals=int(row["signals"]),
                trades=int(row["trades"]),
                ret=float(row["return_pct"]),
                dd=float(row["max_dd_pct"]),
                win=float(row["win_rate"]),
                hold=float(row["avg_hold_days"]),
            )
        )
    ordinary_counts = {}
    if next_day_plan is not None and not next_day_plan.empty and "planned_action" in next_day_plan.columns:
        ordinary_counts = next_day_plan["planned_action"].value_counts(dropna=False).to_dict()
    relay_counts = {}
    if relay_watchlist is not None and not relay_watchlist.empty and "planned_status" in relay_watchlist.columns:
        relay_counts = relay_watchlist["planned_status"].value_counts(dropna=False).to_dict()
    manual_planned = int(manual_open_summary.get("planned_buys", 0)) if manual_open_summary else 0
    manual_trades = int(manual_open_summary.get("trades", 0)) if manual_open_summary else 0
    manual_return = float(manual_open_summary.get("return_pct", 0.0)) if manual_open_summary else 0.0
    relay_hybrid_rule = str(relay_hybrid_summary.get("rule", "")) if relay_hybrid_summary else ""
    relay_hybrid_planned = int(relay_hybrid_summary.get("planned_buys", 0)) if relay_hybrid_summary else 0
    relay_hybrid_trades_count = int(relay_hybrid_summary.get("trades", 0)) if relay_hybrid_summary else 0
    relay_hybrid_return = float(relay_hybrid_summary.get("return_pct", 0.0)) if relay_hybrid_summary else 0.0
    relay_hybrid_dd = float(relay_hybrid_summary.get("max_dd_pct", 0.0)) if relay_hybrid_summary else 0.0
    lines.extend(
        [
            "",
            "## 实盘计划验收",
            f"- 普通模型严格计划：`next_day_plan.csv` 共 {int(len(next_day_plan)) if next_day_plan is not None else 0} 条，动作分布 {ordinary_counts}；`manual_open_v1` 实际计划买入 {manual_planned} 笔，成交回测 {manual_trades} 笔，收益 {manual_return:.2f}%。",
            f"- Relay 候选：`relay_watchlist.csv` 共 {int(len(relay_watchlist)) if relay_watchlist is not None else 0} 条，状态分布 {relay_counts}；`watch_intraday` 只是次日盯盘，不是开盘买入指令。",
            f"- Relay 混合跟随：`{relay_hybrid_rule}` 触发买入 {relay_hybrid_planned} 笔，平仓 {relay_hybrid_trades_count} 笔，收益 {relay_hybrid_return:.2f}%，最大回撤 {relay_hybrid_dd:.2f}%；无分钟线时这是日线代理，不当成最终超短证据。",
            f"- 统一账户约束：每日最多新增 {int(meta.get('max_new_positions_per_day', 1))} 笔，总持仓上限 {float(meta.get('max_total_position_pct', 0.5))*100:.0f}%。",
            "- 风控切片已输出：`risk_summary.csv` 记录最大单笔亏损和连续亏损，`monthly_returns.csv` 记录分月收益。",
            "",
            "## 交易解释",
            "- `ma5_break`：隔日开盘买入后，从第 1 个可卖日开始检查，收盘跌破 MA5 就卖；若一直不破，最多持有 10 个交易日。",
            "- `laowang_style`：隔日开盘买入后，从第 1 个可卖日开始检查，收盘跌破 MA20 就卖；若一直不破，最多持有 20 个交易日。",
            "- `ywcx_style`：隔日开盘买入后，从第 1 个可卖日开始检查，收盘跌破 MA5 就卖；若一直不破，最多持有 5 个交易日。",
            "- `stwg_style`：隔日开盘买入后，从第 1 个可卖日开始检查，收盘跌破 MA10 就卖；若一直不破，最多持有 12 个交易日。",
            "- `fhkq_style`：按事件票处理，买入后第 1 个可卖日收盘离场。",
            "- `relay_style`：沿用接力脚本的规则，T+2 收盘为主，若 T+2 回撤过大则延到 T+3。",
            "",
            "## 说明",
            "- 结果基于当前数据库中已有的 `stock_daily` 和评分表。",
            "- `stock_daily` 目前只覆盖到 2026-04-30，所以 2026-05-01 只能视为报告结束点，不会凭空补行情。",
            "- YWCX 在这个数据库里 2025-2026 区间没有有效信号，说明该样本池在当前数据下基本没打开。",
            "- FHKQ 也几乎没有信号，属于极端事件模型，不能当成常规轮动模型看。",
            "",
            "## 模型原理与训练建议",
            "- LAOWANG：右侧趋势/回踩/低位结构的加权打分模型，适合把卖点放在 MA20 或关键支撑失守；训练优化应把持有期标签从单日涨跌改成 10-20 日风险收益比、最大回撤、是否破位。",
            "- YWCX：弱势次新从极弱到刚站上 MA5 的观察池模型，当前区间无有效样本；训练前先放宽硬过滤或补足次新/流通市值数据，否则没有样本可训。",
            "- STWG：平台缩量压缩后突破模型，MA10/MA20 是结构支撑；训练优化应加入突破后 3/5/10 日收益与假突破标签，单独惩罚次日低开、放量回落和破 MA10。",
            "- FHKQ：连续跌停后的流动性枯竭/开板事件模型，应该按事件交易处理；训练优化要扩大极端事件样本，标签用开板后 T+1/T+2 可执行收益和最差回撤。",
            "- Relay：短线接力是日线涨停结构 + 市场情绪特征的 MLP 二分类，不是真正的分时模型；要提升训练，下一步应把竞价、回封、炸板时间、分时量能加入特征，并按时间切分做 walk-forward，避免把未来行情气味揉进训练集。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    global args
    args = parse_args(argv)
    start_date = _to_iso(args.start_date)
    end_date = _to_iso(args.end_date)
    if start_date > end_date:
        raise SystemExit("start-date must be <= end-date")

    db_target = resolve_db_target(args)
    engine = make_engine(db_target)

    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()
        if engine.dialect.name == "sqlite":
            minute_row = conn.execute(text("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='stock_minute'")).fetchone()
        else:
            minute_row = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                      AND table_name = 'stock_minute'
                    """
                )
            ).fetchone()
    data_end = str(row[0]) if row and row[0] else end_date
    stock_minute_exists = bool(minute_row and int(minute_row[0] or 0) > 0)
    if data_end < end_date:
        end_date = data_end

    load_start = (dt.datetime.strptime(start_date, "%Y-%m-%d").date() - dt.timedelta(days=180)).strftime("%Y-%m-%d")
    daily = load_daily_frame(engine, load_start, end_date)
    if daily.empty:
        raise SystemExit("stock_daily is empty for requested range")
    daily = add_ma(daily)
    trade_dates = sorted(daily["date"].unique().tolist())

    signal_df = load_signals(
        engine,
        start_date=start_date,
        end_date=end_date,
        model=str(args.model),
        thresholds={
            "laowang": float(args.signal_threshold_laowang),
            "ywcx": float(args.signal_threshold_ywcx),
            "stwg": float(args.signal_threshold_stwg),
            "fhkq": float(args.signal_threshold_fhkq),
        },
        relay_threshold=float(args.relay_threshold),
        relay_top_k=int(args.relay_top_k),
        relay_max_board=args.relay_max_board,
    )
    if signal_df.empty:
        print("No signals found in requested window.")
        signal_df = pd.DataFrame()

    model_rules: List[Tuple[str, str, str]] = []
    if args.model in {"all", "laowang"}:
        model_rules.append(("laowang", "laowang_style", "ma20_break"))
        model_rules.append(("laowang", "ma5_break", "ma5_break"))
    if args.model in {"all", "ywcx"}:
        model_rules.append(("ywcx", "ywcx_style", "ma5_break"))
        model_rules.append(("ywcx", "ma5_break", "ma5_break"))
    if args.model in {"all", "stwg"}:
        model_rules.append(("stwg", "stwg_style", "ma10_break"))
        model_rules.append(("stwg", "ma5_break", "ma5_break"))
    if args.model in {"all", "fhkq"}:
        model_rules.append(("fhkq", "fhkq_style", "fixed_t2"))
        model_rules.append(("fhkq", "ma5_break", "ma5_break"))
    if args.model in {"all", "relay"}:
        model_rules.append(("relay", "relay_style", "relay_style"))
        model_rules.append(("relay", "ma5_break", "ma5_break"))

    all_trades: List[Trade] = []
    summaries: List[Dict[str, Any]] = []
    for model_name, rule_name, exit_mode in model_rules:
        model_signals = signal_df[signal_df["model"] == model_name].copy() if not signal_df.empty else pd.DataFrame()
        trades, summary = backtest_model(
            model_name=model_name,
            signals=model_signals,
            daily=daily,
            trade_dates=trade_dates,
            initial_capital=float(args.initial_capital),
            fee_rate=float(args.fee_rate),
            min_fee=float(args.min_fee),
            stamp_tax_rate=float(args.stamp_tax_rate),
            transfer_fee_rate=float(args.transfer_fee_rate),
            slippage_rate=float(args.slippage_rate),
            exit_mode=exit_mode,
        )
        for trade in trades:
            trade.rule = rule_name
        summary["rule"] = rule_name
        summaries.append(summary)
        all_trades.extend(trades)

    out_dir = Path(args.output_dir).expanduser()
    meta = {
        "start_date": start_date,
        "end_date": end_date,
        "data_end_date": data_end,
        "initial_capital": float(args.initial_capital),
        "fee_rate": float(args.fee_rate),
        "min_fee": float(args.min_fee),
        "stamp_tax_rate": float(args.stamp_tax_rate),
        "transfer_fee_rate": float(args.transfer_fee_rate),
        "slippage_rate": float(args.slippage_rate),
        "max_total_position_pct": float(args.max_total_position_pct),
        "max_new_positions_per_day": int(args.max_new_positions_per_day),
        "min_open_gap_override": None if args.min_open_gap is None else float(args.min_open_gap),
        "max_open_gap_override": None if args.max_open_gap is None else float(args.max_open_gap),
        "block_limit_up_open": bool(args.block_limit_up_open),
        "relay_hybrid_position_pct": float(args.relay_hybrid_position_pct),
        "relay_trigger_ret_pct": float(args.relay_trigger_ret_pct),
        "relay_max_entry_drawdown_pct": float(args.relay_max_entry_drawdown_pct),
        "relay_weak_open_stop_pct": float(args.relay_weak_open_stop_pct),
        "relay_hard_stop_close_pct": float(args.relay_hard_stop_close_pct),
        "relay_profit_extend_pct": float(args.relay_profit_extend_pct),
        "db_target": db_target,
        "db_dialect": str(engine.dialect.name),
        "stock_minute_exists": stock_minute_exists,
        "signal_thresholds": {
            "laowang": float(args.signal_threshold_laowang),
            "ywcx": float(args.signal_threshold_ywcx),
            "stwg": float(args.signal_threshold_stwg),
            "fhkq": float(args.signal_threshold_fhkq),
            "relay": float(args.relay_threshold),
        },
    }
    next_day_plan = build_next_day_plan(signal_df, daily, args)
    relay_watchlist = build_relay_watchlist(signal_df, daily, args)
    manual_open_trades, manual_open_summary, manual_open_equity = backtest_manual_open_v1(next_day_plan, daily, args)
    relay_hybrid_trades, relay_hybrid_summary, relay_hybrid_equity, relay_hybrid_decisions = backtest_relay_hybrid_v1(
        relay_watchlist,
        daily,
        args,
        stock_minute_exists=stock_minute_exists,
    )
    summaries.append(manual_open_summary)
    summaries.append(relay_hybrid_summary)
    write_outputs(
        out_dir,
        summaries,
        all_trades,
        meta,
        next_day_plan=next_day_plan,
        relay_watchlist=relay_watchlist,
        manual_open_trades=manual_open_trades,
        manual_open_summary=manual_open_summary,
        manual_open_equity=manual_open_equity,
        relay_hybrid_trades=relay_hybrid_trades,
        relay_hybrid_summary=relay_hybrid_summary,
        relay_hybrid_equity=relay_hybrid_equity,
        relay_hybrid_decisions=relay_hybrid_decisions,
    )
    print(f"report={out_dir / 'report.md'}")
    print(f"summary={out_dir / 'summary.json'}")
    print(f"trades={out_dir / 'trades.csv'}")
    print(f"next_day_plan={out_dir / 'next_day_plan.csv'}")
    print(f"relay_watchlist={out_dir / 'relay_watchlist.csv'}")
    print(f"signal_audit={out_dir / 'signal_audit.csv'}")
    print(f"manual_open_trades={out_dir / 'manual_open_trades.csv'}")
    print(f"manual_open_summary={out_dir / 'manual_open_summary.json'}")
    print(f"relay_hybrid_trades={out_dir / 'relay_hybrid_trades.csv'}")
    print(f"relay_hybrid_summary={out_dir / 'relay_hybrid_summary.json'}")
    print(f"relay_hybrid_decisions={out_dir / 'relay_hybrid_decisions.csv'}")
    print(f"risk_summary={out_dir / 'risk_summary.csv'}")
    print(f"monthly_returns={out_dir / 'monthly_returns.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
