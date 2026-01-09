# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from a_stock_analyzer import db
from a_stock_analyzer.runtime import (
    add_db_args,
    make_engine_for_workers,
    normalize_trade_date,
    resolve_db_from_args,
    resolve_latest_stock_daily_date,
    setup_logging,
    write_dataframe_csv,
    yyyymmdd_from_date,
)


OUTPUT_COLUMNS = [
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
]


STRATEGY_POSITIONING = """\
本模块是【极端情绪博弈模型】：
- 不用于中长期持仓
- 建议：A/B 级 -> 人工确认；仓位 <= 常规策略的 20%
"""


RISK_DISCLOSURE = """\
风险声明：
连续跌停博弈存在极高风险。本模块仅用于研究与辅助决策，不构成投资建议。
"""


def _round_half_up_2(v: pd.Series) -> pd.Series:
    arr = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
    out = np.floor(arr * 100.0 + 0.5) / 100.0
    return pd.Series(out, index=v.index)


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
    up = name.upper()
    return "ST" in up


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


def _calc_fhkq_for_one_stock(
    df_daily_one: pd.DataFrame,
    *,
    trade_date: str,
    stock_code: str,
    stock_name: str,
) -> Optional[Dict[str, Any]]:
    """
    df_daily_one: single-stock daily bars, sorted by date asc.
    Required columns: date, open, high, low, close, volume, amount
    """
    if df_daily_one is None or df_daily_one.empty:
        return None

    d = df_daily_one.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    d = d[d["date"] <= trade_date].reset_index(drop=True)
    if d.empty or str(d["date"].iloc[-1]) != trade_date:
        return None

    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c not in d.columns:
            raise ValueError(f"df_daily missing column: {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")

    is_st_flag = False
    if "is_st" in d.columns:
        try:
            is_st_flag = int(pd.to_numeric(d["is_st"], errors="coerce").fillna(0).iloc[-1]) == 1
        except Exception:  # noqa: BLE001
            is_st_flag = False

    # Hard exclusions
    if is_st_flag or _is_st_name(stock_name):
        return None

    limit_pct = _infer_limit_pct(stock_code, stock_name, is_st_flag)
    if "limit_down" not in d.columns:
        prev_close = d["close"].shift(1)
        d["limit_down"] = _round_half_up_2(prev_close * (1.0 - float(limit_pct)))
    else:
        d["limit_down"] = pd.to_numeric(d["limit_down"], errors="coerce")

    eps = 1e-3
    is_limit_down = (d["close"] - d["limit_down"]).abs() <= eps
    is_limit_down = is_limit_down.fillna(False)
    if not bool(is_limit_down.iloc[-1]):
        return None

    # Consecutive limit-down count (from trade_date backwards)
    consecutive = 0
    for v in is_limit_down.iloc[::-1].to_list():
        if bool(v):
            consecutive += 1
        else:
            break

    # Focus only: 2+ consecutive limit-down
    if consecutive < 2:
        return None

    # Filter: last 5 days all limit-down and all volume==0 (one-word lock)
    if len(d) >= 5:
        vol5 = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0).tail(5)
        if bool(is_limit_down.tail(5).all()) and bool((vol5 == 0).all()):
            return None

    # Filter: last 10-day cumulative drop > 60%
    if len(d) >= 10:
        c0 = float(d["close"].iloc[-10]) if pd.notna(d["close"].iloc[-10]) else np.nan
        c1 = float(d["close"].iloc[-1]) if pd.notna(d["close"].iloc[-1]) else np.nan
        if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 and (c1 / c0 - 1.0) <= -0.60:
            return None

    # Ratios: today vs last 5-day mean (including today, per spec)
    vol_today = float(d["volume"].iloc[-1]) if pd.notna(d["volume"].iloc[-1]) else 0.0
    amt_today = float(d["amount"].iloc[-1]) if pd.notna(d["amount"].iloc[-1]) else 0.0
    vol_mean5 = float(pd.to_numeric(d["volume"], errors="coerce").fillna(0.0).tail(5).mean())
    amt_mean5 = float(pd.to_numeric(d["amount"], errors="coerce").fillna(0.0).tail(5).mean())

    volume_ratio = float(vol_today / vol_mean5) if vol_mean5 > 0 else 0.0
    amount_ratio = float(amt_today / amt_mean5) if amt_mean5 > 0 else 0.0

    # Open-board flag: prefer OHLC: limit-down close but intraday traded above limit-down.
    ld_today = float(d["limit_down"].iloc[-1]) if pd.notna(d["limit_down"].iloc[-1]) else np.nan
    high_today = float(d["high"].iloc[-1]) if pd.notna(d["high"].iloc[-1]) else np.nan
    low_today = float(d["low"].iloc[-1]) if pd.notna(d["low"].iloc[-1]) else np.nan

    open_board_flag = 0
    if np.isfinite(ld_today):
        if np.isfinite(high_today) and high_today > ld_today + eps:
            open_board_flag = 1
        # Fallback to spec literal (rare; helps if high is missing/dirty)
        elif np.isfinite(low_today) and low_today < ld_today - eps:
            open_board_flag = 1

    liquidity_exhaust = int(consecutive >= 3 and volume_ratio >= 1.0 and open_board_flag == 1)

    structure_score = _score_structure(consecutive)
    volume_score = _score_volume_ratio(volume_ratio)
    amount_score = _score_amount_ratio(amount_ratio)
    open_board_score = 20 if open_board_flag == 1 else 0
    exhaust_score = 20 if liquidity_exhaust == 1 else 0

    score = float(structure_score + volume_score + amount_score + open_board_score + exhaust_score)

    # Out-of-range penalty (spec: >6 risk up, score down)
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


def run_fhkq(df_daily: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """
    输入：标准日K DataFrame
    输出：评分结果 DataFrame

    支持两种形态：
    1) 多股票：包含 stock_code 列（可选 stock_name / is_st / limit_down）
    2) 单股票：仍需包含 stock_code（固定值）以便输出
    """
    if df_daily is None or df_daily.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if "stock_code" not in df_daily.columns:
        raise ValueError("df_daily must include 'stock_code' column")
    if "date" not in df_daily.columns:
        raise ValueError("df_daily must include 'date' column")

    trade_date_norm = normalize_trade_date(trade_date)

    df = df_daily.copy()
    if "stock_name" not in df.columns:
        df["stock_name"] = ""

    out_rows: list[Dict[str, Any]] = []
    for code, grp in df.groupby("stock_code"):
        name = str(grp["stock_name"].iloc[-1]) if "stock_name" in grp.columns else ""
        row = _calc_fhkq_for_one_stock(
            grp,
            trade_date=trade_date_norm,
            stock_code=str(code),
            stock_name=name,
        )
        if row:
            out_rows.append(row)

    if not out_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(out_rows)
    out = out[OUTPUT_COLUMNS]
    out = out.sort_values(["fhkq_score", "consecutive_limit_down", "stock_code"], ascending=[False, False, True])
    out = out.reset_index(drop=True)
    return out


def run_fhkq_from_db(
    *,
    db_target: str,
    trade_date: str,
    output_csv: Path,
    workers: int = 8,
) -> pd.DataFrame:
    trade_date_norm = normalize_trade_date(trade_date)

    w = max(1, int(workers))
    engine, w = make_engine_for_workers(db_target, w)

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT d.stock_code, COALESCE(i.name, '') AS name
                FROM stock_daily d
                LEFT JOIN stock_info i ON i.stock_code = d.stock_code
                WHERE d.date = :trade_date
                """
            ),
            {"trade_date": trade_date_norm},
        ).fetchall()

    stocks = [(str(r[0]), str(r[1] or "")) for r in rows if r and r[0]]
    if not stocks:
        raise RuntimeError(f"No stock_daily rows found for trade_date={trade_date_norm}. Did you run pipeline?")

    hist_limit = 120

    def process_stock(code: str, name: str) -> Optional[Dict[str, Any]]:
        try:
            with engine.connect() as conn:
                hist = db.load_daily_until(conn, code, end_date=trade_date_norm, limit=hist_limit)
            if hist is None or hist.empty:
                return None
            return _calc_fhkq_for_one_stock(
                hist,
                trade_date=trade_date_norm,
                stock_code=code,
                stock_name=name,
            )
        except Exception:  # noqa: BLE001
            logging.exception("FHKQ failed: %s", code)
            return None

    out_rows: list[Dict[str, Any]] = []
    if w <= 1:
        for code, name in stocks:
            r = process_stock(code, name)
            if r:
                out_rows.append(r)
    else:
        with ThreadPoolExecutor(max_workers=w) as ex:
            futures = [ex.submit(process_stock, code, name) for code, name in stocks]
            for f in as_completed(futures):
                r = f.result()
                if r:
                    out_rows.append(r)

    out = pd.DataFrame(out_rows, columns=OUTPUT_COLUMNS)
    out = out.sort_values(["fhkq_score", "consecutive_limit_down", "stock_code"], ascending=[False, False, True])
    out = out.reset_index(drop=True)
    write_dataframe_csv(out, output_csv, columns=OUTPUT_COLUMNS)

    logging.info("Exported %d rows -> %s", len(out), output_csv)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="fhkq.py - 连续跌停开板 / 反抽博弈评分模块")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    add_db_args(p)
    p.add_argument("--trade-date", default=None, help="YYYYMMDD or YYYY-MM-DD (default: latest stock_daily date)")
    p.add_argument("--output", default=None, help="CSV output path (default: output/fhkq_YYYYMMDD.csv)")
    p.add_argument("--workers", type=int, default=8, help="Thread workers (MySQL recommended)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_from_args(args)

    trade_date = args.trade_date or resolve_latest_stock_daily_date(db_target)
    trade_date_norm = normalize_trade_date(trade_date)
    yyyymmdd = yyyymmdd_from_date(trade_date_norm)
    output_csv = Path(args.output) if args.output else Path(f"output/fhkq_{yyyymmdd}.csv")

    run_fhkq_from_db(
        db_target=db_target,
        trade_date=trade_date_norm,
        output_csv=output_csv,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
