#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search Relay hybrid follow parameters on existing model outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text

import backtest_model_follow as bt


def parse_csv_float(value: str) -> List[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_csv_int(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize relay_hybrid_v1 proxy parameters.")
    p.add_argument("--config", default="config.ini")
    p.add_argument("--db-url", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--start-date", default="2025-01-01")
    p.add_argument("--end-date", default="2026-05-01")
    p.add_argument("--output-dir", default="reports/relay_hybrid_optimization")
    p.add_argument("--thresholds", default="0.70,0.75,0.80,0.85")
    p.add_argument("--top-ks", default="1,2")
    p.add_argument("--trigger-rets", default="0.000,0.005,0.010")
    p.add_argument("--max-entry-drawdowns", default="-0.02,-0.03,-0.04")
    p.add_argument("--position-pct", type=float, default=0.04)
    p.add_argument("--initial-capital", type=float, default=bt.DEFAULT_INITIAL_CAPITAL)
    p.add_argument("--fee-rate", type=float, default=bt.DEFAULT_FEE_RATE)
    p.add_argument("--min-fee", type=float, default=bt.DEFAULT_MIN_FEE)
    p.add_argument("--stamp-tax-rate", type=float, default=bt.DEFAULT_STAMP_TAX_RATE)
    p.add_argument("--transfer-fee-rate", type=float, default=bt.DEFAULT_TRANSFER_FEE_RATE)
    p.add_argument("--slippage-rate", type=float, default=bt.DEFAULT_SLIPPAGE_RATE)
    p.add_argument("--max-total-position-pct", type=float, default=bt.DEFAULT_MAX_TOTAL_POSITION_PCT)
    p.add_argument("--max-new-positions-per-day", type=int, default=bt.DEFAULT_MAX_NEW_POSITIONS_PER_DAY)
    return p.parse_args(argv)


def clone_args(base: argparse.Namespace, **overrides: Any) -> SimpleNamespace:
    data = vars(base).copy()
    data.update(overrides)
    return SimpleNamespace(**data)


def score_candidate(summary: Dict[str, Any], risk_row: Dict[str, Any]) -> float:
    trades = int(summary.get("trades", 0) or 0)
    ret = float(summary.get("return_pct", 0.0) or 0.0)
    dd = float(summary.get("max_dd_pct", 0.0) or 0.0)
    max_loss = abs(float(risk_row.get("max_single_loss_pct", 0.0) or 0.0))
    loss_streak = int(risk_row.get("max_consecutive_losses", 0) or 0)
    trade_penalty = max(0, 15 - trades) * 0.25
    return float(ret - 1.5 * dd - 0.25 * max_loss - 0.50 * loss_streak - trade_penalty)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    start_date = bt._to_iso(args.start_date)
    end_date = bt._to_iso(args.end_date)
    db_target = bt.resolve_db_target(args)
    engine = bt.make_engine(db_target)

    with engine.connect() as conn:
        data_end = str(conn.execute(text("SELECT MAX(date) FROM stock_daily")).scalar() or end_date)
        if engine.dialect.name == "sqlite":
            stock_minute_exists = bool(
                conn.execute(text("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='stock_minute'")).scalar() or 0
            )
        else:
            stock_minute_exists = bool(
                conn.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM information_schema.tables
                        WHERE table_schema = DATABASE()
                          AND table_name = 'stock_minute'
                        """
                    )
                ).scalar()
                or 0
            )
    if data_end < end_date:
        end_date = data_end

    load_start = (dt.datetime.strptime(start_date, "%Y-%m-%d").date() - dt.timedelta(days=180)).strftime("%Y-%m-%d")
    daily = bt.add_ma(bt.load_daily_frame(engine, load_start, end_date))
    signal_df = bt.load_signals(
        engine,
        start_date=start_date,
        end_date=end_date,
        model="relay",
        thresholds={"laowang": 60.0, "ywcx": 55.0, "stwg": 55.0, "fhkq": 60.0},
        relay_threshold=0.62,
        relay_top_k=max(parse_csv_int(args.top_ks)),
        relay_max_board=None,
    )

    rows: List[Dict[str, Any]] = []
    thresholds = parse_csv_float(args.thresholds)
    top_ks = parse_csv_int(args.top_ks)
    trigger_rets = parse_csv_float(args.trigger_rets)
    drawdowns = parse_csv_float(args.max_entry_drawdowns)
    for threshold, top_k, trigger_ret, drawdown in itertools.product(thresholds, top_ks, trigger_rets, drawdowns):
        run_args = clone_args(
            args,
            plan_threshold_relay=threshold,
            plan_relay_top_k=top_k,
            relay_hybrid_position_pct=float(args.position_pct),
            relay_trigger_ret_pct=trigger_ret,
            relay_max_entry_drawdown_pct=drawdown,
            relay_weak_open_stop_pct=-0.03,
            relay_hard_stop_close_pct=-0.035,
            relay_profit_extend_pct=0.05,
        )
        watchlist = bt.build_relay_watchlist(signal_df, daily, run_args)
        trades, summary, _, decisions = bt.backtest_relay_hybrid_v1(
            watchlist,
            daily,
            run_args,
            stock_minute_exists=stock_minute_exists,
        )
        risk_summary, _ = bt.build_risk_tables(trades)
        if risk_summary.empty:
            risk_row: Dict[str, Any] = {
                "max_single_loss_pct": 0.0,
                "max_consecutive_losses": 0,
                "avg_pnl_pct": 0.0,
            }
        else:
            risk_row = risk_summary.iloc[0].to_dict()
        buy_count = int((decisions["decision"] == "buy_proxy").sum()) if not decisions.empty else 0
        skip_reasons = decisions["decision_reason"].value_counts(dropna=False).head(5).to_dict() if not decisions.empty else {}
        row = {
            "plan_threshold_relay": float(threshold),
            "plan_relay_top_k": int(top_k),
            "relay_trigger_ret_pct": float(trigger_ret),
            "relay_max_entry_drawdown_pct": float(drawdown),
            "watchlist_rows": int(len(watchlist)),
            "watch_intraday": int(summary.get("watch_intraday", 0) or 0),
            "buy_count": int(buy_count),
            "trades": int(summary.get("trades", 0) or 0),
            "return_pct": float(summary.get("return_pct", 0.0) or 0.0),
            "max_dd_pct": float(summary.get("max_dd_pct", 0.0) or 0.0),
            "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
            "avg_hold_days": float(summary.get("avg_hold_days", 0.0) or 0.0),
            "avg_pnl_pct": float(risk_row.get("avg_pnl_pct", 0.0) or 0.0),
            "max_single_loss_pct": float(risk_row.get("max_single_loss_pct", 0.0) or 0.0),
            "max_consecutive_losses": int(risk_row.get("max_consecutive_losses", 0) or 0),
            "top_skip_reasons_json": json.dumps(skip_reasons, ensure_ascii=False),
        }
        row["objective"] = score_candidate(summary, row)
        rows.append(row)

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(
        ["objective", "trades", "return_pct"],
        ascending=[False, False, False],
    )
    df.to_csv(out_dir / "candidates.csv", index=False, encoding="utf-8-sig")
    best = df.iloc[0].to_dict() if not df.empty else {}
    (out_dir / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Relay Hybrid 参数扫描",
        "",
        f"- 区间：{start_date} ~ {end_date}",
        f"- 数据模式：{'minute' if stock_minute_exists else 'daily_proxy'}",
        f"- 组合数：{len(df)}",
        "",
        "## 最优候选",
        "",
    ]
    if best:
        for key in [
            "plan_threshold_relay",
            "plan_relay_top_k",
            "relay_trigger_ret_pct",
            "relay_max_entry_drawdown_pct",
            "trades",
            "return_pct",
            "max_dd_pct",
            "win_rate",
            "max_single_loss_pct",
            "max_consecutive_losses",
            "objective",
        ]:
            lines.append(f"- {key}: {best.get(key)}")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 这是样本内参数扫描，用于找整改方向，不是上线结论。",
            "- 当前无 `stock_minute` 时，所有盘中触发都是日线 OHLC 代理。",
            "- 真正上线必须用 walk-forward 和真实分钟线重新验收。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"candidates={out_dir / 'candidates.csv'}")
    print(f"best_config={out_dir / 'best_config.json'}")
    print(f"report={out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
