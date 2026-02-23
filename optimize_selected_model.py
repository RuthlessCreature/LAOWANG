#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

import mock_backtest as backtest_model
import relay_strategy_model as relay_model


@dataclass
class RiskProfile:
    name: str
    max_broken_rate: Optional[float]
    min_red_rate: Optional[float]
    max_limit_down: Optional[int]
    max_pullback: Optional[float]


@dataclass
class CandidateResult:
    round_no: int
    seed_offset: int
    variant: str
    alpha: int
    threshold: float
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
    ret: float
    max_dd: float
    trades: int
    final_capital: float
    fee: float


def _to_iso(s: str) -> str:
    v = str(s or "").strip()
    if len(v) == 8 and v.isdigit():
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    return dt.datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")


def _fmt_opt(v: Optional[float]) -> str:
    if v is None:
        return "None"
    return f"{float(v):.6f}".rstrip("0").rstrip(".")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optimize selected-group relay strategy with DD constraint.")
    p.add_argument("--config", default="config.ini")
    p.add_argument("--db-url", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--rules-file", default="rules.txt")
    p.add_argument("--train-start-date", default="2020-01-01")
    p.add_argument("--backtest-start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2026-02-13")
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--max-dd-limit", type=float, default=0.10, help="Maximum allowed drawdown ratio.")
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--patience", type=int, default=3, help="Stop when return does not improve in consecutive rounds.")
    p.add_argument("--seed-step", type=int, default=97)
    p.add_argument("--samples-per-key", type=int, default=10)
    p.add_argument(
        "--variant-profile",
        choices=["base", "expanded", "aggressive"],
        default="base",
        help="Model architecture search profile.",
    )
    p.add_argument("--alphas", default="10,15")
    p.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--report-dir", default="reports/selected_opt")
    return p


def _parse_float_list(text: str, *, lo: float, hi: float) -> List[float]:
    vals: List[float] = []
    for x in str(text or "").split(","):
        s = str(x).strip()
        if not s:
            continue
        try:
            v = float(s)
        except Exception:
            continue
        vals.append(max(lo, min(hi, v)))
    out = sorted(set(vals))
    return out


def _parse_int_list(text: str, *, lo: int, hi: int) -> List[int]:
    vals: List[int] = []
    for x in str(text or "").split(","):
        s = str(x).strip()
        if not s:
            continue
        try:
            v = int(float(s))
        except Exception:
            continue
        vals.append(max(lo, min(hi, v)))
    out = sorted(set(vals))
    return out


def _required_pool_columns() -> List[str]:
    return [
        "date",
        "stock_code",
        "stock_name",
        "is_st",
        "close",
        "next_date",
        "next_open",
        "next2_date",
        "next2_close",
        "next3_date",
        "next3_close",
        "board_count",
        "ret1",
        "amp",
        "pullback",
        "close_open_ret",
        "broken_rate",
        "red_rate",
        "limit_up_count",
        "limit_down_count",
        "amount_change5",
        "tomorrow_stage",
        "tomorrow_prob",
    ]


def _ensure_pool_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults: Dict[str, Any] = {
        "stock_name": "",
        "is_st": False,
        "close": np.nan,
        "next_date": "",
        "next_open": np.nan,
        "next2_date": "",
        "next2_close": np.nan,
        "next3_date": "",
        "next3_close": np.nan,
        "board_count": 0,
        "ret1": np.nan,
        "amp": np.nan,
        "pullback": np.nan,
        "close_open_ret": np.nan,
        "broken_rate": np.nan,
        "red_rate": np.nan,
        "limit_up_count": 0,
        "limit_down_count": 0,
        "amount_change5": np.nan,
        "tomorrow_stage": "N/A",
        "tomorrow_prob": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    cols = _required_pool_columns()
    return out[cols].copy()


def _risk_profiles() -> List[RiskProfile]:
    return [
        RiskProfile("strict", 0.40, 0.28, 15, 0.08),
        RiskProfile("balanced", 0.45, 0.24, 20, 0.09),
        RiskProfile("off", None, None, None, None),
    ]


def _is_better(a: Optional[CandidateResult], b: CandidateResult, *, max_dd_limit: float) -> bool:
    if a is None:
        return True
    a_ok = a.max_dd <= max_dd_limit
    b_ok = b.max_dd <= max_dd_limit
    if b_ok and not a_ok:
        return True
    if a_ok and not b_ok:
        return False
    if b_ok and a_ok:
        if b.ret > a.ret + 1e-12:
            return True
        if b.ret < a.ret - 1e-12:
            return False
        if b.max_dd < a.max_dd - 1e-12:
            return True
        if b.max_dd > a.max_dd + 1e-12:
            return False
        return b.trades > a.trades
    if b.max_dd < a.max_dd - 1e-12:
        return True
    if b.max_dd > a.max_dd + 1e-12:
        return False
    return b.ret > a.ret + 1e-12


def _ops_headers() -> List[str]:
    return [
        "\u64cd\u4f5c\u65e5\u671f",
        "\u64cd\u4f5c\u65f6\u95f4",
        "\u80a1\u7968\u4ee3\u7801",
        "\u80a1\u7968\u540d\u79f0",
        "\u64cd\u4f5c\uff08\u4e70or\u5356\uff09",
        "\u64cd\u4f5c\u4ef7\u683c",
        "\u603b\u8d44\u4ea7",
    ]


def _write_operations_csv(path: Path, trades: Sequence[Dict[str, Any]], initial_capital: float) -> None:
    headers = _ops_headers()
    cash = float(initial_capital)
    rows: List[Dict[str, str]] = []
    for t in trades:
        buy_fee = float(t["buy_fee"])
        buy_asset = cash - buy_fee
        rows.append(
            {
                headers[0]: str(t["buy_date"]),
                headers[1]: str(t.get("buy_time", "09:30:00")),
                headers[2]: str(t["stock_code"]),
                headers[3]: str(t["stock_name"]),
                headers[4]: "buy",
                headers[5]: f"{float(t['buy_price']):.3f}",
                headers[6]: f"{buy_asset:.2f}",
            }
        )
        cash = float(t["equity_after"])
        rows.append(
            {
                headers[0]: str(t["sell_date"]),
                headers[1]: str(t.get("sell_time", "15:00:00")),
                headers[2]: str(t["stock_code"]),
                headers[3]: str(t["stock_name"]),
                headers[4]: "sell",
                headers[5]: f"{float(t['sell_price']):.3f}",
                headers[6]: f"{cash:.2f}",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def _cfg_text(c: CandidateResult) -> str:
    return (
        f"{c.variant}/a{c.alpha}/{c.sell_rule}/k{c.top_k}/"
        f"gap[{_fmt_opt(c.gap_min)},{_fmt_opt(c.gap_max)}]/"
        f"board<={c.max_board if c.max_board is not None else 'None'}/"
        f"risk={c.risk_profile}/alloc={c.alloc_pct:.2f}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    train_start = _to_iso(args.train_start_date)
    start_date = _to_iso(args.backtest_start_date)
    end_date = _to_iso(args.end_date)
    initial_capital = float(args.initial_capital)
    max_dd_limit = float(args.max_dd_limit)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    ns = argparse.Namespace(config=args.config, db_url=args.db_url, db=args.db)
    engine = relay_model.make_engine(relay_model.resolve_db_target(ns))
    rules = backtest_model._load_execution_rules(args.rules_file)

    load_start = (dt.datetime.strptime(train_start, "%Y-%m-%d").date() - dt.timedelta(days=60)).strftime("%Y-%m-%d")
    base_raw = relay_model.load_base_frame(engine, load_start, end_date)
    if base_raw.empty:
        raise SystemExit(f"no stock_daily data in range {load_start}~{end_date}")
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
        raise SystemExit("no candidate rows after sample build")

    full_pool["date"] = full_pool["date"].astype(str)
    full_pool["stock_code"] = full_pool["stock_code"].astype(str)
    full_pool = backtest_model._attach_future_fields(engine, full_pool, train_start)

    train_df = full_pool.copy()
    train_df = train_df[(train_df["next_open"].notna()) & (train_df["next2_close"].notna())].copy()
    train_df["next_open"] = pd.to_numeric(train_df["next_open"], errors="coerce")
    train_df["next2_close"] = pd.to_numeric(train_df["next2_close"], errors="coerce")
    train_df = train_df[(train_df["next_open"] > 0) & (train_df["next2_close"] > 0)].copy()
    train_df["t1_hold_ret"] = train_df["next2_close"] / train_df["next_open"] - 1.0
    if train_df.empty:
        raise SystemExit("no train samples after target construction")

    base_pool = full_pool[full_pool["date"] >= start_date].copy()
    if base_pool.empty:
        raise SystemExit(f"no candidate rows from {start_date}")
    base_pool = _ensure_pool_columns(base_pool)

    trade_dates = backtest_model._load_trade_dates(engine, start_date)
    if not trade_dates:
        raise SystemExit(f"no trade dates from {start_date}")

    feat_cols = backtest_model._feature_cols()
    x_train_raw = train_df[feat_cols].to_numpy(dtype=float)
    x_base_raw = full_pool.loc[base_pool.index, feat_cols].to_numpy(dtype=float)
    x_train, x_base, _, _ = relay_model.fill_and_scale(x_train_raw, x_base_raw)

    variants = [
        backtest_model.ModelVariant(
            name=str(spec["name"]),
            hidden=int(spec["hidden"]),
            epochs=int(spec["epochs"]),
            lr=float(spec["lr"]),
            weight_decay=float(spec["weight_decay"]),
            target_gt=float(spec["target_gt"]),
            seed=int(spec["seed"]),
        )
        for spec in backtest_model.list_variant_specs(str(args.variant_profile))
    ]
    alphas = _parse_int_list(args.alphas, lo=1, hi=30) or [10, 15]
    thresholds = _parse_float_list(args.thresholds, lo=0.50, hi=0.99) or [0.50, 0.55, 0.60, 0.65, 0.70]
    sell_rules = ["t2_close", "strong_hold_t3", "profit_lock_t3"]
    top_ks = [1, 2, 3]
    gap_windows: List[Tuple[Optional[float], Optional[float]]] = [(None, None), (-0.005, 0.02), (-0.01, 0.02)]
    board_caps: List[Optional[int]] = [2, 3, None]
    alloc_pcts = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    risk_profiles = _risk_profiles()
    samples_per_key = max(1, int(args.samples_per_key))

    best_global: Optional[CandidateResult] = None
    history_rows: List[Dict[str, Any]] = []
    no_improve = 0

    max_rounds = max(1, int(args.max_rounds))
    patience = max(1, int(args.patience))
    seed_step = int(args.seed_step)

    pool_base = base_pool.copy()

    for round_no in range(1, max_rounds + 1):
        seed_offset = seed_step * (round_no - 1)
        round_best: Optional[CandidateResult] = None
        round_best_any: Optional[CandidateResult] = None
        rng = random.Random(10007 + round_no * 7919)

        for mv in variants:
            y = (train_df["t1_hold_ret"] > float(mv.target_gt)).astype(np.int8).to_numpy(np.int8)
            w1, b1, w2, b2 = relay_model.train_binary_mlp(
                x=x_train,
                y=y,
                hidden_size=int(mv.hidden),
                epochs=int(mv.epochs),
                learning_rate=float(mv.lr),
                weight_decay=float(mv.weight_decay),
                seed=int(mv.seed) + int(seed_offset),
            )
            train_pred, _ = relay_model.predict_binary_mlp(x_train, w1, b1, w2, b2)
            train_acc = float((train_pred == y).mean()) if len(y) else 0.0
            _, base_prob = relay_model.predict_binary_mlp(x_base, w1, b1, w2, b2)

            for alpha in alphas:
                scored = pool_base.copy()
                scored["raw_prob"] = base_prob
                scored["score"] = backtest_model._score_from_raw_prob(base_prob, alpha=alpha)
                ranked_by_date = backtest_model._build_ranked_by_date(scored)

                sampled_cfgs: List[Tuple[float, str, int, Optional[float], Optional[float], Optional[int], RiskProfile, float]] = []
                for _ in range(samples_per_key):
                    sampled_cfgs.append(
                        (
                            float(rng.choice(thresholds)),
                            str(rng.choice(sell_rules)),
                            int(rng.choice(top_ks)),
                            rng.choice(gap_windows)[0],
                            rng.choice(gap_windows)[1],
                            rng.choice(board_caps),
                            rng.choice(risk_profiles),
                            float(rng.choice(alloc_pcts)),
                        )
                    )

                if (
                    best_global
                    and best_global.variant == mv.name
                    and int(best_global.alpha) == int(alpha)
                ):
                    sampled_cfgs.append(
                        (
                            float(best_global.threshold),
                            str(best_global.sell_rule),
                            int(best_global.top_k),
                            best_global.gap_min,
                            best_global.gap_max,
                            best_global.max_board,
                            RiskProfile(
                                best_global.risk_profile,
                                best_global.max_broken_rate,
                                best_global.min_red_rate,
                                best_global.max_limit_down,
                                best_global.max_pullback,
                            ),
                            float(best_global.alloc_pct),
                        )
                    )

                for th, sell_rule, top_k, gap_min, gap_max, max_board, rp, alloc_pct in sampled_cfgs:
                    m = backtest_model._simulate_fast(
                        trade_dates=trade_dates,
                        ranked_by_date=ranked_by_date,
                        threshold=float(th),
                        initial_capital=initial_capital,
                        sell_rule=sell_rule,
                        top_k=int(top_k),
                        max_board=max_board,
                        gap_min=gap_min,
                        gap_max=gap_max,
                        max_broken_rate=rp.max_broken_rate,
                        min_red_rate=rp.min_red_rate,
                        max_limit_down=rp.max_limit_down,
                        max_pullback=rp.max_pullback,
                        alloc_pct=float(alloc_pct),
                        rules=rules,
                    )
                    cand = CandidateResult(
                        round_no=int(round_no),
                        seed_offset=int(seed_offset),
                        variant=str(mv.name),
                        alpha=int(alpha),
                        threshold=float(th),
                        sell_rule=str(sell_rule),
                        top_k=int(top_k),
                        gap_min=gap_min,
                        gap_max=gap_max,
                        max_board=max_board,
                        risk_profile=str(rp.name),
                        max_broken_rate=rp.max_broken_rate,
                        min_red_rate=rp.min_red_rate,
                        max_limit_down=rp.max_limit_down,
                        max_pullback=rp.max_pullback,
                        alloc_pct=float(alloc_pct),
                        train_acc=float(train_acc),
                        ret=float(m["ret"]),
                        max_dd=float(m["max_dd"]),
                        trades=int(m["trades"]),
                        final_capital=float(m["final"]),
                        fee=float(m["fee"]),
                    )
                    if _is_better(round_best_any, cand, max_dd_limit=max_dd_limit):
                        round_best_any = cand
                    if cand.max_dd <= max_dd_limit and _is_better(round_best, cand, max_dd_limit=max_dd_limit):
                        round_best = cand

        if round_best is None:
            row = {
                "round": round_no,
                "seed_offset": seed_offset,
                "feasible": False,
                "best_ret_pct": None,
                "best_dd_pct": None,
                "best_cfg": None,
                "fallback_ret_pct": float(round_best_any.ret * 100.0) if round_best_any else None,
                "fallback_dd_pct": float(round_best_any.max_dd * 100.0) if round_best_any else None,
                "fallback_cfg": _cfg_text(round_best_any) if round_best_any else None,
            }
            history_rows.append(row)
            no_improve += 1
            print(
                f"[round {round_no}] no feasible candidate under dd<={max_dd_limit*100:.2f}%, "
                f"fallback dd={round_best_any.max_dd*100:.2f}% ret={round_best_any.ret*100:.2f}%"
                if round_best_any
                else f"[round {round_no}] no candidate"
            )
        else:
            improved = (best_global is None) or (round_best.ret > best_global.ret + 1e-12)
            if improved:
                best_global = round_best
                no_improve = 0
            else:
                no_improve += 1
            row = {
                "round": round_no,
                "seed_offset": seed_offset,
                "feasible": True,
                "best_ret_pct": float(round_best.ret * 100.0),
                "best_dd_pct": float(round_best.max_dd * 100.0),
                "best_cfg": _cfg_text(round_best),
                "improved": bool(improved),
            }
            history_rows.append(row)
            print(
                f"[round {round_no}] best ret={round_best.ret*100:.2f}% dd={round_best.max_dd*100:.2f}% "
                f"trades={round_best.trades} {'(improved)' if improved else '(no improve)'}"
            )

        if no_improve >= patience:
            print(f"[stop] no return improvement for {no_improve} consecutive rounds.")
            break

    if best_global is None:
        raise SystemExit("no feasible result under drawdown constraint")

    best_seed_offset = int(best_global.seed_offset)
    best_variant = next(v for v in variants if v.name == best_global.variant)
    y = (train_df["t1_hold_ret"] > float(best_variant.target_gt)).astype(np.int8).to_numpy(np.int8)
    w1, b1, w2, b2 = relay_model.train_binary_mlp(
        x=x_train,
        y=y,
        hidden_size=int(best_variant.hidden),
        epochs=int(best_variant.epochs),
        learning_rate=float(best_variant.lr),
        weight_decay=float(best_variant.weight_decay),
        seed=int(best_variant.seed) + int(best_seed_offset),
    )
    _, best_base_prob = relay_model.predict_binary_mlp(x_base, w1, b1, w2, b2)
    scored = pool_base.copy()
    scored["raw_prob"] = best_base_prob
    scored["score"] = backtest_model._score_from_raw_prob(best_base_prob, alpha=int(best_global.alpha))
    ranked = backtest_model._build_ranked_by_date(scored)

    detail = backtest_model._simulate_detailed(
        trade_dates=trade_dates,
        ranked_by_date=ranked,
        threshold=float(best_global.threshold),
        initial_capital=initial_capital,
        sell_rule=str(best_global.sell_rule),
        variant_name=str(best_global.variant),
        alpha=int(best_global.alpha),
        top_k=int(best_global.top_k),
        max_board=best_global.max_board,
        gap_min=best_global.gap_min,
        gap_max=best_global.gap_max,
        max_broken_rate=best_global.max_broken_rate,
        min_red_rate=best_global.min_red_rate,
        max_limit_down=best_global.max_limit_down,
        max_pullback=best_global.max_pullback,
        risk_profile=str(best_global.risk_profile),
        alloc_pct=float(best_global.alloc_pct),
        rules=rules,
    )

    trades_csv = report_dir / "best_selected_trades.csv"
    pd.DataFrame(detail["trades"]).to_csv(trades_csv, index=False, encoding="utf-8-sig")
    operations_csv = report_dir / "best_selected_operations.csv"
    _write_operations_csv(operations_csv, detail["trades"], initial_capital=initial_capital)

    history_csv = report_dir / "history.csv"
    pd.DataFrame(history_rows).to_csv(history_csv, index=False, encoding="utf-8-sig")

    summary = {
        "train_start_date": train_start,
        "backtest_start_date": start_date,
        "end_date": end_date,
        "max_dd_limit": max_dd_limit,
        "initial_capital": initial_capital,
        "rules_file": args.rules_file,
        "patience": patience,
        "max_rounds": max_rounds,
        "history_rows": len(history_rows),
        "best": asdict(best_global),
        "best_cfg_text": _cfg_text(best_global),
        "best_detail": {
            "final_capital": float(detail["final_capital"]),
            "ret_pct": float(detail["total_ret"] * 100.0),
            "max_dd_pct": float(detail["max_dd"] * 100.0),
            "trade_n": int(detail["trade_n"]),
            "total_fee": float(detail["total_fee"]),
        },
        "outputs": {
            "history_csv": history_csv.as_posix(),
            "trades_csv": trades_csv.as_posix(),
            "operations_csv": operations_csv.as_posix(),
        },
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[best] ret={detail['total_ret']*100:.2f}% dd={detail['max_dd']*100:.2f}% "
        f"trades={detail['trade_n']} cfg={_cfg_text(best_global)}"
    )
    print(f"[out] {summary_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
