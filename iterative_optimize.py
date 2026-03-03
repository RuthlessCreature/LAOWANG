# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


LINE_RE = re.compile(
    r"\[mock-backtest\]\s+th=(?P<th>\d+\.\d+)\s+final=(?P<final>[-\d.]+)\s+"
    r"ret=(?P<ret>[-\d.]+)%\s+dd=(?P<dd>[-\d.]+)%\s+fee=(?P<fee>[-\d.]+)\s+"
    r"trades=(?P<trades>\d+)\s+cfg=(?P<cfg>.+)$",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Iterate mock_backtest until return plateaus.")
    p.add_argument("--config", default="config.ini")
    p.add_argument("--db-url", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--rules-file", default="rules.txt")
    p.add_argument("--start-date", default=None, help="Backtest start date (default follows rules file).")
    p.add_argument("--end-date", default=None, help="Backtest end date (default follows rules file).")
    p.add_argument("--train-start-date", default=None, help="Train start date (default follows rules file).")
    p.add_argument("--train-end-date", default=None, help="Train end date (default follows rules file).")
    p.add_argument("--thresholds", default="0.5,0.75,0.99")
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument(
        "--variant-profile",
        choices=["base", "expanded", "aggressive"],
        default="base",
        help="Pass-through model-structure search profile for mock_backtest.",
    )
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--infinite", action="store_true", help="Do not stop on max-rounds/patience; keep adaptive iterations.")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--metric-mode",
        choices=["ret_over_dd", "ret_minus_dd"],
        default="ret_over_dd",
        help="optimization indicator: return/drawdown ratio or return-penalty(drawdown)",
    )
    p.add_argument(
        "--dd-penalty",
        type=float,
        default=1.0,
        help="drawdown penalty weight when metric-mode=ret_minus_dd",
    )
    p.add_argument("--base-retain-ratio", type=float, default=0.80)
    p.add_argument("--retain-step", type=float, default=0.05)
    p.add_argument("--retain-cycle", type=int, default=5)
    p.add_argument("--min-return-floor", type=float, default=4.0)
    p.add_argument("--seed-step", type=int, default=97)
    p.add_argument("--adaptive-retain-drop", type=float, default=0.03, help="Plateau trigger: reduce base retain ratio by this step.")
    p.add_argument("--adaptive-floor-decay", type=float, default=0.90, help="Plateau trigger: multiply min-return-floor by this factor.")
    p.add_argument("--adaptive-penalty-step", type=float, default=0.25, help="Plateau trigger: increase dd-penalty by this step.")
    p.add_argument("--report-dir", default="reports/iter_rounds")
    return p.parse_args()


def threshold_tag(v: float) -> str:
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def parse_thresholds(text: str) -> List[float]:
    out = [float(x.strip()) for x in str(text).split(",") if str(x).strip()]
    out = [max(0.50, min(0.99, x)) for x in out]
    return sorted(set(out))


def parse_metrics(run_text: str) -> Dict[float, Dict[str, object]]:
    out: Dict[float, Dict[str, object]] = {}
    for m in LINE_RE.finditer(run_text):
        th = round(float(m.group("th")), 2)
        out[th] = {
            "final": float(m.group("final")),
            "ret_pct": float(m.group("ret")),
            "dd_pct": float(m.group("dd")),
            "fee": float(m.group("fee")),
            "trades": int(m.group("trades")),
            "cfg": m.group("cfg").strip(),
        }
    return out


def calc_indicator(
    *,
    avg_ret_pct: float,
    avg_dd_pct: float,
    metric_mode: str,
    dd_penalty: float,
) -> float:
    ret_v = float(avg_ret_pct)
    dd_v = max(0.0, float(avg_dd_pct))
    if str(metric_mode) == "ret_minus_dd":
        return ret_v - float(dd_penalty) * dd_v
    # Use 1.0 (one percentage point) as floor to avoid unstable blow-up when dd is near zero.
    denom = max(dd_v, 1.0)
    return ret_v / denom


PROFILE_ORDER = ["base", "expanded", "aggressive"]


def _next_variant_profile(current: str) -> str:
    key = str(current or "base").strip().lower()
    if key not in PROFILE_ORDER:
        return "aggressive"
    idx = PROFILE_ORDER.index(key)
    if idx >= len(PROFILE_ORDER) - 1:
        return PROFILE_ORDER[-1]
    return PROFILE_ORDER[idx + 1]


def run_one_round(
    *,
    round_idx: int,
    args: argparse.Namespace,
    thresholds: List[float],
    seed_offset: int,
    retain_ratio: float,
    variant_profile: str,
    min_return_floor: float,
) -> Tuple[Dict[float, Dict[str, object]], str, str]:
    cmd = [
        sys.executable,
        "-u",
        "mock_backtest.py",
        "--config",
        str(args.config),
        "--rules-file",
        str(args.rules_file),
        "--thresholds",
        str(args.thresholds),
        "--initial-capital",
        str(float(args.initial_capital)),
        "--variant-profile",
        str(variant_profile),
        "--seed-offset",
        str(int(seed_offset)),
        "--retain-ratio",
        f"{float(retain_ratio):.4f}",
        "--min-return-floor",
        str(float(min_return_floor)),
    ]
    if args.start_date:
        cmd.extend(["--start-date", str(args.start_date)])
    if args.end_date:
        cmd.extend(["--end-date", str(args.end_date)])
    if args.train_start_date:
        cmd.extend(["--train-start-date", str(args.train_start_date)])
    if args.train_end_date:
        cmd.extend(["--train-end-date", str(args.train_end_date)])
    if args.db_url:
        cmd.extend(["--db-url", str(args.db_url)])
    if args.db:
        cmd.extend(["--db", str(args.db)])

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Round {round_idx} failed.\nCommand: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    metrics = parse_metrics(merged)
    missing = [th for th in thresholds if round(float(th), 2) not in metrics]
    if missing:
        raise RuntimeError(
            f"Round {round_idx} missing thresholds {missing}; parsed={sorted(metrics.keys())}\nOutput:\n{merged}"
        )
    return metrics, proc.stdout, proc.stderr


def write_history_md(
    path: Path,
    rows: List[Dict[str, object]],
    thresholds: List[float],
    best_round: int,
    *,
    patience: int,
    metric_mode: str,
    dd_penalty: float,
    variant_profile: str,
    infinite: bool,
) -> None:
    headers = [
        "Round",
        "AdaptCycle",
        "SeedOffset",
        "Variant",
        "RetainRatio",
        "MinFloor",
        "Metric",
        "DDPenalty",
        "AvgRet",
        "AvgDD",
        "Indicator",
        "Improved",
    ]
    for th in thresholds:
        tag = threshold_tag(th)
        headers.extend([f"R@{tag}", f"DD@{tag}"])

    lines = [
        "# Iteration History",
        "",
        (
            f"- Stop rule: consecutive {int(patience)} rounds without indicator improvement "
            f"(initial metric={metric_mode}, initial dd_penalty={float(dd_penalty):.2f})"
        ),
        f"- Variant profile (initial): {variant_profile}",
        f"- Infinite adaptive mode: {'on' if infinite else 'off'}",
        f"- Best round: {best_round}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        cols = [
            str(row["round"]),
            str(row.get("adapt_cycle", 0)),
            str(row["seed_offset"]),
            str(row.get("variant_profile", "N/A")),
            f"{float(row['retain_ratio']):.2f}",
            f"{float(row.get('min_return_floor', 0.0)):.2f}",
            str(row.get("metric_mode", "")),
            f"{float(row.get('dd_penalty', 0.0)):.2f}",
            f"{float(row['avg_ret_pct']):.2f}%",
            f"{float(row['avg_dd_pct']):.2f}%",
            f"{float(row['indicator']):.6f}",
            "yes" if row["improved"] else "no",
        ]
        by_th = row["metrics_by_threshold"]
        for th in thresholds:
            m = by_th[round(float(th), 2)]
            cols.append(f"{float(m['ret_pct']):.2f}%")
            cols.append(f"{float(m['dd_pct']):.2f}%")
        lines.append("| " + " | ".join(cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    if not thresholds:
        raise SystemExit("No thresholds provided.")

    report_dir = Path(args.report_dir)
    rounds_dir = report_dir / "rounds"
    best_dir = report_dir / "best"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    history_rows: List[Dict[str, object]] = []

    initial_metric_mode = str(args.metric_mode)
    initial_dd_penalty = float(args.dd_penalty)
    metric_mode = initial_metric_mode
    dd_penalty = initial_dd_penalty

    variant_profile = str(args.variant_profile)
    min_return_floor = float(args.min_return_floor)
    base_retain_ratio = float(args.base_retain_ratio)
    retain_step = float(args.retain_step)
    retain_cycle = max(1, int(args.retain_cycle))

    seed_step = int(args.seed_step)
    max_rounds = max(1, int(args.max_rounds))
    patience = max(1, int(args.patience))
    infinite = bool(args.infinite)

    best_score = float("-inf")
    best_avg_ret_pct = float("-inf")
    best_avg_dd_pct = float("inf")
    best_round = 0
    no_improve = 0
    adapt_cycle = 0
    rnd = 1

    while True:
        if (not infinite) and rnd > max_rounds:
            print(f"[iter] stop: reached max-rounds={max_rounds}", flush=True)
            break

        retain_ratio = min(
            1.0,
            max(0.01, float(base_retain_ratio) + float(retain_step) * ((rnd - 1) % retain_cycle)),
        )
        seed_offset = int((rnd - 1) * seed_step)
        metrics, stdout_text, stderr_text = run_one_round(
            round_idx=rnd,
            args=args,
            thresholds=thresholds,
            seed_offset=seed_offset,
            retain_ratio=retain_ratio,
            variant_profile=variant_profile,
            min_return_floor=min_return_floor,
        )

        round_dir = rounds_dir / f"round_{rnd:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "stdout.log").write_text(stdout_text or "", encoding="utf-8")
        (round_dir / "stderr.log").write_text(stderr_text or "", encoding="utf-8")

        for th in thresholds:
            tag = threshold_tag(th)
            report_src = Path(f"mockReport_{tag}.md")
            if report_src.exists():
                shutil.copy2(report_src, round_dir / report_src.name)
            ops_src = Path(f"mockOperations_{tag}.csv")
            if ops_src.exists():
                shutil.copy2(ops_src, round_dir / ops_src.name)

        ret_values = [float(metrics[round(float(th), 2)]["ret_pct"]) for th in thresholds]
        dd_values = [float(metrics[round(float(th), 2)]["dd_pct"]) for th in thresholds]
        avg_ret_pct = float(mean(ret_values))
        avg_dd_pct = float(mean(dd_values))
        indicator = calc_indicator(
            avg_ret_pct=avg_ret_pct,
            avg_dd_pct=avg_dd_pct,
            metric_mode=metric_mode,
            dd_penalty=dd_penalty,
        )

        improved = indicator > best_score + 1e-9
        if improved:
            best_score = indicator
            best_avg_ret_pct = avg_ret_pct
            best_avg_dd_pct = avg_dd_pct
            best_round = rnd
            no_improve = 0
            for th in thresholds:
                tag = threshold_tag(th)
                report_src = round_dir / f"mockReport_{tag}.md"
                if report_src.exists():
                    shutil.copy2(report_src, best_dir / report_src.name)
                ops_src = round_dir / f"mockOperations_{tag}.csv"
                if ops_src.exists():
                    shutil.copy2(ops_src, best_dir / ops_src.name)
            best_payload = {
                "best_round": best_round,
                "best_indicator": best_score,
                "best_avg_ret_pct": best_avg_ret_pct,
                "best_avg_dd_pct": best_avg_dd_pct,
                "metric_mode": metric_mode,
                "dd_penalty": dd_penalty,
                "seed_offset": seed_offset,
                "retain_ratio": retain_ratio,
                "min_return_floor": min_return_floor,
                "adapt_cycle": adapt_cycle,
                "thresholds": thresholds,
                "metrics_by_threshold": metrics,
                "rules_file": args.rules_file,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "train_start_date": args.train_start_date,
                "train_end_date": args.train_end_date,
                "variant_profile": variant_profile,
                "infinite_mode": infinite,
            }
            (best_dir / "best_model.json").write_text(
                json.dumps(best_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            no_improve += 1

        row = {
            "round": rnd,
            "adapt_cycle": adapt_cycle,
            "seed_offset": seed_offset,
            "variant_profile": variant_profile,
            "retain_ratio": retain_ratio,
            "min_return_floor": min_return_floor,
            "metric_mode": metric_mode,
            "dd_penalty": dd_penalty,
            "avg_ret_pct": avg_ret_pct,
            "avg_dd_pct": avg_dd_pct,
            "indicator": indicator,
            "improved": improved,
            "metrics_by_threshold": metrics,
        }
        history_rows.append(row)
        (round_dir / "summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"[iter] round={rnd} cycle={adapt_cycle} seed={seed_offset} "
            f"variant={variant_profile} retain={retain_ratio:.2f} floor={min_return_floor:.2f} "
            f"metric={metric_mode}(dd_penalty={dd_penalty:.2f}) "
            f"avg_ret={avg_ret_pct:.2f}% avg_dd={avg_dd_pct:.2f}% indicator={indicator:.6f} "
            f"{'IMPROVED' if improved else f'no_improve={no_improve}'}",
            flush=True,
        )

        if no_improve >= patience:
            if not infinite:
                print(f"[iter] stop: reached patience={patience} at round={rnd}", flush=True)
                break

            adapt_cycle += 1
            no_improve = 0
            old_variant = variant_profile
            variant_profile = _next_variant_profile(variant_profile)
            base_retain_ratio = max(0.50, min(1.0, base_retain_ratio - float(args.adaptive_retain_drop)))
            floor_decay = max(0.10, min(0.99, float(args.adaptive_floor_decay)))
            min_return_floor = max(0.0, min_return_floor * floor_decay)
            dd_penalty = min(20.0, dd_penalty + float(args.adaptive_penalty_step))
            metric_mode = "ret_minus_dd" if metric_mode == "ret_over_dd" else "ret_over_dd"
            print(
                "[iter][adaptive] plateau detected -> "
                f"cycle={adapt_cycle} variant={old_variant}->{variant_profile} "
                f"base_retain={base_retain_ratio:.2f} floor={min_return_floor:.2f} "
                f"metric={metric_mode} dd_penalty={dd_penalty:.2f}",
                flush=True,
            )
        rnd += 1

    if best_round <= 0:
        raise SystemExit("No valid round produced.")

    for th in thresholds:
        tag = threshold_tag(th)
        src = best_dir / f"mockReport_{tag}.md"
        if src.exists():
            shutil.copy2(src, Path(f"mockReport_{tag}.md"))
        ops_src = best_dir / f"mockOperations_{tag}.csv"
        if ops_src.exists():
            shutil.copy2(ops_src, Path(f"mockOperations_{tag}.csv"))

    write_history_md(
        report_dir / "iteration_history.md",
        history_rows,
        thresholds,
        best_round,
        patience=patience,
        metric_mode=initial_metric_mode,
        dd_penalty=initial_dd_penalty,
        variant_profile=str(args.variant_profile),
        infinite=infinite,
    )
    print(
        f"[iter] best_round={best_round} best_indicator={best_score:.6f} "
        f"best_avg_ret={best_avg_ret_pct:.2f}% best_avg_dd={best_avg_dd_pct:.2f}% "
        f"history={str((report_dir / 'iteration_history.md').as_posix())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
