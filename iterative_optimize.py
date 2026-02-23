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
    p.add_argument("--start-date", default="2025-01-01")
    p.add_argument("--thresholds", default="0.5,0.75,0.99")
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument(
        "--variant-profile",
        choices=["base", "expanded", "aggressive"],
        default="base",
        help="Pass-through model-structure search profile for mock_backtest.",
    )
    p.add_argument("--max-rounds", type=int, default=20)
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


def run_one_round(
    *,
    round_idx: int,
    args: argparse.Namespace,
    thresholds: List[float],
    seed_offset: int,
    retain_ratio: float,
) -> Tuple[Dict[float, Dict[str, object]], str, str]:
    cmd = [
        sys.executable,
        "-u",
        "mock_backtest.py",
        "--config",
        str(args.config),
        "--rules-file",
        str(args.rules_file),
        "--start-date",
        str(args.start_date),
        "--thresholds",
        str(args.thresholds),
        "--initial-capital",
        str(float(args.initial_capital)),
        "--variant-profile",
        str(args.variant_profile),
        "--seed-offset",
        str(int(seed_offset)),
        "--retain-ratio",
        f"{float(retain_ratio):.4f}",
        "--min-return-floor",
        str(float(args.min_return_floor)),
    ]
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
) -> None:
    headers = ["Round", "SeedOffset", "RetainRatio", "AvgRet", "AvgDD", "Indicator", "Improved"]
    for th in thresholds:
        tag = threshold_tag(th)
        headers.extend([f"R@{tag}", f"DD@{tag}"])

    lines = [
        "# Iteration History",
        "",
        (
            f"- Stop rule: consecutive {int(patience)} rounds without indicator improvement "
            f"(metric={metric_mode}, dd_penalty={float(dd_penalty):.2f})"
        ),
        f"- Variant profile: {variant_profile}",
        f"- Best round: {best_round}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        cols = [
            str(row["round"]),
            str(row["seed_offset"]),
            f"{float(row['retain_ratio']):.2f}",
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
    metric_mode = str(args.metric_mode)
    dd_penalty = float(args.dd_penalty)

    best_score = float("-inf")
    best_avg_ret_pct = float("-inf")
    best_avg_dd_pct = float("inf")
    best_round = 0
    no_improve = 0

    for rnd in range(1, max(1, int(args.max_rounds)) + 1):
        retain_ratio = min(
            1.0,
            max(0.01, float(args.base_retain_ratio) + float(args.retain_step) * ((rnd - 1) % max(1, int(args.retain_cycle)))),
        )
        seed_offset = int((rnd - 1) * int(args.seed_step))
        metrics, stdout_text, stderr_text = run_one_round(
            round_idx=rnd,
            args=args,
            thresholds=thresholds,
            seed_offset=seed_offset,
            retain_ratio=retain_ratio,
        )

        round_dir = rounds_dir / f"round_{rnd:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "stdout.log").write_text(stdout_text or "", encoding="utf-8")
        (round_dir / "stderr.log").write_text(stderr_text or "", encoding="utf-8")

        for th in thresholds:
            tag = threshold_tag(th)
            src = Path(f"mockReport_{tag}.md")
            if src.exists():
                shutil.copy2(src, round_dir / src.name)

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
                src = round_dir / f"mockReport_{tag}.md"
                if src.exists():
                    shutil.copy2(src, best_dir / src.name)
            best_payload = {
                "best_round": best_round,
                "best_indicator": best_score,
                "best_avg_ret_pct": best_avg_ret_pct,
                "best_avg_dd_pct": best_avg_dd_pct,
                "metric_mode": metric_mode,
                "dd_penalty": dd_penalty,
                "seed_offset": seed_offset,
                "retain_ratio": retain_ratio,
                "thresholds": thresholds,
                "metrics_by_threshold": metrics,
                "rules_file": args.rules_file,
                "start_date": args.start_date,
                "variant_profile": args.variant_profile,
            }
            (best_dir / "best_model.json").write_text(
                json.dumps(best_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            no_improve += 1

        row = {
            "round": rnd,
            "seed_offset": seed_offset,
            "retain_ratio": retain_ratio,
            "avg_ret_pct": avg_ret_pct,
            "avg_dd_pct": avg_dd_pct,
            "indicator": indicator,
            "improved": improved,
            "metrics_by_threshold": metrics,
        }
        history_rows.append(row)
        (round_dir / "summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"[iter] round={rnd} seed={seed_offset} retain={retain_ratio:.2f} "
            f"avg_ret={avg_ret_pct:.2f}% avg_dd={avg_dd_pct:.2f}% "
            f"indicator={indicator:.6f} "
            f"{'IMPROVED' if improved else f'no_improve={no_improve}'}",
            flush=True,
        )
        if no_improve >= int(args.patience):
            print(f"[iter] stop: reached patience={int(args.patience)} at round={rnd}", flush=True)
            break

    if best_round <= 0:
        raise SystemExit("No valid round produced.")

    for th in thresholds:
        tag = threshold_tag(th)
        src = best_dir / f"mockReport_{tag}.md"
        if src.exists():
            shutil.copy2(src, Path(f"mockReport_{tag}.md"))

    write_history_md(
        report_dir / "iteration_history.md",
        history_rows,
        thresholds,
        best_round,
        patience=int(args.patience),
        metric_mode=metric_mode,
        dd_penalty=dd_penalty,
        variant_profile=str(args.variant_profile),
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
