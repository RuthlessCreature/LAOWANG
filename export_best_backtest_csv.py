# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export best-model backtest operations to CSV.")
    p.add_argument("--best-json", default="reports/iter_rounds/best/best_model.json")
    p.add_argument("--report", default=None, help="Optional explicit mockReport markdown path.")
    p.add_argument("--output", default="best_model_backtest_records.csv")
    return p.parse_args()


def threshold_tag(v: float) -> str:
    return f"{float(v):.2f}".rstrip("0").rstrip(".")


def choose_best_threshold(best_payload: Dict[str, object]) -> float:
    metrics = best_payload.get("metrics_by_threshold", {})
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("best_model.json missing metrics_by_threshold.")
    parsed: List[Tuple[float, float]] = []
    for k, v in metrics.items():
        th = float(k)
        ret = float((v or {}).get("ret_pct", float("-inf")))
        parsed.append((th, ret))
    parsed.sort(key=lambda x: (-x[1], x[0]))
    return parsed[0][0]


def to_float(text: str) -> float:
    return float(str(text).replace(",", "").strip())


def parse_report(path: Path) -> Tuple[float, List[Dict[str, object]]]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()

    init_cap = None
    for line in lines:
        if line.startswith("- Initial capital:"):
            init_cap = to_float(line.split(":", 1)[1])
            break
    if init_cap is None:
        raise ValueError(f"Cannot find initial capital in {path.as_posix()}")

    trades: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None
    mode: Optional[str] = None

    def flush_trade() -> None:
        nonlocal current
        if current is None:
            return
        required = [
            "code",
            "name",
            "buy_date",
            "buy_price",
            "buy_fee",
            "sell_date",
            "sell_price",
            "equity_after",
        ]
        if all(k in current for k in required):
            trades.append(current)
        current = None

    for raw in lines:
        line = raw.rstrip()
        m = re.match(r"^###\s+\d+\.\s+(\S+)\s*(.*)$", line)
        if m:
            flush_trade()
            current = {"code": m.group(1).strip(), "name": (m.group(2) or "").strip()}
            mode = None
            continue

        if current is None:
            continue

        stripped = line.strip()
        if stripped == "- Buy:":
            mode = "buy"
            continue
        if stripped == "- Sell:":
            mode = "sell"
            continue
        if stripped == "- Result:":
            mode = "result"
            continue

        if not stripped.startswith("- "):
            continue

        if mode == "buy":
            if stripped.startswith("- Time:"):
                current["buy_date"] = stripped.split(":", 1)[1].strip().split()[0]
            elif stripped.startswith("- Price:"):
                current["buy_price"] = to_float(stripped.split(":", 1)[1])
            elif stripped.startswith("- Fee:"):
                current["buy_fee"] = to_float(stripped.split(":", 1)[1])
        elif mode == "sell":
            if stripped.startswith("- Time:"):
                current["sell_date"] = stripped.split(":", 1)[1].strip().split()[0]
            elif stripped.startswith("- Price:"):
                current["sell_price"] = to_float(stripped.split(":", 1)[1])
        elif mode == "result":
            if stripped.startswith("- Equity after trade:"):
                current["equity_after"] = to_float(stripped.split(":", 1)[1])

    flush_trade()
    if not trades:
        raise ValueError(f"No trade records parsed from {path.as_posix()}")
    return init_cap, trades


def export_csv(path: Path, init_cap: float, trades: List[Dict[str, object]]) -> None:
    headers = [
        "\u65e5\u671f",
        "\u80a1\u7968",
        "\u4ee3\u7801",
        "\u64cd\u4f5c",
        "\u64cd\u4f5c\u4ef7\u683c",
        "\u8d44\u4ea7\u603b\u503c",
    ]
    rows: List[List[str]] = []
    equity = float(init_cap)
    for t in trades:
        buy_asset = equity - float(t["buy_fee"])
        rows.append(
            [
                str(t["buy_date"]),
                str(t["name"]),
                str(t["code"]),
                "buy",
                f"{float(t['buy_price']):.3f}",
                f"{buy_asset:.2f}",
            ]
        )
        equity = float(t["equity_after"])
        rows.append(
            [
                str(t["sell_date"]),
                str(t["name"]),
                str(t["code"]),
                "sell",
                f"{float(t['sell_price']):.3f}",
                f"{equity:.2f}",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    best_json = Path(args.best_json)
    if not best_json.exists():
        raise SystemExit(f"best json not found: {best_json.as_posix()}")
    payload = json.loads(best_json.read_text(encoding="utf-8"))

    if args.report:
        report_path = Path(args.report)
    else:
        th = choose_best_threshold(payload)
        tag = threshold_tag(th)
        report_path = best_json.parent / f"mockReport_{tag}.md"
    if not report_path.exists():
        raise SystemExit(f"report file not found: {report_path.as_posix()}")

    init_cap, trades = parse_report(report_path)
    out = Path(args.output)
    export_csv(out, init_cap, trades)

    # Also keep a copy next to best model artifacts.
    best_copy = best_json.parent / out.name
    if best_copy.resolve() != out.resolve():
        export_csv(best_copy, init_cap, trades)

    print(f"source_report={report_path.as_posix()}")
    print(f"rows={len(trades) * 2}")
    print(f"csv={out.as_posix()}")
    if best_copy.resolve() != out.resolve():
        print(f"csv_best_copy={best_copy.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
