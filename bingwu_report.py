# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from a_stock_analyzer import db
from a_stock_analyzer.runtime import (
    add_db_args,
    normalize_trade_date,
    resolve_db_from_args,
    setup_logging,
    yyyymmdd_from_date,
)

from bingwu import analyze_bingwu  # noqa: E402


CSV_COLUMNS = [
    "trade_date",
    "market_emotion",
    "emotion_score",
    "operation_permission",
    "limit_up_count",
    "limit_down_count",
    "max_consecutive",
    "prev_max_consecutive",
    "broken_count",
    "sealed_count",
    "broken_rate",
    "max_position",
    "risk_level",
    "top_concepts",
    "top_industries",
    "symbol",
    "stock_name",
    "role",
    "lb_count",
    "structure_stage",
    "structure_score",
    "primary_concept",
    "primary_industry",
    "tradable",
    "stock_score",
    "operation_type",
    "entry_conditions",
    "stop_loss_rules",
    "take_profit_logic",
    "sell_plan",
]


def _join_list(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ; ".join([str(x) for x in v if str(x).strip()])
    return str(v)


def _themes_to_str(items: Any, *, limit: int = 5) -> str:
    if not isinstance(items, list):
        return ""
    items = [
        it
        for it in items
        if isinstance(it, dict) and int(it.get("theme_score", 0) or 0) > 0
    ]
    parts: List[str] = []
    for it in items[: max(0, int(limit))]:
        name = str(it.get("theme_name") or "").strip()
        if not name:
            continue
        is_core = bool(it.get("is_core_theme"))
        score = it.get("theme_score")
        n = it.get("limit_up_count")
        mx = it.get("max_consecutive")
        parts.append(f"{name}{'*' if is_core else ''}({score}/{n}/{mx})")
    return " ; ".join(parts)


def bingwu_result_to_csv_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    trade_date = str(result.get("trade_date") or "").strip()
    market = result.get("market") or {}
    risk = result.get("risk") or {}
    themes = result.get("themes") or {}

    top_concepts = _themes_to_str(themes.get("concept"))
    top_industries = _themes_to_str(themes.get("industry"))

    plans = result.get("plans") or []
    plan_by_symbol: Dict[str, Dict[str, Any]] = {}
    for p in plans:
        if isinstance(p, dict) and p.get("symbol"):
            plan_by_symbol[str(p["symbol"])] = p

    candidates = result.get("candidates") or []
    rows: List[Dict[str, Any]] = []
    if not candidates:
        rows.append(
            {
                "trade_date": trade_date,
                "market_emotion": market.get("market_emotion", ""),
                "emotion_score": market.get("emotion_score", ""),
                "operation_permission": market.get("operation_permission", ""),
                "limit_up_count": market.get("limit_up_count", ""),
                "limit_down_count": market.get("limit_down_count", ""),
                "max_consecutive": market.get("max_consecutive", ""),
                "prev_max_consecutive": market.get("prev_max_consecutive", ""),
                "broken_count": market.get("broken_count", ""),
                "sealed_count": market.get("sealed_count", ""),
                "broken_rate": market.get("broken_rate", ""),
                "max_position": risk.get("max_position", ""),
                "risk_level": risk.get("risk_level", ""),
                "top_concepts": top_concepts,
                "top_industries": top_industries,
            }
        )
        return rows

    for c in candidates:
        if not isinstance(c, dict):
            continue
        sym = str(c.get("symbol") or "").strip()
        p = plan_by_symbol.get(sym) or {}
        rows.append(
            {
                "trade_date": trade_date,
                "market_emotion": market.get("market_emotion", ""),
                "emotion_score": market.get("emotion_score", ""),
                "operation_permission": market.get("operation_permission", ""),
                "limit_up_count": market.get("limit_up_count", ""),
                "limit_down_count": market.get("limit_down_count", ""),
                "max_consecutive": market.get("max_consecutive", ""),
                "prev_max_consecutive": market.get("prev_max_consecutive", ""),
                "broken_count": market.get("broken_count", ""),
                "sealed_count": market.get("sealed_count", ""),
                "broken_rate": market.get("broken_rate", ""),
                "max_position": risk.get("max_position", ""),
                "risk_level": risk.get("risk_level", ""),
                "top_concepts": top_concepts,
                "top_industries": top_industries,
                "symbol": sym,
                "stock_name": str(c.get("stock_name") or ""),
                "role": str(c.get("role") or ""),
                "lb_count": c.get("lb_count", ""),
                "structure_stage": str(c.get("structure_stage") or ""),
                "structure_score": c.get("structure_score", ""),
                "primary_concept": str(c.get("primary_concept") or ""),
                "primary_industry": str(c.get("primary_industry") or ""),
                "tradable": c.get("tradable", ""),
                "stock_score": c.get("stock_score", ""),
                "operation_type": str(p.get("operation_type") or ""),
                "entry_conditions": _join_list(p.get("entry_conditions")),
                "stop_loss_rules": _join_list(p.get("stop_loss_rules")),
                "take_profit_logic": str(p.get("take_profit_logic") or ""),
                "sell_plan": str(p.get("sell_plan") or ""),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="bingwu_report.py - 导出 bingwu 每日 CSV 报告")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    add_db_args(p)
    p.add_argument("--trade-date", default=None, help="YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--output", default=None, help="CSV output path (default: outputs/bingwu_YYYYMMDD.csv)")
    p.add_argument("--dump-json", default=None, help="Optional: dump full result json to path")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_from_args(args)

    trade_date_in = args.trade_date
    if not trade_date_in:
        raise SystemExit("bingwu_report.py requires --trade-date (use everyday.bat).")
    trade_date_norm = normalize_trade_date(str(trade_date_in))
    yyyymmdd = yyyymmdd_from_date(trade_date_norm)

    out_csv = Path(args.output) if args.output else Path(f"outputs/bingwu_{yyyymmdd}.csv")

    result = analyze_bingwu(db_target=db_target, trade_date=trade_date_norm)
    rows = bingwu_result_to_csv_rows(result)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    db.ensure_parent_dir(out_csv)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info("Wrote csv: %s", out_csv)

    if args.dump_json:
        dump_path = Path(args.dump_json)
        db.ensure_parent_dir(dump_path)
        dump_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote json: %s", dump_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
