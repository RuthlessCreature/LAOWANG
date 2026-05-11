#!/usr/bin/env python3
"""Record manual executions and export realized trade journal reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from generate_trade_plan import load_config, make_engine, resolve_db_url, sql_text
from strategy_journal import (
    TRADE_JOURNAL_COLUMNS,
    ensure_strategy_trade_journal_schema,
    upsert_strategy_trade_journal,
)


DAILY_PNL_COLUMNS = [
    "date",
    "closed_trades",
    "wins",
    "losses",
    "win_rate",
    "net_pnl",
    "avg_pnl_pct",
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_date(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")


def optional_float(value: Any, default: float | None = None) -> float | None:
    if value is None or str(value).strip() == "":
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def make_trade_id(signal_date: str, model: str, stock_code: str, buy_date: str, lot_id: str = "main") -> str:
    raw = "|".join(
        [
            normalize_date(signal_date) or "",
            str(model).strip().lower(),
            str(stock_code).strip(),
            normalize_date(buy_date) or "",
            str(lot_id or "main").strip(),
        ]
    )
    return "manual_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def parse_json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(str(value))
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        return {"legacy_note": str(value)}


def merge_manual_notes(existing: Any, note: str | None, extra: Mapping[str, Any]) -> dict[str, Any]:
    notes = parse_json_dict(existing)
    notes.setdefault("events", [])
    if note:
        notes["events"].append({"ts": now_text(), "text": note})
    clean_extra = {key: value for key, value in extra.items() if value is not None}
    if clean_extra:
        notes.setdefault("execution", {}).update(clean_extra)
    return notes


def load_signal_context(conn, signal_date: str | None, model: str, stock_code: str) -> dict[str, Any]:
    if signal_date:
        query = """
            SELECT *
            FROM strategy_signal_log
            WHERE signal_date = :signal_date AND model = :model AND stock_code = :stock_code
            LIMIT 1
        """
        params = {"signal_date": signal_date, "model": model, "stock_code": stock_code}
    else:
        query = """
            SELECT *
            FROM strategy_signal_log
            WHERE model = :model AND stock_code = :stock_code
            ORDER BY signal_date DESC
            LIMIT 1
        """
        params = {"model": model, "stock_code": stock_code}

    row = conn.execute(sql_text(query), params).mappings().first()
    if not row:
        return {}
    result = dict(row)
    result["features"] = parse_json_dict(result.get("features_json"))
    result["action_plan"] = parse_json_dict(result.get("action_plan_json"))
    return result


def load_trade(conn, trade_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        sql_text("SELECT * FROM strategy_trade_journal WHERE trade_id = :trade_id"),
        {"trade_id": trade_id},
    ).mappings().first()
    return dict(row) if row else None


def find_open_trade(conn, args: argparse.Namespace) -> dict[str, Any]:
    if args.trade_id:
        row = load_trade(conn, args.trade_id)
        if not row:
            raise RuntimeError(f"trade_id not found: {args.trade_id}")
        return row

    if args.signal_date and args.model and args.stock_code and args.buy_date:
        trade_id = make_trade_id(args.signal_date, args.model, args.stock_code, args.buy_date, args.lot_id)
        row = load_trade(conn, trade_id)
        if row:
            return row

    if not (args.model and args.stock_code):
        raise RuntimeError("sell needs --trade-id, or --model and --stock-code")

    rows = conn.execute(
        sql_text(
            """
            SELECT *
            FROM strategy_trade_journal
            WHERE model = :model
              AND stock_code = :stock_code
              AND COALESCE(trade_status, 'OPEN') != 'CLOSED'
            ORDER BY buy_date DESC, updated_at DESC
            """
        ),
        {"model": args.model, "stock_code": args.stock_code},
    ).mappings().all()
    if not rows:
        raise RuntimeError("no open trade found; pass --trade-id or full signal/buy identifiers")
    if len(rows) > 1:
        raise RuntimeError("multiple open trades found; pass --trade-id to disambiguate")
    return dict(rows[0])


def build_buy_row(engine, args: argparse.Namespace) -> dict[str, Any]:
    signal_date = normalize_date(args.signal_date)
    model = str(args.model).strip().lower()
    stock_code = str(args.stock_code).strip()
    buy_date = normalize_date(args.buy_date)
    if not buy_date:
        raise RuntimeError("--buy-date is required")

    with engine.connect() as conn:
        signal = load_signal_context(conn, signal_date, model, stock_code)

    if not signal_date:
        signal_date = normalize_date(signal.get("signal_date"))
    if not signal_date:
        raise RuntimeError("--signal-date is required when no matching strategy_signal_log row exists")

    action_plan = signal.get("action_plan", {})
    stock_name = args.stock_name or signal.get("stock_name") or ""
    planned_position = optional_float(args.planned_position, optional_float(action_plan.get("position_pct")))
    buy_price = optional_float(args.buy_price)
    shares = optional_int(args.shares)
    buy_fee = optional_float(args.buy_fee, 0.0)
    buy_amount = buy_price * shares if buy_price is not None and shares is not None else None
    trade_id = args.trade_id or make_trade_id(signal_date, model, stock_code, buy_date, args.lot_id)

    with engine.connect() as conn:
        existing = load_trade(conn, trade_id)
    if existing and existing.get("trade_status") == "CLOSED":
        raise RuntimeError(f"trade is already CLOSED; refuse to overwrite buy fields: {trade_id}")

    notes = merge_manual_notes(
        existing.get("manual_notes") if existing else None,
        args.notes,
        {
            "lot_id": args.lot_id,
            "source": "manual_buy",
            "signal_model_version": signal.get("model_version"),
            "signal_action_state": action_plan.get("action_state"),
        },
    )

    return {
        "trade_id": trade_id,
        "signal_date": signal_date,
        "model": model,
        "stock_code": stock_code,
        "stock_name": stock_name,
        "buy_date": buy_date,
        "buy_price": buy_price,
        "buy_shares": shares,
        "buy_amount": buy_amount,
        "buy_fee": buy_fee,
        "planned_position": planned_position,
        "sell_date": None,
        "sell_price": None,
        "sell_amount": None,
        "sell_fee": None,
        "pnl": None,
        "pnl_pct": None,
        "exit_reason": None,
        "trade_status": "OPEN",
        "manual_notes": notes,
    }


def build_sell_row(engine, args: argparse.Namespace) -> dict[str, Any]:
    sell_date = normalize_date(args.sell_date)
    sell_price = optional_float(args.sell_price)
    if not sell_date or sell_price is None:
        raise RuntimeError("--sell-date and --sell-price are required")

    with engine.connect() as conn:
        existing = find_open_trade(conn, args)

    shares = optional_int(args.shares, optional_int(existing.get("buy_shares")))
    buy_price = optional_float(existing.get("buy_price"))
    buy_fee = optional_float(existing.get("buy_fee"), 0.0) or 0.0
    sell_fee = optional_float(args.sell_fee, 0.0) or 0.0
    buy_amount = optional_float(existing.get("buy_amount"))
    if buy_amount is None and buy_price is not None and shares is not None:
        buy_amount = buy_price * shares
    sell_amount = sell_price * shares if shares is not None else None

    pnl = None
    pnl_pct = None
    if buy_amount is not None and sell_amount is not None:
        pnl = sell_amount - buy_amount - buy_fee - sell_fee
        denominator = buy_amount + buy_fee
        pnl_pct = pnl / denominator if denominator > 0 else None
    elif buy_price is not None and buy_price > 0:
        pnl_pct = sell_price / buy_price - 1.0

    notes = merge_manual_notes(
        existing.get("manual_notes"),
        args.notes,
        {
            "source": "manual_sell",
            "sell_fee": sell_fee,
            "exit_reason": args.exit_reason,
        },
    )

    row = dict(existing)
    row.update(
        {
            "buy_shares": shares,
            "buy_amount": buy_amount,
            "buy_fee": buy_fee,
            "sell_date": sell_date,
            "sell_price": sell_price,
            "sell_amount": sell_amount,
            "sell_fee": sell_fee,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": args.exit_reason,
            "trade_status": "CLOSED",
            "manual_notes": notes,
        }
    )
    return row


def write_csv(path: Path, columns: list[str] | tuple[str, ...], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_journal(engine, args: argparse.Namespace) -> dict[str, Any]:
    start_date = normalize_date(args.start_date)
    end_date = normalize_date(args.end_date)
    output_dir = Path(args.output_dir or f"reports/trade_journal/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    query = """
        SELECT *
        FROM strategy_trade_journal
        WHERE (:start_date IS NULL OR COALESCE(sell_date, buy_date, signal_date) >= :start_date)
          AND (:end_date IS NULL OR COALESCE(sell_date, buy_date, signal_date) <= :end_date)
        ORDER BY COALESCE(sell_date, buy_date, signal_date), model, stock_code
    """
    with engine.connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                sql_text(query),
                {"start_date": start_date, "end_date": end_date},
            ).mappings().all()
        ]

    trades_path = output_dir / "trade_journal.csv"
    daily_path = output_dir / "daily_pnl.csv"
    summary_path = output_dir / "summary.json"

    write_csv(trades_path, TRADE_JOURNAL_COLUMNS, rows)

    daily_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("trade_status") == "CLOSED" and row.get("sell_date"):
            daily_groups[str(row["sell_date"])].append(row)

    daily_rows = []
    all_closed = []
    for date in sorted(daily_groups):
        closed = daily_groups[date]
        all_closed.extend(closed)
        pnl_values = [optional_float(row.get("pnl")) for row in closed]
        pnl_values = [value for value in pnl_values if value is not None]
        pct_values = [optional_float(row.get("pnl_pct")) for row in closed]
        pct_values = [value for value in pct_values if value is not None]
        wins = sum(1 for value in pct_values if value > 0)
        losses = sum(1 for value in pct_values if value < 0)
        daily_rows.append(
            {
                "date": date,
                "closed_trades": len(closed),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(closed) if closed else 0.0,
                "net_pnl": sum(pnl_values) if pnl_values else "",
                "avg_pnl_pct": sum(pct_values) / len(pct_values) if pct_values else "",
            }
        )

    write_csv(daily_path, DAILY_PNL_COLUMNS, daily_rows)

    pct_values = [optional_float(row.get("pnl_pct")) for row in all_closed]
    pct_values = [value for value in pct_values if value is not None]
    pnl_values = [optional_float(row.get("pnl")) for row in all_closed]
    pnl_values = [value for value in pnl_values if value is not None]
    wins = sum(1 for value in pct_values if value > 0)
    summary = {
        "rows": len(rows),
        "open_trades": sum(1 for row in rows if row.get("trade_status") != "CLOSED"),
        "closed_trades": len(all_closed),
        "wins": wins,
        "losses": sum(1 for value in pct_values if value < 0),
        "win_rate": wins / len(all_closed) if all_closed else 0.0,
        "net_pnl": sum(pnl_values) if pnl_values else None,
        "avg_pnl_pct": sum(pct_values) / len(pct_values) if pct_values else None,
        "trade_journal_csv": str(trades_path),
        "daily_pnl_csv": str(daily_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def add_common_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="config.ini or JSON config path.")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL.")
    parser.add_argument("--db", default=None, help="SQLite path or SQLAlchemy DB URL.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record manual trades and export trade journal reports.")
    add_common_db_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    buy = sub.add_parser("buy", help="Record or update a manual buy execution.")
    buy.add_argument("--trade-id", default=None)
    buy.add_argument("--signal-date", default=None)
    buy.add_argument("--model", required=True)
    buy.add_argument("--stock-code", required=True)
    buy.add_argument("--stock-name", default=None)
    buy.add_argument("--buy-date", required=True)
    buy.add_argument("--buy-price", required=True)
    buy.add_argument("--shares", default=None)
    buy.add_argument("--buy-fee", default=0.0)
    buy.add_argument("--planned-position", default=None)
    buy.add_argument("--lot-id", default="main")
    buy.add_argument("--notes", default=None)

    sell = sub.add_parser("sell", help="Close a manual trade.")
    sell.add_argument("--trade-id", default=None)
    sell.add_argument("--signal-date", default=None)
    sell.add_argument("--model", default=None)
    sell.add_argument("--stock-code", default=None)
    sell.add_argument("--buy-date", default=None)
    sell.add_argument("--lot-id", default="main")
    sell.add_argument("--sell-date", required=True)
    sell.add_argument("--sell-price", required=True)
    sell.add_argument("--shares", default=None)
    sell.add_argument("--sell-fee", default=0.0)
    sell.add_argument("--exit-reason", required=True)
    sell.add_argument("--notes", default=None)

    export = sub.add_parser("export", help="Export trade journal and daily PnL CSVs.")
    export.add_argument("--start-date", default=None)
    export.add_argument("--end-date", default=None)
    export.add_argument("--output-dir", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    engine = make_engine(resolve_db_url(args, cfg))
    ensure_strategy_trade_journal_schema(engine)

    if args.command == "buy":
        row = build_buy_row(engine, args)
        upsert_strategy_trade_journal(engine, [row])
        print(json.dumps({"trade_id": row["trade_id"], "status": row["trade_status"]}, ensure_ascii=False, indent=2))
        return

    if args.command == "sell":
        row = build_sell_row(engine, args)
        upsert_strategy_trade_journal(engine, [row])
        print(
            json.dumps(
                {
                    "trade_id": row["trade_id"],
                    "status": row["trade_status"],
                    "pnl": row["pnl"],
                    "pnl_pct": row["pnl_pct"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "export":
        summary = export_journal(engine, args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
