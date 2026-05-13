#!/usr/bin/env python3
"""Generate T-close manual trading plans from ordinary model pools.

This script intentionally avoids T+1 market data. It turns today's model pools
into conditional next-day action plans and writes an audit trail.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from strategy_journal import json_dumps, upsert_strategy_signal_log
from strategy_protocols import ORDINARY_MODEL_ORDER, ordinary_model_plan_config


ORDINARY_PLAN_COLUMNS = [
    "signal_date",
    "buy_date",
    "model",
    "stock_code",
    "stock_name",
    "score",
    "strategy_version",
    "signal_close",
    "amount_ma20",
    "status_tags",
    "entry_filter_status",
    "planned_action",
    "action_state",
    "t1_buy_condition",
    "t2_sell_condition",
    "planned_position_pct",
    "hard_stop_pct",
    "structure_stop",
    "max_hold_days",
    "skip_reason",
]

SIGNAL_AUDIT_COLUMNS = [
    "signal_date",
    "model",
    "stock_code",
    "stock_name",
    "score",
    "strategy_version",
    "model_version",
    "action_state",
    "planned_position_pct",
    "reason",
    "features_json",
    "action_plan_json",
]


def load_config(path: str | None) -> dict[str, Any]:
    cfg_path = Path(path or "config.ini")
    if not cfg_path.exists():
        return {}

    if cfg_path.suffix.lower() in {".ini", ".cfg"}:
        parser = configparser.ConfigParser()
        parser.read(cfg_path, encoding="utf-8")
        db_url = parser.get("database", "db_url", fallback="").strip()
        return {
            "db_url": db_url or None,
            "mysql": {
                "host": parser.get("mysql", "host", fallback="127.0.0.1").strip() or "127.0.0.1",
                "port": parser.getint("mysql", "port", fallback=3306),
                "user": parser.get("mysql", "user", fallback="").strip(),
                "password": parser.get("mysql", "password", fallback=""),
                "database": parser.get("mysql", "database", fallback="").strip(),
                "charset": parser.get("mysql", "charset", fallback="utf8mb4").strip() or "utf8mb4",
            },
        }

    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_mysql_url(db_cfg: Mapping[str, Any] | None) -> str:
    db_cfg = db_cfg or {}
    from urllib.parse import quote_plus

    host = os.getenv("MYSQL_HOST") or db_cfg.get("host", "localhost")
    port = int(os.getenv("MYSQL_PORT") or db_cfg.get("port", 3306))
    user = os.getenv("MYSQL_USER") or db_cfg.get("user", "root")
    password = os.getenv("MYSQL_PASSWORD") or db_cfg.get("password", "")
    database = os.getenv("MYSQL_DATABASE") or db_cfg.get("database", "laowang")
    charset = os.getenv("MYSQL_CHARSET") or db_cfg.get("charset", "utf8mb4")
    auth = quote_plus(str(user))
    password = quote_plus(str(password))
    if password:
        auth = f"{auth}:{password}"
    return f"mysql+pymysql://{auth}@{host}:{port}/{database}?charset={charset}"


def resolve_db_url(args: argparse.Namespace, cfg: Mapping[str, Any]) -> str:
    if args.db_url:
        return args.db_url
    env_url = os.getenv("ASTOCK_DB_URL") or os.getenv("DATABASE_URL")
    if env_url and env_url.strip():
        return env_url.strip()

    database_value = cfg.get("database")
    db_arg = args.db or cfg.get("db")
    if not db_arg and isinstance(database_value, str):
        db_arg = database_value
    if db_arg and "://" in str(db_arg):
        return str(db_arg)
    if db_arg:
        return f"sqlite:///{Path(db_arg).expanduser().resolve()}"

    if cfg.get("db_url"):
        return str(cfg["db_url"])

    database_cfg = database_value if isinstance(database_value, Mapping) else {}
    if database_cfg:
        db_type = database_cfg.get("type", "sqlite")
        if db_type == "mysql":
            return build_mysql_url(database_cfg)
        sqlite_path = database_cfg.get("path") or database_cfg.get("db") or "data/laowang.db"
        return f"sqlite:///{Path(sqlite_path).expanduser().resolve()}"

    mysql_cfg = cfg.get("mysql") if isinstance(cfg.get("mysql"), Mapping) else {}
    if mysql_cfg and mysql_cfg.get("user") and mysql_cfg.get("database"):
        return build_mysql_url(mysql_cfg)

    return f"sqlite:///{Path('data/stock.db').resolve()}"


def make_engine(db_url: str):
    from sqlalchemy import create_engine, event

    engine = create_engine(db_url, future=True)
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _conn_record):  # type: ignore[no-untyped-def]
            dbapi_conn.execute("PRAGMA journal_mode=WAL;")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL;")

    return engine


def sql_text(query: str):
    from sqlalchemy import text

    return text(query)


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt_float(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def tags_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value)
    for sep in "|,;，；":
        text = text.replace(sep, " ")
    return {part.strip() for part in text.split() if part.strip()}


def latest_trade_date(engine) -> str:
    with engine.connect() as conn:
        row = conn.execute(sql_text("SELECT MAX(date) FROM stock_daily")).first()
    if not row or not row[0]:
        raise RuntimeError("stock_daily has no trade date; run data ingestion first")
    return str(row[0])


def amount_ma20(conn, stock_code: str, trade_date: str) -> float:
    rows = conn.execute(
        sql_text(
            """
            SELECT amount
            FROM stock_daily
            WHERE stock_code = :stock_code AND date <= :trade_date
            ORDER BY date DESC
            LIMIT 20
            """
        ),
        {"stock_code": stock_code, "trade_date": trade_date},
    ).all()
    values = [safe_float(row[0]) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def fetch_rows(conn, query: str, trade_date: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(sql_text(query), {"trade_date": trade_date}).mappings().all()
    ]


def load_ordinary_pool(conn, model: str, trade_date: str) -> list[dict[str, Any]]:
    queries = {
        "laowang": """
            SELECT
                rank_no,
                stock_code,
                stock_name,
                close,
                total_score AS score,
                support_level,
                resistance_level,
                status_tags
            FROM model_laowang_pool
            WHERE trade_date = :trade_date
            ORDER BY rank_no, total_score DESC
        """,
        "stwg": """
            SELECT
                rank_no,
                stock_code,
                stock_name,
                close,
                total_score AS score,
                stageB_compression_score,
                breakout_confirmation_score,
                status_tags
            FROM model_stwg_pool
            WHERE trade_date = :trade_date
            ORDER BY rank_no, total_score DESC
        """,
        "ywcx": """
            SELECT
                rank_no,
                stock_code,
                stock_name,
                close,
                total_score AS score,
                weak_position_score,
                volume_dry_score,
                low_volatility_score,
                status_tags
            FROM model_ywcx_pool
            WHERE trade_date = :trade_date
            ORDER BY rank_no, total_score DESC
        """,
        "fhkq": """
            SELECT
                NULL AS rank_no,
                stock_code,
                stock_name,
                NULL AS close,
                fhkq_score AS score,
                consecutive_limit_down,
                last_limit_down,
                volume_ratio,
                amount_ratio,
                open_board_flag,
                liquidity_exhaust,
                fhkq_level AS status_tags
            FROM model_fhkq
            WHERE trade_date = :trade_date
            ORDER BY fhkq_score DESC
        """,
    }
    return fetch_rows(conn, queries[model], trade_date)


def ordinary_reasons(row: Mapping[str, Any], cfg: Mapping[str, Any], liquidity: float) -> list[str]:
    reasons: list[str] = []
    score = safe_float(row.get("score"))
    tag_set = tags_set(row.get("status_tags"))
    min_amount_ma20 = safe_float(cfg.get("min_amount_ma20"), 0.0)

    if cfg.get("paused"):
        reasons.append("model_paused")
    if math.isnan(score) or score < safe_float(cfg.get("min_score"), 0.0):
        reasons.append("score_below_threshold")

    missing_required = [tag for tag in cfg.get("required_tags", []) if tag not in tag_set]
    if missing_required:
        reasons.append("missing_required_tags:" + "|".join(missing_required))

    any_tags = list(cfg.get("any_tags", []))
    if any_tags and not any(tag in tag_set for tag in any_tags):
        reasons.append("missing_any_tags:" + "|".join(any_tags))

    blocked = [tag for tag in cfg.get("blocked_tags", []) if tag in tag_set]
    if blocked:
        reasons.append("blocked_tags:" + "|".join(blocked))

    if min_amount_ma20 > 0:
        if math.isnan(liquidity):
            reasons.append("missing_amount_ma20")
        elif liquidity < min_amount_ma20:
            reasons.append("amount_ma20_below_threshold")

    if row.get("open_board_flag") is not None:
        consecutive = safe_int(row.get("consecutive_limit_down"))
        open_board = safe_int(row.get("open_board_flag"), 0)
        liquidity_exhaust = safe_int(row.get("liquidity_exhaust"), 0)
        if consecutive is None or consecutive < 2 or consecutive > 4:
            reasons.append("fhkq_limit_down_count_out_of_range")
        if open_board != 1:
            reasons.append("fhkq_not_open_board")
        if liquidity_exhaust != 1:
            reasons.append("fhkq_liquidity_exhaust_not_confirmed")

    return reasons


def build_ordinary_plan_row(
    trade_date: str,
    model: str,
    row: Mapping[str, Any],
    cfg: Mapping[str, Any],
    liquidity: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reasons = ordinary_reasons(row, cfg, liquidity)
    action_state = "skip_precheck" if reasons else "pending_t1_buy_check"
    planned_action = "skip" if reasons else "conditional_buy"
    planned_position = 0.0 if reasons else safe_float(cfg.get("planned_position_pct"), 0.0)

    action_plan = {
        "strategy_version": cfg.get("strategy_version"),
        "entry_gap_min": cfg.get("min_gap"),
        "entry_gap_max": cfg.get("max_gap"),
        "action_state": action_state,
        "planned_action": planned_action,
        "position_pct": planned_position,
        "hard_stop_pct": cfg.get("hard_stop_pct"),
        "structure_stop": cfg.get("structure_stop"),
        "profit_arm_pct": cfg.get("profit_arm_pct"),
        "trailing_stop": cfg.get("profit_trailing_stop"),
        "max_hold_days": cfg.get("max_hold_days"),
        "skip_reason": reasons,
        "t1_buy_condition": cfg.get("t1_buy_condition"),
        "t2_sell_condition": cfg.get("t2_sell_condition"),
    }
    if model == "fhkq":
        action_plan["requires_liquidity_recovery_check"] = True
        action_plan["manual_note"] = "FHKQ is event/liquidity recovery only; default to no trade unless opening liquidity confirms."

    features = {
        key: row.get(key)
        for key in row.keys()
        if key not in {"stock_code", "stock_name"}
    }
    features["amount_ma20"] = liquidity

    plan_row = {
        "signal_date": trade_date,
        "buy_date": "T+1",
        "model": model,
        "stock_code": row.get("stock_code"),
        "stock_name": row.get("stock_name"),
        "score": fmt_float(row.get("score"), 2),
        "strategy_version": cfg.get("strategy_version"),
        "signal_close": fmt_float(row.get("close"), 3),
        "amount_ma20": fmt_float(liquidity, 2),
        "status_tags": row.get("status_tags") or "",
        "entry_filter_status": "pass" if not reasons else "fail",
        "planned_action": planned_action,
        "action_state": action_state,
        "t1_buy_condition": cfg.get("t1_buy_condition"),
        "t2_sell_condition": cfg.get("t2_sell_condition"),
        "planned_position_pct": fmt_float(planned_position, 4),
        "hard_stop_pct": fmt_float(cfg.get("hard_stop_pct"), 4),
        "structure_stop": cfg.get("structure_stop"),
        "max_hold_days": cfg.get("max_hold_days"),
        "skip_reason": "|".join(reasons),
    }

    audit_row = {
        "signal_date": trade_date,
        "model": model,
        "stock_code": row.get("stock_code"),
        "stock_name": row.get("stock_name"),
        "score": safe_float(row.get("score"), 0.0),
        "strategy_version": cfg.get("strategy_version"),
        "model_version": cfg.get("strategy_version"),
        "action_state": action_state,
        "planned_position_pct": planned_position,
        "reason": "|".join(reasons),
        "features_json": json_dumps(features),
        "action_plan_json": json_dumps(action_plan),
    }

    signal_log_row = {
        "signal_date": trade_date,
        "model": model,
        "stock_code": row.get("stock_code"),
        "stock_name": row.get("stock_name"),
        "score": safe_float(row.get("score"), 0.0),
        "features_json": features,
        "model_version": cfg.get("strategy_version"),
        "action_plan_json": action_plan,
    }
    return plan_row, audit_row, signal_log_row


def write_csv(path: Path, columns: list[str], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_trade_plan(engine, trade_date: str, output_dir: Path, write_db: bool) -> dict[str, Any]:
    ordinary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    signal_log_rows: list[dict[str, Any]] = []

    with engine.connect() as conn:
        for model in ORDINARY_MODEL_ORDER:
            cfg = ordinary_model_plan_config(model)
            for row in load_ordinary_pool(conn, model, trade_date):
                liquidity = amount_ma20(conn, str(row.get("stock_code")), trade_date)
                plan_row, audit_row, signal_row = build_ordinary_plan_row(
                    trade_date, model, row, cfg, liquidity
                )
                ordinary_rows.append(plan_row)
                audit_rows.append(audit_row)
                signal_log_rows.append(signal_row)

    write_csv(output_dir / "ordinary_next_day_plan.csv", ORDINARY_PLAN_COLUMNS, ordinary_rows)
    write_csv(output_dir / "signal_audit.csv", SIGNAL_AUDIT_COLUMNS, audit_rows)

    persisted_rows = 0
    if write_db:
        persisted_rows = upsert_strategy_signal_log(engine, signal_log_rows)

    return {
        "trade_date": trade_date,
        "output_dir": str(output_dir),
        "ordinary_rows": len(ordinary_rows),
        "audit_rows": len(audit_rows),
        "persisted_rows": persisted_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate T-close manual trading plans.")
    parser.add_argument("--config", default=None, help="Optional JSON config path.")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL.")
    parser.add_argument("--db", default=None, help="SQLite path or SQLAlchemy DB URL.")
    parser.add_argument("--trade-date", default=None, help="Signal trade date, e.g. 2026-05-11.")
    parser.add_argument("--output-dir", default=None, help="Plan output directory.")
    parser.add_argument(
        "--write-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write strategy_signal_log audit rows. Use --no-write-db for dry output only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    db_url = resolve_db_url(args, cfg)
    engine = make_engine(db_url)
    trade_date = args.trade_date or latest_trade_date(engine)
    output_dir = Path(args.output_dir or f"reports/trade_plans/{trade_date.replace('-', '')}")

    result = generate_trade_plan(engine, trade_date, output_dir, args.write_db)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
