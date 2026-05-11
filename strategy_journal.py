"""Persistence helpers for strategy signal audit rows.

The live workflow writes one row per model/date/code into strategy_signal_log.
This module stays intentionally small so generators and later workers can share
the same idempotent write path.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.engine import Engine


SIGNAL_LOG_COLUMNS = (
    "signal_date",
    "model",
    "stock_code",
    "stock_name",
    "score",
    "features_json",
    "model_version",
    "action_plan_json",
    "created_at",
)

TRADE_JOURNAL_COLUMNS = (
    "trade_id",
    "signal_date",
    "model",
    "stock_code",
    "stock_name",
    "buy_date",
    "buy_price",
    "buy_shares",
    "buy_amount",
    "buy_fee",
    "planned_position",
    "sell_date",
    "sell_price",
    "sell_amount",
    "sell_fee",
    "pnl",
    "pnl_pct",
    "exit_reason",
    "trade_status",
    "manual_notes",
    "created_at",
    "updated_at",
)

TRADE_JOURNAL_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS strategy_trade_journal (
    trade_id VARCHAR(64) PRIMARY KEY,
    signal_date VARCHAR(10) NOT NULL,
    model VARCHAR(32) NOT NULL,
    stock_code VARCHAR(16) NOT NULL,
    stock_name VARCHAR(255) NULL,
    buy_date VARCHAR(10) NULL,
    buy_price DOUBLE NULL,
    buy_shares BIGINT NULL,
    buy_amount DOUBLE NULL,
    buy_fee DOUBLE NULL,
    planned_position DOUBLE NULL,
    sell_date VARCHAR(10) NULL,
    sell_price DOUBLE NULL,
    sell_amount DOUBLE NULL,
    sell_fee DOUBLE NULL,
    pnl DOUBLE NULL,
    pnl_pct DOUBLE NULL,
    exit_reason VARCHAR(255) NULL,
    trade_status VARCHAR(16) NULL,
    manual_notes TEXT NULL,
    created_at VARCHAR(19) NULL,
    updated_at VARCHAR(19) NULL
)
"""

TRADE_JOURNAL_EXTRA_COLUMNS = {
    "stock_name": "VARCHAR(255) NULL",
    "buy_shares": "BIGINT NULL",
    "buy_amount": "DOUBLE NULL",
    "buy_fee": "DOUBLE NULL",
    "sell_amount": "DOUBLE NULL",
    "sell_fee": "DOUBLE NULL",
    "trade_status": "VARCHAR(16) NULL",
}


def json_dumps(value: Any) -> str:
    """Serialize a JSON-ish value with stable, Chinese-friendly output."""
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {column: row.get(column) for column in SIGNAL_LOG_COLUMNS}
    normalized["features_json"] = json_dumps(normalized.get("features_json"))
    normalized["action_plan_json"] = json_dumps(normalized.get("action_plan_json"))
    normalized["created_at"] = normalized.get("created_at") or _now_text()
    return normalized


def _normalize_trade_row(row: Mapping[str, Any]) -> dict[str, Any]:
    now = _now_text()
    normalized = {column: row.get(column) for column in TRADE_JOURNAL_COLUMNS}
    normalized["manual_notes"] = json_dumps(normalized.get("manual_notes"))
    normalized["created_at"] = normalized.get("created_at") or now
    normalized["updated_at"] = normalized.get("updated_at") or now
    return normalized


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _upsert_sql(dialect_name: str) -> str:
    columns_sql = ", ".join(SIGNAL_LOG_COLUMNS)
    values_sql = ", ".join(f":{column}" for column in SIGNAL_LOG_COLUMNS)

    if dialect_name == "sqlite":
        update_sql = ", ".join(
            f"{column}=excluded.{column}"
            for column in SIGNAL_LOG_COLUMNS
            if column not in {"signal_date", "model", "stock_code"}
        )
        return (
            f"INSERT INTO strategy_signal_log ({columns_sql}) VALUES ({values_sql}) "
            "ON CONFLICT(signal_date, model, stock_code) DO UPDATE SET "
            f"{update_sql}"
        )

    update_sql = ", ".join(
        f"{column}=VALUES({column})"
        for column in SIGNAL_LOG_COLUMNS
        if column not in {"signal_date", "model", "stock_code"}
    )
    return (
        f"INSERT INTO strategy_signal_log ({columns_sql}) VALUES ({values_sql}) "
        f"ON DUPLICATE KEY UPDATE {update_sql}"
    )


def _trade_upsert_sql(dialect_name: str) -> str:
    columns_sql = ", ".join(TRADE_JOURNAL_COLUMNS)
    values_sql = ", ".join(f":{column}" for column in TRADE_JOURNAL_COLUMNS)

    if dialect_name == "sqlite":
        update_parts = []
        for column in TRADE_JOURNAL_COLUMNS:
            if column == "trade_id":
                continue
            if column == "created_at":
                update_parts.append("created_at=COALESCE(strategy_trade_journal.created_at, excluded.created_at)")
            else:
                update_parts.append(f"{column}=excluded.{column}")
        return (
            f"INSERT INTO strategy_trade_journal ({columns_sql}) VALUES ({values_sql}) "
            "ON CONFLICT(trade_id) DO UPDATE SET "
            f"{', '.join(update_parts)}"
        )

    update_parts = []
    for column in TRADE_JOURNAL_COLUMNS:
        if column == "trade_id":
            continue
        if column == "created_at":
            update_parts.append("created_at=COALESCE(strategy_trade_journal.created_at, VALUES(created_at))")
        else:
            update_parts.append(f"{column}=VALUES({column})")
    return (
        f"INSERT INTO strategy_trade_journal ({columns_sql}) VALUES ({values_sql}) "
        f"ON DUPLICATE KEY UPDATE {', '.join(update_parts)}"
    )


def _is_duplicate_column_error(exc: Exception) -> bool:
    msg = str(getattr(exc, "orig", exc)).lower()
    return "duplicate" in msg or "exists" in msg


def upsert_strategy_signal_log(engine: "Engine", rows: Sequence[Mapping[str, Any]], chunk_size: int = 500) -> int:
    """Idempotently persist rows into strategy_signal_log.

    Supports the repo's two expected SQLAlchemy dialects: SQLite for local
    development and MySQL for production-like runs.
    """
    normalized_rows = [_normalize_row(row) for row in rows]
    if not normalized_rows:
        return 0

    from sqlalchemy import text

    stmt = text(_upsert_sql(engine.dialect.name))
    with engine.begin() as conn:
        for batch in _chunks(normalized_rows, chunk_size):
            conn.execute(stmt, batch)
    return len(normalized_rows)


def ensure_strategy_trade_journal_schema(engine: "Engine") -> None:
    """Create/upgrade strategy_trade_journal for manual execution logging."""
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    with engine.begin() as conn:
        conn.execute(text(TRADE_JOURNAL_CREATE_SQL))

    for column, ddl in TRADE_JOURNAL_EXTRA_COLUMNS.items():
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE strategy_trade_journal ADD COLUMN {column} {ddl}"))
        except SQLAlchemyError as exc:
            if _is_duplicate_column_error(exc):
                continue
            raise


def upsert_strategy_trade_journal(engine: "Engine", rows: Sequence[Mapping[str, Any]], chunk_size: int = 500) -> int:
    """Idempotently persist manual trade journal rows by trade_id."""
    normalized_rows = [_normalize_trade_row(row) for row in rows]
    if not normalized_rows:
        return 0

    from sqlalchemy import text

    stmt = text(_trade_upsert_sql(engine.dialect.name))
    with engine.begin() as conn:
        for batch in _chunks(normalized_rows, chunk_size):
            conn.execute(stmt, batch)
    return len(normalized_rows)
