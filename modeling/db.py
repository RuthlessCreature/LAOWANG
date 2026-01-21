# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import text


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_model_runs_table(engine) -> None:  # noqa: ANN001
    if engine.dialect.name == "mysql":
        ddl = """
        CREATE TABLE IF NOT EXISTS `model_runs` (
          `model_name` VARCHAR(32) NOT NULL,
          `trade_date` VARCHAR(10) NOT NULL,
          `status` VARCHAR(16) NOT NULL,
          `row_count` INT NOT NULL,
          `message` TEXT NULL,
          `started_at` VARCHAR(19) NULL,
          `finished_at` VARCHAR(19) NULL,
          PRIMARY KEY (`model_name`, `trade_date`),
          KEY `idx_model_runs_trade_date` (`trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    else:
        ddl = """
        CREATE TABLE IF NOT EXISTS model_runs (
          model_name TEXT NOT NULL,
          trade_date TEXT NOT NULL,
          status TEXT NOT NULL,
          row_count INTEGER NOT NULL,
          message TEXT NULL,
          started_at TEXT NULL,
          finished_at TEXT NULL,
          PRIMARY KEY (model_name, trade_date)
        );
        """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def ensure_table(engine, ddl_sql: str) -> None:  # noqa: ANN001
    if not ddl_sql or not str(ddl_sql).strip():
        return
    with engine.begin() as conn:
        conn.execute(text(ddl_sql))


def upsert_sql(conn, table: str, cols: Sequence[str], key_cols: Sequence[str]) -> str:  # noqa: ANN001
    placeholders = ", ".join([f":{c}" for c in cols])
    col_list = ", ".join([str(c) for c in cols])
    if conn.engine.dialect.name == "mysql":
        update_cols = [c for c in cols if c not in set(key_cols)]
        updates = ", ".join([f"{c}=VALUES({c})" for c in update_cols])
        return f"INSERT INTO {table}({col_list}) VALUES({placeholders}) ON DUPLICATE KEY UPDATE {updates}"
    return f"INSERT OR REPLACE INTO {table}({col_list}) VALUES({placeholders})"


def write_model_run(
    engine,  # noqa: ANN001
    *,
    model_name: str,
    trade_date: str,
    status: str,
    row_count: int,
    message: str = "",
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> None:
    cols = ["model_name", "trade_date", "status", "row_count", "message", "started_at", "finished_at"]
    row = {
        "model_name": str(model_name),
        "trade_date": str(trade_date),
        "status": str(status),
        "row_count": int(row_count),
        "message": str(message or ""),
        "started_at": str(started_at or ""),
        "finished_at": str(finished_at or ""),
    }
    with engine.begin() as conn:
        sql = upsert_sql(conn, "model_runs", cols, ["model_name", "trade_date"])
        conn.execute(text(sql), row)


def is_model_ok(engine, *, model_name: str, trade_date: str) -> bool:  # noqa: ANN001
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT 1 FROM model_runs WHERE model_name = :m AND trade_date = :d AND status = 'ok' LIMIT 1"
            ),
            {"m": model_name, "d": trade_date},
        ).fetchone()
    return row is not None


def last_ok_date(engine, *, model_name: str) -> Optional[str]:  # noqa: ANN001
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT MAX(trade_date) FROM model_runs WHERE model_name = :m AND status = 'ok'"),
            {"m": model_name},
        ).fetchone()
    if not row:
        return None
    return str(row[0]) if row[0] else None


def latest_stock_daily_date(engine) -> Optional[str]:  # noqa: ANN001
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()
    if not row:
        return None
    return str(row[0]) if row[0] else None


def list_stock_daily_dates(
    engine,  # noqa: ANN001
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    after_date: Optional[str] = None,
) -> List[str]:
    where: List[str] = []
    params: Dict[str, Any] = {}
    if start_date:
        where.append("date >= :s")
        params["s"] = start_date
    if end_date:
        where.append("date <= :e")
        params["e"] = end_date
    if after_date:
        where.append("date > :a")
        params["a"] = after_date

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT DISTINCT date FROM stock_daily {where_sql} ORDER BY date"
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def delete_by_trade_date(conn, table: str, trade_date: str) -> None:  # noqa: ANN001
    conn.execute(text(f"DELETE FROM {table} WHERE trade_date = :d"), {"d": trade_date})


def bulk_insert(
    conn,  # noqa: ANN001
    *,
    table: str,
    cols: Sequence[str],
    key_cols: Sequence[str],
    rows: Iterable[Dict[str, Any]],
) -> int:
    rows_list = list(rows)
    if not rows_list:
        return 0
    sql = upsert_sql(conn, table, cols, key_cols)
    conn.execute(text(sql), rows_list)
    return len(rows_list)

