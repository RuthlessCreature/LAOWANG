# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from . import db
from .config import resolve_db_target


if TYPE_CHECKING:
    import pandas as pd


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s")


def add_db_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Path to config.ini (optional)")
    p.add_argument("--db", default=None, help="SQLite db path (optional)")
    p.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")


def resolve_db_from_args(args) -> str:  # noqa: ANN001
    return resolve_db_target(
        db_url_arg=getattr(args, "db_url", None),
        db_arg=getattr(args, "db", None),
        config_path=Path(args.config) if getattr(args, "config", None) else None,
    )


def normalize_trade_date(trade_date: str) -> str:
    s = str(trade_date).strip()
    if not s:
        raise ValueError("trade_date is required")
    if s.isdigit() and len(s) == 8:
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as e:  # noqa: BLE001
        raise ValueError(f"Invalid trade_date: {trade_date}. Expect YYYYMMDD or YYYY-MM-DD") from e


def yyyymmdd_from_date(date_yyyy_mm_dd: str) -> str:
    s = str(date_yyyy_mm_dd).strip()
    if s.isdigit() and len(s) == 8:
        return s
    return s.replace("-", "")


def make_engine_for_workers(
    db_target: str,
    workers: int,
    *,
    pool_size_min: int = 5,
) -> Tuple[Engine, int]:
    w = max(1, int(workers))
    engine = db.make_engine(
        db_target,
        pool_size=max(int(pool_size_min), w * 2),
        max_overflow=w * 2,
    )
    if engine.dialect.name == "sqlite" and w > 1:
        logging.warning("SQLite does not support concurrency well; forcing workers=1.")
        w = 1
    return engine, w


def resolve_latest_stock_daily_date(db_target: str) -> str:
    engine = db.make_engine(db_target, pool_size=3, max_overflow=3)
    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()[0]
    if not latest:
        raise RuntimeError("No stock_daily data found; run pipeline first.")
    return str(latest)


def write_csv_rows(
    output_csv: Path,
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> None:
    db.ensure_parent_dir(output_csv)
    import csv

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(columns))
        for r in rows:
            w.writerow(list(r))


def write_dataframe_csv(
    df: "pd.DataFrame",
    output_csv: Path,
    *,
    columns: Optional[Sequence[str]] = None,
) -> None:
    import pandas as pd

    if columns is not None:
        df = df[list(columns)]
    db.ensure_parent_dir(output_csv)
    df.to_csv(output_csv, index=False, encoding="utf-8")
