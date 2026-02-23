#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nightly incremental pipeline for next-day relay review."""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine

import getDataDailyReview as review_data_mod
import relay_model_file
import scoring_relay as relay_score_mod


def _normalize_iso_date(value: str) -> str:
    s = str(value or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def _ensure_sqlalchemy_url(db_target: str) -> str:
    tgt = str(db_target or "").strip()
    if "://" in tgt:
        return tgt
    path = Path(tgt).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def _make_engine(db_target: str) -> Engine:
    url = _ensure_sqlalchemy_url(db_target)
    connect_args = {}
    if url.startswith("sqlite:///"):
        connect_args["check_same_thread"] = False
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600, connect_args=connect_args)
    if engine.dialect.name == "sqlite":
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.close()
    return engine


def _max_trade_date(engine: Engine) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()
    if not row or not row[0]:
        return None
    return str(row[0])


def _max_relay_date(engine: Engine) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(trade_date) FROM model_relay_pool")).fetchone()
    if not row or not row[0]:
        return None
    return str(row[0])


def _next_day(iso_date: str) -> str:
    d = dt.datetime.strptime(iso_date, "%Y-%m-%d").date()
    return (d + dt.timedelta(days=1)).strftime("%Y-%m-%d")


def _build_common_cli(args: argparse.Namespace) -> List[str]:
    cli: List[str] = []
    if getattr(args, "config", None):
        cli.extend(["--config", str(args.config)])
    if getattr(args, "db_url", None):
        cli.extend(["--db-url", str(args.db_url)])
    elif getattr(args, "db", None):
        cli.extend(["--db", str(args.db)])
    return cli


def _run_pipeline(args: argparse.Namespace, *, setup_logging: bool) -> None:
    if setup_logging:
        logging.basicConfig(
            level=getattr(logging, str(args.log_level).upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(message)s",
        )

    db_target = review_data_mod.resolve_db_target(args)
    engine = _make_engine(db_target)
    latest_daily = _max_trade_date(engine)
    if not latest_daily:
        logging.warning("[everydayReview] stock_daily is empty, skip")
        return

    if args.end_date:
        end_date = _normalize_iso_date(args.end_date)
        if end_date > latest_daily:
            end_date = latest_daily
    else:
        end_date = latest_daily

    if args.start_date:
        start_date = _normalize_iso_date(args.start_date)
    elif bool(args.full_rebuild):
        start_date = _normalize_iso_date(args.initial_start_date)
    else:
        prev_relay = _max_relay_date(engine)
        if prev_relay:
            # Recompute from the last scored date to keep latest day aligned.
            start_date = prev_relay
        else:
            start_date = _normalize_iso_date(args.initial_start_date)

    if start_date > end_date:
        logging.info("[everydayReview] no new range to run (%s > %s), skip", start_date, end_date)
        return

    base_cli = _build_common_cli(args)

    data_cli = base_cli + [
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--workers",
        str(max(1, int(args.data_workers))),
        "--xgb-timeout",
        str(max(1, int(args.xgb_timeout))),
    ]
    logging.info("[everydayReview] getDataDailyReview: %s -> %s", start_date, end_date)
    review_data_mod.main(data_cli)

    score_cli = base_cli + [
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--model-file",
        str(args.model_file),
    ]
    if bool(args.full_rebuild):
        score_cli.append("--full-rebuild")
    logging.info("[everydayReview] scoring_relay: %s -> %s", start_date, end_date)
    relay_score_mod.main(score_cli)

    logging.info("[everydayReview] done latest=%s start=%s", end_date, start_date)


def run_once(
    *,
    config: Optional[str],
    db_url: Optional[str],
    db: Optional[str],
    initial_start_date: str,
    model_file: str,
    data_workers: int,
    xgb_timeout: int,
    full_rebuild: bool = False,
) -> None:
    args = argparse.Namespace(
        config=config,
        db_url=db_url,
        db=db,
        initial_start_date=initial_start_date,
        model_file=model_file,
        data_workers=int(data_workers),
        xgb_timeout=int(xgb_timeout),
        full_rebuild=bool(full_rebuild),
        start_date=None,
        end_date=None,
        log_level=logging.getLevelName(logging.getLogger().level),
    )
    _run_pipeline(args, setup_logging=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly incremental relay pipeline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--initial-start-date", default="2025-01-01")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--model-file", default=str(relay_model_file.default_model_path()))
    parser.add_argument("--data-workers", type=int, default=1)
    parser.add_argument("--xgb-timeout", type=int, default=10)
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _run_pipeline(args, setup_logging=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
