# -*- coding: utf-8 -*-
"""
everyday.py

Cross-platform daily runner:
- (Optional) update daily OHLCV + indicators/levels + scores into MySQL
- materialize per-day model outputs into MySQL (LAOWANG/FHKQ)
- export bingwu daily CSV to outputs/

This is designed to be scheduled (cron / Task Scheduler) after close (15:05).

Run:
  python everyday.py --config config.ini
"""

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

from a_stock_analyzer import base_ops
from a_stock_analyzer import db as adb
from a_stock_analyzer.runtime import add_db_args, normalize_trade_date, resolve_db_from_args, setup_logging, yyyymmdd_from_date

from bingwu_report import main as bingwu_report_main
from modeling.registry import build_models
from modeling.runner import ensure_tables, update_models


def _today_yyyymmdd() -> str:
    return dt.date.today().strftime("%Y%m%d")


def _date_from_yyyymmdd(s: str) -> dt.date:
    s = str(s or "").strip()
    if not (s.isdigit() and len(s) == 8):
        raise ValueError("Expect YYYYMMDD")
    return dt.date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def _max_date(engine, table: str, col: str) -> Optional[str]:  # noqa: ANN001
    with engine.connect() as conn:
        row = conn.execute(text(f"SELECT MAX({col}) FROM {table}")).fetchone()
    if not row:
        return None
    return str(row[0]) if row[0] else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily pipeline + models + bingwu report (MySQL only)")
    p.add_argument("--log-level", default="INFO")
    add_db_args(p)

    p.add_argument("--end-date", default=None, help="YYYYMMDD (default: today)")
    p.add_argument("--workers", type=int, default=16, help="Workers for pipeline (MySQL recommended)")
    p.add_argument("--models-workers", type=int, default=16, help="Workers for model compute")
    p.add_argument("--laowang-top", type=int, default=200)
    p.add_argument("--laowang-min-score", type=float, default=0.0)

    p.add_argument("--skip-pipeline", action="store_true", help="Skip market data update/scoring (not recommended)")
    p.add_argument("--skip-bingwu", action="store_true", help="Skip bingwu CSV export")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_from_args(args)
    mw = max(1, int(args.models_workers))
    pool_size = max(10, min(64, mw * 2))
    max_overflow = max(20, min(128, mw * 2))
    engine = adb.make_engine(db_target, pool_size=pool_size, max_overflow=max_overflow)
    if engine.dialect.name != "mysql":
        raise SystemExit("everyday.py requires MySQL (check config.ini / ASTOCK_DB_URL).")

    # Ensure schema (idempotent)
    base_ops.init_db(db_target)
    models = build_models(
        only="both",
        workers=int(args.models_workers),
        laowang_top=int(args.laowang_top),
        laowang_min_score=float(args.laowang_min_score),
    )
    ensure_tables(engine, models)

    end_yyyymmdd = str(args.end_date).strip() if args.end_date else _today_yyyymmdd()
    end_date = _date_from_yyyymmdd(end_yyyymmdd)

    # Decide whether pipeline is needed (avoid slow network calls if already up-to-date).
    latest_daily = _max_date(engine, "stock_daily", "date")
    latest_score = _max_date(engine, "stock_scores_v3", "score_date")

    need_pipeline = True
    if args.skip_pipeline:
        need_pipeline = False
    else:
        if latest_daily:
            try:
                latest_daily_date = dt.datetime.strptime(str(latest_daily), "%Y-%m-%d").date()
                if latest_daily_date >= end_date and latest_score and str(latest_score) >= str(latest_daily):
                    need_pipeline = False
            except Exception:
                need_pipeline = True

    if need_pipeline:
        logging.info("Pipeline update: end=%s workers=%d", end_yyyymmdd, int(args.workers))
        # For stocks that have never been pulled before, fetching "all history"
        # (e.g. from 20000101) is extremely slow. We only need a rolling window
        # to compute indicators/levels/scores reliably.
        start_yyyymmdd = (end_date - dt.timedelta(days=1500)).strftime("%Y%m%d")
        if start_yyyymmdd < "20000101":
            start_yyyymmdd = "20000101"
        base_ops.update_daily_and_score_v3(
            db_target=db_target,
            start_date=start_yyyymmdd,
            end_date=end_yyyymmdd,
            workers=int(args.workers),
        )
    else:
        logging.info("Pipeline already up-to-date; skip.")

    # Resolve effective trade date from DB (latest available trading day).
    latest_daily2 = _max_date(engine, "stock_daily", "date")
    if not latest_daily2:
        raise SystemExit("No stock_daily data found after pipeline; cannot proceed.")

    trade_date_norm = normalize_trade_date(str(latest_daily2))
    trade_yyyymmdd = yyyymmdd_from_date(trade_date_norm)
    logging.info("Effective trade date: %s", trade_yyyymmdd)

    # Materialize models (smart incremental based on model_runs)
    logging.info("Materialize models into MySQL...")
    update_models(engine=engine, models=models, workers=int(args.models_workers))

    # Export bingwu daily CSV
    if not args.skip_bingwu:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"bingwu_{trade_yyyymmdd}.csv"
        logging.info("Export bingwu CSV: %s", out_csv)
        # Call bingwu_report.py main directly to avoid subprocess.
        # Pass --db-url explicitly to avoid accidental SQLite fallback.
        bingwu_report_main(["--db-url", str(db_target), "--trade-date", trade_yyyymmdd, "--output", str(out_csv)])

    logging.info("Everyday OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
