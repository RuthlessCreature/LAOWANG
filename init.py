# -*- coding: utf-8 -*-
"""
init.py

Cross-platform initializer for this project (MySQL only).

What it does (idempotent):
- Create core tables/views via a_stock_analyzer (stock_daily / indicators / scores...)
- Create materialized model tables via modeling (model_runs / model_laowang_pool / model_fhkq)

Run:
  python init.py --config config.ini
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import logging
from typing import List, Optional

from a_stock_analyzer import base_ops
from a_stock_analyzer import db as adb
from a_stock_analyzer.runtime import add_db_args, resolve_db_from_args, setup_logging

from modeling.registry import build_models
from modeling.runner import ensure_tables


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Initialize MySQL schema for LAOWANG/FHKQ/BINGWU")
    p.add_argument("--log-level", default="INFO")
    add_db_args(p)
    p.add_argument("--workers", type=int, default=16, help="Workers hint (model compute)")
    p.add_argument("--laowang-top", type=int, default=200)
    p.add_argument("--laowang-min-score", type=float, default=0.0)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_from_args(args)
    engine = adb.make_engine(db_target, pool_size=5, max_overflow=10)
    if engine.dialect.name != "mysql":
        raise SystemExit("init.py requires MySQL (check config.ini / ASTOCK_DB_URL).")

    logging.info("Init core tables/views...")
    base_ops.init_db(db_target)

    logging.info("Init model tables...")
    models = build_models(
        only="both",
        workers=int(args.workers),
        laowang_top=int(args.laowang_top),
        laowang_min_score=float(args.laowang_min_score),
    )
    ensure_tables(engine, models)

    logging.info("Init OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

