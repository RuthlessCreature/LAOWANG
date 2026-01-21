# -*- coding: utf-8 -*-
"""
models_update.py

CLI entry to materialize per-day model outputs into MySQL.

Commands:
- init-tables: create model_runs + each model's output table (idempotent)
- update: incremental update (smart; no date args needed)
- full: full recompute (all trading days in stock_daily; can be slow)
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
from typing import List, Optional

from a_stock_analyzer import db
from a_stock_analyzer.runtime import add_db_args, resolve_db_from_args, setup_logging

from modeling.registry import build_models
from modeling.runner import ensure_tables, full_recompute, update_models        


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Materialize model outputs into MySQL tables")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    add_db_args(p)

    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--only", default="both", choices=["both", "laowang", "fhkq"])
    p.add_argument("--laowang-top", type=int, default=200)
    p.add_argument("--laowang-min-score", type=float, default=0.0)

    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-tables", help="Create model tables (idempotent)")
    sub.add_parser("update", help="Incremental update (smart; no date args)")
    sub.add_parser("full", help="Full recompute (all trading days; can be slow)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    # Allow `full/update/init-tables` to appear anywhere in argv for ergonomics,
    # since users often type: `... full --workers 16`.
    if argv is None:
        import sys

        argv = sys.argv[1:]
    cmds = {"init-tables", "update", "full"}
    for i, tok in enumerate(list(argv)):
        if tok in cmds and i != len(argv) - 1:
            argv = list(argv[:i]) + list(argv[i + 1 :]) + [tok]
            break

    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_from_args(args)
    # Models can use threads internally; don't let SQLAlchemy pool_size become a
    # hard bottleneck, but also keep it within a sane cap to avoid exhausting
    # MySQL max_connections.
    w = max(1, int(args.workers))
    pool_size = max(10, min(64, w * 2))
    max_overflow = max(20, min(128, w * 2))
    engine = db.make_engine(db_target, pool_size=pool_size, max_overflow=max_overflow)

    models = build_models(
        only=str(args.only),
        workers=int(args.workers),
        laowang_top=int(args.laowang_top),
        laowang_min_score=float(args.laowang_min_score),
    )

    if args.cmd == "init-tables":
        ensure_tables(engine, models)
        return 0

    if args.cmd == "update":
        update_models(engine=engine, models=models, workers=int(args.workers))
        return 0

    if args.cmd == "full":
        full_recompute(engine=engine, models=models, workers=int(args.workers))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
