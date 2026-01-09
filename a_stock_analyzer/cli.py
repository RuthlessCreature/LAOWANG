# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Optional

from .base_ops import init_db, update_daily_and_score_v3
from .backtest import run_backtest
from .future_perf import run_future_perf
from .runtime import resolve_db_from_args, setup_logging
from .settings import today_yyyymmdd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A股中线右侧交易分析系统（SQLite + AkShare）")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    p.add_argument("--config", default=None, help="Path to config.ini (optional)")

    sub = p.add_subparsers(dest="cmd", required=True)

    s_init = sub.add_parser("init-db", help="Create tables")
    s_init.add_argument("--db", default=None, help="SQLite db path (optional)")
    s_init.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")

    s_run = sub.add_parser("run", help="Update + calculate + score")
    s_run.add_argument("--db", default=None, help="SQLite db path (optional)")
    s_run.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")
    s_run.add_argument("--start-date", default="20000101")
    s_run.add_argument("--end-date", default=today_yyyymmdd())
    s_run.add_argument("--workers", type=int, default=8, help="Thread workers (MySQL recommended)")
    s_run.add_argument("--limit-stocks", type=int, default=None)

    s_bt = sub.add_parser("backtest", help="Random day backtest based on score table")
    s_bt.add_argument("--db", default=None, help="SQLite db path (optional)")
    s_bt.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")
    s_bt.add_argument("--nd", type=int, default=20, help="Random sample days")
    s_bt.add_argument("--ne", type=int, default=20, help="Forward trading days")
    s_bt.add_argument("--k", type=float, default=80.0, help="Score threshold")
    s_bt.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    s_bt.add_argument("--workers", type=int, default=8, help="Thread workers")
    s_bt.add_argument("--out-dir", default="output", help="Output directory")

    s_fp = sub.add_parser("future-perf", help="Signal future performance (ne=5/10/20/30)")
    s_fp.add_argument("--db", default=None, help="SQLite db path (optional)")
    s_fp.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")
    s_fp.add_argument("--ne", default="5,10,20,30", help="Comma-separated, e.g. 5,10,20,30")
    s_fp.add_argument("--min-score", type=float, default=80.0, help="Signal threshold k")
    s_fp.add_argument("--signal-date", default=None, help="YYYY-MM-DD (optional). Default: latest eligible day per ne")
    s_fp.add_argument("--top", type=int, default=None, help="Limit signals per day (optional)")
    s_fp.add_argument("--workers", type=int, default=8, help="Thread workers")
    s_fp.add_argument("--out-dir", default="output", help="Output directory")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_from_args(args)

    if args.cmd == "init-db":
        init_db(db_target)
        return 0

    if args.cmd == "run":
        update_daily_and_score_v3(
            db_target=db_target,
            start_date=args.start_date,
            end_date=args.end_date,
            workers=args.workers,
            limit_stocks=args.limit_stocks,
        )
        return 0

    if args.cmd == "backtest":
        run_backtest(
            db_target=db_target,
            nd=args.nd,
            ne=args.ne,
            k=args.k,
            seed=args.seed,
            workers=args.workers,
            out_dir=Path(args.out_dir),
        )
        return 0

    if args.cmd == "future-perf":
        ne_list = [int(x.strip()) for x in str(args.ne).split(",") if x.strip()]
        run_future_perf(
            db_target=db_target,
            ne_list=ne_list,
            min_score=args.min_score,
            signal_date=args.signal_date,
            workers=args.workers,
            top_n=args.top,
            out_dir=Path(args.out_dir),
        )
        return 0

    parser.error("Unknown command")
    return 2
