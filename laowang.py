# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from a_stock_analyzer.pool_export import export_pool
from a_stock_analyzer.runtime import add_db_args, resolve_db_from_args, setup_logging
from a_stock_analyzer.settings import today_yyyymmdd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="laowang.py - 评分结果导出（v3 股票池）")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    add_db_args(p)

    p.add_argument("--output", default=f"output/pool_{today_yyyymmdd()}.csv")
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--require-tags", default=None, help="Comma-separated, e.g. TREND_UP,AT_SUPPORT")
    p.add_argument("--min-resistance-distance", type=float, default=None, help="e.g. 0.10 for 10%")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_from_args(args)
    require_tags = args.require_tags.split(",") if args.require_tags else None

    export_pool(
        db_target=db_target,
        output_csv=Path(args.output),
        top_n=args.top,
        min_score=args.min_score,
        require_tags=require_tags,
        min_resistance_distance=args.min_resistance_distance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

