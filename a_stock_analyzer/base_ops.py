# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from typing import Optional

from . import db
from .pipeline import run_pipeline
from .settings import Settings


def init_db(db_target: str) -> None:
    engine = db.make_engine(db_target, pool_size=5, max_overflow=10)
    db.init_db(engine)
    logging.info("DB initialized: %s", db_target)


def update_daily_and_score_v3(
    *,
    db_target: str,
    start_date: str,
    end_date: str,
    workers: int,
    limit_stocks: Optional[int] = None,
) -> None:
    settings = Settings(db=db_target, start_date=start_date, end_date=end_date, workers=workers)
    run_pipeline(settings, limit_stocks=limit_stocks)

