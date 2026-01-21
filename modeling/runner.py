# -*- coding: utf-8 -*-
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Optional, Sequence, Tuple

from sqlalchemy import text

from . import db as mdb
from .base import Model


def ensure_tables(engine, models: Sequence[Model]) -> None:  # noqa: ANN001
    mdb.ensure_model_runs_table(engine)
    for m in models:
        m.ensure_tables(engine)


def _latest_stock_daily_date(engine) -> Optional[str]:  # noqa: ANN001
    return mdb.latest_stock_daily_date(engine)


def _delete_model_runs(engine, *, model_name: str) -> None:  # noqa: ANN001
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_runs WHERE model_name = :m"),
            {"m": model_name},
        )


def _worker_budget(workers: int, *, outer_cap: int) -> Tuple[int, int]:
    """
    We have two nested concurrency layers:
    - outer: across trade dates
    - inner: passed down into each model's compute(), which may use threads

    Keep the product bounded to avoid thread explosions in "full" runs.
    """
    total = max(1, int(workers))
    outer = max(1, min(int(outer_cap), total))
    inner = max(1, total // outer)
    inner = min(64, inner)
    return int(outer), int(inner)


def update_models(
    *,
    engine,
    models: Sequence[Model],
    workers: int,
) -> None:
    """
    Incremental update:
    - Determine latest trading day from stock_daily.
    - For each model, compute dates after its last_ok_date; if never ran,
      compute latest date only.
    """
    ensure_tables(engine, models)

    latest = _latest_stock_daily_date(engine)
    if not latest:
        logging.warning("No stock_daily data; skip models update.")
        return

    date_set: set[str] = set()
    for m in models:
        last_ok = mdb.last_ok_date(engine, model_name=m.name)
        if last_ok:
            dates = mdb.list_stock_daily_dates(engine, after_date=last_ok, end_date=latest)
        else:
            dates = [latest]
        for d in dates:
            date_set.add(str(d))

    dates_all = sorted(date_set)
    if not dates_all:
        logging.info("No new trading days for models.")
        return

    date_workers, model_workers = _worker_budget(int(workers), outer_cap=4)

    def run_date(d: str) -> None:
        for m in models:
            if mdb.is_model_ok(engine, model_name=m.name, trade_date=d):
                continue

            started = mdb.now_ts()
            try:
                logging.info("[%s] compute %s", m.name, d)
                df = m.compute(engine=engine, trade_date=d, workers=int(model_workers))
                n = m.save(engine=engine, trade_date=d, df=df)
                mdb.write_model_run(
                    engine,
                    model_name=m.name,
                    trade_date=d,
                    status="ok",
                    row_count=int(n),
                    message="",
                    started_at=started,
                    finished_at=mdb.now_ts(),
                )
                logging.info("[%s] ok %s rows=%d", m.name, d, n)
            except Exception as e:  # noqa: BLE001
                mdb.write_model_run(
                    engine,
                    model_name=m.name,
                    trade_date=d,
                    status="error",
                    row_count=0,
                    message=f"{type(e).__name__}: {e}",
                    started_at=started,
                    finished_at=mdb.now_ts(),
                )
                logging.exception("[%s] failed %s", m.name, d)

    if date_workers <= 1 or len(dates_all) <= 1:
        for d in dates_all:
            run_date(str(d))
        return

    logging.info(
        "Models incremental: dates=%d outer_workers=%d model_workers=%d",
        len(dates_all),
        date_workers,
        model_workers,
    )
    with ThreadPoolExecutor(max_workers=int(date_workers)) as ex:
        futs = {ex.submit(run_date, str(d)): str(d) for d in dates_all}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:  # noqa: BLE001
                logging.exception("Model update worker crashed for date=%s", futs.get(f))


def full_recompute(
    *,
    engine,
    models: Sequence[Model],
    workers: int,
) -> None:
    """
    Full recompute for all trading days in stock_daily.
    Warning: can be very slow if a model is heavy for historical backfill.
    """
    ensure_tables(engine, models)
    latest = _latest_stock_daily_date(engine)
    if not latest:
        logging.warning("No stock_daily data; skip models full recompute.")
        return

    dates = mdb.list_stock_daily_dates(engine, end_date=latest)
    if not dates:
        logging.warning("No stock_daily dates found.")
        return

    for m in models:
        _delete_model_runs(engine, model_name=m.name)

    date_workers, model_workers = _worker_budget(int(workers), outer_cap=32)

    def run_date(d: str) -> None:
        for m in models:
            started = mdb.now_ts()
            try:
                logging.info("[%s] compute %s (full)", m.name, d)
                df = m.compute(engine=engine, trade_date=d, workers=int(model_workers))
                n = m.save(engine=engine, trade_date=d, df=df)
                mdb.write_model_run(
                    engine,
                    model_name=m.name,
                    trade_date=d,
                    status="ok",
                    row_count=int(n),
                    message="",
                    started_at=started,
                    finished_at=mdb.now_ts(),
                )
            except Exception as e:  # noqa: BLE001
                mdb.write_model_run(
                    engine,
                    model_name=m.name,
                    trade_date=d,
                    status="error",
                    row_count=0,
                    message=f"{type(e).__name__}: {e}",
                    started_at=started,
                    finished_at=mdb.now_ts(),
                )
                logging.exception("[%s] failed %s (full)", m.name, d)

    if date_workers <= 1 or len(dates) <= 1:
        for d in dates:
            run_date(str(d))
        return

    logging.info(
        "Models full recompute: dates=%d outer_workers=%d model_workers=%d",
        len(dates),
        date_workers,
        model_workers,
    )
    with ThreadPoolExecutor(max_workers=int(date_workers)) as ex:
        futs = {ex.submit(run_date, str(d)): str(d) for d in dates}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:  # noqa: BLE001
                logging.exception("Model full worker crashed for date=%s", futs.get(f))
