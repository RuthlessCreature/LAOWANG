# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import logging
from typing import Optional

import pandas as pd
from tqdm import tqdm

from . import db
from .akshare_client import (
    fetch_daily,
    fetch_float_market_cap_snapshot,
    fetch_stock_list,
    market_cap_score_from_billion,
)
from .indicators import calc_indicators
from .levels import calc_levels
from .scoring_v3 import calc_score_v3
from .settings import Settings


def run_pipeline(settings: Settings, limit_stocks: Optional[int] = None) -> None:
    workers = max(1, int(settings.workers))
    engine = db.make_engine(
        settings.db,
        pool_size=max(5, workers * 2),
        max_overflow=workers * 2,
    )
    if engine.dialect.name == "sqlite" and workers > 1:
        logging.warning("SQLite does not support concurrent writes well; forcing workers=1.")
        workers = 1

    db.init_db(engine)

    stocks = fetch_stock_list()
    if limit_stocks is not None:
        stocks = stocks[:limit_stocks]

    market_cap_map: Optional[dict[str, float]] = None
    try:
        snapshot = fetch_float_market_cap_snapshot()
        if snapshot:
            market_cap_map = snapshot
            logging.info("Loaded float market cap snapshot: %d stocks", len(snapshot))
        else:
            logging.warning("Float market cap snapshot empty; market cap score will default to 0.")
    except Exception:  # noqa: BLE001
        logging.exception("Failed to load float market cap snapshot; market cap score will default to 0.")

    def process_stock(code: str, name: str) -> None:
        try:
            with engine.connect() as conn:
                date_max = db.get_max_date(conn, "stock_daily", code, "date")
                last_score = db.get_max_date(conn, "stock_scores_v3", code, "score_date")

            if date_max:
                start = (dt.datetime.strptime(date_max, "%Y-%m-%d").date() + dt.timedelta(days=1)).strftime("%Y%m%d")
            else:
                start = settings.start_date

            df_new = pd.DataFrame()
            if start <= settings.end_date:
                df_new = fetch_daily(code, start_date=start, end_date=settings.end_date)
            if date_max:
                df_new = df_new[df_new["date"] > date_max]

            with engine.begin() as conn:
                db.upsert_stock_info(conn, code, name)
                inserted = db.upsert_daily(conn, code, df_new)
            if inserted:
                logging.info("%s daily +%d (%s -> %s)", code, inserted, start, settings.end_date)

            with engine.connect() as conn:
                latest_daily = db.get_max_date(conn, "stock_daily", code, "date")
                if not latest_daily:
                    return
                if last_score == latest_daily:
                    return

                df_hist = db.load_daily(
                    conn,
                    code,
                    limit=max(settings.indicator_lookback, settings.level_lookback) + 50,
                )
                last_ind = db.get_max_date(conn, "stock_indicators", code, "date")

            if len(df_hist) < 150:
                return

            df_ind = calc_indicators(df_hist)
            df_ind_to_save = df_ind[df_ind["date"] > last_ind] if last_ind else df_ind

            df_level_hist = df_hist.tail(settings.level_lookback).reset_index(drop=True)
            df_level_ind = df_ind.tail(len(df_level_hist)).reset_index(drop=True)
            support, resistance, support_type, resistance_type = calc_levels(df_level_hist, df_level_ind, settings)

            mcap_score = market_cap_score_from_billion(market_cap_map.get(code)) if market_cap_map else 0
            (
                total,
                s_trend,
                s_pullback,
                s_vp,
                s_rsi,
                s_macd,
                s_base,
                s_space,
                s_mcap,
                status_tags,
                _risk_flags,
            ) = calc_score_v3(
                df_daily=df_hist,
                df_ind=df_ind,
                support_level=support,
                resistance_level=resistance,
                settings=settings,
                market_cap_score=mcap_score,
            )

            with engine.begin() as conn:
                db.upsert_indicators(conn, code, df_ind_to_save)
                db.upsert_levels(
                    conn,
                    code,
                    calc_date=latest_daily,
                    support_level=support,
                    resistance_level=resistance,
                    support_type=support_type,
                    resistance_type=resistance_type,
                )
                db.upsert_score_v3(
                    conn,
                    code,
                    score_date=latest_daily,
                    total_score=total,
                    trend_score=s_trend,
                    pullback_score=s_pullback,
                    volume_price_score=s_vp,
                    rsi_score=s_rsi,
                    macd_score=s_macd,
                    base_structure_score=s_base,
                    space_score=s_space,
                    market_cap_score=s_mcap,
                    status_tags=status_tags,
                )
        except Exception as e:  # noqa: BLE001
            logging.exception("Failed processing %s: %s", code, e)

    if workers <= 1:
        for code, name in tqdm(stocks, desc="Stocks"):
            process_stock(code, name)
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_stock, code, name) for code, name in stocks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Stocks"):
            try:
                f.result()
            except Exception:  # noqa: BLE001
                logging.exception("Worker crashed")


# Backward compatible re-export: pool export lives in a separate module now.
from .pool_export import export_pool  # noqa: E402,F401
