# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd
from sqlalchemy import text

from a_stock_analyzer import db as adb
from a_stock_analyzer.indicators import calc_indicators
from a_stock_analyzer.levels import calc_levels
from a_stock_analyzer.scoring_v3 import calc_score_v3
from a_stock_analyzer.settings import Settings

from . import db as mdb


def _json_is_risk_filtered(tags_str: object) -> bool:
    # Avoid json import for speed; simple substring is enough for our tags format.
    if tags_str is None:
        return False
    s = str(tags_str)
    return "RISK_FILTERED" in s


@dataclass
class LaowangModel:
    """
    Materialized LAOWANG pool per trade day.

    By default this uses `stock_scores_v3` if that day exists (fast).
    Otherwise it falls back to on-the-fly scoring from `stock_daily` history (slow).
    """

    top_n: int = 200
    min_score: float = 0.0
    workers: int = 16
    name: str = "laowang"
    allow_fallback: bool = False

    def ensure_tables(self, engine) -> None:  # noqa: ANN001
        if engine.dialect.name == "mysql":
            ddl = """
            CREATE TABLE IF NOT EXISTS `model_laowang_pool` (
              `trade_date` VARCHAR(10) NOT NULL,
              `stock_code` VARCHAR(16) NOT NULL,
              `stock_name` VARCHAR(255) NULL,
              `close` DOUBLE NULL,
              `support_level` DOUBLE NULL,
              `resistance_level` DOUBLE NULL,
              `total_score` DOUBLE NULL,
              `status_tags` TEXT NULL,
              `rank_no` INT NULL,
              PRIMARY KEY (`trade_date`, `stock_code`),
              KEY `idx_model_laowang_pool_trade_date` (`trade_date`),
              KEY `idx_model_laowang_pool_score` (`trade_date`, `total_score`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        else:
            ddl = """
            CREATE TABLE IF NOT EXISTS model_laowang_pool (
              trade_date TEXT NOT NULL,
              stock_code TEXT NOT NULL,
              stock_name TEXT NULL,
              close REAL NULL,
              support_level REAL NULL,
              resistance_level REAL NULL,
              total_score REAL NULL,
              status_tags TEXT NULL,
              rank_no INTEGER NULL,
              PRIMARY KEY (trade_date, stock_code)
            );
            """
        mdb.ensure_table(engine, ddl)

    def compute(self, *, engine, trade_date: str, workers: int) -> pd.DataFrame:  # noqa: ANN001
        top_n = max(1, min(5000, int(self.top_n)))
        min_score = float(self.min_score)

        # Prefer pipeline scores when coverage is good; otherwise fallback to
        # on-the-fly scoring from raw daily bars (ensures correctness if the
        # score table is only partially populated).
        with engine.connect() as conn:
            score_cnt = int(
                conn.execute(
                    text("SELECT COUNT(1) FROM stock_scores_v3 WHERE score_date = :d"),
                    {"d": trade_date},
                ).fetchone()[0]
                or 0
            )

        if score_cnt <= 0:
            if self.allow_fallback:
                logging.warning("[laowang] no stock_scores_v3 on %s; using fallback (slow)", trade_date)
                return self._compute_fallback(
                    engine,
                    trade_date=trade_date,
                    top_n=top_n,
                    min_score=min_score,
                    workers=int(workers),
                )
            raise RuntimeError(f"[laowang] no stock_scores_v3 on {trade_date}; run pipeline first.")

        with engine.connect() as conn:
            daily_cnt = int(
                conn.execute(
                    text("SELECT COUNT(DISTINCT stock_code) FROM stock_daily WHERE date = :d"),
                    {"d": trade_date},
                ).fetchone()[0]
                or 0
            )

        cov = float(score_cnt / daily_cnt) if daily_cnt > 0 else 1.0
        # If daily table is large but score coverage is low, the score table is
        # incomplete (common when pipeline was interrupted). Fallback is *very*
        # slow; prefer to fail fast so users re-run pipeline to fill scores.
        if daily_cnt >= 1000 and cov < 0.98:
            if self.allow_fallback:
                logging.warning(
                    "[laowang] score table coverage low on %s: scores=%d daily=%d (%.1f%%). Using fallback (slow).",
                    trade_date,
                    score_cnt,
                    daily_cnt,
                    cov * 100.0,
                )
                return self._compute_fallback(
                    engine,
                    trade_date=trade_date,
                    top_n=top_n,
                    min_score=min_score,
                    workers=int(workers),
                )
            raise RuntimeError(
                f"[laowang] stock_scores_v3 incomplete on {trade_date}: scores={score_cnt} daily={daily_cnt} ({cov*100.0:.1f}%). Re-run pipeline."
            )

        logging.info(
            "[laowang] using stock_scores_v3 on %s: scores=%d daily=%d (%.1f%%)",
            trade_date,
            score_cnt,
            daily_cnt,
            cov * 100.0,
        )
        return self._compute_from_scores_v3(
            engine,
            trade_date=trade_date,
            top_n=top_n,
            min_score=min_score,
        )

    def save(self, *, engine, trade_date: str, df: pd.DataFrame) -> int:  # noqa: ANN001
        cols = [
            "trade_date",
            "stock_code",
            "stock_name",
            "close",
            "support_level",
            "resistance_level",
            "total_score",
            "status_tags",
            "rank_no",
        ]
        rows: List[Dict[str, Any]] = []
        if df is not None and not df.empty:
            for r in df.itertuples(index=False):
                rows.append(
                    {
                        "trade_date": str(getattr(r, "trade_date")),
                        "stock_code": str(getattr(r, "stock_code")),
                        "stock_name": str(getattr(r, "stock_name") or ""),
                        "close": float(getattr(r, "close")) if pd.notna(getattr(r, "close")) else None,
                        "support_level": float(getattr(r, "support_level")) if pd.notna(getattr(r, "support_level")) else None,
                        "resistance_level": float(getattr(r, "resistance_level")) if pd.notna(getattr(r, "resistance_level")) else None,
                        "total_score": float(getattr(r, "total_score")) if pd.notna(getattr(r, "total_score")) else None,
                        "status_tags": str(getattr(r, "status_tags") or ""),
                        "rank_no": int(getattr(r, "rank_no")) if getattr(r, "rank_no", None) is not None else None,
                    }
                )

        with engine.begin() as conn:
            mdb.delete_by_trade_date(conn, "model_laowang_pool", trade_date)
            if rows:
                mdb.bulk_insert(conn, table="model_laowang_pool", cols=cols, key_cols=["trade_date", "stock_code"], rows=rows)
        return int(len(rows))

    def _compute_from_scores_v3(self, engine, *, trade_date: str, top_n: int, min_score: float) -> pd.DataFrame:  # noqa: ANN001
        lim = int(max(top_n * 10, top_n))
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT
                        s.stock_code,
                        COALESCE(i.name, '') AS stock_name,
                        d.close AS close,
                        l.support_level AS support_level,
                        l.resistance_level AS resistance_level,
                        s.total_score AS total_score,
                        s.status_tags AS status_tags
                    FROM stock_scores_v3 s
                    LEFT JOIN stock_info i ON i.stock_code = s.stock_code
                    LEFT JOIN stock_daily d ON d.stock_code = s.stock_code AND d.date = s.score_date
                    LEFT JOIN stock_levels l ON l.stock_code = s.stock_code AND l.calc_date = s.score_date
                    WHERE s.score_date = :d
                      AND s.total_score IS NOT NULL
                      AND s.total_score >= :min_score
                    ORDER BY s.total_score DESC
                    LIMIT :lim
                    """
                ),
                {"d": trade_date, "min_score": float(min_score), "lim": lim},
            ).fetchall()

        cols = ["stock_code", "stock_name", "close", "support_level", "resistance_level", "total_score", "status_tags"]
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return pd.DataFrame(columns=["trade_date", "rank_no"] + cols)

        df = df[~df["status_tags"].apply(_json_is_risk_filtered)]
        df = df.sort_values(["total_score", "stock_code"], ascending=[False, True]).head(int(top_n)).reset_index(drop=True)
        df["rank_no"] = np.arange(1, len(df) + 1, dtype=int)
        df["trade_date"] = trade_date
        df = df[
            [
                "trade_date",
                "rank_no",
                "stock_code",
                "stock_name",
                "close",
                "support_level",
                "resistance_level",
                "total_score",
                "status_tags",
            ]
        ]
        return df

    def _compute_fallback(self, engine, *, trade_date: str, top_n: int, min_score: float, workers: int) -> pd.DataFrame:  # noqa: ANN001
        # This is heavy; intended for small backfill or when score table is absent.
        settings = Settings(db="")
        limit_per_stock = max(int(settings.indicator_lookback), int(settings.level_lookback)) + 50

        with engine.connect() as conn:
            code_rows = conn.execute(
                text(
                    """
                    SELECT d.stock_code, COALESCE(i.name, '') AS stock_name
                    FROM stock_daily d
                    LEFT JOIN stock_info i ON i.stock_code = d.stock_code
                    WHERE d.date = :d
                    """
                ),
                {"d": trade_date},
            ).fetchall()
        codes = [(str(r[0]), str(r[1] or "")) for r in code_rows if r and r[0]]
        if not codes:
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "rank_no",
                    "stock_code",
                    "stock_name",
                    "close",
                    "support_level",
                    "resistance_level",
                    "total_score",
                    "status_tags",
                ]
            )
        name_map = {c: n for c, n in codes}

        df_all = pd.DataFrame()
        if engine.dialect.name == "mysql":
            with engine.connect() as conn:
                df_all = pd.read_sql_query(
                    text(
                        """
                        SELECT stock_code, date, open, high, low, close, volume, amount
                        FROM (
                          SELECT
                            sd.stock_code,
                            sd.date,
                            sd.open,
                            sd.high,
                            sd.low,
                            sd.close,
                            sd.volume,
                            sd.amount,
                            ROW_NUMBER() OVER (PARTITION BY sd.stock_code ORDER BY sd.date DESC) AS rn
                          FROM stock_daily sd
                          INNER JOIN (
                            SELECT DISTINCT stock_code FROM stock_daily WHERE date = :d
                          ) c ON c.stock_code = sd.stock_code
                          WHERE sd.date <= :d
                        ) t
                        WHERE t.rn <= :lim
                        ORDER BY stock_code, date
                        """
                    ),
                    conn,
                    params={"d": trade_date, "lim": int(limit_per_stock)},
                )
        else:
            parts: List[pd.DataFrame] = []
            with engine.connect() as conn:
                for code, _ in codes:
                    h = adb.load_daily_until(conn, code, end_date=trade_date, limit=int(limit_per_stock))
                    if not h.empty:
                        h["stock_code"] = code
                        parts.append(h)
            if parts:
                df_all = pd.concat(parts, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)

        if df_all is None or df_all.empty:
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "rank_no",
                    "stock_code",
                    "stock_name",
                    "close",
                    "support_level",
                    "resistance_level",
                    "total_score",
                    "status_tags",
                ]
            )

        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df_all.columns:
                df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

        def compute_one(stock_code: str, grp: pd.DataFrame) -> Optional[Dict[str, Any]]:
            g = grp.reset_index(drop=True)
            if g.empty or str(g["date"].iloc[-1]) != trade_date or len(g) < 150:
                return None
            df_hist = g[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
            df_ind = calc_indicators(df_hist)
            df_level_hist = df_hist.tail(int(settings.level_lookback)).reset_index(drop=True)
            df_level_ind = df_ind.tail(len(df_level_hist)).reset_index(drop=True)
            support, resistance, _st, _rt = calc_levels(df_level_hist, df_level_ind, settings)
            (
                total,
                _s_trend,
                _s_pull,
                _s_vp,
                _s_rsi,
                _s_macd,
                _s_base,
                _s_space,
                _s_mcap,
                status_tags,
                risk,
            ) = calc_score_v3(
                df_daily=df_hist,
                df_ind=df_ind,
                support_level=support,
                resistance_level=resistance,
                settings=settings,
                market_cap_score=0.0,
            )
            if risk.crash_filtered or risk.high_pos_filtered:
                return None
            if float(total) < float(min_score):
                return None
            close = float(df_hist["close"].iloc[-1]) if pd.notna(df_hist["close"].iloc[-1]) else None
            return {
                "trade_date": trade_date,
                "stock_code": stock_code,
                "stock_name": name_map.get(stock_code, ""),
                "close": close,
                "support_level": support,
                "resistance_level": resistance,
                "total_score": float(total),
                "status_tags": status_tags,
            }

        rows: List[Dict[str, Any]] = []
        groups = list(df_all.groupby("stock_code", sort=False))
        w = max(1, int(workers))
        if w <= 1:
            for code, grp in groups:
                r = compute_one(str(code), grp)
                if r:
                    rows.append(r)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=w) as ex:
                futs = [ex.submit(compute_one, str(code), grp) for code, grp in groups]
                for f in as_completed(futs):
                    r = f.result()
                    if r:
                        rows.append(r)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "rank_no",
                    "stock_code",
                    "stock_name",
                    "close",
                    "support_level",
                    "resistance_level",
                    "total_score",
                    "status_tags",
                ]
            )

        df = pd.DataFrame(rows)
        df = df.sort_values(["total_score", "stock_code"], ascending=[False, True]).head(int(top_n)).reset_index(drop=True)
        df["rank_no"] = np.arange(1, len(df) + 1, dtype=int)
        df = df[
            [
                "trade_date",
                "rank_no",
                "stock_code",
                "stock_name",
                "close",
                "support_level",
                "resistance_level",
                "total_score",
                "status_tags",
            ]
        ]
        return df
