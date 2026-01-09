# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from sqlalchemy import text

from . import db


def export_pool(
    db_target: str,
    output_csv: Path,
    top_n: int,
    min_score: Optional[float] = None,
    require_tags: Optional[List[str]] = None,
    min_resistance_distance: Optional[float] = None,
) -> None:
    db.ensure_parent_dir(output_csv)
    engine = db.make_engine(db_target, pool_size=5, max_overflow=10)
    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(score_date) FROM stock_scores_v3")).fetchone()[0]
        if not latest:
            raise RuntimeError("No scores found; run pipeline first.")

        rows = conn.execute(
            text(
                """
                SELECT
                    s.stock_code,
                    i.name,
                    d.close,
                    l.support_level,
                    l.resistance_level,
                    s.total_score,
                    s.status_tags
                FROM stock_scores_v3 s
                LEFT JOIN stock_info i ON i.stock_code = s.stock_code
                LEFT JOIN stock_daily d ON d.stock_code = s.stock_code AND d.date = s.score_date
                LEFT JOIN stock_levels l ON l.stock_code = s.stock_code AND l.calc_date = s.score_date
                WHERE s.score_date = :latest
                ORDER BY s.total_score DESC
                """
            ),
            {"latest": latest},
        ).fetchall()

        def passes_filters(r) -> bool:
            close = float(r[2]) if r[2] is not None else np.nan
            resistance = float(r[4]) if r[4] is not None else None
            total = float(r[5]) if r[5] is not None else 0.0
            tags_str = r[6] or "[]"
            try:
                parsed = json.loads(tags_str) if isinstance(tags_str, str) else []
                tags = set(parsed) if isinstance(parsed, list) else set()
            except Exception:  # noqa: BLE001
                tags = set()

            if "RISK_FILTERED" in tags:
                return False

            if min_score is not None and total < min_score:
                return False
            if require_tags and not set(require_tags).issubset(tags):
                return False
            if min_resistance_distance is not None and resistance is not None and np.isfinite(close) and close > 0:
                if (resistance - close) / close < min_resistance_distance:
                    return False
            return True

        filtered = [r for r in rows if passes_filters(r)]
        limited = filtered[:top_n]

        import csv

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stock_code", "name", "close", "support_level", "resistance_level", "total_score", "status_tags"])
            for r in limited:
                w.writerow(r)

    logging.info("Exported %d rows to %s", len(limited), output_csv)

