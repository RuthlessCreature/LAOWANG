# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text

from a_stock_analyzer import db as adb

from . import db as mdb

import fhkq


def _to_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


@dataclass
class FhkqModel:
    name: str = "fhkq"

    def ensure_tables(self, engine) -> None:  # noqa: ANN001
        if engine.dialect.name == "mysql":
            ddl = """
            CREATE TABLE IF NOT EXISTS `model_fhkq` (
              `trade_date` VARCHAR(10) NOT NULL,
              `stock_code` VARCHAR(16) NOT NULL,
              `stock_name` VARCHAR(255) NULL,
              `consecutive_limit_down` INT NULL,
              `last_limit_down` INT NULL,
              `volume_ratio` DOUBLE NULL,
              `amount_ratio` DOUBLE NULL,
              `open_board_flag` INT NULL,
              `liquidity_exhaust` INT NULL,
              `fhkq_score` INT NULL,
              `fhkq_level` VARCHAR(8) NULL,
              PRIMARY KEY (`trade_date`, `stock_code`),
              KEY `idx_model_fhkq_trade_date` (`trade_date`),
              KEY `idx_model_fhkq_score` (`trade_date`, `fhkq_score`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
        else:
            ddl = """
            CREATE TABLE IF NOT EXISTS model_fhkq (
              trade_date TEXT NOT NULL,
              stock_code TEXT NOT NULL,
              stock_name TEXT NULL,
              consecutive_limit_down INTEGER NULL,
              last_limit_down INTEGER NULL,
              volume_ratio REAL NULL,
              amount_ratio REAL NULL,
              open_board_flag INTEGER NULL,
              liquidity_exhaust INTEGER NULL,
              fhkq_score INTEGER NULL,
              fhkq_level TEXT NULL,
              PRIMARY KEY (trade_date, stock_code)
            );
            """
        mdb.ensure_table(engine, ddl)

    def compute(self, *, engine, trade_date: str, workers: int) -> pd.DataFrame:  # noqa: ANN001
        td = trade_date
        prev = self._prev_trade_date(engine, td)
        if not prev:
            return pd.DataFrame(columns=list(fhkq.OUTPUT_COLUMNS))

        # Candidate prefilter: only stocks that are limit-down on td.
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT
                      d.stock_code,
                      COALESCE(i.name, '') AS stock_name,
                      d.close AS close,
                      p.close AS prev_close
                    FROM stock_daily d
                    INNER JOIN stock_daily p
                      ON p.stock_code = d.stock_code AND p.date = :prev
                    LEFT JOIN stock_info i
                      ON i.stock_code = d.stock_code
                    WHERE d.date = :d
                    """
                ),
                {"d": td, "prev": prev},
            ).fetchall()

        candidates: List[Tuple[str, str]] = []
        eps = 1e-3
        for r in rows:
            code = str(r[0])
            name = str(r[1] or "")
            close = _to_float(r[2])
            prev_close = _to_float(r[3])
            if not (np.isfinite(close) and np.isfinite(prev_close) and prev_close > 0):
                continue
            if fhkq._is_st_name(name):  # noqa: SLF001
                continue
            limit_pct = float(fhkq._infer_limit_pct(code, name, False))  # noqa: SLF001
            limit_down = float(
                fhkq._round_half_up_2(pd.Series([prev_close * (1.0 - limit_pct)])).iloc[0]  # noqa: SLF001
            )
            if abs(close - limit_down) <= eps:
                candidates.append((code, name))

        if not candidates:
            return pd.DataFrame(columns=list(fhkq.OUTPUT_COLUMNS))

        hist_limit = 120
        name_map = {c: n for c, n in candidates}

        # Load all candidate histories in as few DB round-trips as possible.
        # This avoids per-thread engine.connect() bottlenecks when workers >> pool_size.
        df_all = pd.DataFrame()
        if engine.dialect.name == "mysql":
            codes_only = [c for c, _n in candidates]
            with engine.connect() as conn:
                q = (
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
                          WHERE sd.stock_code IN :codes
                            AND sd.date <= :d
                        ) t
                        WHERE t.rn <= :lim
                        ORDER BY stock_code, date
                        """
                    )
                    .bindparams(bindparam("codes", expanding=True))
                )
                df_all = pd.read_sql_query(q, conn, params={"codes": codes_only, "d": td, "lim": int(hist_limit)})
        else:
            parts: List[pd.DataFrame] = []
            with engine.connect() as conn:
                for code, _name in candidates:
                    hist = adb.load_daily_until(conn, code, end_date=td, limit=hist_limit)
                    if hist is None or hist.empty:
                        continue
                    hist = hist.copy()
                    hist["stock_code"] = code
                    parts.append(hist)
            if parts:
                df_all = pd.concat(parts, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)

        if df_all is None or df_all.empty:
            return pd.DataFrame(columns=list(fhkq.OUTPUT_COLUMNS))

        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df_all.columns:
                df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

        groups = list(df_all.groupby("stock_code", sort=False))

        def process_one(code: str, grp: pd.DataFrame) -> Optional[Dict[str, Any]]:
            try:
                hist = grp[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
                return fhkq._calc_fhkq_for_one_stock(  # noqa: SLF001
                    hist,
                    trade_date=td,
                    stock_code=code,
                    stock_name=name_map.get(code, ""),
                )
            except Exception:
                return None

        out_rows: List[Dict[str, Any]] = []
        w = max(1, int(workers))
        if w <= 1 or len(groups) <= 1:
            for code, grp in groups:
                r = process_one(str(code), grp)
                if r:
                    out_rows.append(r)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            w2 = min(int(w), len(groups), 64)
            with ThreadPoolExecutor(max_workers=w2) as ex:
                futs = [ex.submit(process_one, str(code), grp) for code, grp in groups]
                for f in as_completed(futs):
                    r = f.result()
                    if r:
                        out_rows.append(r)

        out = pd.DataFrame(out_rows, columns=list(fhkq.OUTPUT_COLUMNS))
        if out.empty:
            return out
        out = out.sort_values(["fhkq_score", "consecutive_limit_down", "stock_code"], ascending=[False, False, True]).reset_index(drop=True)
        return out

    def save(self, *, engine, trade_date: str, df: pd.DataFrame) -> int:  # noqa: ANN001
        cols = list(fhkq.OUTPUT_COLUMNS)
        rows: List[Dict[str, Any]] = []
        if df is not None and not df.empty:
            for r in df.itertuples(index=False):
                rows.append({c: getattr(r, c, None) for c in cols})

        with engine.begin() as conn:
            mdb.delete_by_trade_date(conn, "model_fhkq", trade_date)
            if rows:
                mdb.bulk_insert(conn, table="model_fhkq", cols=cols, key_cols=["trade_date", "stock_code"], rows=rows)
        return int(len(rows))

    def _prev_trade_date(self, engine, trade_date: str) -> Optional[str]:  # noqa: ANN001
        with engine.connect() as conn:
            row = conn.execute(text("SELECT MAX(date) FROM stock_daily WHERE date < :d"), {"d": trade_date}).fetchone()
        if not row:
            return None
        return str(row[0]) if row[0] else None
