#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score next-day relay candidates with a swappable one-file model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine

import relay_model_file
import relay_strategy_model as relay_model


def _normalize_iso_date(value: str) -> str:
    s = str(value or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if math.isfinite(x):
            return float(x)
    except Exception:
        pass
    return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except Exception:
        return None


def _chunked(rows: Sequence[Dict[str, Any]], size: int = 500) -> Iterable[List[Dict[str, Any]]]:
    buff: List[Dict[str, Any]] = []
    for row in rows:
        buff.append(row)
        if len(buff) >= size:
            yield buff
            buff = []
    if buff:
        yield buff


def ensure_relay_tables(engine: Engine) -> None:
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS model_relay_pool (
            trade_date VARCHAR(10) NOT NULL,
            rank_no INT NULL,
            stock_code VARCHAR(16) NOT NULL,
            stock_name VARCHAR(255) NULL,
            model_prob DOUBLE NULL,
            model_score DOUBLE NULL,
            board_count INT NULL,
            ret1 DOUBLE NULL,
            close DOUBLE NULL,
            next_trade_date VARCHAR(10) NULL,
            next_open DOUBLE NULL,
            is_st INT NULL,
            broken_rate DOUBLE NULL,
            red_rate DOUBLE NULL,
            limit_down_count INT NULL,
            pullback DOUBLE NULL,
            limit_up_count INT NULL,
            max_board INT NULL,
            broken_count INT NULL,
            amount_change5 DOUBLE NULL,
            one_word_flag INT NULL,
            model_version VARCHAR(64) NULL,
            alpha INT NULL,
            default_threshold DOUBLE NULL,
            top_k INT NULL,
            max_board_filter INT NULL,
            gap_min DOUBLE NULL,
            gap_max DOUBLE NULL,
            max_broken_rate_filter DOUBLE NULL,
            min_red_rate_filter DOUBLE NULL,
            max_limit_down_filter INT NULL,
            max_pullback_filter DOUBLE NULL,
            risk_profile VARCHAR(32) NULL,
            updated_at VARCHAR(19) NULL,
            PRIMARY KEY (trade_date, stock_code)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS model_relay_registry (
            model_version VARCHAR(64) PRIMARY KEY,
            model_file VARCHAR(255) NULL,
            model_sha256 VARCHAR(64) NULL,
            model_meta TEXT NULL,
            updated_at VARCHAR(19) NULL
        )
        """,
    ]
    with engine.begin() as conn:
        for stmt in ddl:
            conn.execute(text(stmt))


def delete_range(engine: Engine, start_date: str, end_date: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_relay_pool WHERE trade_date BETWEEN :s AND :e"),
            {"s": start_date, "e": end_date},
        )


def upsert_rows(engine: Engine, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    stmt = text(
        """
        INSERT INTO model_relay_pool (
            trade_date, rank_no, stock_code, stock_name, model_prob, model_score, board_count, ret1, close,
            next_trade_date, next_open, is_st, broken_rate, red_rate, limit_down_count, pullback, limit_up_count,
            max_board, broken_count, amount_change5, one_word_flag, model_version, alpha, default_threshold, top_k,
            max_board_filter, gap_min, gap_max, max_broken_rate_filter, min_red_rate_filter, max_limit_down_filter,
            max_pullback_filter, risk_profile, updated_at
        ) VALUES (
            :trade_date, :rank_no, :stock_code, :stock_name, :model_prob, :model_score, :board_count, :ret1, :close,
            :next_trade_date, :next_open, :is_st, :broken_rate, :red_rate, :limit_down_count, :pullback, :limit_up_count,
            :max_board, :broken_count, :amount_change5, :one_word_flag, :model_version, :alpha, :default_threshold, :top_k,
            :max_board_filter, :gap_min, :gap_max, :max_broken_rate_filter, :min_red_rate_filter, :max_limit_down_filter,
            :max_pullback_filter, :risk_profile, :updated_at
        )
        ON CONFLICT(trade_date, stock_code) DO UPDATE SET
            rank_no=excluded.rank_no,
            stock_name=excluded.stock_name,
            model_prob=excluded.model_prob,
            model_score=excluded.model_score,
            board_count=excluded.board_count,
            ret1=excluded.ret1,
            close=excluded.close,
            next_trade_date=excluded.next_trade_date,
            next_open=excluded.next_open,
            is_st=excluded.is_st,
            broken_rate=excluded.broken_rate,
            red_rate=excluded.red_rate,
            limit_down_count=excluded.limit_down_count,
            pullback=excluded.pullback,
            limit_up_count=excluded.limit_up_count,
            max_board=excluded.max_board,
            broken_count=excluded.broken_count,
            amount_change5=excluded.amount_change5,
            one_word_flag=excluded.one_word_flag,
            model_version=excluded.model_version,
            alpha=excluded.alpha,
            default_threshold=excluded.default_threshold,
            top_k=excluded.top_k,
            max_board_filter=excluded.max_board_filter,
            gap_min=excluded.gap_min,
            gap_max=excluded.gap_max,
            max_broken_rate_filter=excluded.max_broken_rate_filter,
            min_red_rate_filter=excluded.min_red_rate_filter,
            max_limit_down_filter=excluded.max_limit_down_filter,
            max_pullback_filter=excluded.max_pullback_filter,
            risk_profile=excluded.risk_profile,
            updated_at=excluded.updated_at
        """
        if engine.dialect.name == "sqlite"
        else """
        INSERT INTO model_relay_pool (
            trade_date, rank_no, stock_code, stock_name, model_prob, model_score, board_count, ret1, close,
            next_trade_date, next_open, is_st, broken_rate, red_rate, limit_down_count, pullback, limit_up_count,
            max_board, broken_count, amount_change5, one_word_flag, model_version, alpha, default_threshold, top_k,
            max_board_filter, gap_min, gap_max, max_broken_rate_filter, min_red_rate_filter, max_limit_down_filter,
            max_pullback_filter, risk_profile, updated_at
        ) VALUES (
            :trade_date, :rank_no, :stock_code, :stock_name, :model_prob, :model_score, :board_count, :ret1, :close,
            :next_trade_date, :next_open, :is_st, :broken_rate, :red_rate, :limit_down_count, :pullback, :limit_up_count,
            :max_board, :broken_count, :amount_change5, :one_word_flag, :model_version, :alpha, :default_threshold, :top_k,
            :max_board_filter, :gap_min, :gap_max, :max_broken_rate_filter, :min_red_rate_filter, :max_limit_down_filter,
            :max_pullback_filter, :risk_profile, :updated_at
        )
        ON DUPLICATE KEY UPDATE
            rank_no=VALUES(rank_no),
            stock_name=VALUES(stock_name),
            model_prob=VALUES(model_prob),
            model_score=VALUES(model_score),
            board_count=VALUES(board_count),
            ret1=VALUES(ret1),
            close=VALUES(close),
            next_trade_date=VALUES(next_trade_date),
            next_open=VALUES(next_open),
            is_st=VALUES(is_st),
            broken_rate=VALUES(broken_rate),
            red_rate=VALUES(red_rate),
            limit_down_count=VALUES(limit_down_count),
            pullback=VALUES(pullback),
            limit_up_count=VALUES(limit_up_count),
            max_board=VALUES(max_board),
            broken_count=VALUES(broken_count),
            amount_change5=VALUES(amount_change5),
            one_word_flag=VALUES(one_word_flag),
            model_version=VALUES(model_version),
            alpha=VALUES(alpha),
            default_threshold=VALUES(default_threshold),
            top_k=VALUES(top_k),
            max_board_filter=VALUES(max_board_filter),
            gap_min=VALUES(gap_min),
            gap_max=VALUES(gap_max),
            max_broken_rate_filter=VALUES(max_broken_rate_filter),
            min_red_rate_filter=VALUES(min_red_rate_filter),
            max_limit_down_filter=VALUES(max_limit_down_filter),
            max_pullback_filter=VALUES(max_pullback_filter),
            risk_profile=VALUES(risk_profile),
            updated_at=VALUES(updated_at)
        """
    )
    with engine.begin() as conn:
        for chunk in _chunked(rows):
            conn.execute(stmt, chunk)


def upsert_registry(engine: Engine, *, model_version: str, model_file: Path, model_sha: str, meta: Dict[str, Any], now_text: str) -> None:
    stmt = text(
        """
        INSERT INTO model_relay_registry(model_version, model_file, model_sha256, model_meta, updated_at)
        VALUES(:model_version, :model_file, :model_sha256, :model_meta, :updated_at)
        ON CONFLICT(model_version) DO UPDATE SET
            model_file=excluded.model_file,
            model_sha256=excluded.model_sha256,
            model_meta=excluded.model_meta,
            updated_at=excluded.updated_at
        """
        if engine.dialect.name == "sqlite"
        else """
        INSERT INTO model_relay_registry(model_version, model_file, model_sha256, model_meta, updated_at)
        VALUES(:model_version, :model_file, :model_sha256, :model_meta, :updated_at)
        ON DUPLICATE KEY UPDATE
            model_file=VALUES(model_file),
            model_sha256=VALUES(model_sha256),
            model_meta=VALUES(model_meta),
            updated_at=VALUES(updated_at)
        """
    )
    payload = {
        "model_version": model_version,
        "model_file": model_file.as_posix(),
        "model_sha256": model_sha,
        "model_meta": json.dumps(meta or {}, ensure_ascii=False),
        "updated_at": now_text,
    }
    with engine.begin() as conn:
        conn.execute(stmt, payload)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score relay candidates and write model_relay_pool.")
    p.add_argument("--config", default="config.ini")
    p.add_argument("--db-url", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--start-date", default="2025-01-01")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD / YYYYMMDD; default max(stock_daily.date)")
    p.add_argument("--model-file", default=str(relay_model_file.default_model_path()))
    p.add_argument("--full-rebuild", action="store_true", help="semantic flag for full range recompute")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    model_path = Path(args.model_file).expanduser()
    model = relay_model_file.load_model(model_path)
    meta: Dict[str, Any] = dict(model.get("meta") or {})
    model_version = str(meta.get("model_version") or model_path.stem)

    start_date = _normalize_iso_date(args.start_date)
    engine = relay_model.make_engine(relay_model.resolve_db_target(args))
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()
    end_date = _normalize_iso_date(args.end_date) if args.end_date else (str(row[0]) if row and row[0] else None)
    if not end_date:
        raise SystemExit("stock_daily is empty")
    if start_date > end_date:
        raise SystemExit("start-date must be <= end-date")

    ensure_relay_tables(engine)

    buffer_start = (dt.datetime.strptime(start_date, "%Y-%m-%d").date() - dt.timedelta(days=60)).strftime("%Y-%m-%d")
    base = relay_model.load_base_frame(engine, buffer_start, end_date)
    if base.empty:
        delete_range(engine, start_date, end_date)
        logging.warning("[scoring-relay] no base rows in range")
        return 0

    full = relay_model.compute_flags_and_stock_features(base)
    market = relay_model.compute_market_features(full)
    pred_pool, _, _ = relay_model.build_sample_frames(
        full_df=full,
        market_df=market,
        start_iso=start_date,
        end_iso=end_date,
        tp=0.02,
        sl=-0.01,
    )

    delete_range(engine, start_date, end_date)

    if pred_pool.empty:
        logging.warning("[scoring-relay] no relay candidates in range %s~%s", start_date, end_date)
        return 0

    pred_pool = pred_pool.copy()
    pred_pool["date"] = pred_pool["date"].astype(str)
    pred_pool["stock_code"] = pred_pool["stock_code"].astype(str)

    feature_cols = list(model.get("feature_cols") or [])
    missing = [c for c in feature_cols if c not in pred_pool.columns]
    if missing:
        raise SystemExit(f"model feature columns missing in data: {missing[:6]}")

    x_raw = pred_pool[feature_cols].to_numpy(dtype=float)
    raw_prob = relay_model_file.predict_raw_prob(model, x_raw)
    alpha = _safe_int(meta.get("alpha")) or 10
    score = relay_model_file.score_from_raw_prob(model, raw_prob, alpha=alpha)

    pred_pool["model_prob"] = raw_prob
    pred_pool["model_score"] = score
    pred_pool = pred_pool.sort_values(["date", "model_score", "model_prob", "stock_code"], ascending=[True, False, False, True]).copy()
    pred_pool["rank_no"] = pred_pool.groupby("date", sort=True).cumcount() + 1

    default_threshold = relay_model_file.clamp_threshold(_safe_float(meta.get("default_threshold")), 0.75)
    top_k = _safe_int(meta.get("top_k")) or 3
    max_board_filter = _safe_int(meta.get("max_board_filter"))
    gap_min = _safe_float(meta.get("gap_min"))
    gap_max = _safe_float(meta.get("gap_max"))
    max_broken_rate_filter = _safe_float(meta.get("max_broken_rate_filter"))
    min_red_rate_filter = _safe_float(meta.get("min_red_rate_filter"))
    max_limit_down_filter = _safe_int(meta.get("max_limit_down_filter"))
    max_pullback_filter = _safe_float(meta.get("max_pullback_filter"))
    risk_profile = str(meta.get("risk_profile") or "off")

    now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: List[Dict[str, Any]] = []
    for row in pred_pool.itertuples(index=False):
        rows.append(
            {
                "trade_date": str(getattr(row, "date")),
                "rank_no": int(getattr(row, "rank_no")),
                "stock_code": str(getattr(row, "stock_code", "")),
                "stock_name": str(getattr(row, "stock_name", "")),
                "model_prob": _safe_float(getattr(row, "model_prob")),
                "model_score": _safe_float(getattr(row, "model_score")),
                "board_count": _safe_int(getattr(row, "board_count")),
                "ret1": _safe_float(getattr(row, "ret1")),
                "close": _safe_float(getattr(row, "close")),
                "next_trade_date": str(getattr(row, "next_date", "") or ""),
                "next_open": _safe_float(getattr(row, "next_open")),
                "is_st": 1 if bool(getattr(row, "is_st", False)) else 0,
                "broken_rate": _safe_float(getattr(row, "broken_rate")),
                "red_rate": _safe_float(getattr(row, "red_rate")),
                "limit_down_count": _safe_int(getattr(row, "limit_down_count")),
                "pullback": _safe_float(getattr(row, "pullback")),
                "limit_up_count": _safe_int(getattr(row, "limit_up_count")),
                "max_board": _safe_int(getattr(row, "max_board")),
                "broken_count": _safe_int(getattr(row, "broken_count")),
                "amount_change5": _safe_float(getattr(row, "amount_change5")),
                "one_word_flag": 1 if bool(getattr(row, "one_word", False)) else 0,
                "model_version": model_version,
                "alpha": int(alpha),
                "default_threshold": float(default_threshold),
                "top_k": int(top_k),
                "max_board_filter": max_board_filter,
                "gap_min": gap_min,
                "gap_max": gap_max,
                "max_broken_rate_filter": max_broken_rate_filter,
                "min_red_rate_filter": min_red_rate_filter,
                "max_limit_down_filter": max_limit_down_filter,
                "max_pullback_filter": max_pullback_filter,
                "risk_profile": risk_profile,
                "updated_at": now_text,
            }
        )

    upsert_rows(engine, rows)
    model_sha = relay_model_file.file_sha256(model_path)
    upsert_registry(
        engine,
        model_version=model_version,
        model_file=model_path,
        model_sha=model_sha,
        meta=meta,
        now_text=now_text,
    )

    with engine.connect() as conn:
        dates = conn.execute(
            text("SELECT COUNT(DISTINCT trade_date) FROM model_relay_pool WHERE trade_date BETWEEN :s AND :e"),
            {"s": start_date, "e": end_date},
        ).fetchone()
    day_count = int(dates[0] or 0) if dates else 0

    logging.info(
        "[scoring-relay] done range=%s~%s rows=%d dates=%d model=%s full=%s",
        start_date,
        end_date,
        len(rows),
        day_count,
        model_version,
        bool(args.full_rebuild),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

