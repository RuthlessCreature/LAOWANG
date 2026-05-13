#!/usr/bin/env python3
"""Smoke test for generate_trade_plan.py.

Run against a fixture SQLite DB to verify:
1. ordinary_next_day_plan.csv has correct columns and pass/fail logic
2. ordinary rows include T+1 buy and T+2+ sell conditions
3. signal_audit.csv has all signal rows
4. DB upsert into strategy_signal_log works (SQLite)
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_trade_plan import (
    ORDINARY_PLAN_COLUMNS,
    SIGNAL_AUDIT_COLUMNS,
    generate_trade_plan,
    load_config,
    make_engine,
    resolve_db_url,
)


FIXTURE_TRADE_DATE = "2026-04-28"


def create_fixture_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS stock_info (
            stock_code VARCHAR(16) PRIMARY KEY,
            name VARCHAR(255)
        );

        CREATE TABLE IF NOT EXISTS stock_daily (
            stock_code VARCHAR(16) NOT NULL,
            date VARCHAR(10) NOT NULL,
            open DOUBLE NULL,
            high DOUBLE NULL,
            low DOUBLE NULL,
            close DOUBLE NULL,
            volume DOUBLE NULL,
            amount DOUBLE NULL,
            PRIMARY KEY (stock_code, date)
        );

        CREATE TABLE IF NOT EXISTS stock_scores_v3 (
            stock_code VARCHAR(16) NOT NULL,
            score_date VARCHAR(10) NOT NULL,
            total_score DOUBLE NULL,
            status_tags TEXT NULL,
            support_level DOUBLE NULL,
            resistance_level DOUBLE NULL,
            PRIMARY KEY (stock_code, score_date)
        );

        CREATE TABLE IF NOT EXISTS model_laowang_pool (
            trade_date VARCHAR(10) NOT NULL,
            rank_no INT NULL,
            stock_code VARCHAR(16) NOT NULL,
            stock_name VARCHAR(255) NULL,
            close DOUBLE NULL,
            total_score DOUBLE NULL,
            support_level DOUBLE NULL,
            resistance_level DOUBLE NULL,
            status_tags TEXT NULL,
            PRIMARY KEY (trade_date, stock_code)
        );

        CREATE TABLE IF NOT EXISTS stock_scores_stwg (
            stock_code VARCHAR(16) NOT NULL,
            score_date VARCHAR(10) NOT NULL,
            total_score DOUBLE NULL,
            stageB_compression_score DOUBLE NULL,
            breakout_confirmation_score DOUBLE NULL,
            status_tags TEXT NULL,
            PRIMARY KEY (stock_code, score_date)
        );

        CREATE TABLE IF NOT EXISTS model_stwg_pool (
            trade_date VARCHAR(10) NOT NULL,
            rank_no INT NULL,
            stock_code VARCHAR(16) NOT NULL,
            stock_name VARCHAR(255) NULL,
            close DOUBLE NULL,
            total_score DOUBLE NULL,
            stageB_compression_score DOUBLE NULL,
            breakout_confirmation_score DOUBLE NULL,
            status_tags TEXT NULL,
            PRIMARY KEY (trade_date, stock_code)
        );

        CREATE TABLE IF NOT EXISTS stock_scores_ywcx (
            stock_code VARCHAR(16) NOT NULL,
            score_date VARCHAR(10) NOT NULL,
            total_score DOUBLE NULL,
            weak_position_score DOUBLE NULL,
            volume_dry_score DOUBLE NULL,
            low_volatility_score DOUBLE NULL,
            status_tags TEXT NULL,
            PRIMARY KEY (stock_code, score_date)
        );

        CREATE TABLE IF NOT EXISTS model_ywcx_pool (
            trade_date VARCHAR(10) NOT NULL,
            rank_no INT NULL,
            stock_code VARCHAR(16) NOT NULL,
            stock_name VARCHAR(255) NULL,
            close DOUBLE NULL,
            total_score DOUBLE NULL,
            weak_position_score DOUBLE NULL,
            volume_dry_score DOUBLE NULL,
            low_volatility_score DOUBLE NULL,
            status_tags TEXT NULL,
            PRIMARY KEY (trade_date, stock_code)
        );

        CREATE TABLE IF NOT EXISTS model_fhkq (
            trade_date VARCHAR(10) NOT NULL,
            stock_code VARCHAR(16) NOT NULL,
            stock_name VARCHAR(255) NULL,
            consecutive_limit_down INT NULL,
            last_limit_down INT NULL,
            volume_ratio DOUBLE NULL,
            amount_ratio DOUBLE NULL,
            open_board_flag INT NULL,
            liquidity_exhaust INT NULL,
            fhkq_score INT NULL,
            fhkq_level VARCHAR(8) NULL,
            PRIMARY KEY (trade_date, stock_code)
        );

        CREATE TABLE IF NOT EXISTS strategy_signal_log (
            signal_date VARCHAR(10) NOT NULL,
            model VARCHAR(32) NOT NULL,
            stock_code VARCHAR(16) NOT NULL,
            stock_name VARCHAR(255) NULL,
            score DOUBLE NULL,
            features_json TEXT NULL,
            model_version VARCHAR(64) NULL,
            action_plan_json TEXT NULL,
            created_at VARCHAR(19) NULL,
            PRIMARY KEY (signal_date, model, stock_code)
        );

        CREATE TABLE IF NOT EXISTS strategy_trade_journal (
            trade_id VARCHAR(64) PRIMARY KEY,
            signal_date VARCHAR(10) NOT NULL,
            model VARCHAR(32) NOT NULL,
            stock_code VARCHAR(16) NOT NULL,
            stock_name VARCHAR(255) NULL,
            buy_date VARCHAR(10) NULL,
            buy_price DOUBLE NULL,
            buy_shares BIGINT NULL,
            buy_amount DOUBLE NULL,
            buy_fee DOUBLE NULL,
            planned_position DOUBLE NULL,
            sell_date VARCHAR(10) NULL,
            sell_price DOUBLE NULL,
            sell_amount DOUBLE NULL,
            sell_fee DOUBLE NULL,
            pnl DOUBLE NULL,
            pnl_pct DOUBLE NULL,
            exit_reason VARCHAR(255) NULL,
            trade_status VARCHAR(16) NULL,
            manual_notes TEXT NULL,
            created_at VARCHAR(19) NULL,
            updated_at VARCHAR(19) NULL
        );
    """)

    cur.execute(
        """
        INSERT INTO stock_info (stock_code, name) VALUES
        ('000001', '平安银行'),
        ('000002', '万科A'),
        ('600000', '浦发银行'),
        ('600519', '贵州茅台'),
        ('000005', '世纪星源')
        """
    )

    for code, close_price in [
        ("000001", 12.50),
        ("000002", 8.30),
        ("600000", 8.10),
        ("600519", 1680.00),
        ("000005", 3.20),
    ]:
        for days_ago in range(30, 0, -1):
            from datetime import timedelta

            d = date.today() - timedelta(days=days_ago)
            cur.execute(
                """
                INSERT INTO stock_daily (stock_code, date, open, high, low, close, volume, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    code,
                    d.isoformat(),
                    close_price * 0.98,
                    close_price * 1.02,
                    close_price * 0.97,
                    close_price,
                    1000000,
                    close_price * 1000000,
                ),
            )

    cur.execute(
        """
        INSERT INTO model_laowang_pool
        (trade_date, rank_no, stock_code, stock_name, close, total_score, support_level, resistance_level, status_tags)
        VALUES
        (?, 1, '000001', '平安银行', 12.50, 72.0, 12.0, 13.5, 'TREND_UP|LOW_BASE|SPACE_OK'),
        (?, 2, '000002', '万科A', 8.30, 68.0, 8.0, 9.0, 'TREND_UP|LOW_BASE|RISK_FILTERED'),
        (?, 3, '600000', '浦发银行', 8.10, 55.0, 7.8, 8.8, 'TREND_DOWN')
        """,
        (FIXTURE_TRADE_DATE, FIXTURE_TRADE_DATE, FIXTURE_TRADE_DATE),
    )

    cur.execute(
        """
        INSERT INTO model_stwg_pool
        (trade_date, rank_no, stock_code, stock_name, close, total_score, stageB_compression_score, breakout_confirmation_score, status_tags)
        VALUES
        (?, 1, '600519', '贵州茅台', 1680.0, 75.0, 0.8, 0.9, 'STAGE_B_COMPRESSED|BREAKOUT_R|VOLUME_EXPANSION'),
        (?, 2, '000001', '平安银行', 12.50, 65.0, 0.6, 0.5, 'STAGE_B_COMPRESSED')
        """,
        (FIXTURE_TRADE_DATE, FIXTURE_TRADE_DATE),
    )

    cur.execute(
        """
        INSERT INTO model_ywcx_pool
        (trade_date, rank_no, stock_code, stock_name, close, total_score, weak_position_score, volume_dry_score, low_volatility_score, status_tags)
        VALUES
        (?, 1, '000005', '世纪星源', 3.20, 70.0, 0.8, 0.7, 0.6, 'BROKEN_IPO|NEAR_IPO_LOW|VOLUME_DRY|JUST_ABOVE_MA5')
        """,
        (FIXTURE_TRADE_DATE,),
    )

    cur.execute(
        """
        INSERT INTO model_fhkq
        (trade_date, stock_code, stock_name, consecutive_limit_down, last_limit_down, volume_ratio, amount_ratio, open_board_flag, liquidity_exhaust, fhkq_score, fhkq_level)
        VALUES
        (?, '000005', '世纪星源', 3, 1, 0.5, 0.3, 1, 1, 85, 'L3')
        """,
        (FIXTURE_TRADE_DATE,),
    )

    conn.commit()
    conn.close()


def validate_csv_columns(path: Path, expected_columns: list[str], name: str) -> list[str]:
    errors = []
    if not path.exists():
        errors.append(f"{name}: file not found at {path}")
        return errors
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        actual_cols = reader.fieldnames or []
        missing = set(expected_columns) - set(actual_cols)
        extra = set(actual_cols) - set(expected_columns)
        if missing:
            errors.append(f"{name}: missing columns: {sorted(missing)}")
        if extra:
            errors.append(f"{name}: extra columns: {sorted(extra)}")
    return errors


def validate_plan_rows(path: Path, model_name: str) -> list[str]:
    errors = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            status = row.get("entry_filter_status", "")
            planned_action = row.get("planned_action", "")
            skip_reason = row.get("skip_reason", "")
            score = row.get("score", "")
            if status == "pass" and skip_reason:
                errors.append(
                    f"{model_name} row {i}: entry_filter_status=pass but skip_reason='{skip_reason}'"
                )
            if status == "fail" and not skip_reason:
                errors.append(
                    f"{model_name} row {i}: entry_filter_status=fail but skip_reason is empty"
                )
            if planned_action not in ("skip", "conditional_buy"):
                errors.append(
                    f"{model_name} row {i}: planned_action='{planned_action}' not in (skip, conditional_buy)"
                )
            if not row.get("t1_buy_condition"):
                errors.append(f"{model_name} row {i}: missing t1_buy_condition")
            if not row.get("t2_sell_condition"):
                errors.append(f"{model_name} row {i}: missing t2_sell_condition")
    return errors


def check_signal_log(engine_path: str, trade_date: str) -> list[str]:
    errors = []
    conn = sqlite3.connect(engine_path.replace("sqlite:///", ""))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT model, COUNT(*) FROM strategy_signal_log
        WHERE signal_date = ?
        GROUP BY model
        """,
        (trade_date,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        errors.append("strategy_signal_log: no rows found after upsert")
    else:
        print(f"  strategy_signal_log rows by model: {dict(rows)}")
    return errors


def run() -> int:
    fixture_dir = Path(tempfile.mkdtemp(prefix="laowang_smoke_"))
    fixture_db = fixture_dir / "fixture.db"
    output_dir = fixture_dir / "output"

    try:
        create_fixture_db(fixture_db)
        print(f"Fixture DB created: {fixture_db}")

        db_url = f"sqlite:///{fixture_db}"
        engine = make_engine(db_url)
        trade_date = FIXTURE_TRADE_DATE

        result = generate_trade_plan(engine, trade_date, output_dir, write_db=True)
        print(f"generate_trade_plan result: {json.dumps(result, ensure_ascii=False, indent=2)}")

        all_errors: list[str] = []

        ordinary_csv = output_dir / "ordinary_next_day_plan.csv"
        audit_csv = output_dir / "signal_audit.csv"

        all_errors.extend(validate_csv_columns(ordinary_csv, ORDINARY_PLAN_COLUMNS, "ordinary_next_day_plan"))
        all_errors.extend(validate_csv_columns(audit_csv, SIGNAL_AUDIT_COLUMNS, "signal_audit"))

        all_errors.extend(validate_plan_rows(ordinary_csv, "laowang"))
        all_errors.extend(validate_plan_rows(ordinary_csv, "stwg"))
        all_errors.extend(validate_plan_rows(ordinary_csv, "ywcx"))
        all_errors.extend(validate_plan_rows(ordinary_csv, "fhkq"))

        all_errors.extend(check_signal_log(db_url, trade_date))

        if all_errors:
            print("\n=== SMOKE TEST FAILED ===")
            for err in all_errors:
                print(f"  ERROR: {err}")
            return 1
        else:
            print("\n=== SMOKE TEST PASSED ===")
            return 0

    finally:
        shutil.rmtree(fixture_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(run())
