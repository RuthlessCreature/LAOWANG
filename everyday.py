#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
everyday.py

单文件“每日流程”脚本：自动执行
1) getData.py：补齐 K 线
2) scoring_laowang.py：更新老王评分
3) scoring_stwg.py：更新缩头乌龟评分
4) scoring_fhkq.py：更新粪海狂蛆连板模型

- 自动根据 stock_daily 中的最新日期决定 start-date
- 用于 CLI 手动运行，也用于 ui.py 的后台计划任务
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine

import scoring_fhkq as fhkq_mod
import getData as getdata_mod
import scoring_laowang as laowang_mod
import scoring_stwg as stwg_mod


def _normalize_yyyymmdd(date_str: str) -> str:
    s = str(date_str or "").strip()
    if not s:
        raise ValueError("date required")
    if len(s) == 8 and s.isdigit():
        return s
    return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")


def _yyyymmdd_to_iso(s: str) -> str:
    if "-" in s:
        return s
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Invalid date: {s}")
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def _ensure_sqlalchemy_url(db_target: str) -> str:
    tgt = str(db_target or "").strip()
    if "://" in tgt:
        return tgt
    path = Path(tgt).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def _make_engine(db_target: str) -> Engine:
    url = _ensure_sqlalchemy_url(db_target)
    connect_args = {}
    if url.startswith("sqlite:///"):
        connect_args["check_same_thread"] = False
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600, connect_args=connect_args)
    if engine.dialect.name == "sqlite":
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.close()
    return engine


def _max_stock_daily(engine: Engine) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()
    if not row or not row[0]:
        return None
    return str(row[0])


def _build_common_cli(args: argparse.Namespace) -> List[str]:
    cli: List[str] = []
    if getattr(args, "config", None):
        cli.extend(["--config", str(args.config)])
    if getattr(args, "db_url", None):
        cli.extend(["--db-url", str(args.db_url)])
    elif getattr(args, "db", None):
        cli.extend(["--db", str(args.db)])
    return cli


def _run_pipeline(args: argparse.Namespace, *, setup_logging: bool) -> None:
    if setup_logging:
        logging.basicConfig(
            level=getattr(logging, str(args.log_level).upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(message)s",
        )

    db_target = getdata_mod.resolve_db_target(args)
    engine = _make_engine(db_target)

    prev_latest_iso = _max_stock_daily(engine)
    today_yyyymmdd = dt.date.today().strftime("%Y%m%d")

    if prev_latest_iso:
        prev_date = dt.datetime.strptime(prev_latest_iso, "%Y-%m-%d").date()
        fetch_start = (prev_date + dt.timedelta(days=1)).strftime("%Y%m%d")
    else:
        fetch_start = _normalize_yyyymmdd(args.initial_start_date)

    need_fetch = fetch_start <= today_yyyymmdd
    base_cli = _build_common_cli(args)

    if need_fetch:
        get_cli = base_cli + [
            "--start-date",
            fetch_start,
            "--end-date",
            today_yyyymmdd,
            "--workers",
            str(args.getdata_workers),
        ]
        logging.info("[everyday] getData: %s -> %s", fetch_start, today_yyyymmdd)
        getdata_mod.main(get_cli)
    else:
        logging.info("[everyday] getData: K 线已最新，跳过")

    latest_iso = _max_stock_daily(engine)
    if not latest_iso:
        logging.warning("[everyday] 数据库仍没有 K 线，终止")
        return

    if need_fetch:
        score_start_yyyymmdd = fetch_start
    else:
        if prev_latest_iso:
            score_start_yyyymmdd = prev_latest_iso.replace("-", "")
        else:
            score_start_yyyymmdd = today_yyyymmdd

    score_start_iso = _yyyymmdd_to_iso(score_start_yyyymmdd)
    score_end_iso = latest_iso

    lw_cli = base_cli + [
        "--start-date",
        score_start_iso,
        "--end-date",
        score_end_iso,
        "--workers",
        str(args.laowang_workers),
        "--top",
        str(args.laowang_top),
        "--min-score",
        str(args.laowang_min_score),
    ]
    logging.info("[everyday] scoring_laowang: %s -> %s", score_start_iso, score_end_iso)
    laowang_mod.main(lw_cli)

    st_cli = base_cli + [
        "--start-date",
        score_start_iso,
        "--end-date",
        score_end_iso,
        "--workers",
        str(args.stwg_workers),
        "--top",
        str(args.stwg_top),
        "--min-score",
        str(args.stwg_min_score),
    ]
    logging.info("[everyday] scoring_stwg: %s -> %s", score_start_iso, score_end_iso)
    stwg_mod.main(st_cli)

    fk_cli = base_cli + [
        "--start-date",
        score_start_iso,
        "--end-date",
        score_end_iso,
        "--workers",
        str(args.fhkq_workers),
    ]
    logging.info("[everyday] scoring_fhkq: %s -> %s", score_start_iso, score_end_iso)
    fhkq_mod.main(fk_cli)

    logging.info("[everyday] 完成：latest=%s", score_end_iso)


def run_once(
    *,
    config: Optional[str],
    db_url: Optional[str],
    db: Optional[str],
    initial_start_date: str,
    getdata_workers: int,
    laowang_workers: int,
    stwg_workers: int,
    fhkq_workers: int,
    laowang_top: int,
    laowang_min_score: float,
    stwg_top: int,
    stwg_min_score: float,
) -> None:
    args = argparse.Namespace(
        config=config,
        db_url=db_url,
        db=db,
        initial_start_date=initial_start_date,
        getdata_workers=int(getdata_workers),
        laowang_workers=int(laowang_workers),
        stwg_workers=int(stwg_workers),
        fhkq_workers=int(fhkq_workers),
        laowang_top=int(laowang_top),
        laowang_min_score=float(laowang_min_score),
        stwg_top=int(stwg_top),
        stwg_min_score=float(stwg_min_score),
        log_level=logging.getLevelName(logging.getLogger().level),
    )
    _run_pipeline(args, setup_logging=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="每日自动流程（getData → scoring_laowang → scoring_stwg → scoring_fhkq）")
    parser.add_argument("--config", default=None, help="config.ini 路径")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL")
    parser.add_argument("--db", default=None, help="SQLite 文件")
    parser.add_argument("--initial-start-date", default="2000-01-01", help="数据库为空时的起始日期")
    parser.add_argument("--getdata-workers", type=int, default=16)
    parser.add_argument("--laowang-workers", type=int, default=16)
    parser.add_argument("--stwg-workers", type=int, default=16)
    parser.add_argument("--fhkq-workers", type=int, default=8)
    parser.add_argument("--laowang-top", type=int, default=200)
    parser.add_argument("--laowang-min-score", type=float, default=60.0)
    parser.add_argument("--stwg-top", type=int, default=150)
    parser.add_argument("--stwg-min-score", type=float, default=55.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    _run_pipeline(args, setup_logging=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
