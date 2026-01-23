#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init.py

项目首次部署用的唯一入口：
- 自动解析 config.ini / 环境变量 / CLI DB 参数
- 如为 MySQL，会先 CREATE DATABASE IF NOT EXISTS
- 创建所需表：stock_info / stock_daily / stock_scores_v3 / stock_levels / model_laowang_pool / model_fhkq / stock_scores_stwg / model_stwg_pool
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError


DEFAULT_DB = "data/stock.db"


@dataclass
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    charset: str = "utf8mb4"


@dataclass
class AppConfig:
    db_url: Optional[str] = None
    mysql: MySQLConfig = field(default_factory=MySQLConfig)


def load_config(path: Path) -> AppConfig:
    import configparser

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    db_url = parser.get("database", "db_url", fallback=None)
    db_url = db_url.strip() if db_url else None
    mysql = MySQLConfig(
        host=parser.get("mysql", "host", fallback="127.0.0.1").strip() or "127.0.0.1",
        port=parser.getint("mysql", "port", fallback=3306),
        user=parser.get("mysql", "user", fallback="").strip(),
        password=parser.get("mysql", "password", fallback=""),
        database=parser.get("mysql", "database", fallback="").strip(),
        charset=parser.get("mysql", "charset", fallback="utf8mb4").strip() or "utf8mb4",
    )
    return AppConfig(db_url=db_url, mysql=mysql)


def build_mysql_url(cfg: MySQLConfig) -> Optional[str]:
    if not (cfg.user and cfg.database):
        return None
    from urllib.parse import quote_plus

    user = quote_plus(cfg.user)
    password = quote_plus(cfg.password or "")
    auth = f"{user}:{password}" if password else user
    return f"mysql+pymysql://{auth}@{cfg.host}:{int(cfg.port)}/{cfg.database}?charset={cfg.charset}"


def resolve_db_target(args: argparse.Namespace) -> str:
    if getattr(args, "db_url", None):
        return str(args.db_url)
    import os

    env = os.getenv("ASTOCK_DB_URL")
    if env and env.strip():
        return env.strip()
    if getattr(args, "db", None):
        return str(args.db)
    cfg_path = getattr(args, "config", None)
    cfg_file = Path(cfg_path) if cfg_path else Path("config.ini")
    if cfg_file.exists():
        cfg = load_config(cfg_file)
        if cfg.db_url:
            return cfg.db_url
        url = build_mysql_url(cfg.mysql)
        if url:
            return url
    return DEFAULT_DB


def ensure_database(db_target: str) -> str:
    if "://" not in db_target and db_target.endswith(".db"):
        path = Path(db_target).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"
    from sqlalchemy.engine.url import make_url

    try:
        url = make_url(db_target)
    except Exception:  # noqa: BLE001
        return db_target
    if url.get_backend_name() != "mysql":
        return db_target
    database = (url.database or "").strip()
    if not database:
        return db_target
    server = url.set(database=None)
    connect_args = {}
    server_str = str(server)
    engine = create_engine(server_str, pool_size=1, max_overflow=0, pool_pre_ping=True, connect_args=connect_args)
    safe_db = database.replace("`", "")
    charset = url.query.get("charset", "utf8mb4")
    charset_safe = "".join(ch for ch in charset if ch.isalnum() or ch in {"_", "-"}).lower() or "utf8mb4"
    try:
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{safe_db}` CHARACTER SET {charset_safe}"))
    except Exception as exc:  # noqa: BLE001
        logging.warning("无法创建数据库 %s（可能无权限）：%s", safe_db, exc)
        return db_target
    finally:
        engine.dispose()
    return str(url)


def make_engine(db_target: str) -> Engine:
    connect_args = {}
    if db_target.startswith("sqlite:///"):
        connect_args["check_same_thread"] = False
    engine = create_engine(db_target, pool_pre_ping=True, pool_recycle=3600, connect_args=connect_args)
    if engine.dialect.name == "sqlite":
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.close()
    return engine


DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS stock_info (
        stock_code VARCHAR(16) PRIMARY KEY,
        name VARCHAR(255)
    )
    """,
    """
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_levels (
        stock_code VARCHAR(16) NOT NULL,
        calc_date VARCHAR(10) NOT NULL,
        support_level DOUBLE NULL,
        resistance_level DOUBLE NULL,
        support_type VARCHAR(32) NULL,
        resistance_type VARCHAR(32) NULL,
        PRIMARY KEY (stock_code, calc_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_scores_v3 (
        stock_code VARCHAR(16) NOT NULL,
        score_date VARCHAR(10) NOT NULL,
        total_score DOUBLE NULL,
        trend_score DOUBLE NULL,
        pullback_score DOUBLE NULL,
        volume_price_score DOUBLE NULL,
        rsi_score DOUBLE NULL,
        macd_score DOUBLE NULL,
        base_structure_score DOUBLE NULL,
        space_score DOUBLE NULL,
        market_cap_score DOUBLE NULL,
        status_tags TEXT NULL,
        PRIMARY KEY (stock_code, score_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_laowang_pool (
        trade_date VARCHAR(10) NOT NULL,
        rank_no INT NULL,
        stock_code VARCHAR(16) NOT NULL,
        stock_name VARCHAR(255) NULL,
        close DOUBLE NULL,
        support_level DOUBLE NULL,
        resistance_level DOUBLE NULL,
        total_score DOUBLE NULL,
        status_tags TEXT NULL,
        PRIMARY KEY (trade_date, stock_code)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_scores_stwg (
        stock_code VARCHAR(16) NOT NULL,
        score_date VARCHAR(10) NOT NULL,
        total_score DOUBLE NULL,
        trend_base_score DOUBLE NULL,
        stageA_structure_score DOUBLE NULL,
        stageB_compression_score DOUBLE NULL,
        platform_support_score DOUBLE NULL,
        breakout_confirmation_score DOUBLE NULL,
        rsi_state_score DOUBLE NULL,
        space_after_breakout_score DOUBLE NULL,
        status_tags TEXT NULL,
        PRIMARY KEY (stock_code, score_date)
    )
    """,
    """
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
    )
    """,
    """
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
    )
    """,
]


def _is_table_exists_error(exc: SQLAlchemyError) -> bool:
    msg = ""
    orig = getattr(exc, "orig", exc)
    if hasattr(orig, "args") and orig.args:
        msg = " ".join(str(a) for a in orig.args if a)
    if not msg:
        msg = str(exc)
    msg = msg.lower()
    return "already exists" in msg or "exists" in msg and "table" in msg


def run_init(engine: Engine) -> None:
    with engine.begin() as conn:
        for idx, stmt in enumerate(DDL_STATEMENTS, start=1):
            try:
                conn.execute(text(stmt))
            except SQLAlchemyError as exc:
                if _is_table_exists_error(exc):
                    logging.info("DDL #%d 表已存在，跳过", idx)
                    continue
                logging.error("DDL #%d 执行失败，需要处理：%s", idx, exc)
                raise
    logging.info("数据库表创建完成（如已存在则跳过）")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="初始化数据库表结构")
    parser.add_argument("--config", default=None, help="config.ini 路径")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL")
    parser.add_argument("--db", default=None, help="SQLite 文件路径")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    db_target = resolve_db_target(args)
    normalized = ensure_database(db_target)
    engine = make_engine(normalized)
    run_init(engine)
    logging.info("Init OK: %s", normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
