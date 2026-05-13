#!/usr/bin/env python3
"""Relay intraday minute features for T+1 auction and盘中触发.

Computes auction and intraday features from stock_minute data for Relay
watchlist stocks. Results are written to relay_intraday_features table.

Features computed
---------------
Auction (T+1 9:15-9:25 proxy):
  auction_gap          : (first_open - prev_close) / prev_close
  auction_amount       : sum of amount in first 3 5-min bars
  auction_amount_ratio  : auction_amount / prev_20d_avg_amount
  auction_price_slope  : (last_price - first_price) / first_price over auction window

Intraday (9:30-15:00):
  first_5m_ret         : (close_at_935 - open_at_930) / open_at_930
  first_15m_ret        : (close_at_945 - open_at_930) / open_at_930
  first_15m_low_drawdown: (low_min_930_945 - open_at_930) / open_at_930
  vwap_15m             : sum(amount_930_945) / sum(volume_930_945)
  vwap_reclaim         : (close_15m - vwap_15m) / vwap_15m
  reclaim_vwap_flag    : 1 if close_15m > vwap_15m else 0
  reclaim_open_flag    : 1 if close_15m > open_930 else 0
  re_seal_flag         : 1 if hit limit-up then reopened within 30 min
  re_seal_minutes      : minutes from first seal to reopen
  reopen_count         : number of times price <= 1% above open then recovers
  intraday_break_vwap_count : # of times price crossed VWAP from above
  volume_efficiency_15m: (realized_volume_930_945) / (theoretical_volume)

Data dependency
---------------
prev_close   : stock_daily close on (trade_date - 1 trading day)
amount_ma20  : 20-day avg amount from stock_daily (pre-computed by generate_trade_plan)
minute data  : stock_minute for trade_date at frequency='5'
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.engine import Engine


MINUTE_TABLE = "stock_minute"
FEATURES_TABLE = "relay_intraday_features"
DUMMY_DATE = "1970-01-01"


def _resolve_db_url(args: argparse.Namespace, cfg: dict[str, Any]) -> str:
    if args.db_url:
        return args.db_url
    env = os.getenv("ASTOCK_DB_URL") or os.getenv("DATABASE_URL")
    if env and env.strip():
        return env.strip()
    db_arg = args.db or cfg.get("db")
    if db_arg and "://" in str(db_arg):
        return str(db_arg)
    if db_arg:
        return f"sqlite:///{Path(db_arg).expanduser().resolve()}"
    if cfg.get("db_url"):
        return str(cfg["db_url"])
    mysql = cfg.get("mysql", {})
    if isinstance(mysql, dict) and mysql.get("user") and mysql.get("database"):
        from urllib.parse import quote_plus
        u = quote_plus(mysql.get("user", ""))
        p = quote_plus(mysql.get("password", ""))
        auth = f"{u}:{p}" if p else u
        host = mysql.get("host", "localhost")
        port = mysql.get("port", 3306)
        db = mysql.get("database", "laowang")
        charset = mysql.get("charset", "utf8mb4")
        return f"mysql+pymysql://{auth}@{host}:{port}/{db}?charset={charset}"
    return "sqlite:///data/stock.db"


def _make_engine(db_url: str) -> Engine:
    from sqlalchemy import create_engine
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite:///") else {}
    return create_engine(db_url, pool_pre_ping=True, connect_args=connect_args)


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    if p.suffix.lower() in {".ini", ".cfg"}:
        parser = configparser.ConfigParser()
        parser.read(p, encoding="utf-8")
        return {
            "db_url": parser.get("database", "db_url", fallback="").strip() or None,
            "mysql": {
                "host": parser.get("mysql", "host", fallback="127.0.0.1").strip(),
                "port": parser.getint("mysql", "port", fallback=3306),
                "user": parser.get("mysql", "user", fallback="").strip(),
                "password": parser.get("mysql", "password", fallback=""),
                "database": parser.get("mysql", "database", fallback="").strip(),
                "charset": parser.get("mysql", "charset", fallback="utf8mb4").strip(),
            },
        }
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        val = float(v)
        return val if math.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def ensure_features_table(engine: Engine) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {FEATURES_TABLE} (
        trade_date VARCHAR(10) NOT NULL,
        stock_code VARCHAR(16) NOT NULL,
        signal_date VARCHAR(10) NULL,
        data_mode VARCHAR(16) NULL,
        auction_gap DOUBLE NULL,
        auction_amount_ratio DOUBLE NULL,
        first_5m_ret DOUBLE NULL,
        first_15m_ret DOUBLE NULL,
        first_15m_vwap_ret DOUBLE NULL,
        first_30m_low_drawdown DOUBLE NULL,
        reclaim_open_price_flag INT NULL,
        reclaim_vwap_flag INT NULL,
        re_seal_flag INT NULL,
        re_seal_minutes INT NULL,
        reopen_count INT NULL,
        volume_efficiency_15m DOUBLE NULL,
        data_quality VARCHAR(16) NULL,
        created_at VARCHAR(19) NULL,
        PRIMARY KEY (trade_date, stock_code)
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def prev_close_and_amount_ma20(
    conn, stock_code: str, trade_date: str
) -> tuple[float, float]:
    rows = conn.execute(
        text("""
            SELECT close, amount
            FROM stock_daily
            WHERE stock_code = :code AND date < :td
            ORDER BY date DESC
            LIMIT 20
        """),
        {"code": stock_code, "td": trade_date},
    ).all()
    if not rows:
        return (float("nan"), float("nan"))
    closes = [_safe_float(r[0]) for r in rows]
    amounts = [_safe_float(r[1]) for r in rows]
    closes = [c for c in closes if not math.isnan(c)]
    amounts = [a for a in amounts if not math.isnan(a)]
    prev_close = closes[0] if closes else float("nan")
    ma20 = sum(amounts[:20]) / len(amounts[:20]) if amounts else float("nan")
    return (prev_close, ma20)


def load_minute_bars(
    conn,
    stock_code: str,
    trade_date: str,
    frequency: str = "5",
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        text("""
            SELECT time, open, high, low, close, volume, amount
            FROM stock_minute
            WHERE stock_code = :code AND date = :td AND frequency = :freq
            ORDER BY time
        """),
        {"code": stock_code, "td": trade_date, "freq": frequency},
    ).mappings().all()
    return [dict(row) for row in rows]


def compute_auction_features(
    bars: List[Dict[str, Any]],
    prev_close: float,
) -> Dict[str, Any]:
    if math.isnan(prev_close) or prev_close <= 0:
        return {
            "auction_gap": None,
            "auction_amount": None,
            "auction_amount_ratio": None,
            "auction_price_slope": None,
        }
    if not bars:
        return {
            "auction_gap": None,
            "auction_amount": None,
            "auction_amount_ratio": None,
            "auction_price_slope": None,
        }
    first_bar = bars[0]
    last_bar = bars[-1]
    first_open = _safe_float(first_bar.get("open"))
    last_close = _safe_float(last_bar.get("close"))
    auction_amount = sum(_safe_float(b.get("amount")) for b in bars)
    gap = (first_open - prev_close) / prev_close if first_open else None
    slope = (last_close - first_open) / first_open if first_open else None
    return {
        "auction_gap": gap,
        "auction_amount": auction_amount,
        "auction_price_slope": slope,
    }


def compute_intraday_features(
    bars: List[Dict[str, Any]],
    prev_close: float,
    amount_ma20: float,
) -> Dict[str, Any]:
    if not bars:
        return _empty_intraday_features()

    def bar_time(b: Dict[str, Any]) -> str:
        t = str(b.get("time") or "")
        return t.zfill(8) if len(t) < 8 else t

    bars = sorted(bars, key=bar_time)

    open_930 = _safe_float(bars[0].get("open")) if bars else float("nan")
    if math.isnan(open_930) or open_930 <= 0:
        return _empty_intraday_features()

    early_bars = []
    for b in bars:
        t = bar_time(b)
        if t <= "09:35:00":
            early_bars.append(b)

    first_5m = early_bars
    first_15m_bars = []
    for b in bars:
        t = bar_time(b)
        if t <= "09:45:00":
            first_15m_bars.append(b)

    first_5m_close = _safe_float(first_5m[-1].get("close")) if first_5m else open_930
    first_15m_close = _safe_float(first_15m_bars[-1].get("close")) if first_15m_bars else open_930
    first_30m_bars = []
    for b in bars:
        t = bar_time(b)
        if t <= "10:00:00":
            first_30m_bars.append(b)

    first_30m_low = min((_safe_float(b.get("low")) for b in first_30m_bars), default=open_930)

    first_5m_ret = (first_5m_close - open_930) / open_930
    first_15m_ret = (first_15m_close - open_930) / open_930
    first_30m_low_drawdown = (first_30m_low - open_930) / open_930

    vol_sum = sum(_safe_float(b.get("volume")) for b in first_15m_bars)
    amt_sum = sum(_safe_float(b.get("amount")) for b in first_15m_bars)
    vwap_15m = amt_sum / vol_sum if vol_sum > 0 else open_930
    first_15m_vwap_ret = (first_15m_close - vwap_15m) / vwap_15m

    reclaim_vwap_flag = 1 if first_15m_close > vwap_15m else 0
    reclaim_open_flag = 1 if first_15m_close > open_930 else 0

    in_seal = False
    seal_start = None
    re_seal_minutes = None
    reopen_count = 0
    for b in bars:
        t_str = bar_time(b)
        if t_str < "09:30:00":
            continue
        b_close = _safe_float(b.get("close"))
        if not in_seal and b_close >= prev_close * 1.099:
            in_seal = True
            seal_start = t_str
        elif in_seal and b_close < prev_close * 1.05:
            in_seal = False
            if seal_start:
                t1 = datetime.strptime(seal_start, "%H:%M:%S")
                t2 = datetime.strptime(t_str, "%H:%M:%S")
                diff_min = int((t2 - t1).total_seconds() / 60)
                if re_seal_minutes is None or diff_min < re_seal_minutes:
                    re_seal_minutes = diff_min
                reopen_count += 1

    re_seal_flag = 1 if re_seal_minutes is not None and re_seal_minutes <= 30 else 0

    break_count = 0
    prev_cross = None
    for b in first_15m_bars:
        b_close = _safe_float(b.get("close"))
        cross = "above" if b_close > vwap_15m else "below"
        if prev_cross and prev_cross != cross:
            break_count += 1
        prev_cross = cross

    total_vol = sum(_safe_float(b.get("volume")) for b in bars)
    vol_eff = vol_sum / (total_vol * 0.25) if total_vol > 0 else float("nan")

    return {
        "first_5m_ret": first_5m_ret,
        "first_15m_ret": first_15m_ret,
        "first_15m_vwap_ret": first_15m_vwap_ret,
        "first_30m_low_drawdown": first_30m_low_drawdown,
        "reclaim_vwap_flag": reclaim_vwap_flag,
        "reclaim_open_price_flag": reclaim_open_flag,
        "re_seal_flag": re_seal_flag,
        "re_seal_minutes": re_seal_minutes,
        "reopen_count": reopen_count,
        "intraday_break_vwap_count": break_count,
        "volume_efficiency_15m": vol_eff,
    }


def _empty_intraday_features() -> Dict[str, Any]:
    return {
        "first_5m_ret": None,
        "first_15m_ret": None,
        "first_15m_vwap_ret": None,
        "first_30m_low_drawdown": None,
        "reclaim_vwap_flag": 0,
        "reclaim_open_price_flag": 0,
        "re_seal_flag": 0,
        "re_seal_minutes": None,
        "reopen_count": 0,
        "intraday_break_vwap_count": 0,
        "volume_efficiency_15m": None,
    }


def compute_auction_amount_ratio(
    auction_amount: Optional[float],
    amount_ma20: float,
) -> Optional[float]:
    if auction_amount is None or math.isnan(amount_ma20) or amount_ma20 <= 0:
        return None
    return auction_amount / amount_ma20


def _loads_json(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if raw in (None, "", "nan", "NaN"):
        return {}
    try:
        payload = json.loads(str(raw))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_watchlist(
    conn, signal_date: str
) -> List[Dict[str, Any]]:
    signal_rows = conn.execute(
        text("""
            SELECT stock_code, stock_name, score, features_json, action_plan_json
            FROM strategy_signal_log
            WHERE signal_date = :sd AND model = 'relay'
        """),
        {"sd": signal_date},
    ).mappings().all()
    watch_rows: List[Dict[str, Any]] = []
    for row in signal_rows:
        action_plan = _loads_json(row.get("action_plan_json"))
        action_state = str(action_plan.get("action_state") or "")
        if action_state not in {"watch_auction", "pending_open_check"}:
            continue
        features = _loads_json(row.get("features_json"))
        features.update(
            {
                "stock_code": row.get("stock_code"),
                "stock_name": row.get("stock_name"),
                "model_score": row.get("score"),
            }
        )
        watch_rows.append(features)
    if watch_rows:
        return watch_rows

    fallback_rows = conn.execute(
        text("""
            SELECT stock_code, stock_name, model_score, rank_no,
                   board_count, broken_rate, red_rate, limit_down_count,
                   pullback, max_board, close
            FROM model_relay_pool
            WHERE trade_date = :sd
            ORDER BY rank_no, model_score DESC
        """),
        {"sd": signal_date},
    ).mappings().all()
    return [dict(row) for row in fallback_rows]


def next_trading_day(conn, trade_date: str) -> str:
    row = conn.execute(
        text("""
            SELECT MIN(date) FROM stock_daily
            WHERE date > :td LIMIT 1
        """),
        {"td": trade_date},
    ).first()
    return str(row[0]) if row and row[0] else trade_date


def upsert_features(engine: Engine, rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols = [
        "trade_date", "stock_code", "signal_date", "data_mode",
        "auction_gap", "auction_amount_ratio", "first_5m_ret",
        "first_15m_ret", "first_15m_vwap_ret", "first_30m_low_drawdown",
        "reclaim_open_price_flag", "reclaim_vwap_flag", "re_seal_flag",
        "re_seal_minutes", "reopen_count", "volume_efficiency_15m",
        "data_quality", "created_at",
    ]
    placeholders = ", ".join(f":{c}" for c in cols)
    if engine.dialect.name == "sqlite":
        update_parts = [f"{c}=excluded.{c}" for c in cols if c not in ("trade_date", "stock_code")]
        sql = f"""
            INSERT INTO {FEATURES_TABLE} ({", ".join(cols)}) VALUES ({placeholders})
            ON CONFLICT(trade_date, stock_code) DO UPDATE SET
            {", ".join(update_parts)}
        """
    else:
        update_parts = [f"{c}=VALUES({c})" for c in cols if c not in ("trade_date", "stock_code")]
        sql = f"""
            INSERT INTO {FEATURES_TABLE} ({", ".join(cols)}) VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {", ".join(update_parts)}
        """
    normalized = []
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        r = {c: _db_value(row.get(c)) for c in cols}
        r["created_at"] = now_text
        normalized.append(r)
    with engine.begin() as conn:
        for batch in _chunked(normalized, 200):
            conn.execute(text(sql), batch)
    return len(normalized)


def _db_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _chunked(rows: Sequence[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    result = []
    for i in range(0, len(rows), size):
        result.append(list(rows[i : i + size]))
    return result


def compute_features_for_watchlist(
    engine: Engine,
    signal_date: str,
) -> Dict[str, Any]:
    ensure_features_table(engine)
    processed = 0
    skipped_no_minute = 0
    errors = []

    with engine.connect() as conn:
        rows = load_watchlist(conn, signal_date)
        trade_date = next_trading_day(conn, signal_date)

    if not rows:
        return {
            "signal_date": signal_date,
            "trade_date": trade_date,
            "processed": 0,
            "skipped_no_minute": 0,
            "errors": [],
        }

    feature_rows = []
    for row in rows:
        code = str(row.get("stock_code", "")).strip()
        if not code:
            continue
        try:
            with engine.connect() as conn:
                prev_close, amount_ma20 = prev_close_and_amount_ma20(conn, code, trade_date)
                minute_bars = load_minute_bars(conn, code, trade_date)
        except Exception as exc:
            if _is_missing_table_error(exc):
                prev_close, amount_ma20 = float("nan"), float("nan")
                minute_bars = []
            else:
                errors.append({"stock_code": code, "error": str(exc)})
                continue

        if not minute_bars:
            skipped_no_minute += 1
            auction_feats = {
                "auction_gap": None,
                "auction_amount": None,
                "auction_price_slope": None,
            }
            intra_feats = _empty_intraday_features()
            data_mode = "missing"
            data_quality = "missing"
        else:
            auction_feats = compute_auction_features(minute_bars, prev_close)
            intra_feats = compute_intraday_features(minute_bars, prev_close, amount_ma20)
            data_mode = "minute"
            data_quality = "ok" if len(minute_bars) >= 40 else "degraded"

        auction_amount = auction_feats.get("auction_amount")
        auction_amount_ratio = compute_auction_amount_ratio(auction_amount, amount_ma20)

        feature_rows.append({
            "stock_code": code,
            "trade_date": trade_date,
            "signal_date": signal_date,
            "data_mode": data_mode,
            **auction_feats,
            "auction_amount_ratio": auction_amount_ratio,
            **intra_feats,
            "data_quality": data_quality,
        })
        processed += 1

    upsert_features(engine, feature_rows)
    return {
        "signal_date": signal_date,
        "trade_date": trade_date,
        "processed": processed,
        "skipped_no_minute": skipped_no_minute,
        "errors": errors,
    }


def _is_missing_table_error(exc: Exception) -> bool:
    text = str(getattr(exc, "orig", exc)).lower()
    return "no such table" in text or "doesn't exist" in text or "unknown table" in text


def _engine_url_from_engine(engine: Engine) -> str:
    return str(engine.url)


def make_engine_from_url(db_url: str) -> Engine:
    return _make_engine(db_url)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Relay intraday minute features.")
    p.add_argument("--config", default=None)
    p.add_argument("--db-url", default=None)
    p.add_argument("--db", default=None)
    p.add_argument(
        "--signal-date",
        default=None,
        help="Signal date (T) to compute features for T+1 intraday. Defaults to latest signal_date in relay pool.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    db_url = _resolve_db_url(args, cfg)
    engine = _make_engine(db_url)

    if args.signal_date:
        signal_date = args.signal_date
    else:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT MAX(trade_date) FROM model_relay_pool")
            ).first()
        signal_date = str(row[0]) if row and row[0] else None

    if not signal_date:
        print(json.dumps({"error": "no signal_date found"}))
        return

    result = compute_features_for_watchlist(engine, signal_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
