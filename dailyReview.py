#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dailyReview.py

Generate daily review reports based on latest K-line data in stock_daily.
Output file name: daily-review-YYYYMMDD.md
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import bisect
import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import urllib.request
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine, create_engine


DEFAULT_DB = "data/stock.db"
EPS = 1e-3
LOOKBACK_DAYS = 120
XGB_TIMEOUT = 10
_XGB_LIMIT_UP_CACHE: Dict[str, List[dict]] = {}
XGB_POOL_DETAIL_URL = "https://flash-api.xuangubao.cn/api/pool/detail?pool_name={pool_name}&date={date}"
_XGB_POOL_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}
TRADE_CALENDAR_CODES: Tuple[str, ...] = ("000001", "600000", "601398", "600519", "300750")
_TRADE_CALENDAR_CACHE: Dict[str, List[str]] = {}


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


def make_engine(db_target: str) -> Engine:
    connect_args: Dict[str, object] = {}
    if "://" not in db_target and db_target.endswith(".db"):
        db_path = Path(db_target).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_target = f"sqlite:///{db_path.as_posix()}"
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


def parse_date_arg(value: str) -> str:
    v = (value or "").strip()
    if not v:
        raise ValueError("date is required")
    if len(v) == 8 and v.isdigit():
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    try:
        return dt.datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date: {value}") from exc


def yyyymmdd(value: str) -> str:
    v = str(value or "").strip()
    if len(v) == 8 and v.isdigit():
        return v
    return dt.datetime.strptime(v, "%Y-%m-%d").strftime("%Y%m%d")


def list_trade_dates(engine: Engine, start_date: str, end_date: str) -> List[str]:
    dates = _trade_calendar(engine)
    if not dates:
        return []
    left = bisect.bisect_left(dates, start_date)
    right = bisect.bisect_right(dates, end_date)
    return dates[left:right]


def latest_trade_date(engine: Engine) -> Optional[str]:
    dates = _trade_calendar(engine)
    if not dates:
        return None
    return dates[-1]


def prev_trade_date(engine: Engine, trade_date: str) -> Optional[str]:
    if not trade_date:
        return None
    dates = _trade_calendar(engine)
    if not dates:
        return None
    idx = bisect.bisect_left(dates, trade_date)
    if idx <= 0:
        return None
    return dates[idx - 1]


def last_n_trade_dates(engine: Engine, end_date: str, limit: int) -> List[str]:
    if limit <= 0:
        return []
    dates = _trade_calendar(engine)
    if not dates:
        return []
    idx = bisect.bisect_right(dates, end_date)
    if idx <= 0:
        return []
    start = max(0, idx - int(limit))
    return dates[start:idx]


def _trade_calendar(engine: Engine) -> List[str]:
    key = str(engine.url)
    cached = _TRADE_CALENDAR_CACHE.get(key)
    if cached is not None:
        return cached

    with engine.connect() as conn:
        for code in TRADE_CALENDAR_CODES:
            rows = conn.execute(
                text("SELECT date FROM stock_daily WHERE stock_code = :code ORDER BY date"),
                {"code": code},
            ).fetchall()
            dates = [str(r[0]) for r in rows if r and r[0]]
            if len(dates) >= 10:
                _TRADE_CALENDAR_CACHE[key] = dates
                return dates

        rows = conn.execute(text("SELECT DISTINCT date FROM stock_daily ORDER BY date")).fetchall()
        dates = [str(r[0]) for r in rows if r and r[0]]
        _TRADE_CALENDAR_CACHE[key] = dates
        return dates


def _round_half_up_2(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.floor(arr * 100.0 + 0.5) / 100.0
    return pd.Series(out, index=values.index)


def _is_st_name(stock_name: str) -> bool:
    name = str(stock_name or "").strip()
    if not name:
        return False
    if "退" in name:
        return True
    return "ST" in name.upper()


def _infer_limit_pct(stock_code: str, stock_name: str, is_st_flag: bool) -> float:
    if is_st_flag:
        return 0.05
    code = str(stock_code).strip()
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def fetch_trade_frame(engine: Engine, trade_date: str) -> pd.DataFrame:
    if not trade_date:
        return pd.DataFrame(
            columns=[
                "stock_code",
                "stock_name",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "prev_close",
            ]
        )
    prev_date = prev_trade_date(engine, trade_date)
    if prev_date:
        sql = """
            SELECT d.stock_code,
                   COALESCE(i.name, '') AS stock_name,
                   d.open,
                   d.high,
                   d.low,
                   d.close,
                   d.volume,
                   d.amount,
                   p.close AS prev_close
            FROM stock_daily d
            LEFT JOIN stock_info i
                ON i.stock_code = d.stock_code
            LEFT JOIN stock_daily p
                ON p.stock_code = d.stock_code
               AND p.date = :prev_date
            WHERE d.date = :d
        """
    else:
        sql = """
            SELECT d.stock_code,
                   COALESCE(i.name, '') AS stock_name,
                   d.open,
                   d.high,
                   d.low,
                   d.close,
                   d.volume,
                   d.amount,
                   NULL AS prev_close
            FROM stock_daily d
            LEFT JOIN stock_info i
                ON i.stock_code = d.stock_code
            WHERE d.date = :d
        """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"d": trade_date, "prev_date": prev_date}).fetchall()
    df = pd.DataFrame(
        rows,
        columns=[
            "stock_code",
            "stock_name",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "prev_close",
        ],
    )
    return df


def add_limit_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ["open", "high", "low", "close", "volume", "amount", "prev_close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["is_st"] = out["stock_name"].apply(_is_st_name)
    out["limit_pct"] = out.apply(
        lambda r: _infer_limit_pct(r["stock_code"], r["stock_name"], bool(r["is_st"])),
        axis=1,
    )
    out["limit_up"] = _round_half_up_2(out["prev_close"] * (1.0 + out["limit_pct"]))
    out["limit_down"] = _round_half_up_2(out["prev_close"] * (1.0 - out["limit_pct"]))
    valid_prev = pd.to_numeric(out["prev_close"], errors="coerce") > 0
    out["valid_prev"] = valid_prev.fillna(False)
    out["is_limit_up"] = out["valid_prev"] & (out["close"] >= out["limit_up"] - EPS)
    out["is_limit_down"] = out["valid_prev"] & (out["close"] <= out["limit_down"] + EPS)
    out["touched_limit_up"] = (
        out["valid_prev"] & pd.to_numeric(out["high"], errors="coerce").ge(out["limit_up"] - EPS)
    )
    out["broken"] = out["touched_limit_up"] & (~out["is_limit_up"])
    return out


def _normalize_code(raw: object) -> str:
    text = str(raw or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    text = text.replace(".", "").replace("-", "").replace("_", "")
    text = text.upper()
    for prefix in ("SH", "SZ", "BJ"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    if len(digits) < 6:
        digits = digits.zfill(6)
    return digits


def _pick_first(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    col_set = set(columns)
    for name in candidates:
        if name in col_set:
            return name
    return None


def _parse_board_count(raw: object) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:  # noqa: BLE001
        pass
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:  # noqa: BLE001
        return None


def _extract_reason_map(df: pd.DataFrame) -> Dict[str, str]:
    if df is None or df.empty:
        return {}
    cols = list(df.columns)
    code_col = _pick_first(
        cols,
        (
            "代码",
            "股票代码",
            "证券代码",
            "symbol",
            "Symbol",
            "code",
            "Code",
        ),
    )
    if not code_col:
        return {}
    reason_col = _pick_first(
        cols,
        (
            "涨停原因类别",
            "涨停原因",
            "跌停原因",
            "炸板原因",
            "原因",
            "原因类别",
            "入选理由",
            "异动原因",
            "上榜原因",
        ),
    )
    if not reason_col:
        for col in cols:
            if "原因" in str(col):
                reason_col = col
                break
    if not reason_col:
        return {}
    out: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        code = _normalize_code(getattr(row, code_col, ""))
        if not code:
            continue
        reason = str(getattr(row, reason_col, "") or "").strip()
        if not reason or reason.lower() in {"nan", "none"}:
            continue
        out[code] = reason
    return out


def _fetch_xgb_pool_raw(trade_date: str, pool_name: str) -> List[dict]:
    if not trade_date:
        return []
    url = XGB_POOL_DETAIL_URL.format(pool_name=pool_name, date=trade_date)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=XGB_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logging.warning("XGB %s fetch failed for %s: %s", pool_name, trade_date, exc)
        return []

    if not isinstance(payload, dict) or payload.get("code") != 20000:
        logging.warning("XGB %s unexpected response for %s", pool_name, trade_date)
        return []

    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _xgb_reason_text(item: dict) -> str:
    surge_reason = item.get("surge_reason")
    if isinstance(surge_reason, dict):
        for key in ("stock_reason", "reason_type_desc", "reason_type", "reason"):
            value = str(surge_reason.get(key) or "").strip()
            if value:
                return value
    for key in ("stock_reason", "reason", "reason_type_desc", "reason_type"):
        value = item.get(key)
        if isinstance(value, str):
            txt = value.strip()
            if txt:
                return txt
    return ""


def _xgb_board_count(item: dict, pool_name: str) -> Optional[int]:
    candidates: List[object] = []
    if pool_name == "limit_up":
        candidates.extend([item.get("limit_up_days"), item.get("m_days_n_boards_boards")])
    elif pool_name == "limit_down":
        candidates.extend([item.get("limit_down_days"), item.get("m_days_n_boards_boards")])
    elif pool_name == "broken":
        candidates.extend([item.get("limit_up_days"), item.get("m_days_n_boards_boards")])
    for value in candidates:
        val = _parse_board_count(value)
        if val is not None and val > 0:
            return val
    return None


def _normalize_xgb_pool_rows(pool_name: str, items: Sequence[dict]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol") or item.get("code") or item.get("stock_code")
        code = _normalize_code(symbol)
        if not code:
            continue
        rows.append(
            {
                "symbol": str(symbol or code),
                "stock_chi_name": str(
                    item.get("stock_chi_name")
                    or item.get("stock_name")
                    or item.get("name")
                    or ""
                ).strip(),
                "change_percent": item.get("change_percent"),
                "price": item.get("price"),
                "limit_up_days": item.get("limit_up_days"),
                "limit_down_days": item.get("limit_down_days"),
                "break_limit_up_times": item.get("break_limit_up_times"),
                "break_limit_down_times": item.get("break_limit_down_times"),
                "board_count": _xgb_board_count(item, pool_name),
                "reason": _xgb_reason_text(item),
            }
        )
    return rows


def fetch_xgb_pools(trade_date: str) -> Dict[str, pd.DataFrame]:
    if not trade_date:
        return {}
    cached = _XGB_POOL_CACHE.get(trade_date)
    if cached is not None:
        return cached

    pools: Dict[str, pd.DataFrame] = {}
    for pool_name in ("limit_up", "limit_down", "broken"):
        if pool_name == "limit_up":
            raw = fetch_xgb_limit_up_raw(trade_date)
        else:
            raw = _fetch_xgb_pool_raw(trade_date, pool_name)
        if raw:
            rows = _normalize_xgb_pool_rows(pool_name, raw)
            pools[pool_name] = pd.DataFrame(rows if rows else raw)
    _XGB_POOL_CACHE[trade_date] = pools
    return pools


def fetch_xgb_limit_up_raw(trade_date: str) -> List[dict]:
    if not trade_date:
        return []
    cached = _XGB_LIMIT_UP_CACHE.get(trade_date)
    if cached is not None:
        return cached
    data = _fetch_xgb_pool_raw(trade_date, "limit_up")
    _XGB_LIMIT_UP_CACHE[trade_date] = data
    return data


def normalize_xgb_limit_up_items(
    items: Sequence[dict],
    is_st_map: Optional[Dict[str, bool]] = None,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol") or item.get("code") or item.get("stock_code")
        code = _normalize_code(symbol)
        name = str(
            item.get("stock_chi_name")
            or item.get("stock_name")
            or item.get("name")
            or ""
        ).strip()
        if not name and code:
            name = code
        if _is_st_name(name):
            continue
        if is_st_map is not None and code and bool(is_st_map.get(code, False)):
            continue
        limit_up_days = item.get("limit_up_days")
        try:
            limit_up_days = int(limit_up_days)
        except Exception:  # noqa: BLE001
            limit_up_days = 1
        change_percent = item.get("change_percent")
        try:
            change_percent = float(change_percent)
        except Exception:  # noqa: BLE001
            change_percent = 0.0
        plates: List[str] = []
        surge_reason = item.get("surge_reason") or {}
        related = surge_reason.get("related_plates") or []
        if isinstance(related, list):
            for plate in related:
                if not isinstance(plate, dict):
                    continue
                plate_name = str(plate.get("plate_name") or "").strip()
                if plate_name:
                    plates.append(plate_name)
        if not plates:
            continue
        out.append(
            {
                "code": code,
                "name": name,
                "change_percent": change_percent,
                "limit_up_days": limit_up_days,
                "plates": plates,
            }
        )
    return out


def build_theme_index(items: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    themes: Dict[str, List[Dict[str, object]]] = {}
    for item in items:
        plates = item.get("plates") or []
        for plate in plates:
            if not plate:
                continue
            themes.setdefault(str(plate), []).append(item)
    return themes


def eval_theme_breadth(limit_up_count: int, has_multi_tiers: bool) -> Tuple[int, str]:
    if limit_up_count >= 5:
        if has_multi_tiers:
            return 2, "主线候选"
        return 1, "次主线"
    if 3 <= limit_up_count <= 4:
        return 1, "次主线"
    if limit_up_count <= 1:
        return 0, "单票独舞"
    return 0, "轮动"


def eval_theme_height(max_limit_up_days: int) -> Tuple[int, str]:
    if max_limit_up_days >= 5:
        return 2, "空间打开"
    if max_limit_up_days >= 3:
        return 1, "修复阶段"
    return 0, "无高度"


def _max_consecutive_flags(flags: Sequence[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def eval_theme_persistence(
    theme: str,
    daily_counts: Sequence[Dict[str, int]],
) -> Tuple[int, str, str]:
    if not daily_counts:
        return 0, "非主线", "无数据"
    top_flags: List[bool] = []
    top_days = 0
    for counts in daily_counts:
        if not counts:
            top_flags.append(False)
            continue
        max_count = max(counts.values())
        is_top = counts.get(theme, 0) > 0 and counts.get(theme, 0) == max_count
        top_flags.append(is_top)
        if is_top:
            top_days += 1
    max_consec = _max_consecutive_flags(top_flags)
    if max_consec >= 2:
        return 2, "强主线", f"{top_days}天领涨"
    if top_days >= 2:
        return 1, "弱主线", f"{top_days}天领涨"
    if top_days == 1:
        return 0, "一日游", "仅1天领涨"
    return 0, "非主线", "未领涨"


def eval_theme_exclusivity(
    theme: str,
    top_theme: str,
    top_count: int,
    second_count: int,
) -> Tuple[int, str]:
    if not top_theme or top_count <= 0:
        return 0, "无主线"
    if theme != top_theme:
        return 0, "你强我也强"
    if top_count >= max(5, second_count + 2):
        return 2, "明显压制"
    if second_count > 0 and (top_count / float(second_count)) >= 1.5:
        return 2, "明显压制"
    return 1, "多线并行"


def eval_theme_position(total_score: int) -> str:
    if total_score >= 7:
        return "唯一主线"
    if total_score >= 5:
        return "主线"
    if total_score >= 3:
        return "次主线"
    return "轮动/噪音"


def theme_sample_stocks(items: Sequence[Dict[str, object]], limit: int = 3) -> List[str]:
    sorted_items = sorted(
        items,
        key=lambda x: (
            int(x.get("limit_up_days") or 0),
            float(x.get("change_percent") or 0.0),
        ),
        reverse=True,
    )
    names: List[str] = []
    for item in sorted_items:
        name = str(item.get("name") or "").strip()
        if not name:
            name = str(item.get("code") or "").strip()
        if not name or name in names:
            continue
        names.append(name)
        if len(names) >= limit:
            break
    return names


def _pick_item_name(item: Dict[str, object]) -> str:
    name = str(item.get("name") or "").strip()
    if name:
        return name
    return str(item.get("code") or "").strip()


def build_ladder_summary(
    theme: Optional[Dict[str, object]],
    items: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    if not theme or not items:
        return {"has_data": False}

    board_map: Dict[int, List[Dict[str, object]]] = {}
    for item in items:
        try:
            board = int(item.get("limit_up_days") or 0)
        except Exception:  # noqa: BLE001
            board = 0
        if board <= 0:
            continue
        board_map.setdefault(board, []).append(item)
    if not board_map:
        return {"has_data": False}

    for board in board_map:
        board_map[board] = sorted(
            board_map[board],
            key=lambda x: float(x.get("change_percent") or 0.0),
            reverse=True,
        )

    max_board = max(board_map.keys())
    leader_items = board_map.get(max_board, [])
    leader_names = [_pick_item_name(i) for i in leader_items]
    leader_names = [n for n in leader_names if n]
    leader_name = leader_names[0] if leader_names else ""

    if max_board >= 5:
        status = "加速"
    elif max_board >= 3:
        status = "分歧"
    else:
        status = "断板"

    secondary_items: List[Dict[str, object]] = []
    for board in range(max_board - 1, max_board - 3, -1):
        if board < 2:
            continue
        secondary_items.extend(board_map.get(board, []))
    secondary_names = [_pick_item_name(i) for i in secondary_items]
    secondary_names = [n for n in secondary_names if n]
    secondary_names = list(dict.fromkeys(secondary_names))

    supplement_items = board_map.get(1, [])
    supplement_names = [_pick_item_name(i) for i in supplement_items]
    supplement_names = [n for n in supplement_names if n]
    supplement_names = list(dict.fromkeys(supplement_names))
    supplement_label = "、".join(supplement_names[:6]) + (f"...(+{len(supplement_names) - 6})" if len(supplement_names) > 6 else "")
    if not supplement_label:
        supplement_label = "（无）"

    leader_score = 2 if max_board >= 4 else 1 if max_board >= 3 else 0
    secondary_score = 2 if len(secondary_items) >= 2 else 1 if len(secondary_items) == 1 else 0
    supplement_score = 2 if len(supplement_items) >= 3 else 1 if len(supplement_items) > 0 else 0
    efficiency_score = 2 if theme.get("has_multi_tiers") and max_board >= 3 and len(secondary_items) > 0 else 1 if theme.get("has_multi_tiers") else 0
    total_score = int(leader_score + secondary_score + supplement_score + efficiency_score)

    if total_score >= 7:
        health_label = "主升期，可重仓"
    elif total_score >= 5:
        health_label = "可做，控仓"
    elif total_score >= 3:
        health_label = "边走边看"
    else:
        health_label = "准备撤退"

    board_name_map: Dict[int, List[str]] = {}
    for board, values in board_map.items():
        names = [_pick_item_name(i) for i in values]
        names = [n for n in names if n]
        board_name_map[board] = list(dict.fromkeys(names))

    return {
        "has_data": True,
        "leader_name": leader_name,
        "leader_names": leader_names,
        "leader_days": max_board,
        "status": status,
        "secondary_names": secondary_names,
        "supplement_names": supplement_names,
        "supplement_label": supplement_label,
        "health_score": total_score,
        "health_label": health_label,
        "score_breakdown": {
            "leader": leader_score,
            "secondary": secondary_score,
            "supplement": supplement_score,
            "efficiency": efficiency_score,
        },
        "board_name_map": board_name_map,
    }


def build_divergence_summary(
    prev_df: pd.DataFrame,
    today_df: pd.DataFrame,
    prev_prev_df: Optional[pd.DataFrame],
    prev_date: Optional[str],
    stage: str,
    top_theme: Optional[Dict[str, object]],
    items_by_date: Dict[str, Dict[str, Dict[str, object]]],
) -> Dict[str, object]:
    if prev_df is None or prev_df.empty or not prev_date:
        return {"samples": [], "note": "缺少前一交易日数据，无法生成分歧样本。", "source_date": prev_date}

    prev_df = add_limit_flags(prev_df)
    prev_df = prev_df[prev_df["valid_prev"]].copy()
    prev_df = prev_df[~prev_df["is_st"]].copy()
    if prev_df.empty:
        return {"samples": [], "note": "前一交易日无有效样本。", "source_date": prev_date}

    prev_df["opened"] = prev_df["low"].lt(prev_df["limit_up"] - EPS)

    touched = prev_df[prev_df["touched_limit_up"]]
    survivors = touched[touched["is_limit_up"] & touched["opened"]]
    failed = touched[~touched["is_limit_up"]]

    items_prev = items_by_date.get(prev_date) or {}

    amount_prevprev: Dict[str, float] = {}
    if prev_prev_df is not None and not prev_prev_df.empty:
        amount_prevprev = {
            str(row.stock_code): float(row.amount)
            for row in prev_prev_df.itertuples(index=False)
            if row.stock_code and row.amount is not None
        }

    amount_today: Dict[str, float] = {}
    if today_df is not None and not today_df.empty:
        amount_today = {
            str(row.stock_code): float(row.amount)
            for row in today_df.itertuples(index=False)
            if row.stock_code and row.amount is not None
        }
    close_today: Dict[str, float] = {}
    if today_df is not None and not today_df.empty:
        close_today = {
            str(row.stock_code): float(row.close)
            for row in today_df.itertuples(index=False)
            if row.stock_code and row.close is not None
        }

    def _pick_top(df: pd.DataFrame, exclude: set) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        df = df[~df["stock_code"].astype(str).isin(exclude)]
        if df.empty:
            return None
        return df.sort_values("amount", ascending=False).iloc[0]

    def _limit_up_days(code: str) -> Optional[int]:
        item = items_prev.get(code)
        if not item:
            return None
        try:
            return int(item.get("limit_up_days") or 0)
        except Exception:  # noqa: BLE001
            return None

    def _divergence_type(is_survive: bool, board: Optional[int]) -> str:
        if is_survive:
            if board and 2 <= board <= 4:
                return "健康分歧"
            if board and board >= 5:
                return "末期分歧"
            return "健康分歧"
        if board and board >= 5:
            return "末期分歧"
        if board and board >= 2:
            return "结构分歧"
        return "伪分歧"

    def _next_result(code: str, prev_close: float) -> str:
        if code not in close_today or prev_close is None:
            return "N/A"
        return "收红" if float(close_today[code]) > float(prev_close) else "收绿/持平"

    def _expected_match(div_type: str, next_result: str) -> str:
        if next_result == "N/A":
            return "待观察"
        if div_type == "健康分歧":
            return "是" if next_result == "收红" else "否"
        if div_type in {"末期分歧", "伪分歧"}:
            return "是" if next_result == "收绿/持平" else "否"
        return "待观察"

    samples: List[Dict[str, object]] = []
    used: set = set()

    sample_survive = _pick_top(survivors, used)
    if sample_survive is not None:
        code = str(sample_survive.stock_code)
        used.add(code)
        board = _limit_up_days(code)
        volume_label = _amount_trend_label(
            float(sample_survive.amount) if sample_survive.amount is not None else None,
            amount_prevprev.get(code),
        )
        div_type = _divergence_type(True, board)
        next_result = _next_result(code, float(sample_survive.close))
        samples.append(
            {
                "label": "样本1-活下来的",
                "stock": f"{sample_survive.stock_name}({code})",
                "divergence_type": div_type,
                "position": f"{board}板" if board else "未知",
                "volume": volume_label,
                "support": "分歧转一致/回封",
                "environment": f"{stage} / 主线:{top_theme.get('name') if top_theme else '无'}",
                "next_result": next_result,
                "expected_match": _expected_match(div_type, next_result),
            }
        )

    sample_fail = _pick_top(failed, used)
    if sample_fail is not None:
        code = str(sample_fail.stock_code)
        used.add(code)
        board = _limit_up_days(code)
        volume_label = _amount_trend_label(
            float(sample_fail.amount) if sample_fail.amount is not None else None,
            amount_prevprev.get(code),
        )
        div_type = _divergence_type(False, board)
        next_result = _next_result(code, float(sample_fail.close))
        samples.append(
            {
                "label": "样本2-死掉的",
                "stock": f"{sample_fail.stock_name}({code})",
                "divergence_type": div_type,
                "position": f"{board}板" if board else "未知",
                "volume": volume_label,
                "support": "承接不足/封不住",
                "environment": f"{stage} / 主线:{top_theme.get('name') if top_theme else '无'}",
                "next_result": next_result,
                "expected_match": _expected_match(div_type, next_result),
            }
        )

    should_candidates = failed.copy()
    if not should_candidates.empty:
        should_candidates["limit_up_days"] = should_candidates["stock_code"].astype(str).map(
            lambda c: _limit_up_days(c) or 0
        )
        should_candidates = should_candidates.sort_values(
            ["limit_up_days", "amount"], ascending=[False, False]
        )
        for _, row in should_candidates.iterrows():
            code = str(row.stock_code)
            if code in used:
                continue
            used.add(code)
            board = int(row.limit_up_days) if int(row.limit_up_days) > 0 else None
            volume_label = _amount_trend_label(
                float(row.amount) if row.amount is not None else None,
                amount_prevprev.get(code),
            )
            div_type = _divergence_type(False, board)
            next_result = _next_result(code, float(row.close))
            samples.append(
                {
                    "label": "样本3-该活未活",
                    "stock": f"{row.stock_name}({code})",
                    "divergence_type": div_type,
                    "position": f"{board}板" if board else "未知",
                    "volume": volume_label,
                    "support": "承接不足/未回封",
                    "environment": f"{stage} / 主线:{top_theme.get('name') if top_theme else '无'}",
                    "next_result": next_result,
                    "expected_match": _expected_match(div_type, next_result),
                }
            )
            break

    return {
        "samples": samples,
        "note": "样本来自前一交易日炸板/分歧股，次日结果以今日收盘验证。",
        "source_date": prev_date,
    }


def compute_theme_summary(
    engine: Engine,
    trade_date: str,
    is_st_map: Optional[Dict[str, bool]] = None,
    lookback: int = 5,
) -> Dict[str, object]:
    dates = last_n_trade_dates(engine, trade_date, limit=lookback)
    if not dates:
        return {
            "themes": [],
            "top": None,
            "second": None,
            "lookback_dates": [],
            "today_theme_items": {},
            "items_by_date": {},
        }
    dates = sorted(dates)
    daily_counts: List[Dict[str, int]] = []
    daily_theme_items: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    items_by_date: Dict[str, Dict[str, Dict[str, object]]] = {}
    for d in dates:
        raw = fetch_xgb_limit_up_raw(d)
        items = normalize_xgb_limit_up_items(raw, is_st_map=is_st_map)
        theme_items = build_theme_index(items)
        daily_theme_items[d] = theme_items
        daily_counts.append({k: len(v) for k, v in theme_items.items()})
        items_by_date[d] = {str(item.get("code")): item for item in items if item.get("code")}

    today_items = daily_theme_items.get(trade_date) or {}
    if not today_items:
        return {
            "themes": [],
            "top": None,
            "second": None,
            "lookback_dates": dates,
            "lookback_days": len(dates),
            "today_theme_items": {},
            "items_by_date": items_by_date,
        }

    today_counts = daily_counts[-1] if dates and dates[-1] == trade_date else {k: len(v) for k, v in today_items.items()}
    sorted_counts = sorted(today_counts.items(), key=lambda x: x[1], reverse=True)
    top_theme = sorted_counts[0][0] if sorted_counts else ""
    top_count = sorted_counts[0][1] if sorted_counts else 0
    second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0

    themes: List[Dict[str, object]] = []
    for theme, items in today_items.items():
        limit_up_count = len(items)
        if limit_up_count <= 0:
            continue
        max_limit_up_days = max(int(item.get("limit_up_days") or 0) for item in items)
        tiers = sorted({int(item.get("limit_up_days") or 0) for item in items if int(item.get("limit_up_days") or 0) > 0})
        has_multi_tiers = len(tiers) >= 2
        breadth_score, breadth_label = eval_theme_breadth(limit_up_count, has_multi_tiers)
        height_score, height_label = eval_theme_height(max_limit_up_days)
        persistence_score, persistence_label, persistence_detail = eval_theme_persistence(theme, daily_counts)
        exclusivity_score, exclusivity_label = eval_theme_exclusivity(
            theme,
            top_theme,
            top_count,
            second_count,
        )
        total_score = int(breadth_score + height_score + persistence_score + exclusivity_score)
        position = eval_theme_position(total_score)
        themes.append(
            {
                "name": theme,
                "limit_up_count": limit_up_count,
                "max_limit_up_days": max_limit_up_days,
                "tiers": tiers,
                "has_multi_tiers": has_multi_tiers,
                "breadth_score": breadth_score,
                "breadth_label": breadth_label,
                "height_score": height_score,
                "height_label": height_label,
                "persistence_score": persistence_score,
                "persistence_label": persistence_label,
                "persistence_detail": persistence_detail,
                "exclusivity_score": exclusivity_score,
                "exclusivity_label": exclusivity_label,
                "total_score": total_score,
                "position": position,
                "sample_stocks": theme_sample_stocks(items),
            }
        )

    themes.sort(
        key=lambda x: (
            int(x.get("total_score") or 0),
            int(x.get("limit_up_count") or 0),
            int(x.get("max_limit_up_days") or 0),
        ),
        reverse=True,
    )
    top = themes[0] if themes else None
    second = themes[1] if len(themes) > 1 else None
    return {
        "themes": themes,
        "top": top,
        "second": second,
        "lookback_dates": dates,
        "lookback_days": len(dates),
        "today_theme_items": today_items,
        "items_by_date": items_by_date,
    }


def prepare_pool_df(
    pool_df: Optional[pd.DataFrame],
    fallback_name_map: Dict[str, str],
    is_st_map: Optional[Dict[str, bool]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if pool_df is None or pool_df.empty:
        return pd.DataFrame(), []
    df = pool_df.copy()
    cols = list(df.columns)
    code_col = _pick_first(
        cols,
        (
            "stock_code",
            "symbol",
            "Symbol",
            "code",
            "Code",
            "????",
            "????",
            "??",
        ),
    )
    if not code_col:
        return pd.DataFrame(), []
    name_col = _pick_first(
        cols,
        (
            "stock_chi_name",
            "stock_name",
            "name",
            "Name",
            "????",
            "????",
            "??",
        ),
    )
    df["_code_norm"] = df[code_col].apply(_normalize_code)
    if name_col:
        df["_name_norm"] = df[name_col].astype(str).fillna("").str.strip()
    else:
        df["_name_norm"] = df["_code_norm"].map(fallback_name_map).fillna("")

    mask = df["_code_norm"].ne("")
    mask &= ~df["_name_norm"].apply(_is_st_name)
    if is_st_map is not None:
        mask &= ~df["_code_norm"].map(is_st_map).fillna(False)
    df = df.loc[mask].copy()

    codes = df["_code_norm"].tolist()
    df = df.drop(columns=[c for c in ["_code_norm", "_name_norm"] if c in df.columns])
    return df, codes


def build_local_pool_df(df: pd.DataFrame, pool_type: str) -> Tuple[pd.DataFrame, List[str]]:
    if df is None or df.empty:
        return pd.DataFrame(), []

    if pool_type == "limit_up":
        mask = df["is_limit_up"]
    elif pool_type == "limit_down":
        mask = df["is_limit_down"]
    elif pool_type == "broken":
        mask = df["broken"]
    else:
        return pd.DataFrame(), []

    keep_cols = [
        "stock_code",
        "stock_name",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "amount",
        "volume",
        "limit_up",
        "limit_down",
    ]
    available = [c for c in keep_cols if c in df.columns]
    out = df.loc[mask, available].copy()
    if out.empty:
        return pd.DataFrame(), []

    code_series = out["stock_code"].astype(str)
    out = out.rename(
        columns={
            "stock_code": "代码",
            "stock_name": "名称",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "prev_close": "昨收",
            "amount": "成交额",
            "volume": "成交量",
            "limit_up": "涨停价",
            "limit_down": "跌停价",
        }
    )
    if "成交额" in out.columns:
        out = out.sort_values("成交额", ascending=False)
    return out, code_series.tolist()




def fetch_limit_up_codes(engine: Engine, trade_date: str) -> List[str]:
    df = fetch_trade_frame(engine, trade_date)
    if df.empty:
        return []
    df = add_limit_flags(df)
    if df.empty:
        return []
    df = df[df["valid_prev"]]
    return df.loc[df["is_limit_up"], "stock_code"].astype(str).tolist()


def compute_max_consecutive(
    engine: Engine,
    trade_date: str,
    codes: Sequence[str],
    name_map: Dict[str, str],
    is_st_map: Optional[Dict[str, bool]] = None,
    lookback: int = LOOKBACK_DAYS,
) -> int:
    if not codes:
        return 0
    dates = last_n_trade_dates(engine, trade_date, limit=lookback + 1)
    if len(dates) <= 1:
        return 0
    dates = sorted(dates)
    stmt = (
        text(
            """
            SELECT stock_code, date, close
            FROM stock_daily
            WHERE stock_code IN :codes AND date IN :dates
            ORDER BY stock_code, date
            """
        )
        .bindparams(bindparam("codes", expanding=True))
        .bindparams(bindparam("dates", expanding=True))
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"codes": list(codes), "dates": dates}).fetchall()
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=["stock_code", "date", "close"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"]).sort_values(["stock_code", "date"]).reset_index(drop=True)
    max_consecutive = 0
    for code, g in df.groupby("stock_code"):
        g = g.sort_values("date")
        if g.empty or str(g["date"].iloc[-1]) != trade_date:
            continue
        prev_close = g["close"].shift(1)
        is_st_flag = False
        if is_st_map is not None:
            is_st_flag = bool(is_st_map.get(code, False))
        else:
            is_st_flag = _is_st_name(name_map.get(code, ""))
        pct = _infer_limit_pct(code, name_map.get(code, ""), is_st_flag)
        limit_up = _round_half_up_2(prev_close * (1.0 + pct))
        is_limit_up = g["close"].ge(limit_up - EPS)
        if not bool(is_limit_up.iloc[-1]):
            continue
        cnt = 0
        for v in is_limit_up.iloc[::-1].to_list():
            if bool(v):
                cnt += 1
            else:
                break
        if cnt > max_consecutive:
            max_consecutive = cnt
    return int(max_consecutive)


def amount_sum_for_date(engine: Engine, trade_date: str) -> float:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT SUM(COALESCE(amount, 0)) FROM stock_daily WHERE date = :d"),
            {"d": trade_date},
        ).fetchone()
    if not row or row[0] is None:
        return 0.0
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return 0.0


def prev_trade_dates(engine: Engine, trade_date: str, limit: int) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT date FROM stock_daily WHERE date < :d ORDER BY date DESC LIMIT :lim"),
            {"d": trade_date, "lim": int(limit)},
        ).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def eval_limit_up_count(count: int) -> Dict[str, object]:
    thresholds = "≥80 高涨；50–79 可用；30–49 偏弱；<30 冰点"
    if count >= 80:
        return {"label": "情绪高涨", "level": "高", "score": 2, "thresholds": thresholds}
    if count >= 50:
        return {"label": "情绪可用", "level": "中", "score": 1, "thresholds": thresholds}
    if count >= 30:
        return {"label": "情绪偏弱", "level": "低", "score": 0, "thresholds": thresholds}
    return {"label": "情绪冰点", "level": "低", "score": 0, "thresholds": thresholds}


def eval_limit_down_count(count: int) -> Dict[str, object]:
    thresholds = "0–3 风险低；4–10 可控；11–20 警惕；>20 恐慌"
    if count <= 3:
        return {"label": "风险低", "level": "低", "score": 2, "thresholds": thresholds}
    if count <= 10:
        return {"label": "可控", "level": "中", "score": 1, "thresholds": thresholds}
    if count <= 20:
        return {"label": "警惕", "level": "高", "score": 0, "thresholds": thresholds}
    return {"label": "恐慌", "level": "高", "score": 0, "thresholds": thresholds}


def eval_broken_rate(rate: Optional[float], has_sample: bool) -> Dict[str, object]:
    thresholds = "<25% 容错极高；25–40% 正常；40–55% 容错下降；>55% 情绪退潮"
    if not has_sample or rate is None:
        return {"label": "样本不足", "level": "N/A", "score": 0, "thresholds": thresholds}
    pct = rate * 100.0
    if pct < 25:
        return {"label": "容错极高", "level": "低", "score": 2, "thresholds": thresholds}
    if pct <= 40:
        return {"label": "正常", "level": "中", "score": 1, "thresholds": thresholds}
    if pct <= 55:
        return {"label": "容错下降", "level": "高", "score": 0, "thresholds": thresholds}
    return {"label": "情绪退潮", "level": "高", "score": 0, "thresholds": thresholds}


def eval_max_consecutive(value: int) -> Dict[str, object]:
    thresholds = "≥7 主升/高潮；5–6 强势；3–4 修复；≤2 冰点"
    if value >= 7:
        return {"label": "主升/高潮", "level": "高", "score": 2, "thresholds": thresholds}
    if value >= 5:
        return {"label": "强势", "level": "中", "score": 1, "thresholds": thresholds}
    if value >= 3:
        return {"label": "修复", "level": "低", "score": 0, "thresholds": thresholds}
    return {"label": "冰点", "level": "低", "score": 0, "thresholds": thresholds}


def eval_red_rate(rate: Optional[float], has_sample: bool) -> Dict[str, object]:
    thresholds = "≥65% 健康；50–64% 勉强；35–49% 弱；<35% 断层"
    if not has_sample or rate is None:
        return {"label": "样本不足", "level": "N/A", "score": 0, "thresholds": thresholds}
    pct = rate * 100.0
    if pct >= 65:
        return {"label": "情绪健康", "level": "高", "score": 2, "thresholds": thresholds}
    if pct >= 50:
        return {"label": "勉强", "level": "中", "score": 1, "thresholds": thresholds}
    if pct >= 35:
        return {"label": "偏弱", "level": "低", "score": 0, "thresholds": thresholds}
    return {"label": "断层", "level": "低", "score": 0, "thresholds": thresholds}


def eval_prev_broken_red_rate(rate: Optional[float], has_sample: bool) -> Dict[str, object]:
    thresholds = "≥65% 修复健康；50–64% 勉强；35–49% 偏弱；<35% 断层"
    if not has_sample or rate is None:
        return {"label": "样本不足", "level": "N/A", "score": 0, "thresholds": thresholds}
    pct = rate * 100.0
    if pct >= 65:
        return {"label": "修复健康", "level": "高", "score": 2, "thresholds": thresholds}
    if pct >= 50:
        return {"label": "勉强", "level": "中", "score": 1, "thresholds": thresholds}
    if pct >= 35:
        return {"label": "偏弱", "level": "低", "score": 0, "thresholds": thresholds}
    return {"label": "断层", "level": "低", "score": 0, "thresholds": thresholds}


def eval_amount_change(change: Optional[float], has_sample: bool) -> Dict[str, object]:
    thresholds = "放量≥+10% 有增量；持平[-10%,+10%] 存量博弈；缩量≤-10% 风险上升"
    if not has_sample or change is None:
        return {"label": "样本不足", "level": "N/A", "score": 0, "thresholds": thresholds}
    pct = change * 100.0
    if pct >= 10:
        return {"label": "有增量", "level": "高", "score": 2, "thresholds": thresholds}
    if pct <= -10:
        return {"label": "风险上升", "level": "低", "score": 0, "thresholds": thresholds}
    return {"label": "存量博弈", "level": "中", "score": 1, "thresholds": thresholds}


def emotion_stage(total_score: int) -> Tuple[str, str]:
    if total_score >= 12:
        return "主升", "60–100%"
    if total_score >= 9:
        return "修复", "30–50%"
    if total_score >= 6:
        return "混沌", "≤20%"
    return "冰点", "0–10%"


def compute_metrics(engine: Engine, trade_date: str) -> Dict[str, object]:
    prev_date = prev_trade_date(engine, trade_date)
    df = fetch_trade_frame(engine, trade_date)
    df = add_limit_flags(df)
    df_valid = df
    if df is not None and not df.empty:
        df_valid = df[df["valid_prev"]].copy()

    name_map = {}
    is_st_map: Dict[str, bool] = {}
    if not df_valid.empty:
        name_map = {
            str(row.stock_code): str(row.stock_name or "")
            for row in df_valid.itertuples(index=False)
        }
        is_st_map = {
            str(row.stock_code): bool(row.is_st)
            for row in df_valid.itertuples(index=False)
        }
    theme_summary = compute_theme_summary(engine, trade_date, is_st_map=is_st_map)
    pools_today = fetch_xgb_pools(trade_date)
    limit_up_df, limit_up_codes = prepare_pool_df(pools_today.get("limit_up"), name_map, is_st_map=is_st_map)
    limit_down_df, limit_down_codes = prepare_pool_df(pools_today.get("limit_down"), name_map, is_st_map=is_st_map)
    broken_df, broken_codes = prepare_pool_df(pools_today.get("broken"), name_map, is_st_map=is_st_map)
    if not limit_up_codes:
        limit_up_df, limit_up_codes = build_local_pool_df(df_valid, "limit_up")
    if not limit_down_codes:
        limit_down_df, limit_down_codes = build_local_pool_df(df_valid, "limit_down")
    if not broken_codes:
        broken_df, broken_codes = build_local_pool_df(df_valid, "broken")

    limit_up_count = int(len(limit_up_codes))
    limit_down_count = int(len(limit_down_codes))
    broken_count = int(len(broken_codes))
    sealed_count = limit_up_count
    broken_rate = None
    if (broken_count + sealed_count) > 0:
        broken_rate = broken_count / float(broken_count + sealed_count)
    max_consecutive = compute_max_consecutive(
        engine,
        trade_date,
        limit_up_codes,
        name_map,
        is_st_map=is_st_map,
        lookback=LOOKBACK_DAYS,
    )

    prev_limit_up_count = 0
    red_count = 0
    red_rate = None
    prev_broken_count = 0
    prev_broken_red_count = 0
    prev_broken_red_rate = None
    prev_prev = prev_trade_date(engine, prev_date) if prev_date else None
    prev_df = fetch_trade_frame(engine, prev_date) if prev_date else pd.DataFrame()
    prev_df_flagged = add_limit_flags(prev_df) if prev_df is not None and not prev_df.empty else pd.DataFrame()
    prev_df_valid = (
        prev_df_flagged[prev_df_flagged["valid_prev"]].copy()
        if prev_df_flagged is not None and not prev_df_flagged.empty
        else pd.DataFrame()
    )
    prev_prev_df = fetch_trade_frame(engine, prev_prev) if prev_prev else pd.DataFrame()
    if prev_date:
        pools_prev = fetch_xgb_pools(prev_date)
        _, prev_limit_up_codes = prepare_pool_df(pools_prev.get("limit_up"), name_map, is_st_map=is_st_map)
        if not prev_limit_up_codes:
            _, prev_limit_up_codes = build_local_pool_df(prev_df_valid, "limit_up")
        prev_limit_up_count = len(prev_limit_up_codes)
        if prev_limit_up_codes and not df_valid.empty:
            df_today = df_valid.set_index("stock_code")
            for code in prev_limit_up_codes:
                if code not in df_today.index:
                    continue
                row = df_today.loc[code]
                try:
                    if float(row["close"]) > float(row["prev_close"]):
                        red_count += 1
                except Exception:  # noqa: BLE001
                    continue
            if prev_limit_up_count > 0:
                red_rate = red_count / float(prev_limit_up_count)

        _, prev_broken_codes = prepare_pool_df(pools_prev.get("broken"), name_map, is_st_map=is_st_map)
        if not prev_broken_codes:
            _, prev_broken_codes = build_local_pool_df(prev_df_valid, "broken")
        prev_broken_count = len(prev_broken_codes)
        if prev_broken_codes and not df_valid.empty:
            df_today = df_valid.set_index("stock_code")
            for code in prev_broken_codes:
                if code not in df_today.index:
                    continue
                row = df_today.loc[code]
                try:
                    if float(row["close"]) > float(row["prev_close"]):
                        prev_broken_red_count += 1
                except Exception:  # noqa: BLE001
                    continue
            if prev_broken_count > 0:
                prev_broken_red_rate = prev_broken_red_count / float(prev_broken_count)

    amount_today = 0.0
    if df is not None and not df.empty:
        amount_today = float(pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).sum())
    prev_dates = prev_trade_dates(engine, trade_date, limit=5)
    amount_avg5 = None
    amount_change = None
    if prev_dates:
        vals = [amount_sum_for_date(engine, d) for d in prev_dates]
        if vals:
            amount_avg5 = float(sum(vals) / len(vals))
            if amount_avg5 > 0:
                amount_change = (amount_today - amount_avg5) / amount_avg5

    evals = {
        "limit_up": eval_limit_up_count(limit_up_count),
        "limit_down": eval_limit_down_count(limit_down_count),
        "broken_rate": eval_broken_rate(broken_rate, has_sample=(broken_count + sealed_count) > 0),
        "max_consecutive": eval_max_consecutive(max_consecutive),
        "red_rate": eval_red_rate(red_rate, has_sample=prev_limit_up_count > 0),
        "prev_broken_red_rate": eval_prev_broken_red_rate(prev_broken_red_rate, has_sample=prev_broken_count > 0),
        "amount_change": eval_amount_change(amount_change, has_sample=amount_avg5 is not None),
    }
    total_score = int(sum(int(v["score"]) for v in evals.values()))
    stage, position = emotion_stage(total_score)
    divergence_summary = build_divergence_summary(
        prev_df,
        df_valid,
        prev_prev_df if prev_prev else None,
        prev_date,
        stage,
        theme_summary.get("top"),
        theme_summary.get("items_by_date") or {},
    )

    return {
        "trade_date": trade_date,
        "prev_date": prev_date,
        "limit_up_count": limit_up_count,
        "limit_down_count": limit_down_count,
        "broken_count": broken_count,
        "sealed_count": sealed_count,
        "broken_rate": broken_rate,
        "max_consecutive": max_consecutive,
        "prev_limit_up_count": prev_limit_up_count,
        "red_count": red_count,
        "red_rate": red_rate,
        "prev_broken_count": prev_broken_count,
        "prev_broken_red_count": prev_broken_red_count,
        "prev_broken_red_rate": prev_broken_red_rate,
        "amount_today": amount_today,
        "amount_avg5": amount_avg5,
        "amount_change": amount_change,
        "limit_up_df": limit_up_df,
        "limit_down_df": limit_down_df,
        "broken_df": broken_df,
        "evals": evals,
        "total_score": total_score,
        "stage": stage,
        "position": position,
        "prev_prev_date": prev_prev,
        "theme_summary": theme_summary,
        "divergence_summary": divergence_summary,
    }


def fmt_pct(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value * 100.0:.2f}%"


def fmt_amount(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:,.2f}"


def fmt_price(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.2f}"


def fmt_volume(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:,.0f}"


def _sanitize_md(value: str) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    return text.replace("|", "｜")


def render_detail_table(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return ["无"]
    columns = [str(c) for c in df.columns]
    header = "| " + " | ".join(_sanitize_md(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]

    def _fmt_cell(val: object, col_name: str) -> str:
        col_key = str(col_name or "").lower()
        if "代码" in col_name or col_key.endswith("code") or col_key in {"code", "symbol"}:
            code = _normalize_code(val)
            if code:
                return code
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return ""
        if isinstance(val, (int, float, np.floating)):
            if isinstance(val, (int, np.integer)):
                return f"{int(val)}"
            v = float(val)
            if abs(v - int(v)) < 1e-6:
                return f"{int(v)}"
            return f"{v:.2f}"
        return str(val)

    for row in df.itertuples(index=False):
        values = [_fmt_cell(getattr(row, col), col) for col in df.columns]
        lines.append("| " + " | ".join(_sanitize_md(v) for v in values) + " |")
    return lines


def render_theme_table(themes: List[Dict[str, object]], limit: int = 10) -> List[str]:
    if not themes:
        return ["无"]
    rows: List[Dict[str, object]] = []
    for theme in themes[:limit]:
        tiers = theme.get("tiers") or []
        tier_label = "多梯队" if theme.get("has_multi_tiers") else "无梯队"
        if tiers:
            tier_label = f"{tier_label}({','.join(str(t) for t in tiers)})"
        rows.append(
            {
                "题材": theme.get("name") or "",
                "涨停数": theme.get("limit_up_count") or 0,
                "最高板": theme.get("max_limit_up_days") or 0,
                "梯队": tier_label,
                "广度": f"{theme.get('breadth_label')}({theme.get('breadth_score')})",
                "高度": f"{theme.get('height_label')}({theme.get('height_score')})",
                "持续": f"{theme.get('persistence_label')}({theme.get('persistence_score')})",
                "排他": f"{theme.get('exclusivity_label')}({theme.get('exclusivity_score')})",
                "总分": theme.get("total_score") or 0,
                "定位": theme.get("position") or "",
            }
        )
    df = pd.DataFrame(rows)
    return render_detail_table(df)


def render_ladder_distribution(board_name_map: Dict[int, List[str]], limit: int = 6) -> List[str]:
    if not board_name_map:
        return ["无"]
    rows: List[Dict[str, object]] = []
    for board in sorted(board_name_map.keys(), reverse=True):
        names = board_name_map[board]
        if not names:
            continue
        shown = names[:limit]
        more = len(names) - len(shown)
        label = ", ".join(shown)
        if more > 0:
            label = f"{label}...(+{more})"
        rows.append({"连板": f"{board}板", "数量": len(names), "代表票": label})
    df = pd.DataFrame(rows)
    return render_detail_table(df)


def _amount_trend_label(amount: Optional[float], prev_amount: Optional[float]) -> str:
    if amount is None or prev_amount is None or prev_amount <= 0:
        return "N/A"
    ratio = float(amount) / float(prev_amount)
    if ratio >= 1.3:
        return "放量"
    if ratio <= 0.7:
        return "缩量"
    return "平量"


def render_divergence_samples(samples: Sequence[Dict[str, object]]) -> List[str]:
    if not samples:
        return ["无"]
    rows: List[Dict[str, object]] = []
    for sample in samples:
        rows.append(
            {
                "样本": sample.get("label") or "",
                "股票": sample.get("stock") or "",
                "分歧类型": sample.get("divergence_type") or "",
                "位置": sample.get("position") or "",
                "量能": sample.get("volume") or "",
                "承接方式": sample.get("support") or "",
                "环境": sample.get("environment") or "",
                "次日结果": sample.get("next_result") or "",
                "是否符合预期": sample.get("expected_match") or "",
            }
        )
    df = pd.DataFrame(rows)
    return render_detail_table(df)


def render_report(metrics: Dict[str, object]) -> str:
    trade_date = str(metrics["trade_date"])
    prev_date = metrics.get("prev_date") or "无"
    date_obj = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
    title = f"{date_obj.year}年{date_obj.month:02d}月{date_obj.day:02d}日 收盘复盘"

    evals = metrics["evals"]
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## 市场情绪复盘")
    lines.append(f"交易日：{trade_date}（前一交易日：{prev_date}）")
    lines.append("统计口径：Xuangubao 涨停/跌停/炸板池，剔除ST；涨跌停价=昨收×(1±涨停幅度)四舍五入到分；炸板=最高价触及涨停且收盘未封死。")
    lines.append("")

    lines.append("### 涨停家数")
    lines.append(f"数值：{metrics['limit_up_count']} 家")
    lines.append(f"判定：{evals['limit_up']['label']}（档位：{evals['limit_up']['level']}，得分 +{evals['limit_up']['score']}）")
    lines.append(f"阈值：{evals['limit_up']['thresholds']}")
    lines.append("口径：Xuangubao 涨停股池（剔除ST）")
    lines.append("")
    lines.append("#### 涨停明细")
    lines.extend(render_detail_table(metrics.get("limit_up_df")))
    lines.append("")

    lines.append("### 跌停家数")
    lines.append(f"数值：{metrics['limit_down_count']} 家")
    lines.append(f"判定：{evals['limit_down']['label']}（档位：{evals['limit_down']['level']}，得分 +{evals['limit_down']['score']}）")
    lines.append(f"阈值：{evals['limit_down']['thresholds']}")
    lines.append("口径：Xuangubao 跌停股池（剔除ST）")
    lines.append("")
    lines.append("#### 跌停明细")
    lines.extend(render_detail_table(metrics.get("limit_down_df")))
    lines.append("")

    lines.append("### 炸板率")
    lines.append(
        f"数值：{fmt_pct(metrics['broken_rate'])}（炸板 {metrics['broken_count']} / 封板 {metrics['sealed_count']}）"
    )
    lines.append(f"判定：{evals['broken_rate']['label']}（档位：{evals['broken_rate']['level']}，得分 +{evals['broken_rate']['score']}）")
    lines.append(f"阈值：{evals['broken_rate']['thresholds']}")
    lines.append("口径：Xuangubao 炸板股池（剔除ST）")
    lines.append("")
    lines.append("#### 炸板明细")
    lines.extend(render_detail_table(metrics.get("broken_df")))
    lines.append("")

    lines.append("### 最高连板")
    lines.append(f"数值：{metrics['max_consecutive']} 板")
    lines.append(f"判定：{evals['max_consecutive']['label']}（档位：{evals['max_consecutive']['level']}，得分 +{evals['max_consecutive']['score']}）")
    lines.append(f"阈值：{evals['max_consecutive']['thresholds']}")
    lines.append("口径：收盘涨停连续天数（截至当日）")
    lines.append("")

    lines.append("### 昨日涨停红盘率")
    lines.append(
        f"数值：{fmt_pct(metrics['red_rate'])}（红盘 {metrics['red_count']} / 昨涨停 {metrics['prev_limit_up_count']}）"
    )
    lines.append(f"判定：{evals['red_rate']['label']}（档位：{evals['red_rate']['level']}，得分 +{evals['red_rate']['score']}）")
    lines.append(f"阈值：{evals['red_rate']['thresholds']}")
    lines.append("口径：昨日涨停股池（剔除ST），今日收盘价 > 昨日收盘价")
    lines.append("")

    lines.append("### 昨日炸板红盘率")
    lines.append(
        f"数值：{fmt_pct(metrics['prev_broken_red_rate'])}（红盘 {metrics['prev_broken_red_count']} / 昨炸板 {metrics['prev_broken_count']}）"
    )
    lines.append(
        f"判定：{evals['prev_broken_red_rate']['label']}（档位：{evals['prev_broken_red_rate']['level']}，得分 +{evals['prev_broken_red_rate']['score']}）"
    )
    lines.append(f"阈值：{evals['prev_broken_red_rate']['thresholds']}")
    lines.append("口径：昨日炸板股池（剔除ST），今日收盘价 > 昨日收盘价")
    lines.append("")

    lines.append("### 成交额变化")
    lines.append(f"今日成交额：{fmt_amount(metrics['amount_today'])}")
    lines.append(f"5日均值：{fmt_amount(metrics['amount_avg5'])}")
    lines.append(f"变化：{fmt_pct(metrics['amount_change'])}")
    lines.append(f"判定：{evals['amount_change']['label']}（档位：{evals['amount_change']['level']}，得分 +{evals['amount_change']['score']}）")
    lines.append(f"阈值：{evals['amount_change']['thresholds']}")
    lines.append("口径：全市场成交额（stock_daily.amount）求和")
    lines.append("")

    lines.append("## 情绪总评")
    lines.append(f"情绪总分：{metrics['total_score']} / 14")
    lines.append(f"情绪阶段：{metrics['stage']}")
    lines.append(f"明日总仓位上限：{metrics['position']}")

    lines.append("")
    theme_summary = metrics.get("theme_summary") or {}
    themes = theme_summary.get("themes") or []
    top_theme = theme_summary.get("top")
    second_theme = theme_summary.get("second")
    lookback_dates = theme_summary.get("lookback_dates") or []
    today_theme_items = theme_summary.get("today_theme_items") or {}
    lines.append("## 模板二：主线题材复盘（详细）")
    lines.append("### 1) 主线定义")
    lines.append("主线 = 资金共识的最大公约数；不是行业趋势、不是逻辑最硬，而是当下最集中、最具持续性的情绪承载体。")
    lines.append("")
    lines.append("### 2) 四维度量化口径")
    lines.append("- 广度：题材内涨停家数、涨幅≥5%家数、是否有梯队结构")
    lines.append("- 高度：题材内最高连板高度是否打开空间")
    lines.append("- 持续：近 3–5 个交易日是否反复成为涨停最多题材、退潮后是否有资金回流")
    lines.append("- 排他：是否明显压制其他题材、非主线个股是否高开低走")
    lines.append("")
    lines.append("### 3) 量化打分（0/1/2）")
    lines.append("| 维度 | 分值 | 参考 |")
    lines.append("|---|---|---|")
    lines.append("| 广度 | 0/1/2 | ≥5 涨停+多梯队=2；3–4 涨停=1；≤2 涨停=0 |")
    lines.append("| 高度 | 0/1/2 | 最高连板明显上行/打开空间=2；修复=1；无高度=0 |")
    lines.append("| 持续 | 0/1/2 | 2 天以上持续强=2；隔日轮动=1；一日游=0 |")
    lines.append("| 排他 | 0/1/2 | 明显压制其他题材=2；多线并行=1；你强我也强=0 |")
    lines.append("")
    lines.append("### 4) 总分定位")
    lines.append("| 总分 | 定位 |")
    lines.append("|---|---|")
    lines.append("| 7–8 | 唯一主线 |")
    lines.append("| 5–6 | 主线 |")
    lines.append("| 3–4 | 次主线 |")
    lines.append("| ≤2 | 轮动/杂音 |")
    lines.append("")
    lines.append("### 5) 交易过滤（三个必过）")
    lines.append("- 是否末期主线：高位一致、连板一字、题材被过度解读")
    lines.append("- 是否高度压制：龙头 6–7 板、次龙 2–3 板失真")
    lines.append("- 是否监管敏感：涉炒作红线、盘后点名/问询")
    lines.append("")
    lines.append("### 6) 当日题材量化结果")
    if themes:
        lines.extend(render_theme_table(themes, limit=10))
        if lookback_dates:
            lines.append("")
            lines.append(
                "口径：XGB 涨停池（剔除 ST）；持续性回看 "
                f"{len(lookback_dates)} 个交易日：{', '.join(lookback_dates)}"
            )
        else:
            lines.append("")
            lines.append("口径：XGB 涨停池（剔除 ST）")
    else:
        lines.append("无题材数据（非交易日或涨停池为空）。")

    lines.append("")
    lines.append("### 7) 复盘输出")
    lines.append("【主线题材复盘】")
    if top_theme:
        tiers = "多梯队" if top_theme.get("has_multi_tiers") else "无梯队"
        lines.append(f"题材A：{top_theme.get('name')}")
        lines.append(
            f"广度：{top_theme.get('breadth_label')}（涨停{top_theme.get('limit_up_count')}，{tiers}）"
        )
        lines.append(
            f"高度：{top_theme.get('height_label')}（最高板{top_theme.get('max_limit_up_days')}）"
        )
        persistence_detail = top_theme.get("persistence_detail") or ""
        if persistence_detail:
            lines.append(f"持续性：{top_theme.get('persistence_label')}（{persistence_detail}）")
        else:
            lines.append(f"持续性：{top_theme.get('persistence_label')}")
        lines.append(f"排他性：{top_theme.get('exclusivity_label')}")
        lines.append(f"总分：{top_theme.get('total_score')}")
        lines.append(f"定位：{top_theme.get('position')}")
        sample_stocks = top_theme.get("sample_stocks") or []
        if sample_stocks:
            lines.append(f"代表票：{', '.join(sample_stocks)}")
    else:
        lines.append("题材A：N/A")
        lines.append("广度：N/A")
        lines.append("高度：N/A")
        lines.append("持续性：N/A")
        lines.append("排他性：N/A")
        lines.append("总分：N/A")
        lines.append("定位：N/A")

    lines.append("")
    if second_theme:
        tiers_b = "多梯队" if second_theme.get("has_multi_tiers") else "无梯队"
        lines.append(f"题材B（备选）：{second_theme.get('name')}")
        lines.append(
            f"广度：{second_theme.get('breadth_label')}（涨停{second_theme.get('limit_up_count')}，{tiers_b}）"
        )
        lines.append(
            f"高度：{second_theme.get('height_label')}（最高板{second_theme.get('max_limit_up_days')}）"
        )
        persistence_detail = second_theme.get("persistence_detail") or ""
        if persistence_detail:
            lines.append(f"持续性：{second_theme.get('persistence_label')}（{persistence_detail}）")
        else:
            lines.append(f"持续性：{second_theme.get('persistence_label')}")
        lines.append(f"排他性：{second_theme.get('exclusivity_label')}")
        lines.append(f"总分：{second_theme.get('total_score')}")
        lines.append(f"定位：{second_theme.get('position')}")
        sample_stocks = second_theme.get("sample_stocks") or []
        if sample_stocks:
            lines.append(f"代表票：{', '.join(sample_stocks)}")
    else:
        lines.append("题材B（备选）：（无）")

    lines.append("")
    lines.append("> 明日我只围绕：________ 这条题材做交易")

    lines.append("")
    lines.append("## 模板三：梯队结构复盘（详细）")
    lines.append("### 1) 梯队定义")
    lines.append("有效主线内，股票分为：龙头 / 次龙 / 补涨 / 炮灰。")
    lines.append("")
    lines.append("### 2) 龙头量化识别（硬条件）")
    lines.append("- 空间优势：主线内最高连板或唯一打破前高")
    lines.append("- 时间优势：最早启动或第一批封板")
    lines.append("- 资金一致性：封单可回封、分歧日不爆量、多日换手递增")
    lines.append("- 情绪代表性：它动时同题材票跟随")
    lines.append("- 指数脱钩能力（加分项）：指数弱它不弱")
    lines.append("")
    lines.append("### 3) 梯队健康度评分")
    lines.append("| 项目 | 分值 |")
    lines.append("|---|---|")
    lines.append("| 龙头稳定性 | 0/1/2 |")
    lines.append("| 次龙承接力 | 0/1/2 |")
    lines.append("| 补涨尝试 | 0/1/2 |")
    lines.append("| 失败淘汰效率 | 0/1/2 |")
    lines.append("")
    lines.append("### 总分 → 策略")
    lines.append("| 分数 | 结论 |")
    lines.append("|---|---|")
    lines.append("| 7–8 | 主升期，可重仓 |")
    lines.append("| 5–6 | 可做，控仓 |")
    lines.append("| 3–4 | 边走边看 |")
    lines.append("| ≤2 | 准备撤退 |")
    lines.append("")
    lines.append("### 4) 当日梯队结构统计")
    if top_theme:
        mainline = top_theme.get("name") or ""
        ladder_items = today_theme_items.get(mainline) or []
        ladder = build_ladder_summary(top_theme, ladder_items)
        lines.append(f"主线名称：{mainline or 'N/A'}")
        if ladder.get("has_data"):
            board_name_map = ladder.get("board_name_map") or {}
            lines.extend(render_ladder_distribution(board_name_map))
            lines.append("")
            lines.append("口径：基于 XGB 涨停池连板天数构造梯队；失败淘汰效率以梯队层级完整度近似。")
            lines.append("")
            lines.append("### 5) 复盘输出模板（三）")
            lines.append("【梯队结构复盘】")
            lines.append(f"龙头：{ladder.get('leader_name')}")
            lines.append(f"空间：{ladder.get('leader_days')}板")
            lines.append(f"状态（加速 / 分歧 / 断板）：{ladder.get('status')}")
            secondaries = ladder.get("secondary_names") or []
            secondaries += ["（无）", "（无）"]
            lines.append(f"次龙1：{secondaries[0]}")
            lines.append(f"次龙2：{secondaries[1]}")
            lines.append(f"补涨：{ladder.get('supplement_label')}")
            scores = ladder.get("score_breakdown") or {}
            lines.append(
                "梯队健康分："
                f"{ladder.get('health_score')}（{ladder.get('health_label')}）"
                f"；拆解：龙头{scores.get('leader', 0)} / 次龙{scores.get('secondary', 0)} / 补涨{scores.get('supplement', 0)} / 淘汰{scores.get('efficiency', 0)}"
            )
            lines.append(f"明日策略：{ladder.get('health_label')}")
        else:
            lines.append("梯队结构统计：无")
            lines.append("")
            lines.append("### 5) 复盘输出模板（三）")
            lines.append("【梯队结构复盘】")
            lines.append("龙头：N/A")
            lines.append("空间：N/A")
            lines.append("状态（加速 / 分歧 / 断板）：N/A")
            lines.append("次龙1：N/A")
            lines.append("次龙2：N/A")
            lines.append("补涨：N/A")
            lines.append("梯队健康分：N/A")
            lines.append("明日策略：N/A")
    else:
        lines.append("主线名称：N/A")
        lines.append("梯队结构统计：无")
        lines.append("")
        lines.append("### 5) 复盘输出模板（三）")
        lines.append("【梯队结构复盘】")
        lines.append("龙头：N/A")
        lines.append("空间：N/A")
        lines.append("状态（加速 / 分歧 / 断板）：N/A")
        lines.append("次龙1：N/A")
        lines.append("次龙2：N/A")
        lines.append("补涨：N/A")
        lines.append("梯队健康分：N/A")
        lines.append("明日策略：N/A")

    lines.append("")
    divergence_summary = metrics.get("divergence_summary") or {}
    samples = divergence_summary.get("samples") or []
    source_date = divergence_summary.get("source_date")
    note = divergence_summary.get("note") or ""
    lines.append("## 模板四：分歧形态复盘（详细）")
    lines.append("### 1) 底层定义")
    lines.append("分歧不是风险，是“定价权重新分配的过程”；真正危险的是分歧方式选错。")
    lines.append("")
    lines.append("### 2) 四大类分歧")
    lines.append("| 类型 | 本质 | 是否可博 |")
    lines.append("|---|---|---|")
    lines.append("| 健康分歧 | 洗不坚定筹码 | 可博 |")
    lines.append("| 结构分歧 | 层级切换 | 克制 |")
    lines.append("| 末期分歧 | 出货前摇 | 不博 |")
    lines.append("| 伪分歧 | 资金撤退 | 不博 |")
    lines.append("")
    lines.append("### 3) 可量化的分歧判断维度")
    lines.append("1) 位置（第几板/相对高度）")
    lines.append("2) 量能（放量 or 缩量）")
    lines.append("3) 时间（分歧发生在何时）")
    lines.append("4) 承接（谁在接/怎么接）")
    lines.append("5) 环境（情绪阶段/主线状态）")
    lines.append("")
    lines.append("### 4) 四类分歧量化标准（摘要）")
    lines.append("- 健康分歧：2–4板/次龙；缩量或温和放量；早盘/盘中；快速回封；主线明确且情绪非退潮")
    lines.append("- 结构分歧：龙头震荡、次龙尝试补位；同题材强弱分化")
    lines.append("- 末期分歧：≥5板高位首次大分歧、历史最大量、高开低走/反抽无力、板上反复开板")
    lines.append("- 伪分歧：低位无人承接、主线内集体缩量走弱、分歧后无修复尝试")
    lines.append("")
    lines.append("### 5) 分歧量化打分模型")
    lines.append("| 项目 | 分值 |")
    lines.append("|---|---|")
    lines.append("| 位置合理 | 0/1/2 |")
    lines.append("| 量能健康 | 0/1/2 |")
    lines.append("| 承接明确 | 0/1/2 |")
    lines.append("| 环境支持 | 0/1/2 |")
    lines.append("| 梯队未断 | 0/1/2 |")
    lines.append("")
    lines.append("| 分数 | 结论 |")
    lines.append("|---|---|")
    lines.append("| 8–10 | 可重仓 |")
    lines.append("| 6–7 | 轻仓博 |")
    lines.append("| 4–5 | 观察 |")
    lines.append("| ≤3 | 直接放弃 |")
    lines.append("")
    lines.append("### 6) 买卖点SOP")
    lines.append("- 买点：只买“分歧转一致”的瞬间，不提前、不预测")
    lines.append("- 卖点：分歧未修复→次日无条件走；修复失败→盘中走；形态升级→立即走")
    lines.append("")
    lines.append("### 7) 分歧样本（模板四）")
    if source_date:
        lines.append(f"样本日期：{source_date}")
    if note:
        lines.append(f"备注：{note}")
    lines.extend(render_divergence_samples(samples))

    if metrics.get("prev_prev_date") is None:
        lines.append("备注：缺少前一交易日或更早的K线，昨日涨停红盘率可能偏低或无法计算。")
    if metrics.get("prev_date") is None:
        lines.append("备注：缺少前一交易日数据，涨跌停/炸板等指标无法准确计算。")
    return "\n".join(lines).strip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="每日复盘报告（基于 stock_daily）")
    parser.add_argument("--config", default=None, help="config.ini 路径")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy DB URL")
    parser.add_argument("--db", default=None, help="SQLite 文件路径")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD / YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD / YYYYMMDD")
    parser.add_argument("--output-path", default=".", help="输出目录")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    db_target = resolve_db_target(args)
    engine = make_engine(db_target)

    end_date = parse_date_arg(args.end_date) if args.end_date else None
    if not end_date:
        end_date = latest_trade_date(engine)
        if not end_date:
            logging.warning("stock_daily 为空，无法生成复盘报告")
            return 0

    start_date = parse_date_arg(args.start_date) if args.start_date else end_date
    if start_date > end_date:
        raise SystemExit("start-date 必须 <= end-date")

    trade_dates = list_trade_dates(engine, start_date, end_date)
    if not trade_dates:
        logging.warning("区间内没有交易日数据：%s ~ %s", start_date, end_date)
        return 0

    out_path = Path(args.output_path or ".").expanduser()
    if out_path.suffix.lower() == ".md":
        out_path = out_path.parent
    out_path.mkdir(parents=True, exist_ok=True)

    for trade_date in trade_dates:
        metrics = compute_metrics(engine, trade_date)
        content = render_report(metrics)
        filename = f"daily-review-{yyyymmdd(trade_date)}.md"
        file_path = out_path / filename
        file_path.write_text(content, encoding="utf-8")
        logging.info("输出：%s", file_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
