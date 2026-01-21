# -*- coding: utf-8 -*-

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from a_stock_analyzer import db
from a_stock_analyzer.akshare_client import normalize_stock_code, pick_col
from a_stock_analyzer.runtime import (
    add_db_args,
    normalize_trade_date,
    resolve_db_from_args,
    resolve_latest_stock_daily_date,
    setup_logging,
    yyyymmdd_from_date,
)


TAIL_START_MINUTE = 14 * 60 + 30


@dataclass(frozen=True)
class MarketEmotion:
    market_emotion: str
    emotion_score: int
    operation_permission: bool
    limit_up_count: int
    limit_down_count: int
    max_consecutive: int
    prev_max_consecutive: Optional[int]
    broken_count: int
    sealed_count: int
    broken_rate: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_emotion": self.market_emotion,
            "emotion_score": int(self.emotion_score),
            "operation_permission": bool(self.operation_permission),
            "limit_up_count": int(self.limit_up_count),
            "limit_down_count": int(self.limit_down_count),
            "max_consecutive": int(self.max_consecutive),
            "prev_max_consecutive": int(self.prev_max_consecutive)
            if self.prev_max_consecutive is not None
            else None,
            "broken_count": int(self.broken_count),
            "sealed_count": int(self.sealed_count),
            "broken_rate": float(self.broken_rate) if self.broken_rate is not None else None,
        }


def _safe_str(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none"} else s


def _to_int(v: object, default: int = 0) -> int:
    try:
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:  # noqa: BLE001
        return default


def _to_float(v: object) -> Optional[float]:
    try:
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return None
        return float(x)
    except Exception:  # noqa: BLE001
        return None


def _time_to_minute(v: object) -> Optional[int]:
    s = _safe_str(v).replace("：", ":")
    if not s:
        return None

    # 930, 93000, 093000, ...
    if s.isdigit():
        s2 = s.zfill(6)[0:6]
        hh = int(s2[0:2])
        mm = int(s2[2:4])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh * 60 + mm
        return None

    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return hh * 60 + mm
    return None


def _split_themes(raw: object, *, limit: int = 3) -> List[str]:
    s = _safe_str(raw)
    if not s or s in {"-", "—", "--", "暂无", "不明", "无", "未知"}:
        return []
    parts = re.split(r"[;,，；、/\\|&+]+", s)
    out = [p.strip() for p in parts if p and p.strip()]
    return out[: max(1, int(limit))] if out else []


def _pick_best_text_col(df: pd.DataFrame, candidates: List[str], *, label: str) -> Optional[str]:
    """
    Choose the most informative text column among multiple candidates.
    Tie-breaker: prefer earlier candidates when coverage is close.
    """
    if df is None or df.empty:
        return None

    resolved: List[Tuple[str, int]] = []
    for i, cand in enumerate(candidates):
        col = pick_col(df, [cand])
        if col is None or col not in df.columns:
            continue
        if any(col == c for c, _ in resolved):
            continue
        resolved.append((str(col), int(i)))

    if not resolved:
        return None

    bad = {"", "-", "—", "--", "暂无", "不明", "无", "未知"}
    best_col: Optional[str] = None
    best_cov = -1.0
    best_pri = 10**9

    for col, pri in resolved:
        try:
            s = df[col].map(_safe_str)
        except Exception:  # noqa: BLE001
            continue
        cov = float((~s.isin(bad) & s.notna()).mean())
        if cov > best_cov + 1e-6 or (abs(cov - best_cov) <= 0.02 and pri < best_pri):
            best_col = col
            best_cov = cov
            best_pri = pri

    if best_col:
        logging.debug("Selected %s column: %s (coverage=%.1f%%)", label, best_col, best_cov * 100)
    return best_col


def _call_akshare_df(func_names: Iterable[str], *, trade_date_norm: str) -> pd.DataFrame:
    try:
        import akshare as ak  # type: ignore
    except Exception as e:  # noqa: BLE001
        logging.warning("AkShare not available: %s", e)
        return pd.DataFrame()

    date_compact = yyyymmdd_from_date(trade_date_norm)
    for fn in func_names:
        f = getattr(ak, fn, None)
        if f is None:
            continue

        try:
            import inspect

            keys = list(inspect.signature(f).parameters.keys())
        except Exception:  # noqa: BLE001
            keys = []

        if "date" in keys:
            attempts = [("date", date_compact), ("date", trade_date_norm)]
        elif "trade_date" in keys:
            attempts = [("trade_date", date_compact), ("trade_date", trade_date_norm)]
        elif keys:
            attempts = [("pos", date_compact), ("pos", trade_date_norm)]
        else:
            attempts = [("none", None)]

        for mode, ds in attempts:
            try:
                if mode == "date":
                    return f(date=ds)
                if mode == "trade_date":
                    return f(trade_date=ds)
                if mode == "pos":
                    return f(ds)
                return f()
            except ValueError as e:
                msg = str(e)
                if "最近" in msg and "交易日" in msg:
                    logging.info("AkShare %s unavailable for %s: %s", fn, trade_date_norm, msg)
                    return pd.DataFrame()
                logging.warning("AkShare %s failed for %s: %s", fn, ds, msg)
                continue
            except TypeError:
                continue
            except Exception as e:  # noqa: BLE001
                logging.warning("AkShare %s failed for %s: %s", fn, ds, e)
                continue

    return pd.DataFrame()


def _fetch_limit_up_pool(trade_date_norm: str) -> pd.DataFrame:
    return _call_akshare_df(["stock_zt_pool_em"], trade_date_norm=trade_date_norm)


def _fetch_limit_down_pool(trade_date_norm: str) -> pd.DataFrame:
    return _call_akshare_df(["stock_dt_pool_em"], trade_date_norm=trade_date_norm)


def _fetch_broken_pool(trade_date_norm: str) -> pd.DataFrame:
    # 炸板股池：不同 AkShare 版本可能存在不同函数名
    return _call_akshare_df(
        ["stock_zt_pool_zbgc_em", "stock_zt_pool_zb_em", "stock_zt_pool_zb_gc_em"],
        trade_date_norm=trade_date_norm,
    )


def _normalize_basic_pool(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["stock_code", "stock_name"])

    code_col = pick_col(df, ["stock_code", "code", "代码", "股票代码"]) or df.columns[0]
    name_col = pick_col(df, ["stock_name", "name", "名称", "股票简称"])

    out = pd.DataFrame()
    out["stock_code"] = df[code_col].apply(normalize_stock_code)
    out["stock_name"] = df[name_col].astype(str) if name_col in df.columns else ""

    out["stock_code"] = out["stock_code"].astype(str).str.strip()
    out = out[out["stock_code"].str.fullmatch(r"\d{6}", na=False)]
    out = out.drop_duplicates(subset=["stock_code"]).reset_index(drop=True)
    return out


def _parse_lb_count_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().mean() >= 0.5:
        return x.fillna(1).astype(int)

    def parse_one(v: object) -> int:
        t = _safe_str(v)
        if not t:
            return 1
        m = re.search(r"(\d+)\s*板", t)
        if m:
            return int(m.group(1))
        m2 = re.search(r"连板\s*(\d+)", t)
        if m2:
            return int(m2.group(1))
        m3 = re.search(r"(\d+)", t)
        if m3:
            return max(1, int(m3.group(1)))
        return 1

    return s.apply(parse_one).astype(int)


def _normalize_limit_up_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize AkShare 东财涨停池 DataFrame -> a compact table for后续模块。
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "stock_code",
                "stock_name",
                "lb_count",
                "concept",
                "reason",
                "industry",
                "first_time_min",
                "last_time_min",
                "open_count",
                "seal_amount",
            ]
        )

    code_col = pick_col(df, ["stock_code", "code", "代码", "股票代码"]) or df.columns[0]
    name_col = pick_col(df, ["stock_name", "name", "名称", "股票简称"])
    lb_col = pick_col(df, ["连板数", "连板", "涨停统计", "板数"])

    industry_col = _pick_best_text_col(df, ["所属行业", "行业", "industry"], label="industry")
    # “概念”优先使用“所属概念/概念/题材”类字段；不要把“涨停原因”当成概念板块，避免主题误判。
    concept_col = _pick_best_text_col(
        df,
        ["所属概念", "所属题材", "概念", "题材", "concept"],
        label="concept",
    )
    reason_col = _pick_best_text_col(
        df,
        ["涨停原因", "涨停原因类别", "原因", "reason"],
        label="reason",
    )

    first_time_col = pick_col(df, ["首次封板时间", "首次封板", "首封时间", "首次涨停"])
    last_time_col = pick_col(df, ["最后封板时间", "最后封板", "末封时间", "最后涨停"])

    open_cnt_col = pick_col(df, ["炸板次数", "开板次数", "打开次数", "炸板", "开板"])
    seal_amount_col = pick_col(df, ["封板资金", "封板金额", "封单额", "封单金额", "封单资金"])

    out = pd.DataFrame()
    out["stock_code"] = df[code_col].apply(normalize_stock_code)
    out["stock_name"] = df[name_col].astype(str) if name_col in df.columns else ""

    out["lb_count"] = _parse_lb_count_series(df[lb_col]) if lb_col in df.columns else 1
    out["industry"] = df[industry_col].apply(_safe_str) if industry_col in df.columns else ""
    out["concept"] = df[concept_col].apply(_safe_str) if concept_col in df.columns else ""
    out["reason"] = df[reason_col].apply(_safe_str) if reason_col in df.columns else ""

    # Normalize placeholders to empty.
    bad = {"-", "—", "--", "暂无", "不明", "无", "未知"}
    out["industry"] = out["industry"].map(lambda x: "" if _safe_str(x) in bad else _safe_str(x))
    out["concept"] = out["concept"].map(lambda x: "" if _safe_str(x) in bad else _safe_str(x))
    out["reason"] = out["reason"].map(lambda x: "" if _safe_str(x) in bad else _safe_str(x))

    out["first_time_min"] = (
        df[first_time_col].apply(_time_to_minute) if first_time_col in df.columns else None
    )
    out["last_time_min"] = (
        df[last_time_col].apply(_time_to_minute) if last_time_col in df.columns else None
    )

    out["open_count"] = (
        pd.to_numeric(df[open_cnt_col], errors="coerce").fillna(0).astype(int)
        if open_cnt_col in df.columns
        else 0
    )
    out["seal_amount"] = (
        pd.to_numeric(df[seal_amount_col], errors="coerce")
        if seal_amount_col in df.columns
        else np.nan
    )

    out["stock_code"] = out["stock_code"].astype(str).str.strip()
    out = out[out["stock_code"].str.fullmatch(r"\d{6}", na=False)]
    out["lb_count"] = pd.to_numeric(out["lb_count"], errors="coerce").fillna(1).astype(int)

    # Deduplicate by keeping the "strongest" row.
    out = out.sort_values(
        ["lb_count", "seal_amount", "open_count"],
        ascending=[False, False, True],
        na_position="last",
    )
    out = out.drop_duplicates(subset=["stock_code"], keep="first").reset_index(drop=True)
    return out


def _is_st_name(stock_name: str) -> bool:
    name = _safe_str(stock_name)
    if not name:
        return False
    if "退" in name:
        return True
    return "ST" in name.upper()


def _infer_limit_pct(stock_code: str, stock_name: str) -> float:
    if _is_st_name(stock_name):
        return 0.05
    code = str(stock_code).strip()
    if code.startswith(("300", "301", "688")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def _round_half_up_2_series(v: pd.Series) -> pd.Series:
    arr = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
    out = np.floor(arr * 100.0 + 0.5) / 100.0
    return pd.Series(out, index=v.index)


def _calc_limit_up_streak(
    conn,
    *,
    stock_code: str,
    stock_name: str,
    trade_date_norm: str,
    limit_days: int = 12,
) -> int:  # noqa: ANN001
    hist = db.load_daily_until(conn, stock_code, end_date=trade_date_norm, limit=int(limit_days))
    if hist is None or hist.empty:
        return 1

    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    h = h.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if h.empty or str(h["date"].iloc[-1]) != trade_date_norm:
        return 1

    h["close"] = pd.to_numeric(h["close"], errors="coerce")
    limit_pct = float(_infer_limit_pct(stock_code, stock_name))
    eps = 1e-3
    prev_close = h["close"].shift(1)
    limit_up = _round_half_up_2_series(prev_close * (1.0 + limit_pct))
    is_up = (h["close"] - limit_up).abs() <= eps
    is_up = is_up.fillna(False)

    streak = 0
    for v in is_up.iloc[::-1].to_list():
        if bool(v):
            streak += 1
        else:
            break
    return max(1, int(streak))


def _build_pools_from_db(
    engine,
    *,
    trade_date_norm: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # noqa: ANN001
    sql = """
    SELECT
      d.stock_code,
      COALESCE(i.name, '') AS stock_name,
      d.open, d.high, d.low, d.close, d.volume, d.amount,
      (
        SELECT p.close
        FROM stock_daily p
        WHERE p.stock_code = d.stock_code AND p.date < d.date
        ORDER BY p.date DESC
        LIMIT 1
      ) AS prev_close
    FROM stock_daily d
    LEFT JOIN stock_info i ON i.stock_code = d.stock_code
    WHERE d.date = :d
    """

    with engine.connect() as conn:
        snap = pd.read_sql_query(text(sql), conn, params={"d": trade_date_norm})

    if snap is None or snap.empty:
        empty_zt = _normalize_limit_up_pool(pd.DataFrame())
        empty_dt = _normalize_basic_pool(pd.DataFrame())
        empty_zb = _normalize_basic_pool(pd.DataFrame())
        return empty_zt, empty_dt, empty_zb

    df = snap.copy()
    df["stock_code"] = df["stock_code"].apply(normalize_stock_code).astype(str).str.strip()
    df["stock_name"] = df["stock_name"].astype(str)
    df = df[df["stock_code"].str.fullmatch(r"\d{6}", na=False)].reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume", "amount", "prev_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df.empty:
        empty_zt = _normalize_limit_up_pool(pd.DataFrame())
        empty_dt = _normalize_basic_pool(pd.DataFrame())
        empty_zb = _normalize_basic_pool(pd.DataFrame())
        return empty_zt, empty_dt, empty_zb

    df["limit_pct"] = df.apply(lambda r: _infer_limit_pct(r["stock_code"], r["stock_name"]), axis=1)
    eps = 1e-3
    df["limit_up"] = _round_half_up_2_series(df["prev_close"] * (1.0 + df["limit_pct"]))
    df["limit_down"] = _round_half_up_2_series(df["prev_close"] * (1.0 - df["limit_pct"]))

    is_limit_up = (df["close"] - df["limit_up"]).abs() <= eps
    is_limit_down = (df["close"] - df["limit_down"]).abs() <= eps
    is_broken = (df["high"] >= df["limit_up"] - eps) & (df["close"] < df["limit_up"] - eps)

    zt_base = df[is_limit_up].copy()
    dt_base = df[is_limit_down].copy()
    zb_base = df[is_broken].copy()

    df_dt = dt_base[["stock_code", "stock_name"]].drop_duplicates().reset_index(drop=True)
    df_zb = zb_base[["stock_code", "stock_name"]].drop_duplicates().reset_index(drop=True)

    if zt_base.empty:
        df_zt = _normalize_limit_up_pool(pd.DataFrame())
        return df_zt, df_dt, df_zb

    df_zt = pd.DataFrame()
    df_zt["stock_code"] = zt_base["stock_code"].astype(str)
    df_zt["stock_name"] = zt_base["stock_name"].astype(str)
    df_zt["lb_count"] = 1
    df_zt["concept"] = ""
    df_zt["industry"] = ""
    df_zt["first_time_min"] = None
    df_zt["last_time_min"] = None
    df_zt["open_count"] = 0
    df_zt["seal_amount"] = np.nan

    # 连板数：仅对涨停股做小窗口回看，避免全市场扫描。
    with engine.connect() as conn:
        lb_list: List[int] = []
        for code, name in zip(df_zt["stock_code"].to_list(), df_zt["stock_name"].to_list()):
            lb_list.append(
                _calc_limit_up_streak(
                    conn,
                    stock_code=str(code),
                    stock_name=str(name),
                    trade_date_norm=trade_date_norm,
                )
            )
    df_zt["lb_count"] = lb_list
    df_zt["lb_count"] = pd.to_numeric(df_zt["lb_count"], errors="coerce").fillna(1).astype(int)
    df_zt = df_zt.sort_values(["lb_count", "stock_code"], ascending=[False, True]).reset_index(drop=True)
    return df_zt, df_dt, df_zb


def _build_pools(
    engine,
    *,
    trade_date_norm: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:  # noqa: ANN001
    df_zt = _normalize_limit_up_pool(_fetch_limit_up_pool(trade_date_norm))
    df_dt = _normalize_basic_pool(_fetch_limit_down_pool(trade_date_norm))
    df_zb = _normalize_basic_pool(_fetch_broken_pool(trade_date_norm))

    if not df_zt.empty or not df_dt.empty or not df_zb.empty:
        return df_zt, df_dt, df_zb, "akshare"

    df_zt, df_dt, df_zb = _build_pools_from_db(engine, trade_date_norm=trade_date_norm)
    return df_zt, df_dt, df_zb, "db"


def _resolve_prev_trade_date(engine, trade_date_norm: str) -> Optional[str]:  # noqa: ANN001
    try:
        with engine.connect() as conn:
            prev = conn.execute(
                text("SELECT MAX(date) FROM stock_daily WHERE date < :d"),
                {"d": trade_date_norm},
            ).fetchone()[0]
        return str(prev) if prev else None
    except Exception as e:  # noqa: BLE001
        logging.debug("Prev trade date unavailable (no DB?): %s", e)
        return None


def _resolve_effective_trade_date(engine, trade_date_norm: str) -> str:  # noqa: ANN001
    try:
        with engine.connect() as conn:
            effective = conn.execute(
                text("SELECT MAX(date) FROM stock_daily WHERE date <= :d"),
                {"d": trade_date_norm},
            ).fetchone()[0]

        if effective:
            effective_s = str(effective)
            if effective_s != trade_date_norm:
                logging.warning(
                    "trade_date=%s has no DB data; fallback to %s",
                    trade_date_norm,
                    effective_s,
                )
            return effective_s

        logging.debug("No stock_daily rows <= %s; fallback to requested date.", trade_date_norm)
        return trade_date_norm
    except Exception as e:  # noqa: BLE001
        logging.debug("Effective trade date unavailable (no DB?): %s", e)
        return trade_date_norm


def _calc_market_emotion(
    *,
    limit_up_count: int,
    limit_down_count: int,
    max_consecutive: int,
    prev_max_consecutive: Optional[int],
    broken_count: int,
    sealed_count: int,
) -> MarketEmotion:
    broken_rate = None
    # User-defined: 炸板率 = 炸板 / (炸板 + 封板)
    denom = int(broken_count) + int(sealed_count)
    if denom > 0:
        broken_rate = float(broken_count) / float(denom)

    trend = "flat"
    if prev_max_consecutive is not None:
        if max_consecutive > prev_max_consecutive:
            trend = "up"
        elif max_consecutive < prev_max_consecutive:
            trend = "down"

    # Stage判定：贴近 bingwu.md 的示例规则（含阈值），并做保守化处理。
    if limit_down_count > 10 or (broken_rate is not None and broken_rate > 0.50):
        stage = "退潮期"
        permission = False
    elif (
        limit_down_count <= 5
        and (broken_rate is None or broken_rate < 0.30)
        and trend in {"up", "flat"}
        and max_consecutive >= 2
    ):
        stage = "上升期"
        permission = True
    else:
        stage = "震荡期"
        permission = True

    score = 50
    if limit_up_count >= 80:
        score += 10
    elif limit_up_count >= 40:
        score += 5
    elif limit_up_count <= 15:
        score -= 10
    elif limit_up_count <= 30:
        score -= 5

    if limit_down_count <= 5:
        score += 10
    elif limit_down_count > 10:
        score -= 10

    if broken_rate is not None:
        if broken_rate < 0.30:
            score += 10
        elif broken_rate > 0.50:
            score -= 10

    if trend == "up":
        score += 10
    elif trend == "down":
        score -= 10

    score = int(max(0, min(100, round(score))))

    return MarketEmotion(
        market_emotion=stage,
        emotion_score=score,
        operation_permission=permission,
        limit_up_count=limit_up_count,
        limit_down_count=limit_down_count,
        max_consecutive=max_consecutive,
        prev_max_consecutive=prev_max_consecutive,
        broken_count=broken_count,
        sealed_count=sealed_count,
        broken_rate=broken_rate,
    )


def _scan_themes(
    df_zt: pd.DataFrame,
    df_zt_prev: pd.DataFrame,
    *,
    theme_type: str,
    theme_col: str,
    top_n: int = 10,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    if df_zt is None or df_zt.empty or theme_col not in df_zt.columns:
        return [], {}, []

    prev_set: set[str] = set()
    if df_zt_prev is not None and not df_zt_prev.empty and theme_col in df_zt_prev.columns:
        for v in df_zt_prev[theme_col].to_list():
            for t in _split_themes(v, limit=5):
                prev_set.add(t)

    all_max_lb = int(pd.to_numeric(df_zt["lb_count"], errors="coerce").fillna(1).max())

    rows: List[Dict[str, Any]] = []
    for r in df_zt.itertuples(index=False):
        raw = getattr(r, theme_col)
        for t in _split_themes(raw, limit=5):
            rows.append(
                {
                    "theme": t,
                    "stock_code": getattr(r, "stock_code"),
                    "lb_count": int(getattr(r, "lb_count")),
                    "first_time_min": getattr(r, "first_time_min"),
                    "last_time_min": getattr(r, "last_time_min"),
                }
            )

    d = pd.DataFrame(rows)
    if d.empty:
        return [], {}, []

    out: List[Dict[str, Any]] = []
    info: Dict[str, Dict[str, Any]] = {}
    core: List[str] = []

    for theme, grp in d.groupby("theme"):
        theme = _safe_str(theme) or "未知"
        n = int(grp["stock_code"].nunique())
        max_lb = int(pd.to_numeric(grp["lb_count"], errors="coerce").fillna(1).max())

        last_min = pd.to_numeric(grp["last_time_min"], errors="coerce")
        tail = bool(last_min.notna().any() and float(last_min.max()) >= TAIL_START_MINUTE)

        # “新题材”在 bingwu.md 里指“当日首次大规模出现”，因此需要同时满足：
        # 1) 昨日未出现；2) 当日涨停数达到阈值（>=5）
        new_theme = theme not in prev_set and theme != "未知" and n >= 5

        # 题材强度评分（0-8）：新题材 / 涨停数 / 龙头高度 / 尾盘回流
        # "未知" 不是有效题材：不参与核心题材判定，且强制为 0 分，避免误导。
        score = 0
        if theme != "未知":
            if new_theme:
                score += 2
            # bingwu.md: 板块涨停数量 >= 5 只 => +2
            if n >= 5:
                score += 2
            if all_max_lb >= 2 and max_lb >= all_max_lb:
                score += 2
            if tail:
                score += 2
        score = int(max(0, min(8, score)))

        is_core = bool(score >= 5 and theme != "未知")
        if is_core:
            core.append(theme)

        first_min = pd.to_numeric(grp["first_time_min"], errors="coerce")
        q20 = float(first_min.quantile(0.2)) if first_min.notna().any() else None

        out.append(
            {
                "theme_type": theme_type,
                "theme_name": theme,
                "theme_score": score,
                "is_core_theme": is_core,
                "limit_up_count": n,
                "max_consecutive": max_lb,
                "is_new_theme": bool(new_theme),
                "tail_inflow": bool(tail),
            }
        )
        info[theme] = {
            "theme_type": theme_type,
            "theme_name": theme,
            "theme_score": score,
            "is_core_theme": is_core,
            "limit_up_count": n,
            "max_consecutive": max_lb,
            "first_time_q20": q20,
        }

    out.sort(key=lambda x: (x["theme_score"], x["limit_up_count"], x["max_consecutive"]), reverse=True)
    out = out[: max(0, int(top_n))]
    core = sorted(set(core), key=core.index)
    return out, info, core


def _structure_stage(lb_count: int) -> str:
    lb = int(lb_count)
    if lb <= 1:
        return "启动期"
    if lb <= 3:
        return "主升初期"
    return "高位期"


def _pick_candidates(
    df_zt: pd.DataFrame,
    *,
    concept_info: Dict[str, Dict[str, Any]],
    industry_info: Dict[str, Dict[str, Any]],
    core_concepts: List[str],
    core_industries: List[str],
    top_n: int = 2,
) -> List[Dict[str, Any]]:
    if df_zt is None or df_zt.empty:
        return []

    core_concept_set = set(core_concepts)
    core_industry_set = set(core_industries)

    def pick_best_theme(
        raw: object,
        core_set: set[str],
        info_map: Dict[str, Dict[str, Any]],
    ) -> Optional[str]:
        themes = _split_themes(raw, limit=5)
        if not themes:
            return None
        best = None
        best_score = -1
        for t in themes:
            if t not in core_set:
                continue
            sc = int(info_map.get(t, {}).get("theme_score", 0))
            if sc > best_score:
                best = t
                best_score = sc
        return best

    seal_amount_p70 = None
    if "seal_amount" in df_zt.columns:
        arr = pd.to_numeric(df_zt["seal_amount"], errors="coerce").dropna()
        if not arr.empty:
            seal_amount_p70 = float(arr.quantile(0.7))

    cands: List[Dict[str, Any]] = []
    for r in df_zt.itertuples(index=False):
        concept = pick_best_theme(getattr(r, "concept", ""), core_concept_set, concept_info)
        industry = pick_best_theme(getattr(r, "industry", ""), core_industry_set, industry_info)
        if concept is None and industry is None:
            continue

        lb = int(getattr(r, "lb_count"))
        stage = _structure_stage(lb)

        open_cnt = int(getattr(r, "open_count"))
        first_min = getattr(r, "first_time_min")
        last_min = getattr(r, "last_time_min")
        seal_amount = _to_float(getattr(r, "seal_amount"))

        # 结构评分（0-5）
        structure_score = 0
        structure_score += 2 if lb >= 2 else 1
        structure_score += 1 if open_cnt == 0 else 0

        theme_for_time = concept or industry
        q20 = None
        if theme_for_time:
            q20 = (concept_info.get(theme_for_time) or industry_info.get(theme_for_time) or {}).get(
                "first_time_q20"
            )
        if q20 is not None and first_min is not None:
            try:
                if float(first_min) <= float(q20):
                    structure_score += 1
            except Exception:  # noqa: BLE001
                pass

        if last_min is not None and int(last_min) >= TAIL_START_MINUTE:
            structure_score += 1

        structure_score = int(max(0, min(5, structure_score)))

        seal_quality = 0
        if open_cnt == 0:
            seal_quality += 1
        if seal_amount_p70 is not None and seal_amount is not None and seal_amount >= seal_amount_p70:
            seal_quality += 1

        cands.append(
            {
                "symbol": str(getattr(r, "stock_code")),
                "stock_name": str(getattr(r, "stock_name") or ""),
                "primary_concept": concept or "",
                "primary_industry": industry or "",
                "lb_count": lb,
                "structure_stage": stage,
                "structure_score": structure_score,
                "open_count": open_cnt,
                "seal_amount": seal_amount,
                "seal_quality_score": int(seal_quality),
                "first_time_min": first_min,
                "last_time_min": last_min,
            }
        )

    if not cands:
        return []

    cands.sort(
        key=lambda x: (
            -int(x["structure_score"]),
            -int(x["lb_count"]),
            -int(x["seal_quality_score"]),
            -(float(x["seal_amount"] or 0.0)),
            int(x["first_time_min"] or 9999),
        )
    )
    picked = cands[: max(0, int(top_n))]
    for i, r in enumerate(picked):
        r["role"] = "龙头" if i == 0 else "次龙"
    return picked


def _load_daily_row(conn, stock_code: str, date: str) -> Optional[Dict[str, Any]]:  # noqa: ANN001
    try:
        if conn is not None:
            row = conn.execute(
                text(
                    """
                    SELECT open, high, low, close, volume, amount
                    FROM stock_daily
                    WHERE stock_code = :code AND date = :d
                    """
                ),
                {"code": stock_code, "d": date},
            ).fetchone()
            if row:
                return {
                    "open": _to_float(row[0]),
                    "high": _to_float(row[1]),
                    "low": _to_float(row[2]),
                    "close": _to_float(row[3]),
                    "volume": _to_float(row[4]),
                    "amount": _to_float(row[5]),
                }
    except Exception as e:  # noqa: BLE001
        logging.debug("DB daily row unavailable for %s %s: %s", stock_code, date, e)

    # Fallback: AkShare history (only used when DB not ready).
    return _load_daily_row_akshare(stock_code, date)


def _load_daily_row_akshare(stock_code: str, date: str) -> Optional[Dict[str, Any]]:
    try:
        from a_stock_analyzer.akshare_client import fetch_daily  # type: ignore
    except Exception:
        return None

    code = normalize_stock_code(stock_code)
    if not re.fullmatch(r"\d{6}", str(code or "")):
        return None

    end = yyyymmdd_from_date(date)
    try:
        df = fetch_daily(code, start_date=end, end_date=end)
    except Exception as e:  # noqa: BLE001
        logging.debug("AkShare daily fetch failed for %s %s: %s", code, date, e)
        return None
    if df is None or df.empty:
        return None

    df = df.copy()
    df["date"] = df["date"].astype(str)
    df = df[df["date"] == date]
    if df.empty:
        return None
    r = df.iloc[-1]
    return {
        "open": _to_float(r.get("open")),
        "high": _to_float(r.get("high")),
        "low": _to_float(r.get("low")),
        "close": _to_float(r.get("close")),
        "volume": _to_float(r.get("volume")),
        "amount": _to_float(r.get("amount")),
    }


def _load_history_akshare(stock_code: str, trade_date: str, *, days: int) -> pd.DataFrame:
    try:
        from a_stock_analyzer.akshare_client import fetch_daily  # type: ignore
    except Exception:
        return pd.DataFrame()

    code = normalize_stock_code(stock_code)
    if not re.fullmatch(r"\d{6}", str(code or "")):
        return pd.DataFrame()

    try:
        end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    # Fetch a wider window to survive holidays/weekends.
    start_dt = end_dt - timedelta(days=int(max(30, days * 3)))
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    try:
        df = fetch_daily(code, start_date=start, end_date=end)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = df["date"].astype(str)
    df = df[df["date"] <= trade_date].tail(int(days))
    return df.reset_index(drop=True)


def _position_score_from_history(conn, stock_code: str, trade_date: str) -> int:  # noqa: ANN001
    hist = None
    try:
        if conn is not None:
            hist = db.load_daily_until(conn, stock_code, end_date=trade_date, limit=80)
    except Exception as e:  # noqa: BLE001
        logging.debug("DB history unavailable for %s: %s", stock_code, e)
        hist = None

    if hist is None or hist.empty:
        hist = _load_history_akshare(stock_code, trade_date, days=80)

    if hist is None or hist.empty or len(hist) < 20:
        return 0
    w = hist.tail(60).copy()
    hi = pd.to_numeric(w["high"], errors="coerce").max()
    lo = pd.to_numeric(w["low"], errors="coerce").min()
    close = pd.to_numeric(w["close"], errors="coerce").iloc[-1]
    if not (pd.notna(hi) and pd.notna(lo) and pd.notna(close)) or float(hi) <= float(lo):
        return 0
    pos = (float(close) - float(lo)) / (float(hi) - float(lo))
    return 1 if pos <= 0.35 else 0


def _turnover_structure_score(conn, stock_code: str, trade_date: str) -> int:  # noqa: ANN001
    hist = None
    try:
        if conn is not None:
            hist = db.load_daily_until(conn, stock_code, end_date=trade_date, limit=30)
    except Exception as e:  # noqa: BLE001
        logging.debug("DB turnover history unavailable for %s: %s", stock_code, e)
        hist = None

    if hist is None or hist.empty:
        hist = _load_history_akshare(stock_code, trade_date, days=30)

    if hist is None or hist.empty or len(hist) < 6:
        return 0
    v = pd.to_numeric(hist["volume"], errors="coerce").dropna()
    if len(v) < 6:
        return 0
    today = float(v.iloc[-1])
    avg_prev5 = float(v.iloc[-6:-1].mean())
    if not (np.isfinite(today) and np.isfinite(avg_prev5)) or avg_prev5 <= 0:
        return 0
    ratio = today / avg_prev5
    return 1 if ratio <= 3.0 else 0


def _stock_tradability_score(
    *,
    conn,
    candidate: Dict[str, Any],
    market: MarketEmotion,
    trade_date_norm: str,
) -> Tuple[bool, int, Dict[str, Any]]:  # noqa: ANN001
    score = 0

    # 题材归属：主线题材 => +2
    theme_core = bool(candidate.get("primary_concept") or candidate.get("primary_industry"))
    if theme_core:
        score += 2

    # 市场情绪：上升/震荡 => +2
    if market.market_emotion in {"上升期", "震荡期"} and market.operation_permission:
        score += 2

    # 位置：相对低位 => +1
    pos = _position_score_from_history(conn, candidate["symbol"], trade_date=trade_date_norm)
    score += pos

    # 封板质量：0-2（候选阶段得出）=> +2
    seal_q = int(candidate.get("seal_quality_score", 0))
    score += seal_q

    # 换手结构：非天量 => +1
    turn = _turnover_structure_score(conn, candidate["symbol"], trade_date=trade_date_norm)
    score += turn

    # 高位期降权：更保守（避免高位接力）
    if candidate.get("structure_stage") == "高位期":
        score -= 1

    score = int(max(0, min(8, score)))
    tradable = bool(score >= 6)
    detail = {
        "theme_core": theme_core,
        "position_low": bool(pos),
        "seal_quality_score": seal_q,
        "turnover_ok": bool(turn),
        "market_emotion": market.market_emotion,
    }
    return tradable, score, detail


def _plan_for_candidate(
    *,
    candidate: Dict[str, Any],
    prev_bar: Optional[Dict[str, Any]],
    market: MarketEmotion,
) -> Dict[str, Any]:
    lb = int(candidate.get("lb_count", 1))
    stage = str(candidate.get("structure_stage", "启动期"))
    op = "打板" if lb <= 1 else ("半路" if lb == 2 else "低吸")

    prev_close = prev_bar.get("close") if prev_bar else None
    prev_high = prev_bar.get("high") if prev_bar else None

    entry: List[str] = []
    if prev_close is not None and np.isfinite(prev_close) and prev_close > 0:
        entry.append(f"次日开盘涨幅 <= +5%（open <= {prev_close * 1.05:.2f}）")
        entry.append(f"次日开盘不低于昨收 -2%（open >= {prev_close * 0.98:.2f}）")
    else:
        entry.append("次日开盘涨幅 <= +5%")

    if prev_high is not None and np.isfinite(prev_high) and prev_high > 0:
        entry.append(f"盘中突破昨高（price >= {prev_high:.2f}）后再介入")
    else:
        entry.append("盘中突破昨高后再介入")

    if candidate.get("primary_concept"):
        entry.append("同概念板块 10:00 前至少 1 只个股触及涨停（theme_follow_count >= 1）")
    elif candidate.get("primary_industry"):
        entry.append("同产业链 10:00 前至少 1 只个股触及涨停（industry_follow_count >= 1）")

    stop: List[str] = [
        "买入后跌破买入价 -3%（price <= entry_price * 0.97）立即止损",
        "若 10:30 前无法站稳昨高（price < prev_high）则减仓/清仓",
        "题材炸板率显著升高（broken_rate > 50%）时不恋战",
    ]

    take_profit_logic = "分批止盈：+6% 卖 50%，+10% 清仓；若再封板可用“开板”触发减仓"
    sell_plan = "高开 >= 6% 且 10:00 前无法回封时逢冲高减仓；午后首次明显开板/放量长阴则清仓"

    return {
        "symbol": candidate["symbol"],
        "operation_type": op,
        "entry_conditions": entry,
        "stop_loss_rules": stop,
        "take_profit_logic": take_profit_logic,
        "sell_plan": sell_plan,
        "meta": {"lb_count": lb, "structure_stage": stage, "market_emotion": market.market_emotion},
    }


def _risk_control(market: MarketEmotion) -> Dict[str, Any]:
    if market.market_emotion == "上升期":
        return {"max_position": "50%", "risk_level": "低"}
    if market.market_emotion == "震荡期":
        return {"max_position": "30%", "risk_level": "中"}
    return {"max_position": "0%", "risk_level": "高"}


def _load_cache_json(path: Path) -> Dict[str, str]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if k and v}
    except Exception:  # noqa: BLE001
        return {}
    return {}


def _save_cache_json(path: Path, data: Dict[str, str]) -> None:
    try:
        db.ensure_parent_dir(path)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        return


def _fetch_em_concept_board_code_set() -> set[str]:
    """
    Eastmoney 概念板块 code 集合（m:90 t:3）.

    用于把个股 “所属板块” 列表过滤为真正的【概念】板块（避免混入行业/地域/指数等）。
    """
    try:
        import requests
    except Exception:
        return set()

    url = "https://79.push2.eastmoney.com/api/qt/clist/get"
    base_params = {
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:90 t:3 f:!50",
        "fields": "f12",
    }
    codes: set[str] = set()
    pn = 1
    pz = 200
    while True:
        params = dict(base_params)
        params["pn"] = str(pn)
        params["pz"] = str(pz)
        try:
            j = requests.get(url, params=params, timeout=10).json()
        except Exception:  # noqa: BLE001
            break

        data = (j or {}).get("data") or {}
        diff = data.get("diff") or []
        if not diff:
            break
        for item in diff:
            c = _safe_str((item or {}).get("f12"))
            if c:
                codes.add(c)

        total = int(data.get("total") or 0)
        if pn * pz >= total:
            break
        pn += 1
        if pn > 10:  # hard stop
            break

    return codes


def _get_em_concept_board_code_set(*, cache_path: Path) -> set[str]:
    """
    Load Eastmoney 概念板块 code set from cache, otherwise fetch and cache.
    Cache format: {"BKxxxx": "1", ...} (values are dummy).
    """
    cached = _load_cache_json(cache_path)
    if cached:
        return set(cached.keys())

    codes = _fetch_em_concept_board_code_set()
    if codes:
        _save_cache_json(cache_path, {c: "1" for c in sorted(codes)})
    return codes


def _fetch_stock_concepts_em(stock_code: str, *, concept_code_set: set[str]) -> List[str]:
    """
    Eastmoney push2 `api/qt/slist/get` 返回个股所属板块列表（含行业/地域/概念等）。
    我们用 `concept_code_set` 过滤得到概念板块名称列表。
    """
    code = normalize_stock_code(stock_code)
    if not re.fullmatch(r"\d{6}", str(code or "")):
        return []

    try:
        import requests
    except Exception:
        return []

    market_code = 1 if str(code).startswith("6") else 0
    url = "https://push2.eastmoney.com/api/qt/slist/get"
    params = {
        "fltt": "1",
        "invt": "2",
        "spt": "3",
        "secid": f"{market_code}.{code}",
        "pi": "0",
        "po": "1",
        "np": "1",
        "pz": "200",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "fields": "f12,f14",
    }
    try:
        j = requests.get(url, params=params, timeout=8).json()
    except Exception:  # noqa: BLE001
        return []

    diff = ((j or {}).get("data") or {}).get("diff") or []
    out: List[str] = []
    for item in diff:
        c = _safe_str((item or {}).get("f12"))
        n = _safe_str((item or {}).get("f14"))
        if not c or not n:
            continue
        if c not in concept_code_set:
            continue
        if n not in out:
            out.append(n)
    return out


def _fetch_industry_akshare(stock_code: str) -> Optional[str]:
    try:
        import akshare as ak  # type: ignore
    except Exception:
        return None

    try:
        df = ak.stock_individual_info_em(symbol=str(stock_code))
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty:
        return None

    item_col = pick_col(df, ["item", "项目"])
    value_col = pick_col(df, ["value", "值"])
    if item_col is None or value_col is None:
        return None

    m = df[df[item_col].astype(str).str.strip() == "行业"]
    if m.empty:
        return None
    return _safe_str(m[value_col].iloc[0])

def _fetch_concept_individual_info_em(stock_code: str) -> Optional[str]:
    """
    Prefer a lightweight per-stock concept field if the data source provides it.
    This avoids very heavy THS concept-board scans and improves稳定性。
    """
    try:
        import akshare as ak  # type: ignore
    except Exception:
        return None

    try:
        df = ak.stock_individual_info_em(symbol=str(stock_code))
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty:
        return None

    item_col = pick_col(df, ["item", "项目"])
    value_col = pick_col(df, ["value", "值"])
    if item_col is None or value_col is None:
        return None

    wanted = {"所属概念", "概念", "概念板块", "所属题材", "题材"}
    m = df[df[item_col].astype(str).str.strip().isin(wanted)]
    if m.empty:
        return None

    v = _safe_str(m[value_col].iloc[0])
    if not v:
        return None
    bad = {"-", "—", "--", "暂无", "不明", "无", "未知"}
    return None if v in bad else v

def _call_akshare_table(
    ak: Any,
    func_names: Iterable[str],
    *,
    symbol: str,
) -> pd.DataFrame:
    sym = _safe_str(symbol)
    if not sym:
        return pd.DataFrame()

    for fn in func_names:
        f = getattr(ak, fn, None)
        if f is None:
            continue

        try:
            import inspect

            keys = list(inspect.signature(f).parameters.keys())
        except Exception:  # noqa: BLE001
            keys = []

        attempts: List[Tuple[str, Any]] = []
        if "symbol" in keys:
            attempts.append(("kw", {"symbol": sym}))
        if "symbol_code" in keys:
            attempts.append(("kw", {"symbol_code": sym}))
        if "code" in keys:
            attempts.append(("kw", {"code": sym}))
        if "name" in keys:
            attempts.append(("kw", {"name": sym}))
        if not attempts and keys:
            attempts.append(("pos", sym))
        if not keys:
            attempts.append(("none", None))

        for mode, payload in attempts:
            try:
                if mode == "kw":
                    return f(**payload)
                if mode == "pos":
                    return f(payload)
                return f()
            except TypeError:
                continue
            except Exception as e:  # noqa: BLE001
                logging.debug("AkShare %s failed for %s: %s", fn, sym, e)
                continue

    return pd.DataFrame()


def _list_concept_boards_ths(ak: Any) -> List[Dict[str, str]]:
    """
    同花顺概念板块列表：优先用 summary（通常有排序/热度），其次用 name 列表。
    返回 [{'name': 'XXX', 'code': '301xxx'}]；code 可能为空。
    """
    def extract_from_df(df: pd.DataFrame) -> List[Dict[str, str]]:
        if df is None or df.empty:
            return []
        name_col = pick_col(df, ["板块名称", "概念名称", "板块", "概念", "名称", "name"])
        code_col = pick_col(df, ["概念代码", "板块代码", "代码", "concept_code", "symbol", "code"])
        if name_col is None:
            return []
        names = df[name_col].astype(str).tolist()
        codes = df[code_col].astype(str).tolist() if code_col in df.columns else [""] * len(names)
        out: List[Dict[str, str]] = []
        for n, c in zip(names, codes):
            name = _safe_str(n)
            if not name:
                continue
            out.append({"name": name, "code": _safe_str(c)})
        return out

    try:
        f = getattr(ak, "stock_board_concept_summary_ths", None)
        df_sum = f() if callable(f) else pd.DataFrame()
    except Exception:  # noqa: BLE001
        df_sum = pd.DataFrame()

    try:
        f = getattr(ak, "stock_board_concept_name_ths", None)
        df_name = f() if callable(f) else pd.DataFrame()
    except Exception:  # noqa: BLE001
        df_name = pd.DataFrame()

    merged = extract_from_df(df_sum) + extract_from_df(df_name)

    # Deduplicate by name; if later entries have code,补齐 code。
    idx_by_name: Dict[str, int] = {}
    uniq: List[Dict[str, str]] = []
    for b in merged:
        name = _safe_str(b.get("name"))
        code = _safe_str(b.get("code"))
        if not name:
            continue
        if name not in idx_by_name:
            idx_by_name[name] = len(uniq)
            uniq.append({"name": name, "code": code})
            continue
        i = idx_by_name[name]
        if i < 0 or i >= len(uniq):
            continue
        if not _safe_str(uniq[i].get("code")) and code:
            uniq[i]["code"] = code

    return uniq


def _bulk_fetch_concepts_ths_via_akshare(
    stock_codes: Iterable[str],
    *,
    max_boards: int = 360,
    min_concepts_per_stock: int = 1,
    max_concepts_per_stock: int = 3,
    extra_boards_after_min: int = 40,
) -> Dict[str, List[str]]:
    """
    用 AkShare 的同花顺概念板块接口填充个股所属概念：
    - 通过遍历概念板块成份股，反向映射出指定股票的概念集合
    - 为避免耗时过长：
      1) 达到 min_concepts_per_stock 后只再额外扫描 extra_boards_after_min 个板块；
      2) 每只股票最多保留 max_concepts_per_stock 个概念。
    """
    codes = sorted(
        {
            normalize_stock_code(c)
            for c in stock_codes
            if re.fullmatch(r"\d{6}", str(normalize_stock_code(c) or ""))
        }
    )
    if not codes:
        return {}

    try:
        import akshare as ak  # type: ignore
    except Exception:
        return {}

    boards = _list_concept_boards_ths(ak)
    if not boards:
        return {}

    # Discover available "concept constituents" functions (AkShare versions differ).
    cons_funcs: List[str] = []
    for n in dir(ak):
        ln = str(n).lower()
        if "ths" not in ln or "concept" not in ln:
            continue
        if "cons" in ln or "constitu" in ln or "component" in ln:
            cons_funcs.append(str(n))
    preferred = [
        "stock_board_concept_cons_ths",
        "stock_board_concept_info_ths",
        "stock_board_cons_ths",
    ]
    for p in reversed(preferred):
        if p in cons_funcs:
            cons_funcs.remove(p)
        cons_funcs.insert(0, p)
    cons_funcs = [x for x in cons_funcs if hasattr(ak, x)]
    cons_funcs = cons_funcs[:20] if cons_funcs else preferred

    # Ensure bounds sane.
    max_boards = max(1, int(max_boards))
    min_concepts_per_stock = max(1, int(min_concepts_per_stock))
    max_concepts_per_stock = max(min_concepts_per_stock, int(max_concepts_per_stock))
    extra_boards_after_min = max(0, int(extra_boards_after_min))

    concepts_by_code: Dict[str, List[str]] = {c: [] for c in codes}

    def need_more(code: str) -> bool:
        return len(concepts_by_code.get(code) or []) < max_concepts_per_stock

    min_satisfied = False
    extra_scanned = 0
    scanned = 0

    for b in boards:
        if scanned >= max_boards:
            break
        if all(not need_more(c) for c in codes):
            break

        board_name = _safe_str(b.get("name"))
        board_code = _safe_str(b.get("code"))
        if not board_name and not board_code:
            continue

        # Prefer code if available (some AkShare endpoints require it), fallback to name.
        df_cons = pd.DataFrame()
        for sym_try in [board_code, board_name]:
            if not sym_try:
                continue
            df_cons = _call_akshare_table(ak, cons_funcs, symbol=sym_try)
            if df_cons is not None and not df_cons.empty:
                break

        scanned += 1
        if df_cons is None or df_cons.empty:
            continue

        code_col = pick_col(df_cons, ["股票代码", "代码", "成分股代码", "stock_code", "symbol", "code"])
        if code_col is None or code_col not in df_cons.columns:
            continue

        # Extract 6-digit codes and intersect with our target set (small list) to avoid big sets.
        try:
            series = df_cons[code_col].astype(str).str.extract(r"(\d{6})", expand=False)
        except Exception:  # noqa: BLE001
            continue

        candidates = [c for c in codes if need_more(c)]
        if not candidates:
            continue
        hits = set(series[series.isin(candidates)].dropna().unique().tolist())
        if not hits:
            continue

        for c in hits:
            if not need_more(c):
                continue
            lst = concepts_by_code.get(c)
            if lst is None:
                lst = []
                concepts_by_code[c] = lst
            if board_name and board_name not in lst:
                lst.append(board_name)
            if len(lst) > max_concepts_per_stock:
                concepts_by_code[c] = lst[:max_concepts_per_stock]

        if not min_satisfied and all(len(concepts_by_code[c]) >= min_concepts_per_stock for c in codes):
            min_satisfied = True
            extra_scanned = 0
            continue

        if min_satisfied:
            extra_scanned += 1
            if extra_scanned >= extra_boards_after_min:
                break

    return {c: v for c, v in concepts_by_code.items() if v}


def _fill_industry(
    df_zt: pd.DataFrame,
    *,
    cache_path: Path,
) -> pd.DataFrame:
    if df_zt is None or df_zt.empty:
        return df_zt

    if "industry" not in df_zt.columns:
        out = df_zt.copy()
        out["industry"] = ""
    else:
        out = df_zt.copy()

    cache = _load_cache_json(cache_path)
    changed = False

    for idx, row in out.iterrows():
        cur = _safe_str(row.get("industry", ""))
        if cur:
            continue
        code = normalize_stock_code(row.get("stock_code", ""))
        if not re.fullmatch(r"\d{6}", str(code or "")):
            continue
        if code in cache and cache[code]:
            out.at[idx, "industry"] = cache[code]
            continue
        ind = _fetch_industry_akshare(code)
        if ind:
            cache[code] = ind
            out.at[idx, "industry"] = ind
            changed = True

    if changed:
        _save_cache_json(cache_path, cache)
    return out


def _fill_concept(
    df_zt: pd.DataFrame,
    *,
    cache_path: Path,
) -> pd.DataFrame:
    if df_zt is None or df_zt.empty:
        return df_zt

    if "concept" not in df_zt.columns:
        out = df_zt.copy()
        out["concept"] = ""
    else:
        out = df_zt.copy()

    cache = _load_cache_json(cache_path)   
    changed = False

    # Eastmoney 概念板块 code 集合（用来过滤“所属板块” -> 概念板块）
    em_codes_cache = cache_path.parent / "em_concept_board_codes.json"
    concept_code_set = _get_em_concept_board_code_set(cache_path=em_codes_cache)

    def reason_to_concept(raw_reason: object) -> str:
        parts = _split_themes(raw_reason, limit=5)
        if not parts:
            return ""
        cleaned: List[str] = []
        for p in parts:
            s = _safe_str(p)
            if not s:
                continue
            if len(s) > 30:
                s = s[:30]
            cleaned.append(s)
        return "；".join(cleaned[:3])

    # Legacy block below expects `need`; keep it empty so it never runs.
    need: set[str] = set()
    for idx, row in out.iterrows():
        cur = _safe_str(row.get("concept", ""))
        if cur:
            continue
        code = normalize_stock_code(row.get("stock_code", ""))
        if not re.fullmatch(r"\d{6}", str(code or "")):
            continue
        if code in cache and cache[code]:
            out.at[idx, "concept"] = cache[code]
            continue
        concepts = (
            _fetch_stock_concepts_em(code, concept_code_set=concept_code_set)
            if concept_code_set
            else []
        )
        if concepts:
            joined = "；".join([_safe_str(x) for x in concepts if _safe_str(x)][:8])
            if joined:
                cache[code] = joined
                out.at[idx, "concept"] = joined
                changed = True

    if need:
        need_list = list(need)
        # Avoid extremely heavy THS scans when base data lacks concept fields.
        # Prefer to fill "leaders" first; this is enough for题材归因与候选股打分。
        max_uncached = 60
        if len(need_list) > max_uncached:
            tmp = out.copy()
            tmp["__code"] = tmp["stock_code"].apply(normalize_stock_code)
            tmp = tmp[tmp["__code"].isin(need_list)]
            tmp["__lb"] = pd.to_numeric(tmp.get("lb_count"), errors="coerce").fillna(0)
            tmp["__seal"] = pd.to_numeric(tmp.get("seal_amount"), errors="coerce").fillna(0)
            tmp = tmp.sort_values(["__lb", "__seal"], ascending=[False, False], na_position="last")
            need_list = tmp["__code"].astype(str).head(max_uncached).tolist()
            logging.info(
                "Filling THS concepts via AkShare: %d/%d stocks (uncached, leader-priority)",
                len(need_list),
                len(need),
            )
        else:
            logging.info("Filling THS concepts via AkShare: %d stocks (uncached)", len(need_list))

        concepts_map = _bulk_fetch_concepts_ths_via_akshare(need_list)
        if not concepts_map:
            logging.warning("THS concept fill returned empty; concepts may be blocked (401/login) in current network.")
        for idx, row in out.iterrows():
            cur = _safe_str(row.get("concept", ""))
            if cur:
                continue
            code = normalize_stock_code(row.get("stock_code", ""))
            if code in concepts_map and concepts_map[code]:
                joined = "；".join([_safe_str(x) for x in concepts_map[code] if _safe_str(x)][:8])
                if joined:
                    cache[code] = joined
                    out.at[idx, "concept"] = joined
                    changed = True

    # Final fallback: still-empty concepts use daily reason (AkShare limit-up pool field) to avoid empty “概念” table.
    if "reason" in out.columns:
        for idx, row in out.iterrows():
            cur = _safe_str(row.get("concept", ""))
            if cur:
                continue
            derived = reason_to_concept(row.get("reason", ""))
            if derived:
                out.at[idx, "concept"] = derived

    if changed:
        _save_cache_json(cache_path, cache)
    return out


def analyze_bingwu(
    *,
    db_target: str,
    trade_date: str,
    top_themes: int = 10,
    top_candidates: int = 2,
) -> Dict[str, Any]:
    requested_trade_date_norm = normalize_trade_date(trade_date)
    engine = db.make_engine(db_target, pool_size=3, max_overflow=3)
    trade_date_norm = _resolve_effective_trade_date(engine, requested_trade_date_norm)
    prev_trade_date = _resolve_prev_trade_date(engine, trade_date_norm)

    df_zt, df_dt, df_zb, src_today = _build_pools(engine, trade_date_norm=trade_date_norm)

    prev_max_lb = None
    df_zt_prev = pd.DataFrame()
    if prev_trade_date:
        df_zt_prev, _, _, _ = _build_pools(engine, trade_date_norm=prev_trade_date)
        if not df_zt_prev.empty:
            prev_max_lb = int(pd.to_numeric(df_zt_prev["lb_count"], errors="coerce").fillna(1).max())

    market = _calc_market_emotion(
        limit_up_count=int(len(df_zt)),
        limit_down_count=int(len(df_dt)),
        max_consecutive=int(pd.to_numeric(df_zt["lb_count"], errors="coerce").fillna(1).max())
        if not df_zt.empty
        else 0,
        prev_max_consecutive=prev_max_lb,
        broken_count=int(len(df_zb)),
        sealed_count=int(len(df_zt)),
    )

    out: Dict[str, Any] = {
        "requested_trade_date": requested_trade_date_norm,
        "trade_date": trade_date_norm,
        "prev_trade_date": prev_trade_date,
        "data_source": {"trade_date": src_today},
        "market": market.to_dict(),
    }

    # Data not available (AkShare may not support the date, or network failed, and DB has no backup).
    if df_zt.empty and df_dt.empty and df_zb.empty:
        out["advice"] = "空仓建议（该日期缺少涨停/跌停/炸板数据，可能接口不支持或网络失败）"
        out["risk"] = {"max_position": "0%", "risk_level": "高"}
        return out

    ind_cache_path = Path("output/bingwu_industry_cache.json")
    concept_cache_path = Path("output/bingwu_concept_cache.json")
    df_zt = _fill_industry(df_zt, cache_path=ind_cache_path)
    df_zt = _fill_concept(df_zt, cache_path=concept_cache_path)
    if prev_trade_date:
        df_zt_prev = _fill_industry(df_zt_prev, cache_path=ind_cache_path)
        df_zt_prev = _fill_concept(df_zt_prev, cache_path=concept_cache_path)

    concept_themes, concept_info, core_concepts = _scan_themes(
        df_zt, df_zt_prev, theme_type="concept", theme_col="concept", top_n=top_themes
    )
    industry_themes, industry_info, core_industries = _scan_themes(
        df_zt, df_zt_prev, theme_type="industry", theme_col="industry", top_n=top_themes
    )
    out["themes"] = {"concept": concept_themes, "industry": industry_themes}    

    # Allow继续输出“题材观察”即使当日不建议交易；但后续“候选股/次日计划”跳过。
    if not market.operation_permission:
        out["advice"] = "空仓建议（市场退潮/风险高）"
        out["risk"] = _risk_control(market)
        return out

    if not core_concepts and not core_industries:
        out["advice"] = "空仓建议（未形成清晰主线题材）"
        out["risk"] = _risk_control(market)
        return out

    candidates = _pick_candidates(
        df_zt,
        concept_info=concept_info,
        industry_info=industry_info,
        core_concepts=core_concepts,
        core_industries=core_industries,
        top_n=top_candidates,
    )
    if not candidates:
        out["advice"] = "空仓建议（未找到可操作龙头/结构标的）"
        out["risk"] = _risk_control(market)
        return out

    plans: List[Dict[str, Any]] = []
    enriched: List[Dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            for c in candidates:
                bar = _load_daily_row(conn, c["symbol"], trade_date_norm)
                tradable, stock_score, detail = _stock_tradability_score(
                    conn=conn,
                    candidate=c,
                    market=market,
                    trade_date_norm=trade_date_norm,
                )

                c2 = dict(c)
                c2["tradable"] = bool(tradable)
                c2["stock_score"] = int(stock_score)
                c2["score_detail"] = detail
                c2["prev_bar"] = bar or {}
                enriched.append(c2)

                if tradable:
                    plans.append(_plan_for_candidate(candidate=c2, prev_bar=bar, market=market))
    except Exception as e:  # noqa: BLE001
        logging.warning("DB unavailable; fallback to AkShare history only: %s", e)
        for c in candidates:
            bar = _load_daily_row(None, c["symbol"], trade_date_norm)
            tradable, stock_score, detail = _stock_tradability_score(
                conn=None,
                candidate=c,
                market=market,
                trade_date_norm=trade_date_norm,
            )

            c2 = dict(c)
            c2["tradable"] = bool(tradable)
            c2["stock_score"] = int(stock_score)
            c2["score_detail"] = detail
            c2["prev_bar"] = bar or {}
            enriched.append(c2)

            if tradable:
                plans.append(_plan_for_candidate(candidate=c2, prev_bar=bar, market=market))

    out["candidates"] = enriched

    if not plans:
        out["advice"] = "空仓建议（候选股评分不足）"
        out["risk"] = _risk_control(market)
        return out

    out["plans"] = plans
    out["risk"] = _risk_control(market)
    return out


def _fmt_rate(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x * 100:.1f}%"


def render_markdown(result: Dict[str, Any]) -> str:
    trade_date = str(result.get("trade_date", "")).strip()
    requested_trade_date = str(result.get("requested_trade_date", "")).strip()
    dt_obj = None
    try:
        dt_obj = datetime.strptime(trade_date, "%Y-%m-%d")
    except Exception:  # noqa: BLE001
        dt_obj = None

    title_date = (
        f"{dt_obj.year}年{dt_obj.month:02d}月{dt_obj.day:02d}日" if dt_obj else trade_date
    )
    market = result.get("market") or {}
    risk = result.get("risk") or {}

    lines: List[str] = []
    lines.append(f"# {title_date} 收盘复盘（bingwu）")
    if requested_trade_date and requested_trade_date != trade_date:
        lines.append(f"> 请求日期：{requested_trade_date}；实际分析日期：{trade_date}（自动回退到最近有数据交易日）")
    lines.append("")
    lines.append("## 一、市场情绪判定")
    lines.append(
        f"- 涨停：{int(market.get('limit_up_count') or 0)}；跌停：{int(market.get('limit_down_count') or 0)}"
    )
    prev_max = market.get("prev_max_consecutive")
    prev_s = f"（昨 {prev_max}）" if prev_max is not None else ""
    lines.append(f"- 最高连板：{int(market.get('max_consecutive') or 0)} {prev_s}".rstrip())
    lines.append(
        f"- 炸板率：{_fmt_rate(market.get('broken_rate'))}（炸板 {int(market.get('broken_count') or 0)} / 封板 {int(market.get('sealed_count') or 0)}）"
    )
    lines.append(
        f"- 情绪周期：{market.get('market_emotion','')}；情绪分：{int(market.get('emotion_score') or 0)}/100；操作许可：{'是' if market.get('operation_permission') else '否'}"
    )

    advice = result.get("advice")
    if advice:
        lines.append("")
        lines.append("## 空仓建议")
        lines.append(f"- {advice}")
        lines.append(f"- 最大总仓位：{risk.get('max_position','0%')}；风险等级：{risk.get('risk_level','高')}")

    themes = result.get("themes") or {}
    lines.append("")
    lines.append("## 二、核心题材")

    def render_theme_table(items: List[Dict[str, Any]], title: str) -> None:
        lines.append(f"### {title}")
        show_items = [
            it
            for it in (items or [])
            if isinstance(it, dict) and int(it.get("theme_score", 0) or 0) > 0
        ]
        if not show_items:
            lines.append("- （无）")
            return
        lines.append("| 题材 | 得分 | 主线 | 涨停数 | 最高连板 | 新题材 | 尾盘回流 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for it in show_items:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(it.get("theme_name", "")),
                        str(it.get("theme_score", 0)),
                        "是" if it.get("is_core_theme") else "否",
                        str(it.get("limit_up_count", 0)),
                        str(it.get("max_consecutive", 0)),
                        "是" if it.get("is_new_theme") else "否",
                        "是" if it.get("tail_inflow") else "否",
                    ]
                )
                + " |"
            )

    render_theme_table(themes.get("concept") or [], "概念")
    lines.append("")
    render_theme_table(themes.get("industry") or [], "行业")

    lines.append("")
    lines.append("## 三、备选个股")
    cands = result.get("candidates") or []
    if not cands:
        if not market.get("operation_permission"):
            lines.append("- （跳过：操作许可=否）")
        else:
            lines.append("- （无）")
    else:
        lines.append("| 代码 | 名称 | 角色 | 阶段 | 结构分 | 概念 | 行业 | 可交易 | 交易分 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |") 
        for c in cands:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(c.get("symbol", "")),
                        str(c.get("stock_name", "")),
                        str(c.get("role", "")),
                        str(c.get("structure_stage", "")),
                        str(c.get("structure_score", 0)),
                        str(c.get("primary_concept", "")),
                        str(c.get("primary_industry", "")),
                        "是" if c.get("tradable") else "否",
                        str(c.get("stock_score", 0)),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## 四、次日交易计划")
    plans = result.get("plans") or []
    if not plans:
        if not market.get("operation_permission"):
            lines.append("- （跳过：操作许可=否）")
        else:
            lines.append("- （无）")
    else:
        for p in plans:
            lines.append(f"### {p.get('symbol','')}")
            lines.append(f"- 操作类型：{p.get('operation_type','')}")      
            lines.append("- 入场条件：")
            for s in p.get("entry_conditions") or []:
                lines.append(f"  - {s}")
            lines.append("- 止损规则：")
            for s in p.get("stop_loss_rules") or []:
                lines.append(f"  - {s}")
            lines.append(f"- 止盈逻辑：{p.get('take_profit_logic','')}")
            lines.append(f"- 卖出计划：{p.get('sell_plan','')}")
            lines.append("")

    lines.append("## 五、仓位与风险提示")
    lines.append(f"- 最大总仓位：{risk.get('max_position','0%')}")
    lines.append(f"- 风险等级：{risk.get('risk_level','高')}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="bingwu.py - 超短复盘/次日作战文档生成（AkShare + DB）")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    add_db_args(p)
    p.add_argument(
        "--trade-date",
        default=None,
        help="YYYYMMDD or YYYY-MM-DD (default: latest stock_daily date)",
    )
    p.add_argument("--output-md", default=None, help="Markdown output path")
    p.add_argument("--output-json", default=None, help="JSON output path")
    p.add_argument("--top-themes", type=int, default=10, help="Top themes to show")
    p.add_argument("--top-candidates", type=int, default=2, help="Top candidates to pick")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_from_args(args)
    logging.info("DB target: %s", db_target)
    try:
        from a_stock_analyzer.config import build_mysql_db_url, load_config  # type: ignore

        cfg_path = Path(args.config) if getattr(args, "config", None) else Path("config.ini")
        if cfg_path.exists():
            cfg = load_config(cfg_path)
            mysql_url = cfg.db_url or build_mysql_db_url(cfg.mysql)
            resolved_url = db.normalize_db_url(db_target)
            if mysql_url and str(mysql_url).startswith("mysql") and str(resolved_url).startswith("sqlite:///"):
                logging.warning(
                    "MySQL is configured in %s but resolved DB is SQLite; check env ASTOCK_DB_URL or --db override.",
                    cfg_path,
                )
    except Exception:  # noqa: BLE001
        pass
    trade_date_in = args.trade_date or resolve_latest_stock_daily_date(db_target)
    result = analyze_bingwu(
        db_target=db_target,
        trade_date=str(trade_date_in),
        top_themes=args.top_themes,
        top_candidates=args.top_candidates,
    )

    trade_date_norm = str(result.get("trade_date") or normalize_trade_date(trade_date_in))
    yyyymmdd = yyyymmdd_from_date(trade_date_norm)

    out_md = Path(args.output_md) if args.output_md else Path(f"output/bingwu_{yyyymmdd}.md")
    out_json = Path(args.output_json) if args.output_json else Path(f"output/bingwu_{yyyymmdd}.json")

    md = render_markdown(result)
    db.ensure_parent_dir(out_md)
    out_md.write_text(md, encoding="utf-8")

    db.ensure_parent_dir(out_json)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Wrote markdown: %s", out_md)
    logging.info("Wrote json: %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
