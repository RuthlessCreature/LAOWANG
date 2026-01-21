# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lowered = {c: str(c).lower() for c in cols}
    for c, lc in lowered.items():
        if any(k in lc for k in candidates):
            return c
    return None


def normalize_stock_code(raw) -> str:
    s = str(raw).strip()
    m = re.findall(r"\d{6}", s)
    if m:
        return m[-1]
    m2 = re.findall(r"\d+", s)
    if m2:
        return m2[-1].zfill(6)
    return s


def to_ak_symbol(stock_code: str) -> str:
    s = stock_code.strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        return s
    if len(s) == 6 and s.isdigit():
        if s.startswith(("6", "9")):
            return f"sh{s}"
        if s.startswith(("8", "4")):
            return f"bj{s}"
        return f"sz{s}"
    return s


def normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Robust column picking: AkShare endpoints may return either English columns
    # or Chinese columns (e.g. 日期/开盘/最高/最低/收盘/成交量/成交额).
    date_col = pick_col(df, ["date", "日期", "时间", "交易日期"])
    if date_col is None and len(df.columns) >= 1:
        # Some endpoints return date as the first unnamed column.
        c0 = df.columns[0]
        dt0 = pd.to_datetime(df[c0], errors="coerce")
        if dt0.notna().mean() >= 0.8:
            date_col = c0

    if date_col is None:
        # Some endpoints return date in index.
        idx_dt = pd.to_datetime(df.index, errors="coerce")
        if idx_dt.notna().mean() >= 0.8:
            df = df.reset_index()
            df.columns = [str(c).strip() for c in df.columns]
            date_col = str(df.columns[0])
        else:
            raise ValueError(f"Daily DF missing date column; columns={list(df.columns)}")

    open_col = pick_col(df, ["open", "开盘"])
    high_col = pick_col(df, ["high", "最高"])
    low_col = pick_col(df, ["low", "最低"])
    close_col = pick_col(df, ["close", "收盘"])
    vol_col = pick_col(df, ["volume", "成交量"])
    amt_col = pick_col(df, ["amount", "成交额"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d"),
            "open": pd.to_numeric(df[open_col], errors="coerce") if open_col else np.nan,
            "high": pd.to_numeric(df[high_col], errors="coerce") if high_col else np.nan,
            "low": pd.to_numeric(df[low_col], errors="coerce") if low_col else np.nan,
            "close": pd.to_numeric(df[close_col], errors="coerce") if close_col else np.nan,
            "volume": pd.to_numeric(df[vol_col], errors="coerce") if vol_col else np.nan,
            "amount": pd.to_numeric(df[amt_col], errors="coerce") if amt_col else np.nan,
        }
    )
    out = out.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"]).sort_values("date")
    return out


def fetch_stock_list() -> List[Tuple[str, str]]:
    import akshare as ak

    df = ak.stock_info_a_code_name()
    if df is None or df.empty:
        raise RuntimeError("AkShare returned empty stock list")

    code_col = pick_col(df, ["code", "代码"])
    name_col = pick_col(df, ["name", "名称"])
    if code_col is None:
        code_col = df.columns[0]
    if name_col is None:
        name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    out: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        code = normalize_stock_code(row[code_col])
        name = str(row[name_col]) if name_col in row else ""
        if code and code.isdigit() and len(code) == 6:
            out.append((code, name))

    seen = set()
    deduped: List[Tuple[str, str]] = []
    for code, name in out:
        if code in seen:
            continue
        seen.add(code)
        deduped.append((code, name))
    return deduped


def fetch_daily(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    # Prefer Eastmoney history (often more stable), fallback to Sina daily.
    last_error: Optional[Exception] = None

    # Transient upstream failures sometimes manifest as KeyError('date') inside
    # AkShare parsers. Retry a bit before giving up.
    for _attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="",
            )
            return normalize_daily_df(df)
        except Exception as e:  # noqa: BLE001
            last_error = e
            try:
                import time

                time.sleep(0.6)
            except Exception:
                pass

    for sym in [to_ak_symbol(stock_code), stock_code]:
        for _attempt in range(2):
            try:
                df = ak.stock_zh_a_daily(symbol=sym, start_date=start_date, end_date=end_date)
                return normalize_daily_df(df)
            except Exception as e:  # noqa: BLE001
                last_error = e
                try:
                    import time

                    time.sleep(0.6)
                except Exception:
                    pass
                continue

    raise RuntimeError(f"Failed to fetch daily for {stock_code}: {type(last_error).__name__}: {last_error}")


def parse_market_cap_billion(raw) -> Optional[float]:
    """
    Parse market cap into "亿" (1e8 CNY) units.
    Accepts strings like "123.4亿", "1.2万亿", or plain numbers.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().replace(",", "")
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    val = float(m.group(1))

    if "万亿" in s:
        return val * 1e4
    if "亿" in s:
        return val
    if val > 1e8:
        return val / 1e8
    return val


def market_cap_score_from_billion(cap_billion: Optional[float]) -> int:
    # v3.0: core 30-100 亿
    if cap_billion is None or not np.isfinite(cap_billion):
        return 0
    cap_billion = float(cap_billion)
    if 30 <= cap_billion <= 100:
        return 10
    if 100 < cap_billion <= 150:
        return 6
    if 150 < cap_billion <= 300:
        return 3
    return 0


def fetch_float_market_cap_snapshot() -> dict[str, float]:
    """
    Return {stock_code(6 digits): float_market_cap_in_亿}.
    Uses a single "spot" query to avoid per-stock requests.
    """
    import akshare as ak

    df = ak.stock_zh_a_spot_em()
    if df is None or df.empty:
        return {}

    code_col = pick_col(df, ["code", "代码"])
    if code_col is None:
        code_col = df.columns[0]

    cap_col = None
    for c in df.columns:
        sc = str(c)
        if "流通市值" in sc:
            cap_col = c
            break
    if cap_col is None:
        for c in df.columns:
            sc = str(c)
            if "总市值" in sc:
                cap_col = c
                break
    if cap_col is None:
        return {}

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        code = normalize_stock_code(row[code_col])
        if not (code.isdigit() and len(code) == 6):
            continue
        cap_billion = parse_market_cap_billion(row[cap_col])
        if cap_billion is None:
            continue
        out[code] = float(cap_billion)
    return out


def fetch_market_cap_score(stock_code: str) -> int:
    import akshare as ak

    for sym in [to_ak_symbol(stock_code), stock_code]:
        try:
            df = ak.stock_a_indicator_lg(symbol=sym)
            if df is None or df.empty:
                continue

            col = None
            for c in df.columns:
                if "流通市值" in str(c):
                    col = c
                    break
            if col is None:
                for c in df.columns:
                    if "市值" in str(c):
                        col = c
                        break
            if col is None:
                continue

            cap_billion = parse_market_cap_billion(df[col].iloc[-1])
            if cap_billion is None:
                continue
            return market_cap_score_from_billion(cap_billion)
        except Exception:  # noqa: BLE001
            continue
    return 0
