#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build relay candidates from limit-up stocks using market-emotion + recent limit-up context.

Workflow:
1) Load stock_daily + stock_info in target range.
2) Compute market emotion / limit-up features with lag windows.
3) Train a lightweight binary MLP to score "next-day relay success" probability.
4) Select high-confidence relay candidates and generate daily strategy markdown files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from dailyReview import EPS, make_engine, resolve_db_target


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate next-day relay candidates from daily limit-up stocks.")
    p.add_argument("--config", default="config.ini", help="config.ini path.")
    p.add_argument("--db-url", default=None, help="SQLAlchemy DB URL.")
    p.add_argument("--db", default=None, help="SQLite DB path.")
    p.add_argument("--start-date", default="20200101", help="Sample start date (YYYYMMDD).")
    p.add_argument("--end-date", default="20260213", help="Sample end date (YYYYMMDD).")
    p.add_argument("--buffer-days", type=int, default=45, help="Extra calendar days loaded before start-date.")

    p.add_argument("--take-profit-threshold", type=float, default=0.02, help="Success rule: next_high/close-1 >= this.")
    p.add_argument("--stop-loss-threshold", type=float, default=-0.01, help="Success rule: next_close/close-1 >= this.")
    p.add_argument("--target-precision", type=float, default=0.90, help="Target precision for candidate threshold search.")
    p.add_argument("--min-selected", type=int, default=60, help="Min selected samples for threshold search.")

    p.add_argument("--hidden-size", type=int, default=56, help="MLP hidden size.")
    p.add_argument("--epochs", type=int, default=1000, help="MLP epochs.")
    p.add_argument("--learning-rate", type=float, default=0.035, help="MLP learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="MLP L2 penalty.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument("--max-per-day", type=int, default=6, help="Max recommended stocks per day.")
    p.add_argument("--output-dir", default="dailyReview/relayStrategy", help="Output directory.")
    p.add_argument("--report-file", default="dailyReview/relayStrategy/relay-model-report.md", help="Report markdown file.")
    p.add_argument("--csv-file", default="dailyReview/relayStrategy/relay-predictions.csv", help="Prediction CSV output.")
    return p.parse_args(argv)


def _to_iso(d: str) -> str:
    s = str(d or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _to_yyyymmdd(d: str) -> str:
    s = str(d or "").strip()
    if len(s) == 8 and s.isdigit():
        return s
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s.replace("-", "")
    return s


def _is_st_name(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str)
    return text.str.upper().str.contains("ST") | text.str.contains("退")


def _infer_limit_pct_vector(stock_code: pd.Series, is_st: pd.Series) -> np.ndarray:
    code = stock_code.fillna("").astype(str)
    out = np.full(len(code), 0.10, dtype=float)
    st_mask = is_st.to_numpy(dtype=bool)
    out[st_mask] = 0.05
    gem_mask = (~st_mask) & code.str.startswith(("300", "301", "688")).to_numpy(dtype=bool)
    out[gem_mask] = 0.20
    bse_mask = (~st_mask) & code.str.startswith(("8", "4")).to_numpy(dtype=bool)
    out[bse_mask] = 0.30
    return out


def _round_half_up_2(arr: np.ndarray) -> np.ndarray:
    return np.floor(arr * 100.0 + 0.5) / 100.0


def load_base_frame(engine, start_iso: str, end_iso: str) -> pd.DataFrame:
    sql = text(
        """
        SELECT d.stock_code,
               d.date,
               d.open,
               d.high,
               d.low,
               d.close,
               d.volume,
               d.amount,
               COALESCE(i.name, '') AS stock_name,
               i.float_cap_billion
        FROM stock_daily d
        LEFT JOIN stock_info i
            ON i.stock_code = d.stock_code
        WHERE d.date BETWEEN :s AND :e
        ORDER BY d.stock_code, d.date
        """
    )
    df = pd.read_sql(sql, engine, params={"s": start_iso, "e": end_iso})
    if df.empty:
        return df
    for c in ["open", "high", "low", "close", "volume", "amount", "float_cap_billion"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_flags_and_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["prev_close"] = out.groupby("stock_code", sort=False)["close"].shift(1)
    out["prev_amount"] = out.groupby("stock_code", sort=False)["amount"].shift(1)

    out["is_st"] = _is_st_name(out["stock_name"])
    out["limit_pct"] = _infer_limit_pct_vector(out["stock_code"], out["is_st"])

    prev_close = out["prev_close"].to_numpy(dtype=float)
    limit_pct = out["limit_pct"].to_numpy(dtype=float)
    out["limit_up"] = _round_half_up_2(prev_close * (1.0 + limit_pct))
    out["limit_down"] = _round_half_up_2(prev_close * (1.0 - limit_pct))

    out["valid_prev"] = out["prev_close"] > 0
    out["is_limit_up"] = out["valid_prev"] & (out["close"] >= out["limit_up"] - EPS)
    out["is_limit_down"] = out["valid_prev"] & (out["close"] <= out["limit_down"] + EPS)
    out["touched_limit_up"] = out["valid_prev"] & (out["high"] >= out["limit_up"] - EPS)
    out["broken"] = out["touched_limit_up"] & (~out["is_limit_up"])
    out["opened_limit"] = out["is_limit_up"] & (out["low"] < out["limit_up"] - EPS)
    out["one_word"] = out["is_limit_up"] & (out["open"] >= out["limit_up"] - EPS) & (out["low"] >= out["limit_up"] - EPS)

    # Consecutive limit-up count (single pass, data already sorted by stock_code/date).
    code_arr = out["stock_code"].astype(str).to_numpy()
    lu_arr = out["is_limit_up"].to_numpy(dtype=bool)
    board = np.zeros(len(out), dtype=np.int16)
    prev_code = ""
    cnt = 0
    for i, code in enumerate(code_arr):
        if code != prev_code:
            cnt = 0
            prev_code = code
        if lu_arr[i]:
            cnt += 1
        else:
            cnt = 0
        board[i] = cnt
    out["board_count"] = board

    # Per-stock lag features
    grp = out.groupby("stock_code", sort=False)
    out["ret1"] = out["close"] / out["prev_close"] - 1.0
    out["ret2"] = grp["ret1"].shift(1)
    out["ret3"] = grp["ret1"].shift(2)
    out["amount_chg1"] = out["amount"] / out["prev_amount"] - 1.0
    out["amp"] = (out["high"] - out["low"]) / out["prev_close"]
    out["close_open_ret"] = out["close"] / out["open"] - 1.0
    out["pullback"] = (out["high"] - out["close"]) / out["close"]

    for lag in (1, 2, 3):
        out[f"lu_lag{lag}"] = grp["is_limit_up"].shift(lag).fillna(False).astype(np.int8)
        out[f"broken_lag{lag}"] = grp["broken"].shift(lag).fillna(False).astype(np.int8)
    out["lu_recent3"] = out["lu_lag1"] + out["lu_lag2"] + out["lu_lag3"]
    out["broken_recent3"] = out["broken_lag1"] + out["broken_lag2"] + out["broken_lag3"]

    # Next-day fields for labeling and backtest
    for c in ("date", "open", "high", "low", "close"):
        out[f"next_{c}"] = grp[c].shift(-1)
    return out


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mk = df[df["valid_prev"] & (~df["is_st"])].copy()
    grp = mk.groupby("stock_code", sort=False)
    mk["prev_is_limit_up"] = grp["is_limit_up"].shift(1).fillna(False)
    mk["prev_is_broken"] = grp["broken"].shift(1).fillna(False)
    mk["prev_lu_red"] = mk["prev_is_limit_up"] & (mk["close"] > mk["prev_close"])
    mk["prev_broken_red"] = mk["prev_is_broken"] & (mk["close"] > mk["prev_close"])
    mk["board_for_max"] = np.where(mk["is_limit_up"], mk["board_count"], 0)

    daily = (
        mk.groupby("date", as_index=False)
        .agg(
            limit_up_count=("is_limit_up", "sum"),
            limit_down_count=("is_limit_down", "sum"),
            broken_count=("broken", "sum"),
            max_board=("board_for_max", "max"),
            amount_sum=("amount", "sum"),
            prev_lu_count=("prev_is_limit_up", "sum"),
            prev_lu_red=("prev_lu_red", "sum"),
            prev_broken_count=("prev_is_broken", "sum"),
            prev_broken_red=("prev_broken_red", "sum"),
        )
        .sort_values("date")
    )

    daily["broken_rate"] = daily["broken_count"] / (daily["broken_count"] + daily["limit_up_count"]).replace(0, np.nan)
    daily["red_rate"] = daily["prev_lu_red"] / daily["prev_lu_count"].replace(0, np.nan)
    daily["broken_red_rate"] = daily["prev_broken_red"] / daily["prev_broken_count"].replace(0, np.nan)

    daily["amount_ma5"] = daily["amount_sum"].rolling(5, min_periods=1).mean().shift(1)
    daily["amount_change5"] = daily["amount_sum"] / daily["amount_ma5"] - 1.0

    base_cols = [
        "limit_up_count",
        "limit_down_count",
        "broken_rate",
        "max_board",
        "red_rate",
        "broken_red_rate",
        "amount_change5",
    ]
    for col in base_cols:
        for lag in (1, 2, 3):
            daily[f"{col}_lag{lag}"] = daily[col].shift(lag)
    return daily


def build_sample_frames(
    full_df: pd.DataFrame,
    market_df: pd.DataFrame,
    start_iso: str,
    end_iso: str,
    tp: float,
    sl: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    merged = full_df.merge(market_df, on="date", how="left", suffixes=("", "_m"))
    pred_pool = merged[
        (merged["date"] >= start_iso)
        & (merged["date"] <= end_iso)
        & merged["valid_prev"]
        & (~merged["is_st"])
        & merged["is_limit_up"]
        & (~merged["one_word"])
    ].copy()
    if pred_pool.empty:
        return pred_pool, pred_pool.copy(), []

    # Label only rows that have next trade day available in range.
    labeled = pred_pool[pred_pool["next_date"].notna()].copy()
    gap_days = (pd.to_datetime(labeled["next_date"]) - pd.to_datetime(labeled["date"])).dt.days
    labeled = labeled[(gap_days >= 1) & (gap_days <= 7)].copy()
    labeled["next_high_ret"] = labeled["next_high"] / labeled["close"] - 1.0
    labeled["next_close_ret"] = labeled["next_close"] / labeled["close"] - 1.0
    labeled["relay_success"] = (
        (labeled["next_high_ret"] >= float(tp)) & (labeled["next_close_ret"] >= float(sl))
    ).astype(np.int8)

    feature_cols = [
        "board_count",
        "one_word",
        "opened_limit",
        "ret1",
        "ret2",
        "ret3",
        "amount_chg1",
        "amp",
        "close_open_ret",
        "pullback",
        "lu_recent3",
        "broken_recent3",
        "float_cap_billion",
        "limit_up_count",
        "limit_down_count",
        "broken_rate",
        "max_board",
        "red_rate",
        "broken_red_rate",
        "amount_change5",
    ]
    lag_bases = [
        "limit_up_count",
        "limit_down_count",
        "broken_rate",
        "max_board",
        "red_rate",
        "broken_red_rate",
        "amount_change5",
    ]
    for col in lag_bases:
        for lag in (1, 2, 3):
            feature_cols.append(f"{col}_lag{lag}")
    return pred_pool, labeled, feature_cols


def fill_and_scale(train_x: np.ndarray, pred_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros(train_x.shape[1], dtype=float)
    train_fill = train_x.astype(float, copy=True)
    pred_fill = pred_x.astype(float, copy=True)
    for i in range(train_x.shape[1]):
        col = train_fill[:, i]
        finite_mask = np.isfinite(col)
        valid = col[finite_mask]
        mean_i = float(valid.mean()) if valid.size else 0.0
        means[i] = mean_i
        if finite_mask.size:
            col[~finite_mask] = mean_i
        pred_col = pred_fill[:, i]
        pred_finite = np.isfinite(pred_col)
        if pred_finite.size:
            pred_col[~pred_finite] = mean_i
    std = train_fill.std(axis=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    std = std + 1e-6
    return (train_fill - means) / std, (pred_fill - means) / std, means, std


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)


def train_binary_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    y_oh = np.eye(2)[y.astype(int)]

    w1 = rng.normal(scale=0.15, size=(d, hidden_size))
    b1 = np.zeros(hidden_size)
    w2 = rng.normal(scale=0.15, size=(hidden_size, 2))
    b2 = np.zeros(2)

    for _ in range(max(1, int(epochs))):
        z1 = x @ w1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ w2 + b2
        p = softmax(z2)

        dz2 = (p - y_oh) / float(n)
        dw2 = a1.T @ dz2 + 2.0 * weight_decay * w2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ w2.T
        dz1 = da1 * (z1 > 0)
        dw1 = x.T @ dz1 + 2.0 * weight_decay * w1
        db1 = dz1.sum(axis=0)

        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
    return w1, b1, w2, b2


def predict_binary_mlp(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h = np.maximum(x @ w1 + b1, 0.0)
    z = h @ w2 + b2
    p = softmax(z)
    pred = np.argmax(p, axis=1).astype(np.int8)
    prob_pos = p[:, 1]
    return pred, prob_pos


def choose_threshold(
    prob: np.ndarray,
    y: np.ndarray,
    target_precision: float,
    min_selected: int,
) -> Tuple[float, float, float, int]:
    best: Optional[Tuple[float, float, float, int]] = None
    # prioritize thresholds that satisfy target precision and maximize coverage
    for th in np.arange(0.50, 0.991, 0.01):
        mask = prob >= float(th)
        count = int(mask.sum())
        if count < int(min_selected):
            continue
        precision = float(y[mask].mean())
        coverage = float(count / len(y))
        candidate = (float(th), precision, coverage, count)
        if precision >= target_precision:
            if best is None:
                best = candidate
                continue
            # among feasible thresholds, pick larger coverage then larger precision
            if coverage > best[2] + 1e-12 or (abs(coverage - best[2]) <= 1e-12 and precision > best[1]):
                best = candidate
    if best is not None:
        return best

    # fallback: maximize precision with minimum sample count
    for th in np.arange(0.50, 0.991, 0.01):
        mask = prob >= float(th)
        count = int(mask.sum())
        if count < int(min_selected):
            continue
        precision = float(y[mask].mean())
        coverage = float(count / len(y))
        candidate = (float(th), precision, coverage, count)
        if best is None or precision > best[1] or (abs(precision - best[1]) <= 1e-12 and coverage > best[2]):
            best = candidate
    if best is None:
        th = 0.5
        mask = prob >= th
        count = int(mask.sum())
        precision = float(y[mask].mean()) if count > 0 else 0.0
        coverage = float(count / len(y)) if len(y) else 0.0
        return th, precision, coverage, count
    return best


def market_stage(row: pd.Series) -> Tuple[str, str, str]:
    lu = float(row.get("limit_up_count", 0.0) or 0.0)
    br = float(row.get("broken_rate", np.nan))
    mb = float(row.get("max_board", 0.0) or 0.0)
    if np.isnan(br):
        br = 0.5
    if lu >= 60 and br <= 0.30 and mb >= 4:
        return "强趋势", "60%-80%", "优先2-4板换手票，允许2-3只并行。"
    if lu >= 35 and br <= 0.42 and mb >= 2:
        return "修复/震荡", "30%-50%", "只做分时回封确认，单票不超过15%。"
    return "弱势轮动", "10%-20%", "减少出手，仅做高胜率首板/二板回封。"


def render_day_plan(
    trade_date: str,
    day_df: pd.DataFrame,
    threshold: float,
    max_per_day: int,
) -> str:
    d = _to_iso(trade_date)
    m = day_df.iloc[0] if not day_df.empty else pd.Series(dtype=object)
    stage, pos, focus = market_stage(m)
    lu = int(float(m.get("limit_up_count", 0.0) or 0.0)) if not day_df.empty else 0
    ld = int(float(m.get("limit_down_count", 0.0) or 0.0)) if not day_df.empty else 0
    br = float(m.get("broken_rate", np.nan)) if not day_df.empty else np.nan
    mb = int(float(m.get("max_board", 0.0) or 0.0)) if not day_df.empty else 0

    rec_df = day_df[day_df["model_prob"] >= threshold].copy().sort_values("model_prob", ascending=False)
    rec_df = rec_df.head(max(1, int(max_per_day)))

    lines = [
        f"# {d} 次日接力计划",
        "",
        f"- 市场状态：{stage}",
        f"- 建议总仓位：{pos}",
        f"- 聚焦方向：{focus}",
        f"- 当日情绪：涨停 {lu} / 跌停 {ld} / 炸板率 {('N/A' if np.isnan(br) else f'{br*100:.2f}%')} / 最高连板 {mb}",
        f"- 模型阈值：P(接力成功) ≥ {threshold:.2f}",
        "",
        "## 可接力标的",
    ]

    if rec_df.empty:
        lines.append("无（当日未达到阈值的高置信标的，建议观望）。")
    else:
        lines.extend(
            [
                "| 代码 | 名称 | 概率 | 连板 | 当日涨跌 | 次日最高涨幅* | 次日收盘涨幅* |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rec_df.itertuples(index=False):
            next_high = row.next_high_ret * 100.0 if pd.notna(row.next_high_ret) else np.nan
            next_close = row.next_close_ret * 100.0 if pd.notna(row.next_close_ret) else np.nan
            lines.append(
                "| {code} | {name} | {prob:.2%} | {board} | {ret:.2%} | {nh} | {nc} |".format(
                    code=str(row.stock_code),
                    name=str(row.stock_name or ""),
                    prob=float(row.model_prob),
                    board=int(row.board_count or 0),
                    ret=float(row.ret1) if pd.notna(row.ret1) else 0.0,
                    nh=("待验证" if np.isnan(next_high) else f"{next_high:.2f}%"),
                    nc=("待验证" if np.isnan(next_close) else f"{next_close:.2f}%"),
                )
            )

    lines.extend(
        [
            "",
            "## 执行策略",
            "1. 只做“竞价不弱 + 盘中回封确认”的票，不做缩量一字追高。",
            "2. 单票风控：入场后若跌破前收 -3% 或分时放量走弱，直接止损。",
            "3. 止盈节奏：次日冲高 2%-4% 分批减仓，剩余仓位看是否二次回封。",
            "",
            "* 注：次日数据仅用于历史回测日验证，实时交易日会显示“待验证”。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    pred_pool: pd.DataFrame,
    labeled: pd.DataFrame,
    threshold: float,
    max_per_day: int,
    output_dir: Path,
    csv_file: Path,
    report_file: Path,
    start_date: str,
    end_date: str,
    tp: float,
    sl: float,
    target_precision: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("relay-plan-*.md"):
        try:
            old_file.unlink()
        except Exception:  # noqa: BLE001
            pass

    # Daily plan files
    for trade_date, day_df in pred_pool.groupby("date", sort=True):
        path = output_dir / f"relay-plan-{_to_yyyymmdd(str(trade_date))}.md"
        path.write_text(
            render_day_plan(
                trade_date=str(trade_date),
                day_df=day_df.sort_values("model_prob", ascending=False),
                threshold=threshold,
                max_per_day=max_per_day,
            ),
            encoding="utf-8",
        )

    # Candidate CSV
    cols = [
        "date",
        "stock_code",
        "stock_name",
        "board_count",
        "model_prob",
        "model_pred",
        "ret1",
        "amount",
        "limit_up_count",
        "broken_rate",
        "next_date",
        "next_high_ret",
        "next_close_ret",
        "relay_success",
    ]
    for c in cols:
        if c not in pred_pool.columns:
            pred_pool[c] = np.nan
    pred_pool[cols].sort_values(["date", "model_prob"], ascending=[True, False]).to_csv(csv_file, index=False, encoding="utf-8-sig")

    # Report metrics (focus on relay candidate hit-rate)
    mask = labeled["model_prob"] >= threshold
    selected = labeled.loc[mask].copy()
    hit_rate = float(selected["relay_success"].mean()) if not selected.empty else 0.0
    coverage = float(len(selected) / len(labeled)) if len(labeled) else 0.0
    daily_selected = selected.groupby("date").size()
    day_count = int((daily_selected > 0).sum())

    # Unfiltered hit-rate as baseline (if you relay all limit-up stocks)
    baseline_hit = float(labeled["relay_success"].mean()) if len(labeled) else 0.0

    lines = [
        "# 接力模型报告（市场情绪 + 涨停结构）",
        "",
        f"- 样本区间：{start_date} ~ {end_date}",
        f"- 样本总数（可回测涨停样本）：{len(labeled)}",
        f"- 成功定义：次日最高涨幅 ≥ {tp*100:.1f}% 且次日收盘涨幅 ≥ {sl*100:.1f}%",
        "",
        "## 命中率",
        f"- 全量接力基准命中率（不筛选）：{baseline_hit*100:.2f}%",
        f"- 模型筛选阈值：P(成功) ≥ {threshold:.2f}",
        f"- 模型筛选命中率：{hit_rate*100:.2f}%",
        f"- 覆盖率：{coverage*100:.2f}%（{len(selected)}/{len(labeled)}）",
        f"- 有候选交易日：{day_count} 天",
        "",
        "## 指标约束检查",
        f"- 目标命中率：{target_precision*100:.1f}%",
        f"- 实际命中率：{hit_rate*100:.2f}% {'✅' if hit_rate >= target_precision else '⚠️'}",
        "",
        "## 产出文件",
        f"- 每日计划目录：`{output_dir.as_posix()}`",
        f"- 明细CSV：`{csv_file.as_posix()}`",
        "",
    ]
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    start_iso = _to_iso(args.start_date)
    end_iso = _to_iso(args.end_date)
    start_dt = dt.datetime.strptime(start_iso, "%Y-%m-%d").date()
    buffer_start = (start_dt - dt.timedelta(days=max(5, int(args.buffer_days)))).strftime("%Y-%m-%d")

    engine = make_engine(resolve_db_target(args))
    base = load_base_frame(engine, buffer_start, end_iso)
    if base.empty:
        raise SystemExit("No stock_daily data in requested range.")

    full = compute_flags_and_stock_features(base)
    market = compute_market_features(full)
    pred_pool, labeled, feature_cols = build_sample_frames(
        full_df=full,
        market_df=market,
        start_iso=start_iso,
        end_iso=end_iso,
        tp=float(args.take_profit_threshold),
        sl=float(args.stop_loss_threshold),
    )
    if labeled.empty:
        raise SystemExit("No labeled samples generated (check date range / data completeness).")

    train_x_raw = labeled[feature_cols].to_numpy(dtype=float)
    pred_x_raw = pred_pool[feature_cols].to_numpy(dtype=float)
    train_y = labeled["relay_success"].to_numpy(dtype=np.int8)

    train_x, pred_x, _, _ = fill_and_scale(train_x_raw, pred_x_raw)
    w1, b1, w2, b2 = train_binary_mlp(
        x=train_x,
        y=train_y,
        hidden_size=int(args.hidden_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    train_pred, train_prob = predict_binary_mlp(train_x, w1, b1, w2, b2)
    pool_pred, pool_prob = predict_binary_mlp(pred_x, w1, b1, w2, b2)

    # attach predictions
    labeled = labeled.copy()
    labeled["model_pred"] = train_pred
    labeled["model_prob"] = train_prob
    pred_pool = pred_pool.copy()
    pred_pool["model_pred"] = pool_pred
    pred_pool["model_prob"] = pool_prob

    threshold, precision, coverage, selected_n = choose_threshold(
        prob=train_prob,
        y=train_y,
        target_precision=float(args.target_precision),
        min_selected=int(args.min_selected),
    )

    # bring backtest columns to full prediction pool
    merge_cols = [
        "date",
        "stock_code",
        "next_high_ret",
        "next_close_ret",
        "relay_success",
    ]
    backtest = labeled[merge_cols].copy()
    pred_pool = pred_pool.merge(backtest, on=["date", "stock_code"], how="left")

    output_dir = Path(args.output_dir).expanduser()
    csv_file = Path(args.csv_file).expanduser()
    report_file = Path(args.report_file).expanduser()
    write_outputs(
        pred_pool=pred_pool,
        labeled=labeled,
        threshold=float(threshold),
        max_per_day=int(args.max_per_day),
        output_dir=output_dir,
        csv_file=csv_file,
        report_file=report_file,
        start_date=_to_yyyymmdd(start_iso),
        end_date=_to_yyyymmdd(end_iso),
        tp=float(args.take_profit_threshold),
        sl=float(args.stop_loss_threshold),
        target_precision=float(args.target_precision),
    )

    fit_acc = float(np.mean(train_pred == train_y))
    baseline_hit = float(np.mean(train_y))
    print(
        "[relay model] "
        f"samples={len(train_y)} baseline_hit={baseline_hit*100:.2f}% "
        f"fit_acc={fit_acc*100:.2f}% threshold={threshold:.2f} "
        f"selected={selected_n} selected_hit={precision*100:.2f}% coverage={coverage*100:.2f}%"
    )
    print(f"[relay model] output_dir={output_dir.as_posix()}")
    print(f"[relay model] report_md={report_file.as_posix()}")
    print(f"[relay model] detail_csv={csv_file.as_posix()}")

    if precision < float(args.target_precision):
        print("[relay model] WARNING: selected-hit below target precision.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
