#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit and evaluate dailyReview next-day stage model.

What it does:
1) Parse dailyReview markdown files.
2) Build t -> t+1 supervised samples.
3) Add optional 5-minute market microstructure features from stock_minute.
4) Train a lightweight MLP for historical fitting.
5) Run rolling forward (walk-forward) evaluation via kNN.
6) Export CSV and markdown report.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from dailyReview import make_engine, resolve_db_target


STAGE_TO_ID: Dict[str, int] = {
    "冰点": 0,
    "混沌": 1,
    "修复": 2,
    "主升": 3,
}
ID_TO_STAGE: Dict[int, str] = {v: k for k, v in STAGE_TO_ID.items()}

MINUTE_FEATURE_KEYS: Tuple[str, ...] = (
    "m_ret",
    "m_morning_ret",
    "m_afternoon_ret",
    "m_range",
    "m_open30_ratio",
    "m_close30_ratio",
    "m_amount_log",
    "m_vw_std",
    "m_bars",
)
MINUTE_PROXY_CODES: Tuple[str, ...] = ("000001", "600000", "601398", "600519", "300750")


PATTERNS = {
    "stage": re.compile(r"^情绪阶段：(.+)$", re.M),
    "position": re.compile(r"^明日总仓位上限：(.+)$", re.M),
    "score": re.compile(r"^情绪总分：(\d+)\s*/\s*(\d+)", re.M),
    "limit_up": re.compile(r"### 涨停家数\s+数值：([\d.]+)\s*家", re.S),
    "limit_down": re.compile(r"### 跌停家数\s+数值：([\d.]+)\s*家", re.S),
    "broken": re.compile(r"### 炸板率\s+数值：([\d.]+)%（炸板\s*(\d+)\s*/\s*封板\s*(\d+)）", re.S),
    "max_consecutive": re.compile(r"### 最高连板\s+数值：([\d.]+)\s*板", re.S),
    "red_rate": re.compile(r"### 昨日涨停红盘率\s+数值：([\d.]+)%", re.S),
    "prev_broken_red_rate": re.compile(r"### 昨日炸板红盘率\s+数值：([\d.]+)%", re.S),
    "amount_change": re.compile(r"### 成交额变化[\s\S]*?变化：([+\-]?[\d.]+)%", re.S),
}


@dataclass
class DayReview:
    date: str  # YYYYMMDD
    stage_id: int
    pred_stage_id: int
    score: float
    score_max: float
    limit_up: float
    limit_down: float
    broken_rate: float
    broken_count: float
    sealed_count: float
    max_consecutive: float
    red_rate: float
    prev_broken_red_rate: float
    amount_change: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit next-day stage model from dailyReview markdown files.")
    p.add_argument("--review-dir", default="dailyReview", help="Directory storing daily-review-*.md.")
    p.add_argument("--start-date", default="20250701", help="Start date (YYYYMMDD).")
    p.add_argument("--end-date", default=None, help="End date (YYYYMMDD).")
    p.add_argument("--config", default="config.ini", help="Config path for DB resolving.")
    p.add_argument("--db-url", default=None, help="SQLAlchemy DB URL.")
    p.add_argument("--db", default=None, help="SQLite DB path.")
    p.add_argument("--minute-frequency", default="5", help="stock_minute frequency, default 5.")
    p.add_argument("--disable-minute-features", action="store_true", help="Disable 5-minute features.")

    p.add_argument("--hidden-size", type=int, default=64, help="MLP hidden layer width.")
    p.add_argument("--epochs", type=int, default=5000, help="MLP training epochs.")
    p.add_argument("--learning-rate", type=float, default=0.03, help="MLP learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="MLP L2 regularization.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    p.add_argument("--forward-min-train", type=int, default=40, help="Rolling evaluation minimum train samples.")
    p.add_argument("--forward-k", type=int, default=31, help="Rolling kNN k.")
    p.add_argument("--forward-confidence-threshold", type=float, default=0.55, help="Confidence threshold for strong signals.")

    p.add_argument("--report-file", default="dailyReview/model-fit-report.md", help="Markdown report output path.")
    p.add_argument("--csv-file", default="dailyReview/model-prediction-compare.csv", help="CSV comparison output path.")
    return p.parse_args(argv)


def normalize_stage_to_id(stage_text: str) -> int:
    text = str(stage_text or "").strip()
    for k, v in STAGE_TO_ID.items():
        if k in text:
            return v
    return -1


def normalize_position_to_stage_id(position_text: str) -> int:
    text = str(position_text or "").strip().replace("-", "–")
    if "60–100" in text:
        return STAGE_TO_ID["主升"]
    if "30–50" in text:
        return STAGE_TO_ID["修复"]
    if "≤20" in text:
        return STAGE_TO_ID["混沌"]
    if "0–10" in text:
        return STAGE_TO_ID["冰点"]
    return -1


def _grab_float(text: str, key: str, default: float = np.nan, group: int = 1, scale: float = 1.0) -> float:
    m = PATTERNS[key].search(text)
    if not m:
        return default
    try:
        return float(m.group(group)) * scale
    except Exception:  # noqa: BLE001
        return default


def parse_review_file(path: Path) -> Optional[DayReview]:
    text = path.read_text(encoding="utf-8")
    stage_m = PATTERNS["stage"].search(text)
    position_m = PATTERNS["position"].search(text)
    score_m = PATTERNS["score"].search(text)
    if not (stage_m and position_m and score_m):
        return None

    stage_id = normalize_stage_to_id(stage_m.group(1))
    pred_stage_id = normalize_position_to_stage_id(position_m.group(1))
    if stage_id < 0 or pred_stage_id < 0:
        return None

    return DayReview(
        date=path.stem.split("-")[-1],
        stage_id=stage_id,
        pred_stage_id=pred_stage_id,
        score=float(score_m.group(1)),
        score_max=float(score_m.group(2)),
        limit_up=_grab_float(text, "limit_up", default=0.0),
        limit_down=_grab_float(text, "limit_down", default=0.0),
        broken_rate=_grab_float(text, "broken", default=0.0, group=1, scale=0.01),
        broken_count=_grab_float(text, "broken", default=0.0, group=2),
        sealed_count=_grab_float(text, "broken", default=0.0, group=3),
        max_consecutive=_grab_float(text, "max_consecutive", default=0.0),
        red_rate=_grab_float(text, "red_rate", default=0.0, scale=0.01),
        prev_broken_red_rate=_grab_float(text, "prev_broken_red_rate", default=0.0, scale=0.01),
        amount_change=_grab_float(text, "amount_change", default=0.0, scale=0.01),
    )


def load_reviews(review_dir: Path, start_date: str, end_date: Optional[str]) -> List[DayReview]:
    files = sorted(review_dir.glob("daily-review-*.md"))
    rows: List[DayReview] = []
    for file in files:
        date = file.stem.split("-")[-1]
        if date < start_date:
            continue
        if end_date and date > end_date:
            continue
        row = parse_review_file(file)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda x: x.date)
    return rows


def _to_iso_date(date_text: str) -> str:
    d = str(date_text or "").strip()
    if len(d) == 8 and d.isdigit():
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
    return d


def _to_yyyymmdd(date_text: str) -> str:
    d = str(date_text or "").strip()
    if len(d) == 8 and d.isdigit():
        return d
    if len(d) == 10 and d[4] == "-" and d[7] == "-":
        return d.replace("-", "")
    return d


def _to_time8(time_text: object) -> str:
    t = str(time_text or "").strip()
    if not t:
        return ""
    if len(t) == 5:
        return f"{t}:00"
    if len(t) == 7:
        return f"0{t}"
    return t


def load_minute_features(engine: Engine, start_date: str, end_date: str, frequency: str) -> Dict[str, Dict[str, float]]:
    stmt = (
        text(
            """
            SELECT stock_code,
                   date,
                   time,
                   open,
                   high,
                   low,
                   close,
                   amount
            FROM stock_minute
            WHERE frequency = :freq
              AND date BETWEEN :s AND :e
              AND stock_code IN :codes
            ORDER BY stock_code, date, time
            """
        )
        .bindparams(bindparam("codes", expanding=True))
    )

    with engine.connect() as conn:
        rows = conn.execute(
            stmt,
            {
                "freq": str(frequency),
                "s": _to_iso_date(start_date),
                "e": _to_iso_date(end_date),
                "codes": list(MINUTE_PROXY_CODES),
            },
        ).fetchall()

    by_code_date: Dict[Tuple[str, str], List[Tuple[str, float, float, float, float, float]]] = {}
    for row in rows:
        code = str(row[0] or "").strip()
        date = _to_yyyymmdd(str(row[1]))
        tm = _to_time8(row[2])
        op = float(row[3] or 0.0)
        hi = float(row[4] or 0.0)
        lo = float(row[5] or 0.0)
        cl = float(row[6] or 0.0)
        amt = float(row[7] or 0.0)
        if not code or not date or not tm:
            continue
        by_code_date.setdefault((code, date), []).append((tm, op, hi, lo, cl, amt))

    by_date_features: Dict[str, List[Tuple[float, Dict[str, float]]]] = {}
    for (_, date), arr in by_code_date.items():
        if not arr:
            continue
        arr.sort(key=lambda x: x[0])
        times = [x[0] for x in arr]
        open_arr = np.asarray([x[1] for x in arr], dtype=float)
        high_arr = np.asarray([x[2] for x in arr], dtype=float)
        low_arr = np.asarray([x[3] for x in arr], dtype=float)
        close_arr = np.asarray([x[4] for x in arr], dtype=float)
        amount_arr = np.asarray([x[5] for x in arr], dtype=float)
        if close_arr.size == 0:
            continue

        open_px = float(open_arr[0]) if np.isfinite(open_arr[0]) and abs(open_arr[0]) > 1e-9 else float(close_arr[0])
        close_px = float(close_arr[-1])
        if not np.isfinite(open_px) or abs(open_px) < 1e-9:
            continue

        noon_idx = [i for i, t in enumerate(times) if t <= "11:30:00"]
        mid_idx = int(noon_idx[-1]) if noon_idx else int(len(arr) // 2)
        mid_px = float(close_arr[mid_idx])

        total_amount = float(np.nansum(amount_arr))
        first_30 = float(np.nansum(amount_arr[:6]))
        last_30 = float(np.nansum(amount_arr[-6:]))
        day_high = float(np.nanmax(high_arr)) if high_arr.size else open_px
        day_low = float(np.nanmin(low_arr)) if low_arr.size else open_px

        denom_open = max(abs(open_px), 1e-6)
        denom_mid = max(abs(mid_px), 1e-6)
        item = {
            "m_ret": (close_px - open_px) / denom_open,
            "m_morning_ret": (mid_px - open_px) / denom_open,
            "m_afternoon_ret": (close_px - mid_px) / denom_mid,
            "m_range": (day_high - day_low) / denom_open,
            "m_open30_ratio": first_30 / max(total_amount, 1e-6),
            "m_close30_ratio": last_30 / max(total_amount, 1e-6),
            "m_amount_log": float(np.log1p(max(total_amount, 0.0))),
            "m_vw_std": float(np.nanstd(close_arr) / denom_open),
            "m_bars": float(len(arr)),
        }
        by_date_features.setdefault(date, []).append((max(total_amount, 1.0), item))

    out: Dict[str, Dict[str, float]] = {}
    for date, weighted_items in by_date_features.items():
        if not weighted_items:
            continue
        weights = np.asarray([w for w, _ in weighted_items], dtype=float)
        weights = weights / max(float(weights.sum()), 1e-9)
        merged: Dict[str, float] = {}
        for key in MINUTE_FEATURE_KEYS:
            values = np.asarray([float(feats.get(key, 0.0)) for _, feats in weighted_items], dtype=float)
            merged[key] = float(np.sum(weights * values))
        out[date] = merged
    return out


def build_dataset(
    rows: Sequence[DayReview],
    minute_map: Dict[str, Dict[str, float]],
    use_minute_features: bool,
) -> Tuple[np.ndarray, np.ndarray, List[dict], List[str]]:
    if len(rows) < 3:
        raise ValueError("Not enough dailyReview files to build samples.")

    base_feature_names = [
        "limit_up",
        "limit_down",
        "broken_rate",
        "broken_count",
        "sealed_count",
        "max_consecutive",
        "red_rate",
        "prev_broken_red_rate",
        "amount_change",
        "score",
        "score_max",
        "stage_id",
        "pred_stage_id",
    ]
    feature_names = list(base_feature_names)
    if use_minute_features:
        feature_names.extend(MINUTE_FEATURE_KEYS)

    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    meta: List[dict] = []

    for i in range(len(rows) - 1):
        today = rows[i]
        nxt = rows[i + 1]
        feats = [
            today.limit_up,
            today.limit_down,
            today.broken_rate,
            today.broken_count,
            today.sealed_count,
            today.max_consecutive,
            today.red_rate,
            today.prev_broken_red_rate,
            today.amount_change,
            today.score,
            today.score_max,
            float(today.stage_id),
            float(today.pred_stage_id),
        ]
        has_minute = 0
        if use_minute_features:
            minute = minute_map.get(today.date, {})
            has_minute = int(bool(minute))
            for key in MINUTE_FEATURE_KEYS:
                feats.append(float(minute.get(key, 0.0)))

        X_rows.append(feats)
        y_rows.append(int(nxt.stage_id))
        meta.append(
            {
                "date": today.date,
                "next_date": nxt.date,
                "baseline_pred": int(today.pred_stage_id),
                "actual": int(nxt.stage_id),
                "has_minute_features": has_minute,
            }
        )

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)
    return X, y, meta, feature_names


def fill_nan_and_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.size == 0:
        return X.copy(), np.asarray([]), np.asarray([])
    means = np.nanmean(X, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    X_filled = np.where(np.isnan(X), means, X)
    std = X_filled.std(axis=0)
    std = np.where(np.isfinite(std), std, 1.0)
    std = std + 1e-6
    return (X_filled - means) / std, means, std


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    classes = int(max(STAGE_TO_ID.values()) + 1)
    Y = np.eye(classes)[y]

    W1 = rng.normal(scale=0.2, size=(d, hidden_size))
    b1 = np.zeros(hidden_size)
    W2 = rng.normal(scale=0.2, size=(hidden_size, classes))
    b2 = np.zeros(classes)

    for _ in range(max(1, int(epochs))):
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        logits = a1 @ W2 + b2
        probs = softmax(logits)

        dz2 = (probs - Y) / float(n)
        dW2 = a1.T @ dz2 + 2.0 * weight_decay * W2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1 + 2.0 * weight_decay * W1
        db1 = dz1.sum(axis=0)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    return W1, b1, W2, b2


def predict_mlp(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hidden = np.maximum(X @ W1 + b1, 0.0)
    logits = hidden @ W2 + b2
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1).astype(int)
    conf = probs[np.arange(len(preds)), preds]
    return preds, conf


def rolling_knn_forward(
    X_raw: np.ndarray,
    y: np.ndarray,
    min_train: int,
    k: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    preds = np.full(n, -1, dtype=int)
    conf = np.full(n, np.nan, dtype=float)
    if n <= min_train:
        return preds, conf

    for i in range(max(1, int(min_train)), n):
        X_train = X_raw[:i]
        y_train = y[:i]
        x_test = X_raw[i]

        means = np.nanmean(X_train, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        X_train = np.where(np.isnan(X_train), means, X_train)
        x_test = np.where(np.isnan(x_test), means, x_test)

        mu = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-6
        Xs = (X_train - mu) / std
        xs = (x_test - mu) / std

        dist = np.sum((Xs - xs) ** 2, axis=1)
        kk = min(max(1, int(k)), len(dist))
        nearest = np.argpartition(dist, kk - 1)[:kk]
        weights = 1.0 / (dist[nearest] + 1e-6)

        scores = np.zeros(num_classes, dtype=float)
        for idx, w in zip(nearest, weights):
            scores[int(y_train[idx])] += float(w)
        pred = int(np.argmax(scores))
        preds[i] = pred
        conf[i] = float(scores[pred] / (scores.sum() + 1e-12))
    return preds, conf


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def stage_name(stage_id: int) -> str:
    return ID_TO_STAGE.get(int(stage_id), f"未知({stage_id})")


def write_compare_csv(
    path: Path,
    meta: List[dict],
    fit_preds: np.ndarray,
    fit_conf: np.ndarray,
    forward_preds: np.ndarray,
    forward_conf: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trade_date",
                "next_trade_date",
                "baseline_prediction",
                "actual_next_stage",
                "baseline_match",
                "fit_prediction",
                "fit_match",
                "fit_confidence",
                "forward_prediction",
                "forward_match",
                "forward_confidence",
                "has_minute_features",
            ]
        )
        for i, item in enumerate(meta):
            baseline = int(item["baseline_pred"])
            actual = int(item["actual"])
            fit = int(fit_preds[i])
            fwd = int(forward_preds[i]) if i < len(forward_preds) else -1
            fwd_valid = fwd >= 0
            writer.writerow(
                [
                    item["date"],
                    item["next_date"],
                    stage_name(baseline),
                    stage_name(actual),
                    int(baseline == actual),
                    stage_name(fit),
                    int(fit == actual),
                    f"{float(fit_conf[i]):.6f}",
                    stage_name(fwd) if fwd_valid else "",
                    int(fwd == actual) if fwd_valid else "",
                    f"{float(forward_conf[i]):.6f}" if fwd_valid else "",
                    int(item.get("has_minute_features", 0)),
                ]
            )


def write_report(
    path: Path,
    sample_count: int,
    start_date: str,
    end_date: str,
    baseline_acc: float,
    fit_acc: float,
    forward_acc: float,
    forward_samples: int,
    forward_strong_acc: float,
    forward_strong_samples: int,
    forward_conf_threshold: float,
    use_minute_features: bool,
    minute_coverage: float,
    feature_names: Sequence[str],
    csv_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DailyReview 次日结果拟合报告",
        "",
        f"- 样本区间：{start_date} ~ {end_date}",
        f"- 样本数量：{sample_count}",
        "- 标签定义：`t`日预测 对比 `t+1` 交易日情绪阶段（冰点/混沌/修复/主升）",
        "",
        "## 指标结果",
        f"- 原始预测准确率（明日仓位映射）：{baseline_acc * 100:.2f}%",
        f"- 模型拟合准确率（轻量神经网络，训练集）：{fit_acc * 100:.2f}%",
        f"- 滚动前瞻准确率（kNN，walk-forward）：{forward_acc * 100:.2f}%（样本 {forward_samples}）",
        f"- 滚动强信号准确率（置信度≥{forward_conf_threshold:.2f}）：{forward_strong_acc * 100:.2f}%（样本 {forward_strong_samples}）",
        "",
        "## 特征与口径",
        f"- 是否启用5分钟K线特征：{'是' if use_minute_features else '否'}",
        f"- 5分钟特征覆盖率：{minute_coverage * 100:.2f}%",
        f"- 使用特征数：{len(feature_names)}",
        f"- 特征列表：{', '.join(feature_names)}",
        "",
        "## 文件",
        f"- 逐日对比：`{csv_path.as_posix()}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    review_dir = Path(args.review_dir).expanduser()
    if not review_dir.exists():
        raise SystemExit(f"Review directory does not exist: {review_dir}")

    rows = load_reviews(review_dir, start_date=str(args.start_date), end_date=args.end_date)
    if len(rows) < 3:
        raise SystemExit("Not enough parsed dailyReview files to train model.")

    use_minute_features = not bool(args.disable_minute_features)
    minute_map: Dict[str, Dict[str, float]] = {}
    if use_minute_features:
        db_target = resolve_db_target(args)
        engine = make_engine(db_target)
        minute_map = load_minute_features(
            engine=engine,
            start_date=rows[0].date,
            end_date=rows[-1].date,
            frequency=str(args.minute_frequency),
        )

    X_raw, y, meta, feature_names = build_dataset(
        rows=rows,
        minute_map=minute_map,
        use_minute_features=use_minute_features,
    )
    baseline_preds = np.asarray([int(m["baseline_pred"]) for m in meta], dtype=int)
    baseline_acc = accuracy(y, baseline_preds)

    X_norm, _, _ = fill_nan_and_scale(X_raw)
    W1, b1, W2, b2 = train_mlp(
        X=X_norm,
        y=y,
        hidden_size=int(args.hidden_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    fit_preds, fit_conf = predict_mlp(X_norm, W1, b1, W2, b2)
    fit_acc = accuracy(y, fit_preds)

    forward_preds, forward_conf = rolling_knn_forward(
        X_raw=X_raw,
        y=y,
        min_train=int(args.forward_min_train),
        k=int(args.forward_k),
        num_classes=int(max(STAGE_TO_ID.values()) + 1),
    )
    valid_forward = forward_preds >= 0
    forward_acc = accuracy(y[valid_forward], forward_preds[valid_forward]) if np.any(valid_forward) else 0.0
    forward_samples = int(np.sum(valid_forward))

    conf_th = float(args.forward_confidence_threshold)
    strong_mask = valid_forward & np.isfinite(forward_conf) & (forward_conf >= conf_th)
    forward_strong_acc = accuracy(y[strong_mask], forward_preds[strong_mask]) if np.any(strong_mask) else 0.0
    forward_strong_samples = int(np.sum(strong_mask))

    csv_path = Path(args.csv_file).expanduser()
    report_path = Path(args.report_file).expanduser()
    write_compare_csv(csv_path, meta, fit_preds, fit_conf, forward_preds, forward_conf)

    minute_hits = sum(int(m.get("has_minute_features", 0)) for m in meta)
    minute_coverage = float(minute_hits / len(meta)) if meta else 0.0
    write_report(
        path=report_path,
        sample_count=len(y),
        start_date=rows[0].date,
        end_date=rows[-1].date,
        baseline_acc=baseline_acc,
        fit_acc=fit_acc,
        forward_acc=forward_acc,
        forward_samples=forward_samples,
        forward_strong_acc=forward_strong_acc,
        forward_strong_samples=forward_strong_samples,
        forward_conf_threshold=conf_th,
        use_minute_features=use_minute_features,
        minute_coverage=minute_coverage,
        feature_names=feature_names,
        csv_path=csv_path,
    )

    print(
        "[dailyReview model] "
        f"samples={len(y)} baseline_acc={baseline_acc * 100:.2f}% "
        f"fit_acc={fit_acc * 100:.2f}% forward_acc={forward_acc * 100:.2f}% "
        f"forward_strong_acc={forward_strong_acc * 100:.2f}%"
    )
    print(f"[dailyReview model] compare_csv={csv_path.as_posix()}")
    print(f"[dailyReview model] report_md={report_path.as_posix()}")

    if fit_acc < 0.80:
        print("[dailyReview model] WARNING: fit accuracy below 80%.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
