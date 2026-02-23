#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for a single-file relay model."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


FORMAT_VERSION = 1
DEFAULT_MODEL_FILE = "models/relay_model_active.npz"


def default_model_path() -> Path:
    return Path(DEFAULT_MODEL_FILE)


def clamp_threshold(value: Optional[float], default: float) -> float:
    try:
        if value is None:
            raise ValueError("none")
        v = float(value)
    except Exception:
        v = float(default)
    if not np.isfinite(v):
        v = float(default)
    return float(max(0.50, min(0.99, v)))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(
    path: Path,
    *,
    feature_cols: List[str],
    means: np.ndarray,
    std: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    cdf_ref: Optional[np.ndarray],
    meta: Dict[str, Any],
) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    merged_meta = dict(meta or {})
    merged_meta.setdefault("created_at", now_text)
    merged_meta.setdefault("model_version", f"relay-{now_text.replace(' ', 'T').replace(':', '')}")
    payload = {
        "format_version": np.asarray([FORMAT_VERSION], dtype=np.int32),
        "feature_cols": np.asarray(list(feature_cols), dtype=object),
        "means": np.asarray(means, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "w1": np.asarray(w1, dtype=np.float64),
        "b1": np.asarray(b1, dtype=np.float64),
        "w2": np.asarray(w2, dtype=np.float64),
        "b2": np.asarray(b2, dtype=np.float64),
        "cdf_ref": np.asarray(cdf_ref if cdf_ref is not None else np.asarray([], dtype=np.float64), dtype=np.float64),
        "meta_json": np.asarray([json.dumps(merged_meta, ensure_ascii=False)], dtype=object),
    }
    np.savez_compressed(path, **payload)


def load_model(path: Path) -> Dict[str, Any]:
    p = path.expanduser()
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {p.as_posix()}")
    with np.load(p, allow_pickle=True) as data:
        fmt_arr = data.get("format_version")
        fmt = int(fmt_arr[0]) if fmt_arr is not None and len(fmt_arr) else 0
        if fmt != FORMAT_VERSION:
            raise ValueError(f"unsupported model format: {fmt}")

        feature_cols_raw = data["feature_cols"].tolist()
        feature_cols = [str(x) for x in feature_cols_raw]
        means = np.asarray(data["means"], dtype=np.float64)
        std = np.asarray(data["std"], dtype=np.float64)
        w1 = np.asarray(data["w1"], dtype=np.float64)
        b1 = np.asarray(data["b1"], dtype=np.float64)
        w2 = np.asarray(data["w2"], dtype=np.float64)
        b2 = np.asarray(data["b2"], dtype=np.float64)
        cdf_ref = np.asarray(data.get("cdf_ref", np.asarray([], dtype=np.float64)), dtype=np.float64)
        meta_raw = data.get("meta_json")
        meta: Dict[str, Any] = {}
        if meta_raw is not None and len(meta_raw):
            try:
                meta = json.loads(str(meta_raw[0]))
            except Exception:
                meta = {}

    if means.shape[0] != len(feature_cols) or std.shape[0] != len(feature_cols):
        raise ValueError("model means/std shape mismatch feature columns")
    if w1.shape[0] != len(feature_cols):
        raise ValueError("model w1 shape mismatch feature columns")

    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    model = {
        "path": p,
        "feature_cols": feature_cols,
        "means": means,
        "std": std,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "cdf_ref": np.sort(cdf_ref[np.isfinite(cdf_ref)]),
        "meta": meta,
    }
    return model


def predict_raw_prob(model: Dict[str, Any], x_raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(x_raw, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("x_raw must be 2D")
    means = np.asarray(model["means"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    if arr.shape[1] != len(means):
        raise ValueError(f"feature dim mismatch: x={arr.shape[1]} model={len(means)}")

    x = arr.copy()
    for i in range(x.shape[1]):
        col = x[:, i]
        valid = np.isfinite(col)
        col[~valid] = means[i]
    x = (x - means) / std

    w1 = np.asarray(model["w1"], dtype=np.float64)
    b1 = np.asarray(model["b1"], dtype=np.float64)
    w2 = np.asarray(model["w2"], dtype=np.float64)
    b2 = np.asarray(model["b2"], dtype=np.float64)

    h = np.maximum(x @ w1 + b1, 0.0)
    z = h @ w2 + b2
    z = z - np.max(z, axis=1, keepdims=True)
    ex = np.exp(z)
    p = ex / np.sum(ex, axis=1, keepdims=True)
    if p.shape[1] < 2:
        raise ValueError("binary classifier expected 2 output classes")
    return p[:, 1]


def score_from_raw_prob(model: Dict[str, Any], raw_prob: np.ndarray, alpha: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(raw_prob, dtype=np.float64)
    if arr.size == 0:
        return arr

    meta = model.get("meta") or {}
    alpha_v = int(alpha if alpha is not None else meta.get("alpha", 10))
    alpha_v = max(1, alpha_v)

    ref = np.asarray(model.get("cdf_ref", np.asarray([], dtype=np.float64)), dtype=np.float64)
    if ref.size > 0:
        rank = np.searchsorted(ref, arr, side="right").astype(np.float64) / float(ref.size)
    else:
        # Fallback if model has no calibration reference.
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
        rank = ranks / float(len(arr))

    score = 1.0 - np.power(1.0 - np.clip(rank, 0.0, 1.0), alpha_v)
    return np.clip(score, 0.0, 1.0)

