#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit relay model and export as one swappable model file."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy import text

import mock_backtest as backtest_model
import relay_model_file
import relay_strategy_model as relay_model


def _build_variant_specs() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for profile in ("base", "expanded", "aggressive"):
        for spec in backtest_model.list_variant_specs(profile):
            name = str(spec.get("name") or "").strip()
            if not name:
                continue
            out[name] = {
                "hidden": float(spec["hidden"]),
                "epochs": float(spec["epochs"]),
                "lr": float(spec["lr"]),
                "weight_decay": float(spec["weight_decay"]),
                "target_gt": float(spec["target_gt"]),
                "seed": float(spec["seed"]),
            }
    return out


VARIANT_SPECS: Dict[str, Dict[str, float]] = _build_variant_specs()

RISK_PROFILES: Dict[str, Sequence[Optional[float]]] = {
    "off": (None, None, None, None),
    "balanced": (0.45, 0.24, 20, 0.09),
    "strict": (0.40, 0.28, 15, 0.08),
}


def _normalize_iso_date(value: str) -> str:
    s = str(value or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")


def _safe_opt_float(v: Any) -> Optional[float]:
    s = str(v).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    try:
        x = float(s)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None


def _safe_opt_int(v: Any) -> Optional[int]:
    s = str(v).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _parse_cfg(cfg_text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "variant": "mlp_h56_e320_gt1",
        "alpha": 10,
        "top_k": 3,
        "max_board_filter": 3,
        "gap_min": None,
        "gap_max": None,
        "risk_profile": "off",
        "default_threshold": 0.75,
    }
    parts = [p.strip() for p in str(cfg_text or "").split("/") if str(p).strip()]
    if parts:
        out["variant"] = parts[0]
    for p in parts[1:]:
        if p.startswith("a"):
            try:
                out["alpha"] = int(p[1:])
            except Exception:
                pass
            continue
        if p.startswith("k"):
            try:
                out["top_k"] = max(1, int(p[1:]))
            except Exception:
                pass
            continue
        if p.startswith("gap[") and p.endswith("]"):
            body = p[4:-1]
            left, right = (body.split(",", 1) + [""])[:2]
            out["gap_min"] = _safe_opt_float(left)
            out["gap_max"] = _safe_opt_float(right)
            continue
        if p.startswith("board<="):
            out["max_board_filter"] = _safe_opt_int(p.split("<=", 1)[1])
            continue
        if p.startswith("risk="):
            out["risk_profile"] = p.split("=", 1)[1].strip() or "off"
            continue
    risk = RISK_PROFILES.get(str(out["risk_profile"]), RISK_PROFILES["off"])
    out["max_broken_rate_filter"] = risk[0]
    out["min_red_rate_filter"] = risk[1]
    out["max_limit_down_filter"] = _safe_opt_int(risk[2]) if risk[2] is not None else None
    out["max_pullback_filter"] = risk[3]
    return out


def _choose_best_threshold_and_cfg(meta: Dict[str, Any]) -> tuple[float, str]:
    metrics = meta.get("metrics_by_threshold") if isinstance(meta.get("metrics_by_threshold"), dict) else {}
    if not metrics:
        return 0.75, "mlp_h56_e320_gt1/a10/strong_hold_t3/k3/gap[None,None]/board<=3/risk=off/alloc=1.00"
    best_th = 0.75
    best_ret = float("-inf")
    best_cfg = ""
    for k, v in metrics.items():
        try:
            th = float(k)
            ret = float((v or {}).get("ret_pct", float("-inf")))
        except Exception:
            continue
        if ret > best_ret:
            best_ret = ret
            best_th = th
            best_cfg = str((v or {}).get("cfg") or "")
    if not best_cfg:
        best_cfg = "mlp_h56_e320_gt1/a10/strong_hold_t3/k3/gap[None,None]/board<=3/risk=off/alloc=1.00"
    return float(max(0.50, min(0.99, best_th))), best_cfg


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit relay model and export one-file model.")
    p.add_argument("--config", default="config.ini")
    p.add_argument("--db-url", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--train-start-date", default="2020-01-01")
    p.add_argument("--score-base-start-date", default="2025-01-01")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD / YYYYMMDD, default max(stock_daily.date)")
    p.add_argument("--best-meta-path", default="reports/iter_rounds/best/best_model.json")
    p.add_argument("--cfg", default=None, help="Override cfg string like mlp_h56_e320_gt1/a10/.../risk=balanced")
    p.add_argument("--variant", default=None, help="Override variant name")
    p.add_argument("--alpha", type=int, default=None, help="Override score alpha")
    p.add_argument("--default-threshold", type=float, default=None, help="Override default threshold")
    p.add_argument("--seed-offset", type=int, default=None, help="Override seed offset")
    p.add_argument("--model-version", default=None)
    p.add_argument("--model-file", default=str(relay_model_file.default_model_path()))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    db_target = relay_model.resolve_db_target(args)
    engine = relay_model.make_engine(db_target)

    with engine.connect() as conn:
        row = conn.execute(text("SELECT MAX(date) FROM stock_daily")).fetchone()
    end_date = _normalize_iso_date(args.end_date) if args.end_date else (str(row[0]) if row and row[0] else None)
    if not end_date:
        raise SystemExit("stock_daily is empty")

    train_start = _normalize_iso_date(args.train_start_date)
    score_base_start = _normalize_iso_date(args.score_base_start_date)
    if train_start > end_date:
        raise SystemExit("train-start-date must be <= end-date")

    best_meta: Dict[str, Any] = {}
    best_meta_path = Path(args.best_meta_path)
    if best_meta_path.exists():
        try:
            best_meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            best_meta = {}
    default_th, cfg_text = _choose_best_threshold_and_cfg(best_meta)
    if args.cfg:
        cfg_text = str(args.cfg)
    cfg = _parse_cfg(cfg_text)

    if args.variant:
        cfg["variant"] = str(args.variant).strip()
    if args.alpha is not None:
        cfg["alpha"] = max(1, int(args.alpha))
    if args.default_threshold is not None:
        cfg["default_threshold"] = relay_model_file.clamp_threshold(args.default_threshold, default_th)
    else:
        cfg["default_threshold"] = relay_model_file.clamp_threshold(default_th, default_th)

    variant_name = str(cfg.get("variant") or "")
    spec = VARIANT_SPECS.get(variant_name)
    if not spec:
        raise SystemExit(f"unknown variant: {variant_name}")

    if args.seed_offset is not None:
        seed_offset = int(args.seed_offset)
    else:
        try:
            seed_offset = int(float(best_meta.get("seed_offset", 0) or 0))
        except Exception:
            seed_offset = 0

    load_start = (dt.datetime.strptime(train_start, "%Y-%m-%d").date() - dt.timedelta(days=60)).strftime("%Y-%m-%d")
    base = relay_model.load_base_frame(engine, load_start, end_date)
    if base.empty:
        raise SystemExit("no data in stock_daily for fit range")
    full = relay_model.compute_flags_and_stock_features(base)
    market = relay_model.compute_market_features(full)
    full_pool, _, _ = relay_model.build_sample_frames(
        full_df=full,
        market_df=market,
        start_iso=train_start,
        end_iso=end_date,
        tp=0.02,
        sl=-0.01,
    )
    if full_pool.empty:
        raise SystemExit("relay sample pool is empty")

    full_pool = full_pool.copy()
    full_pool["date"] = full_pool["date"].astype(str)
    full_pool["stock_code"] = full_pool["stock_code"].astype(str)
    full_pool = backtest_model._attach_future_fields(engine, full_pool, train_start)

    train_df = full_pool.copy()
    train_df = train_df[(train_df["next_open"].notna()) & (train_df["next2_close"].notna())].copy()
    train_df["next_open"] = np.asarray(train_df["next_open"], dtype=float)
    train_df["next2_close"] = np.asarray(train_df["next2_close"], dtype=float)
    train_df = train_df[(train_df["next_open"] > 0) & (train_df["next2_close"] > 0)].copy()
    train_df["t1_hold_ret"] = train_df["next2_close"] / train_df["next_open"] - 1.0
    if train_df.empty:
        raise SystemExit("no train samples after target construction")

    feature_cols = backtest_model._feature_cols()
    x_train_raw = train_df[feature_cols].to_numpy(dtype=float)
    x_train, _, means, std = relay_model.fill_and_scale(x_train_raw, x_train_raw)

    y = (train_df["t1_hold_ret"] > float(spec["target_gt"])).astype(np.int8).to_numpy(np.int8)
    w1, b1, w2, b2 = relay_model.train_binary_mlp(
        x=x_train,
        y=y,
        hidden_size=int(spec["hidden"]),
        epochs=int(spec["epochs"]),
        learning_rate=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
        seed=int(spec["seed"]) + int(seed_offset),
    )
    train_pred, train_prob = relay_model.predict_binary_mlp(x_train, w1, b1, w2, b2)
    train_acc = float((train_pred == y).mean()) if len(y) else 0.0

    score_base_pool = full_pool[full_pool["date"] >= score_base_start].copy()
    if score_base_pool.empty:
        score_base_pool = full_pool.copy()
    x_base_raw = score_base_pool[feature_cols].to_numpy(dtype=float)
    x_base = x_base_raw.copy()
    for i in range(x_base.shape[1]):
        col = x_base[:, i]
        col[~np.isfinite(col)] = means[i]
    x_base = (x_base - means) / std
    _, base_prob = relay_model.predict_binary_mlp(x_base, w1, b1, w2, b2)
    cdf_ref = np.sort(base_prob[np.isfinite(base_prob)])

    now_text = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_version = str(args.model_version or f"{variant_name}-{now_text.replace(' ', 'T').replace(':', '')}")
    meta = {
        "model_version": model_version,
        "variant": variant_name,
        "alpha": int(cfg.get("alpha") or 10),
        "default_threshold": float(cfg.get("default_threshold") or 0.75),
        "top_k": int(cfg.get("top_k") or 3),
        "max_board_filter": cfg.get("max_board_filter"),
        "gap_min": cfg.get("gap_min"),
        "gap_max": cfg.get("gap_max"),
        "risk_profile": str(cfg.get("risk_profile") or "off"),
        "max_broken_rate_filter": cfg.get("max_broken_rate_filter"),
        "min_red_rate_filter": cfg.get("min_red_rate_filter"),
        "max_limit_down_filter": cfg.get("max_limit_down_filter"),
        "max_pullback_filter": cfg.get("max_pullback_filter"),
        "train_start": train_start,
        "end_date": end_date,
        "score_base_start": score_base_start,
        "train_samples": int(len(train_df)),
        "train_acc": float(train_acc),
        "seed_offset": int(seed_offset),
        "cfg_text": cfg_text,
        "created_at": now_text,
    }

    model_path = Path(args.model_file).expanduser()
    relay_model_file.save_model(
        model_path,
        feature_cols=feature_cols,
        means=means,
        std=std,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        cdf_ref=cdf_ref,
        meta=meta,
    )

    print(
        "[fit-relay-model-file] "
        f"model={model_path.as_posix()} version={model_version} "
        f"samples={len(train_df)} train_acc={train_acc*100:.2f}% "
        f"variant={variant_name} alpha={meta['alpha']} threshold={meta['default_threshold']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
