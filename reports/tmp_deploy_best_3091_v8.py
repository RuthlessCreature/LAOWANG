import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_model_file
import relay_strategy_model as relay_model

RULES_FILE = "rules_1613.txt"
MODEL_PATH = Path("models/relay_model_active.npz")
MODEL_VERSION = "relay_best_3091_20260303"

BEST = {
    "variant": "mlp_h88_e500_gt1_wd2e4",
    "hidden": 88,
    "epochs": 500,
    "lr": 0.0210,
    "weight_decay": 0.00020,
    "target_gt": 0.0100,
    "seed": 440,
    "score_base_start": "2025-03-01",
    "alpha": 15,
    "threshold": 0.62,
    "alloc": 0.9998,
    "gap_min": -0.0010,
    "gap_max": 0.0255,
    "sell_rule": "loss_rebound_t3",
    "top_k": 3,
    "max_board": None,
    "max_broken_rate": 0.412,
    "min_red_rate": 0.287,
    "max_limit_down": 10,
    "max_pullback": 0.065,
}


def prepare_data():
    rules = mb._load_execution_rules(RULES_FILE)
    backtest_start = rules.backtest_start_date or "2025-01-01"
    backtest_end = rules.backtest_end_date
    train_start = rules.train_start_date or "2020-01-01"
    train_end = rules.train_end_date or (
        dt.datetime.strptime(backtest_start, "%Y-%m-%d").date() - dt.timedelta(days=1)
    ).strftime("%Y-%m-%d")

    ns = argparse.Namespace(config="config.ini", db_url=None, db=None)
    engine = relay_model.make_engine(relay_model.resolve_db_target(ns))
    trade_dates = mb._load_trade_dates(engine, backtest_start, backtest_end)
    end_date = trade_dates[-1]

    load_start = (
        dt.datetime.strptime(train_start, "%Y-%m-%d").date() - dt.timedelta(days=60)
    ).strftime("%Y-%m-%d")
    base_raw = relay_model.load_base_frame(engine, load_start, end_date)
    full = relay_model.compute_flags_and_stock_features(base_raw)
    market = relay_model.compute_market_features(full)
    full_pool, _, _ = relay_model.build_sample_frames(
        full_df=full,
        market_df=market,
        start_iso=train_start,
        end_iso=end_date,
        tp=0.02,
        sl=-0.01,
    )
    full_pool["date"] = full_pool["date"].astype(str)
    full_pool["stock_code"] = full_pool["stock_code"].astype(str)
    full_pool = mb._attach_future_fields(engine, full_pool, train_start)
    full_pool["tomorrow_stage"] = "N/A"
    full_pool["tomorrow_prob"] = np.nan

    base = full_pool[(full_pool["date"] >= backtest_start) & (full_pool["date"] <= end_date)].copy()
    train_df = full_pool[(full_pool["date"] >= train_start) & (full_pool["date"] <= train_end)].copy()
    train_df = train_df[(train_df["next_open"].notna()) & (train_df["next2_close"].notna())].copy()
    train_df["next_open"] = pd.to_numeric(train_df["next_open"], errors="coerce")
    train_df["next2_close"] = pd.to_numeric(train_df["next2_close"], errors="coerce")
    train_df = train_df[(train_df["next_open"] > 0) & (train_df["next2_close"] > 0)].copy()
    train_df["t1_hold_ret"] = train_df["next2_close"] / train_df["next_open"] - 1.0

    feat_cols = mb._feature_cols()
    x_train_raw = train_df[feat_cols].to_numpy(dtype=float)
    x_all_raw = full_pool[feat_cols].to_numpy(dtype=float)
    x_train, x_all, means, std = relay_model.fill_and_scale(x_train_raw, x_all_raw)

    return {
        "rules": rules,
        "trade_dates": trade_dates,
        "backtest_start": backtest_start,
        "end_date": end_date,
        "train_start": train_start,
        "train_end": train_end,
        "feat_cols": feat_cols,
        "base": base,
        "full_pool": full_pool,
        "train_df": train_df,
        "x_train": x_train,
        "x_all": x_all,
        "means": means,
        "std": std,
    }


def score_from_ref(raw_prob: np.ndarray, ref: np.ndarray, alpha: int) -> np.ndarray:
    arr = np.asarray(raw_prob, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    cdf = np.searchsorted(ref_arr, arr, side="right").astype(float) / float(ref_arr.size)
    score = 1.0 - np.power(1.0 - np.clip(cdf, 0.0, 1.0), int(alpha))
    return np.clip(score, 0.0, 1.0)


def main() -> None:
    data = prepare_data()
    train_df = data["train_df"]
    x_train = data["x_train"]
    x_all = data["x_all"]
    full_pool = data["full_pool"]
    base = data["base"]

    y = (train_df["t1_hold_ret"] > float(BEST["target_gt"])) .astype(np.int8).to_numpy(np.int8)
    w1, b1, w2, b2 = relay_model.train_binary_mlp(
        x=x_train,
        y=y,
        hidden_size=int(BEST["hidden"]),
        epochs=int(BEST["epochs"]),
        learning_rate=float(BEST["lr"]),
        weight_decay=float(BEST["weight_decay"]),
        seed=int(BEST["seed"]),
    )
    pred_train, _ = relay_model.predict_binary_mlp(x_train, w1, b1, w2, b2)
    train_acc = float((pred_train == y).mean())

    _, all_prob = relay_model.predict_binary_mlp(x_all, w1, b1, w2, b2)
    prob_full = pd.Series(all_prob, index=full_pool.index)
    base_prob = prob_full.reindex(base.index).to_numpy(dtype=float)

    ref = prob_full.reindex(full_pool[full_pool["date"] >= BEST["score_base_start"]].index).to_numpy(dtype=float)
    ref = ref[np.isfinite(ref)]
    ref.sort()

    score = score_from_ref(base_prob, ref, int(BEST["alpha"]))
    scored = base.copy()
    scored["raw_prob"] = base_prob
    scored["score"] = score
    ranked = mb._build_ranked_by_date(scored)

    fast = mb._simulate_fast(
        trade_dates=data["trade_dates"],
        ranked_by_date=ranked,
        threshold=float(BEST["threshold"]),
        initial_capital=10000.0,
        sell_rule=str(BEST["sell_rule"]),
        top_k=int(BEST["top_k"]),
        max_board=BEST["max_board"],
        gap_min=float(BEST["gap_min"]),
        gap_max=float(BEST["gap_max"]),
        max_broken_rate=float(BEST["max_broken_rate"]),
        min_red_rate=float(BEST["min_red_rate"]),
        max_limit_down=int(BEST["max_limit_down"]),
        max_pullback=float(BEST["max_pullback"]),
        alloc_pct=float(BEST["alloc"]),
        rules=data["rules"],
    )

    detail = mb._simulate_detailed(
        trade_dates=data["trade_dates"],
        ranked_by_date=ranked,
        threshold=float(BEST["threshold"]),
        initial_capital=10000.0,
        sell_rule=str(BEST["sell_rule"]),
        variant_name=str(BEST["variant"]),
        alpha=int(BEST["alpha"]),
        top_k=int(BEST["top_k"]),
        max_board=BEST["max_board"],
        gap_min=float(BEST["gap_min"]),
        gap_max=float(BEST["gap_max"]),
        max_broken_rate=float(BEST["max_broken_rate"]),
        min_red_rate=float(BEST["min_red_rate"]),
        max_limit_down=int(BEST["max_limit_down"]),
        max_pullback=float(BEST["max_pullback"]),
        risk_profile="custom",
        alloc_pct=float(BEST["alloc"]),
        rules=data["rules"],
    )

    ret_pct = float(fast["ret"]) * 100.0
    dd_pct = float(fast["max_dd"]) * 100.0
    trades = int(fast["trades"])

    cfg_text = (
        f"{BEST['variant']}/a{int(BEST['alpha'])}/{BEST['sell_rule']}/k{int(BEST['top_k'])}"
        f"/gap[{float(BEST['gap_min']):.4f},{float(BEST['gap_max']):.4f}]"
        f"/board<=None/risk=custom(br<={float(BEST['max_broken_rate']):.3f},"
        f"rr>={float(BEST['min_red_rate']):.3f},ld<={int(BEST['max_limit_down'])},pb<={float(BEST['max_pullback']):.3f})"
        f"/alloc={float(BEST['alloc']):.4f}"
    )

    meta = {
        "model_version": MODEL_VERSION,
        "variant": BEST["variant"],
        "alpha": int(BEST["alpha"]),
        "default_threshold": float(BEST["threshold"]),
        "top_k": int(BEST["top_k"]),
        "max_board_filter": BEST["max_board"],
        "gap_min": float(BEST["gap_min"]),
        "gap_max": float(BEST["gap_max"]),
        "risk_profile": "custom",
        "max_broken_rate_filter": float(BEST["max_broken_rate"]),
        "min_red_rate_filter": float(BEST["min_red_rate"]),
        "max_limit_down_filter": int(BEST["max_limit_down"]),
        "max_pullback_filter": float(BEST["max_pullback"]),
        "train_start": data["train_start"],
        "end_date": data["end_date"],
        "score_base_start": BEST["score_base_start"],
        "train_samples": int(len(train_df)),
        "train_acc": float(train_acc),
        "seed_offset": int(BEST["seed"] - 52),
        "cfg_text": cfg_text,
        "sell_rule": BEST["sell_rule"],
        "deploy_note": "deployed_from_npz_scorebase_v8_3091",
    }

    relay_model_file.save_model(
        MODEL_PATH,
        feature_cols=data["feat_cols"],
        means=data["means"],
        std=data["std"],
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        cdf_ref=ref,
        meta=meta,
    )

    out_dir = Path("reports/manual_1613_best_v7")
    out_dir.mkdir(parents=True, exist_ok=True)
    ops_csv = out_dir / "mockOperations_best_3091.csv"
    mb._write_operations_csv(ops_csv, detail["trades"], initial_capital=10000.0)

    summary = out_dir / "summary_best_3091.txt"
    summary.write_text(
        "\n".join(
            [
                f"model_version={MODEL_VERSION}",
                f"model_path={MODEL_PATH.as_posix()}",
                f"train_range={data['train_start']}~{data['train_end']}",
                f"backtest_range={data['backtest_start']}~{data['end_date']}",
                f"score_base_start={BEST['score_base_start']}",
                f"ret_pct={ret_pct:.6f}",
                f"dd_pct={dd_pct:.6f}",
                f"trades={trades}",
                f"fast_final_cap={float(fast['final']):.2f}",
                f"detail_final_cap={float(detail['final_capital']):.2f}",
                f"ops_csv={ops_csv.as_posix()}",
                f"cfg={cfg_text}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"model_saved={MODEL_PATH.as_posix()}")
    print(f"model_version={MODEL_VERSION}")
    print(f"score_base_start={BEST['score_base_start']} cdf_ref_size={len(ref)}")
    print(f"ret_pct={ret_pct:.6f} dd_pct={dd_pct:.6f} trades={trades}")
    print(f"ops_csv={ops_csv.as_posix()}")
    print(f"summary={summary.as_posix()}")


if __name__ == "__main__":
    main()
