import argparse
import datetime as dt
import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_npz_scorebase_monthly_v10.csv"
OUT_SUMMARY = "reports/search_npz_scorebase_monthly_v10.txt"

RISK_ANCHOR = {
    "sell_rule": "loss_rebound_t3",
    "top_k": 3,
    "max_board": None,
    "max_limit_down": 10,
}

HP_CANDIDATES = [
    {"hidden": 88, "epochs": 500, "lr": 0.0210, "weight_decay": 0.00020, "target_gt": 0.0100, "seed": 440, "tag": "incumbent"},
    {"hidden": 88, "epochs": 520, "lr": 0.0205, "weight_decay": 0.00020, "target_gt": 0.0100, "seed": 440, "tag": "tie_e520"},
    {"hidden": 88, "epochs": 500, "lr": 0.0210, "weight_decay": 0.00016, "target_gt": 0.0100, "seed": 440, "tag": "tie_wd16"},
    {"hidden": 88, "epochs": 500, "lr": 0.0215, "weight_decay": 0.00022, "target_gt": 0.0100, "seed": 440, "tag": "tie_lr215"},
    {"hidden": 88, "epochs": 480, "lr": 0.0220, "weight_decay": 0.00020, "target_gt": 0.0100, "seed": 440, "tag": "local_alt"},
    {"hidden": 88, "epochs": 540, "lr": 0.0200, "weight_decay": 0.00018, "target_gt": 0.0100, "seed": 440, "tag": "local_alt"},
]

SCORE_BASE_STARTS = [
    "2024-12-01",
    "2025-01-01",
    "2025-02-01",
    "2025-03-01",
    "2025-04-01",
    "2025-05-01",
    "2025-06-01",
    "2025-07-01",
    "2025-08-01",
    "2025-09-01",
]

STAGE1_CFGS = [
    {"alpha": 15, "threshold": 0.62, "alloc": 0.9998, "gap_min": -0.0010, "gap_max": 0.0255, "max_broken_rate": 0.412, "min_red_rate": 0.287, "max_pullback": 0.065},
    {"alpha": 15, "threshold": 0.64, "alloc": 0.9998, "gap_min": -0.0005, "gap_max": 0.0265, "max_broken_rate": 0.412, "min_red_rate": 0.287, "max_pullback": 0.065},
    {"alpha": 15, "threshold": 0.68, "alloc": 0.9998, "gap_min": -0.0005, "gap_max": 0.0255, "max_broken_rate": 0.406, "min_red_rate": 0.287, "max_pullback": 0.060},
]

STAGE2_ALPHA = [14, 15, 16, 17, 18]
STAGE2_THRESHOLD = [0.62, 0.64, 0.66, 0.68, 0.70]
STAGE2_ALLOC = [0.9990, 0.9998, 1.0000]
STAGE2_GAPS = [(-0.0010, 0.0255), (-0.0005, 0.0255), (-0.0005, 0.0265)]
STAGE2_MAX_BROKEN = [0.406, 0.412]
STAGE2_MIN_RED = [0.275, 0.287]
STAGE2_PULLBACK = [0.060, 0.065]


def objective(ret_pct: float, dd_pct: float) -> float:
    return ret_pct - 120.0 * max(0.0, dd_pct - DD_LIMIT)


def hp_key(hp: Dict[str, float]) -> Tuple[float, ...]:
    return (
        int(hp["hidden"]),
        int(hp["epochs"]),
        round(float(hp["lr"]), 6),
        round(float(hp["weight_decay"]), 8),
        round(float(hp["target_gt"]), 6),
        int(hp["seed"]),
    )


def score_from_ref(raw_prob: np.ndarray, ref: np.ndarray, alpha: int) -> np.ndarray:
    arr = np.asarray(raw_prob, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    cdf = np.searchsorted(ref_arr, arr, side="right").astype(float) / float(ref_arr.size)
    score = 1.0 - np.power(1.0 - np.clip(cdf, 0.0, 1.0), int(alpha))
    return np.clip(score, 0.0, 1.0)


def prepare_data() -> Dict[str, object]:
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
    x_train, x_all, _, _ = relay_model.fill_and_scale(x_train_raw, x_all_raw)

    return {
        "rules": rules,
        "trade_dates": trade_dates,
        "base": base,
        "train_df": train_df,
        "full_pool": full_pool,
        "x_train": x_train,
        "x_all": x_all,
    }


def train_prob(data: Dict[str, object], hp: Dict[str, float]) -> pd.Series:
    train_df: pd.DataFrame = data["train_df"]
    x_train: np.ndarray = data["x_train"]
    x_all: np.ndarray = data["x_all"]
    full_pool: pd.DataFrame = data["full_pool"]

    y = (train_df["t1_hold_ret"] > float(hp["target_gt"])).astype(np.int8).to_numpy(np.int8)
    w1, b1, w2, b2 = relay_model.train_binary_mlp(
        x=x_train,
        y=y,
        hidden_size=int(hp["hidden"]),
        epochs=int(hp["epochs"]),
        learning_rate=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        seed=int(hp["seed"]),
    )
    _, all_prob = relay_model.predict_binary_mlp(x_all, w1, b1, w2, b2)
    return pd.Series(all_prob, index=full_pool.index)


def simulate(
    data: Dict[str, object],
    ranked_by_date: Dict[str, List[Dict[str, object]]],
    cfg: Dict[str, float],
) -> Tuple[float, float, int]:
    m = mb._simulate_fast(
        trade_dates=data["trade_dates"],
        ranked_by_date=ranked_by_date,
        threshold=float(cfg["threshold"]),
        initial_capital=10000.0,
        sell_rule=RISK_ANCHOR["sell_rule"],
        top_k=int(RISK_ANCHOR["top_k"]),
        max_board=RISK_ANCHOR["max_board"],
        gap_min=float(cfg["gap_min"]),
        gap_max=float(cfg["gap_max"]),
        max_broken_rate=float(cfg["max_broken_rate"]),
        min_red_rate=float(cfg["min_red_rate"]),
        max_limit_down=int(RISK_ANCHOR["max_limit_down"]),
        max_pullback=float(cfg["max_pullback"]),
        alloc_pct=float(cfg["alloc"]),
        rules=data["rules"],
    )
    return float(m["ret"]) * 100.0, float(m["max_dd"]) * 100.0, int(m["trades"])


def main() -> None:
    data = prepare_data()
    base: pd.DataFrame = data["base"]
    full_pool: pd.DataFrame = data["full_pool"]

    rows: List[Dict[str, object]] = []
    best: Dict[str, object] = {}
    best_ret = float("-inf")

    prob_cache: Dict[Tuple[float, ...], pd.Series] = {}
    stage1_rank: List[Tuple[float, float, Dict[str, float], str]] = []

    total_models = len(HP_CANDIDATES)
    for i, hp in enumerate(HP_CANDIDATES, 1):
        prob_full = train_prob(data, hp)
        prob_cache[hp_key(hp)] = prob_full
        base_prob = prob_full.reindex(base.index).to_numpy(dtype=float)

        for s_idx, score_base_start in enumerate(SCORE_BASE_STARTS, 1):
            ref = prob_full.reindex(full_pool[full_pool["date"] >= score_base_start].index).to_numpy(dtype=float)
            ref = ref[np.isfinite(ref)]
            if ref.size < 100:
                continue
            ref.sort()

            alpha_ranked = {}
            local_best_obj = float("-inf")
            local_best_ret = float("-inf")

            for cfg in STAGE1_CFGS:
                alpha = int(cfg["alpha"])
                if alpha not in alpha_ranked:
                    score = score_from_ref(base_prob, ref, alpha)
                    scored = base.copy()
                    scored["raw_prob"] = base_prob
                    scored["score"] = score
                    alpha_ranked[alpha] = mb._build_ranked_by_date(scored)

                ret_pct, dd_pct, trades = simulate(data, alpha_ranked[alpha], cfg)
                obj = objective(ret_pct, dd_pct)
                rec = {
                    "stage": "stage1",
                    **hp,
                    "score_base_start": score_base_start,
                    **cfg,
                    "ret_pct": ret_pct,
                    "dd_pct": dd_pct,
                    "trades": trades,
                    "obj": obj,
                }
                rows.append(rec)
                if obj > local_best_obj:
                    local_best_obj = obj
                    local_best_ret = ret_pct
                if dd_pct <= DD_LIMIT and ret_pct > best_ret:
                    best = rec
                    best_ret = ret_pct
                    print(
                        f"[best<=16.13 stage1] model={i}/{total_models} scorebase={s_idx}/{len(SCORE_BASE_STARTS)} "
                        f"ret={ret_pct:.6f}% dd={dd_pct:.6f}% hp={hp} score_base_start={score_base_start}"
                    )

            stage1_rank.append((local_best_obj, local_best_ret, hp, score_base_start))

        print(f"stage1 progress {i}/{total_models}")

    stage1_rank.sort(key=lambda x: (x[0], x[1]), reverse=True)
    parents = stage1_rank[:8]
    print(f"stage2 parents={len(parents)}")

    stage2_cfgs = []
    for alpha, th, alloc, (gmin, gmax), br, rr, pb in itertools.product(
        STAGE2_ALPHA,
        STAGE2_THRESHOLD,
        STAGE2_ALLOC,
        STAGE2_GAPS,
        STAGE2_MAX_BROKEN,
        STAGE2_MIN_RED,
        STAGE2_PULLBACK,
    ):
        if rr > br - 0.004:
            continue
        stage2_cfgs.append(
            {
                "alpha": alpha,
                "threshold": th,
                "alloc": alloc,
                "gap_min": gmin,
                "gap_max": gmax,
                "max_broken_rate": br,
                "min_red_rate": rr,
                "max_pullback": pb,
            }
        )

    for idx, (_, __, hp, score_base_start) in enumerate(parents, 1):
        prob_full = prob_cache[hp_key(hp)]
        base_prob = prob_full.reindex(base.index).to_numpy(dtype=float)
        ref = prob_full.reindex(full_pool[full_pool["date"] >= score_base_start].index).to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        if ref.size < 100:
            continue
        ref.sort()

        alpha_ranked = {}
        for alpha in STAGE2_ALPHA:
            score = score_from_ref(base_prob, ref, alpha)
            scored = base.copy()
            scored["raw_prob"] = base_prob
            scored["score"] = score
            alpha_ranked[alpha] = mb._build_ranked_by_date(scored)

        for cfg in stage2_cfgs:
            ranked = alpha_ranked[int(cfg["alpha"])]
            ret_pct, dd_pct, trades = simulate(data, ranked, cfg)
            obj = objective(ret_pct, dd_pct)
            rec = {
                "stage": "stage2",
                **hp,
                "score_base_start": score_base_start,
                **cfg,
                "ret_pct": ret_pct,
                "dd_pct": dd_pct,
                "trades": trades,
                "obj": obj,
            }
            rows.append(rec)
            if dd_pct <= DD_LIMIT and ret_pct > best_ret:
                best = rec
                best_ret = ret_pct
                print(
                    f"[best<=16.13 stage2] parent={idx}/{len(parents)} "
                    f"ret={ret_pct:.6f}% dd={dd_pct:.6f}% hp={hp} score_base_start={score_base_start} cfg={cfg}"
                )

        print(f"stage2 progress {idx}/{len(parents)}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    feasible = out[out["dd_pct"] <= DD_LIMIT].sort_values("ret_pct", ascending=False)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(f"rows={len(out)}\n")
        f.write(f"feasible={len(feasible)}\n")
        f.write(f"best={best}\n")
        f.write(f"csv={OUT_CSV}\n")
        if len(feasible):
            f.write("\n[top20_feasible]\n")
            f.write(feasible.head(20).to_string(index=False))
            f.write("\n")

    print(f"saved={OUT_CSV} rows={len(out)} feasible={len(feasible)}")
    print(f"best={best}")
    if len(feasible):
        cols = [
            "stage", "tag", "hidden", "epochs", "lr", "weight_decay", "target_gt", "seed",
            "score_base_start", "alpha", "threshold", "alloc", "gap_min", "gap_max",
            "max_broken_rate", "min_red_rate", "max_pullback",
            "ret_pct", "dd_pct", "trades",
        ]
        print(feasible[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
