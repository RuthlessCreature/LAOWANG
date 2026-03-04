import argparse
import datetime as dt
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_npz_adaptive_gen_v12.csv"
OUT_SUMMARY = "reports/search_npz_adaptive_gen_v12.txt"

CURRENT_BEST_RET = 3091.114719633972

SCORE_BASE_STARTS = ["2025-02-01", "2025-03-01", "2025-04-01"]
SELL_RULES = ["loss_rebound_t3", "strong_hold_t3", "t2_close"]


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


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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
        sell_rule=str(cfg["sell_rule"]),
        top_k=int(cfg["top_k"]),
        max_board=None,
        gap_min=float(cfg["gap_min"]),
        gap_max=float(cfg["gap_max"]),
        max_broken_rate=float(cfg["max_broken_rate"]),
        min_red_rate=float(cfg["min_red_rate"]),
        max_limit_down=int(cfg["max_limit_down"]),
        max_pullback=float(cfg["max_pullback"]),
        alloc_pct=float(cfg["alloc"]),
        rules=data["rules"],
    )
    return float(m["ret"]) * 100.0, float(m["max_dd"]) * 100.0, int(m["trades"])


def random_hp(rng: random.Random) -> Dict[str, float]:
    return {
        "hidden": rng.choice([80, 84, 88, 92, 96]),
        "epochs": rng.choice([420, 460, 500, 520, 560]),
        "lr": round(rng.uniform(0.0180, 0.0240), 4),
        "weight_decay": round(rng.uniform(0.00012, 0.00028), 6),
        "target_gt": rng.choice([0.008, 0.009, 0.010, 0.011, 0.012, 0.015]),
        "seed": rng.randint(428, 452),
        "tag": "random_hp",
    }


def mutate_hp(parent: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    hp = dict(parent)
    hp["hidden"] = rng.choice([hp["hidden"], 80, 84, 88, 92, 96])
    hp["epochs"] = int(clip(int(hp["epochs"]) + rng.choice([-80, -40, 0, 40, 80]), 360, 620))
    hp["lr"] = round(clip(float(hp["lr"]) + rng.uniform(-0.0018, 0.0018), 0.0150, 0.0280), 4)
    hp["weight_decay"] = round(clip(float(hp["weight_decay"]) + rng.uniform(-0.00005, 0.00005), 0.00008, 0.00040), 6)
    hp["target_gt"] = rng.choice([hp["target_gt"], 0.008, 0.009, 0.010, 0.011, 0.012, 0.015])
    hp["seed"] = int(clip(int(hp["seed"]) + rng.randint(-6, 6), 380, 520))
    hp["tag"] = "mut_hp"
    return hp


def random_cfg(rng: random.Random) -> Dict[str, float]:
    cfg = {
        "score_base_start": rng.choice(SCORE_BASE_STARTS),
        "alpha": rng.choice([14, 15, 16, 17, 18]),
        "sell_rule": rng.choice(SELL_RULES),
        "top_k": rng.choice([2, 3, 4]),
        "threshold": round(rng.choice([0.62, 0.64, 0.66, 0.68, 0.70]), 4),
        "alloc": round(rng.choice([0.9990, 0.9998, 1.0000]), 4),
        "gap_min": 0.0,
        "gap_max": 0.0,
        "max_broken_rate": round(rng.choice([0.406, 0.412]), 3),
        "min_red_rate": round(rng.choice([0.275, 0.287]), 3),
        "max_limit_down": int(rng.choice([10, 12])),
        "max_pullback": round(rng.choice([0.060, 0.065]), 3),
    }
    cfg["gap_min"], cfg["gap_max"] = rng.choice([(-0.0010, 0.0255), (-0.0005, 0.0255), (-0.0005, 0.0265)])
    if cfg["min_red_rate"] > cfg["max_broken_rate"] - 0.004:
        cfg["min_red_rate"] = round(max(0.240, cfg["max_broken_rate"] - 0.006), 3)
    return cfg


def baseline_cfgs() -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for s in SCORE_BASE_STARTS:
        out.extend(
            [
                {
                    "score_base_start": s,
                    "alpha": 15,
                    "sell_rule": "loss_rebound_t3",
                    "top_k": 3,
                    "threshold": 0.62,
                    "alloc": 0.9998,
                    "gap_min": -0.0010,
                    "gap_max": 0.0255,
                    "max_broken_rate": 0.412,
                    "min_red_rate": 0.287,
                    "max_limit_down": 10,
                    "max_pullback": 0.065,
                },
                {
                    "score_base_start": s,
                    "alpha": 15,
                    "sell_rule": "loss_rebound_t3",
                    "top_k": 3,
                    "threshold": 0.64,
                    "alloc": 0.9998,
                    "gap_min": -0.0005,
                    "gap_max": 0.0265,
                    "max_broken_rate": 0.412,
                    "min_red_rate": 0.287,
                    "max_limit_down": 10,
                    "max_pullback": 0.065,
                },
            ]
        )
    return out


def evaluate_hp(
    data: Dict[str, object],
    base: pd.DataFrame,
    full_pool: pd.DataFrame,
    hp: Dict[str, float],
    cfgs: List[Dict[str, float]],
    rows: List[Dict[str, object]],
    best: Dict[str, object],
    best_ret: float,
    stage: str,
) -> Tuple[float, float, Dict[str, object], float]:
    prob_full = train_prob(data, hp)
    base_prob = prob_full.reindex(base.index).to_numpy(dtype=float)

    # cache ranked by (score_base_start, alpha)
    ranked_cache: Dict[Tuple[str, int], Dict[str, List[Dict[str, object]]]] = {}

    local_best_obj = float("-inf")
    local_best_ret = float("-inf")

    for cfg in cfgs:
        k = (str(cfg["score_base_start"]), int(cfg["alpha"]))
        if k not in ranked_cache:
            ref = prob_full.reindex(full_pool[full_pool["date"] >= str(cfg["score_base_start"])].index).to_numpy(dtype=float)
            ref = ref[np.isfinite(ref)]
            if ref.size < 100:
                continue
            ref.sort()

            score = score_from_ref(base_prob, ref, int(cfg["alpha"]))
            scored = base.copy()
            scored["raw_prob"] = base_prob
            scored["score"] = score
            ranked_cache[k] = mb._build_ranked_by_date(scored)

        ranked = ranked_cache[k]
        ret_pct, dd_pct, trades = simulate(data, ranked, cfg)
        obj = objective(ret_pct, dd_pct)
        rec = {
            "stage": stage,
            **hp,
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
            hp_txt = (
                f"hidden={hp['hidden']},epochs={hp['epochs']},lr={hp['lr']},"
                f"wd={hp['weight_decay']},gt={hp['target_gt']},seed={hp['seed']}"
            )
            cfg_txt = (
                f"s={cfg['score_base_start']},a={cfg['alpha']},rule={cfg['sell_rule']},"
                f"k={cfg['top_k']},th={cfg['threshold']}"
            )
            print(
                f"[best<=16.13 {stage}] ret={ret_pct:.6f}% dd={dd_pct:.6f}% "
                f"hp={hp_txt} cfg={cfg_txt}"
            )

    return local_best_obj, local_best_ret, best, best_ret


def main() -> None:
    rng = random.Random(202603040137)

    data = prepare_data()
    base: pd.DataFrame = data["base"]
    full_pool: pd.DataFrame = data["full_pool"]

    rows: List[Dict[str, object]] = []
    best: Dict[str, object] = {}
    best_ret = float("-inf")

    initial_hps: List[Dict[str, float]] = [
        {"hidden": 88, "epochs": 500, "lr": 0.0210, "weight_decay": 0.00020, "target_gt": 0.0100, "seed": 440, "tag": "incumbent"},
        {"hidden": 88, "epochs": 520, "lr": 0.0205, "weight_decay": 0.00020, "target_gt": 0.0100, "seed": 440, "tag": "tie_e520"},
        {"hidden": 88, "epochs": 500, "lr": 0.0210, "weight_decay": 0.00016, "target_gt": 0.0100, "seed": 440, "tag": "tie_wd16"},
        {"hidden": 88, "epochs": 500, "lr": 0.0215, "weight_decay": 0.00022, "target_gt": 0.0100, "seed": 440, "tag": "tie_lr215"},
    ]
    for _ in range(12):
        initial_hps.append(random_hp(rng))

    uniq = {}
    for hp in initial_hps:
        uniq[hp_key(hp)] = hp
    initial_hps = list(uniq.values())

    print(f"gen1 hp_count={len(initial_hps)}")

    base_cfgs = baseline_cfgs()
    stage_rank: List[Tuple[float, float, Dict[str, float]]] = []

    for i, hp in enumerate(initial_hps, 1):
        cfgs = list(base_cfgs)
        # add a few random strategy configs per hp in gen1
        for _ in range(12):
            cfgs.append(random_cfg(rng))

        local_obj, local_ret, best, best_ret = evaluate_hp(
            data=data,
            base=base,
            full_pool=full_pool,
            hp=hp,
            cfgs=cfgs,
            rows=rows,
            best=best,
            best_ret=best_ret,
            stage="gen1",
        )
        stage_rank.append((local_obj, local_ret, hp))
        print(f"gen1 progress {i}/{len(initial_hps)} obj={local_obj:.4f} ret={local_ret:.4f}")

    stage_rank.sort(key=lambda x: (x[0], x[1]), reverse=True)
    parents = [x[2] for x in stage_rank[:6]]

    gen2_hps: List[Dict[str, float]] = []
    for p in parents:
        gen2_hps.append(dict(p, tag="parent"))
        for _ in range(2):
            gen2_hps.append(mutate_hp(p, rng))

    uniq2 = {}
    for hp in gen2_hps:
        uniq2[hp_key(hp)] = hp
    gen2_hps = list(uniq2.values())
    print(f"gen2 hp_count={len(gen2_hps)}")

    for i, hp in enumerate(gen2_hps, 1):
        cfgs = []
        # stronger random search for each hp in gen2
        for _ in range(40):
            cfgs.append(random_cfg(rng))
        # add anchors to keep comparability
        cfgs.extend(base_cfgs[:4])

        local_obj, local_ret, best, best_ret = evaluate_hp(
            data=data,
            base=base,
            full_pool=full_pool,
            hp=hp,
            cfgs=cfgs,
            rows=rows,
            best=best,
            best_ret=best_ret,
            stage="gen2",
        )
        print(f"gen2 progress {i}/{len(gen2_hps)} obj={local_obj:.4f} ret={local_ret:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    feasible = out[out["dd_pct"] <= DD_LIMIT].sort_values("ret_pct", ascending=False)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(f"rows={len(out)}\n")
        f.write(f"feasible={len(feasible)}\n")
        f.write(f"best={best}\n")
        f.write(f"current_best_ret={CURRENT_BEST_RET:.6f}\n")
        if best:
            f.write(f"delta_vs_current_best={float(best.get('ret_pct', float('-inf'))) - CURRENT_BEST_RET:.6f}\n")
        f.write(f"csv={OUT_CSV}\n")
        if len(feasible):
            f.write("\n[top20_feasible]\n")
            f.write(feasible.head(20).to_string(index=False))
            f.write("\n")

    print(f"saved={OUT_CSV} rows={len(out)} feasible={len(feasible)}")
    print(f"best={best}")
    if best:
        print(f"delta_vs_current_best={float(best.get('ret_pct', float('-inf'))) - CURRENT_BEST_RET:.6f}")
    if len(feasible):
        cols = [
            "stage", "tag", "hidden", "epochs", "lr", "weight_decay", "target_gt", "seed",
            "score_base_start", "alpha", "sell_rule", "top_k", "threshold", "alloc", "gap_min", "gap_max",
            "max_broken_rate", "min_red_rate", "max_limit_down", "max_pullback",
            "ret_pct", "dd_pct", "trades",
        ]
        print(feasible[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
