import argparse
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_1613_modelseed_adaptive_candidates.csv"
OUT_SUMMARY = "reports/search_1613_modelseed_adaptive.txt"


@dataclass
class Candidate:
    variant: str
    seed: int
    stage1_score: float


def objective(ret_pct: float, dd_pct: float) -> float:
    return ret_pct - 100.0 * max(0.0, dd_pct - DD_LIMIT)


def prepare_data():
    rules = mb._load_execution_rules(RULES_FILE)
    backtest_start = rules.backtest_start_date or "2025-01-01"
    backtest_end = rules.backtest_end_date
    train_start = rules.train_start_date or "2020-01-01"
    train_end = rules.train_end_date or (
        dt.datetime.strptime(backtest_start, "%Y-%m-%d").date() - dt.timedelta(days=1)
    ).strftime("%Y-%m-%d")
    print(f"train={train_start}~{train_end} backtest={backtest_start}~{backtest_end or latest}")

    ns = argparse.Namespace(config="config.ini", db_url=None, db=None)
    engine = relay_model.make_engine(relay_model.resolve_db_target(ns))
    trade_dates = mb._load_trade_dates(engine, backtest_start, backtest_end)
    end_date = trade_dates[-1]

    load_start = (dt.datetime.strptime(train_start, "%Y-%m-%d").date() - dt.timedelta(days=60)).strftime("%Y-%m-%d")
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
        "base": base,
        "full_pool": full_pool,
        "train_df": train_df,
        "x_train": x_train,
        "x_all": x_all,
        "trade_dates": trade_dates,
    }


def train_prob(data, spec, seed):
    train_df = data["train_df"]
    x_train = data["x_train"]
    x_all = data["x_all"]
    full_pool = data["full_pool"]
    base = data["base"]

    y = (train_df["t1_hold_ret"] > float(spec["target_gt"])).astype(np.int8).to_numpy(np.int8)
    w1, b1, w2, b2 = relay_model.train_binary_mlp(
        x=x_train,
        y=y,
        hidden_size=int(spec["hidden"]),
        epochs=int(spec["epochs"]),
        learning_rate=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
        seed=int(seed),
    )
    _, all_prob = relay_model.predict_binary_mlp(x_all, w1, b1, w2, b2)
    prob_full = pd.Series(all_prob, index=full_pool.index)
    base_prob = prob_full.reindex(base.index).to_numpy(dtype=float)
    return base_prob


def simulate(ranked_by_date, data, cfg):
    m = mb._simulate_fast(
        trade_dates=data["trade_dates"],
        ranked_by_date=ranked_by_date,
        threshold=float(cfg["threshold"]),
        initial_capital=10000.0,
        sell_rule=str(cfg["sell_rule"]),
        top_k=int(cfg["top_k"]),
        max_board=cfg["max_board"],
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


def main():
    data = prepare_data()
    base = data["base"]

    specs = mb.list_variant_specs("aggressive")
    offsets = [260, 300, 340, 380, 388, 420, 460, 500, 540, 580]

    stage1_rows = []
    rows = []
    best = None

    anchor_risk = {
        "sell_rule": "loss_rebound_t3",
        "top_k": 3,
        "max_board": None,
        "gap_min": -0.0010,
        "gap_max": 0.0255,
        "max_broken_rate": 0.412,
        "min_red_rate": 0.287,
        "max_limit_down": 10,
        "max_pullback": 0.065,
        "threshold": 0.6000,
        "alloc": 0.9998,
    }

    print(f"stage1 specs={len(specs)} seeds_per_spec={len(offsets)} total_models={len(specs)*len(offsets)}")
    model_idx = 0
    total_models = len(specs) * len(offsets)

    for spec in specs:
        for off in offsets:
            model_idx += 1
            seed = int(spec["seed"]) + int(off)
            base_prob = train_prob(data, spec, seed)

            alpha_ranked = {}
            for alpha in [14, 15, 16]:
                scored = base.copy()
                scored["raw_prob"] = base_prob
                scored["score"] = mb._score_from_raw_prob(base_prob, alpha=alpha)
                alpha_ranked[alpha] = mb._build_ranked_by_date(scored)

            for alpha in [14, 15, 16]:
                cfg = dict(anchor_risk)
                cfg["alpha"] = alpha
                ret_pct, dd_pct, trades = simulate(alpha_ranked[alpha], data, cfg)
                rec = {
                    "stage": "stage1",
                    "variant": spec["name"],
                    "seed": seed,
                    "alpha": alpha,
                    **cfg,
                    "ret_pct": ret_pct,
                    "dd_pct": dd_pct,
                    "trades": trades,
                    "obj": objective(ret_pct, dd_pct),
                }
                stage1_rows.append(rec)
                rows.append(rec)
                if dd_pct <= DD_LIMIT and ((best is None) or (ret_pct > best["ret_pct"])):
                    best = rec
                    print(f"[stage1 best<=16.13] model={model_idx}/{total_models} ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")

            if model_idx % 5 == 0:
                print(f"stage1 progress {model_idx}/{total_models}")

    stage1_df = pd.DataFrame(stage1_rows)
    stage1_df = stage1_df.sort_values(["obj", "ret_pct"], ascending=False)
    candidates = (
        stage1_df[["variant", "seed", "obj"]]
        .drop_duplicates(subset=["variant", "seed"])
        .head(20)
        .reset_index(drop=True)
    )
    cand_list = [Candidate(variant=str(r["variant"]), seed=int(r["seed"]), stage1_score=float(r["obj"])) for _, r in candidates.iterrows()]
    print(f"stage2 candidates={len(cand_list)}")

    alpha_values = [14, 15, 16]
    topk_values = [2, 3, 4]
    gap_min_values = [-0.0015, -0.0010, -0.0005]
    gap_max_values = [0.0250, 0.0255, 0.0260, 0.0265]
    th_values = [0.56, 0.58, 0.60, 0.62, 0.64, 0.66]
    alloc_values = [0.9980, 0.9998]

    for idx, cand in enumerate(cand_list, 1):
        spec = [s for s in specs if s["name"] == cand.variant][0]
        base_prob = train_prob(data, spec, cand.seed)

        alpha_ranked = {}
        for alpha in alpha_values:
            scored = base.copy()
            scored["raw_prob"] = base_prob
            scored["score"] = mb._score_from_raw_prob(base_prob, alpha=alpha)
            alpha_ranked[alpha] = mb._build_ranked_by_date(scored)

        checked_local = 0
        for alpha in alpha_values:
            ranked = alpha_ranked[alpha]
            for top_k in topk_values:
                for gap_min in gap_min_values:
                    for gap_max in gap_max_values:
                        if gap_max < gap_min + 0.012:
                            continue
                        for th in th_values:
                            for alloc in alloc_values:
                                cfg = {
                                    "sell_rule": "loss_rebound_t3",
                                    "top_k": top_k,
                                    "max_board": None,
                                    "gap_min": gap_min,
                                    "gap_max": gap_max,
                                    "max_broken_rate": 0.412,
                                    "min_red_rate": 0.287,
                                    "max_limit_down": 10,
                                    "max_pullback": 0.065,
                                    "threshold": th,
                                    "alloc": alloc,
                                }
                                ret_pct, dd_pct, trades = simulate(ranked, data, cfg)
                                rec = {
                                    "stage": "stage2",
                                    "variant": cand.variant,
                                    "seed": cand.seed,
                                    "alpha": alpha,
                                    **cfg,
                                    "ret_pct": ret_pct,
                                    "dd_pct": dd_pct,
                                    "trades": trades,
                                    "obj": objective(ret_pct, dd_pct),
                                    "stage1_score": cand.stage1_score,
                                }
                                rows.append(rec)
                                checked_local += 1
                                if dd_pct <= DD_LIMIT and ((best is None) or (ret_pct > best["ret_pct"])):
                                    best = rec
                                    print(f"[stage2 best<=16.13] cand={idx}/{len(cand_list)} checked={checked_local} ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")

        print(f"stage2 progress {idx}/{len(cand_list)} checked_local={checked_local}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(f"checked={len(out)}\\n")
        f.write(f"best={best}\\n")
        f.write(f"csv={OUT_CSV}\\n")

    print(f"checked={len(out)}")
    print(f"best={best}")
    print(f"saved={OUT_CSV}")
    print(f"summary={OUT_SUMMARY}")


if __name__ == "__main__":
    main()
