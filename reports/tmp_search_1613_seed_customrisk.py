import argparse
import datetime as dt

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_1613_seed_customrisk_candidates.csv"


def objective(ret_pct: float, dd_pct: float) -> float:
    return ret_pct - 60.0 * max(0.0, dd_pct - DD_LIMIT)


def prepare_data():
    rules = mb._load_execution_rules(RULES_FILE)
    backtest_start = rules.backtest_start_date or "2025-01-01"
    backtest_end = rules.backtest_end_date
    train_start = rules.train_start_date or "2020-01-01"
    train_end = rules.train_end_date or (
        dt.datetime.strptime(backtest_start, "%Y-%m-%d").date() - dt.timedelta(days=1)
    ).strftime("%Y-%m-%d")
    print(f"train={train_start}~{train_end} backtest={backtest_start}~{backtest_end or 'latest'}")

    ns = argparse.Namespace(config="config.ini", db_url=None, db=None)
    engine = relay_model.make_engine(relay_model.resolve_db_target(ns))
    trade_dates = mb._load_trade_dates(engine, backtest_start, backtest_end)
    end_date = trade_dates[-1]

    load_start = (dt.datetime.strptime(train_start, "%Y-%m-%d").date() - dt.timedelta(days=60)).strftime(
        "%Y-%m-%d"
    )
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

    spec = [s for s in mb.list_variant_specs("aggressive") if s["name"] == "mlp_h88_e500_gt1_wd2e4"][0]
    y = (train_df["t1_hold_ret"] > float(spec["target_gt"])).astype(np.int8).to_numpy(np.int8)
    return rules, trade_dates, full_pool, base, x_train, x_all, y, spec


def train_ranked_maps(spec, seed, x_train, x_all, y, full_pool, base):
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

    ranked_map = {}
    for alpha in [14, 15, 16]:
        scored = base.copy()
        scored["raw_prob"] = base_prob
        scored["score"] = mb._score_from_raw_prob(base_prob, alpha=alpha)
        ranked_map[alpha] = mb._build_ranked_by_date(scored)
    return ranked_map


def simulate_one(rules, trade_dates, ranked_by_date, *, alpha, top_k, gap_min, gap_max, max_board, max_broken_rate, min_red_rate, max_limit_down, max_pullback, threshold, alloc, sell_rule="t2_close"):
    m = mb._simulate_fast(
        trade_dates=trade_dates,
        ranked_by_date=ranked_by_date,
        threshold=float(threshold),
        initial_capital=10000.0,
        sell_rule=sell_rule,
        top_k=int(top_k),
        max_board=max_board,
        gap_min=gap_min,
        gap_max=gap_max,
        max_broken_rate=max_broken_rate,
        min_red_rate=min_red_rate,
        max_limit_down=max_limit_down,
        max_pullback=max_pullback,
        alloc_pct=float(alloc),
        rules=rules,
    )
    ret_pct = float(m["ret"]) * 100.0
    dd_pct = float(m["max_dd"]) * 100.0
    return ret_pct, dd_pct, int(m["trades"])


def main():
    rules, trade_dates, full_pool, base, x_train, x_all, y, spec = prepare_data()

    base_seed = int(spec["seed"])
    seed_offsets = list(range(328, 451, 2))  # 62 seeds, includes 440
    print(f"seed_offsets={seed_offsets[0]}..{seed_offsets[-1]} n={len(seed_offsets)}")

    # Baseline family from current best.
    baseline = dict(
        alpha=15,
        top_k=3,
        gap_min=-0.005,
        gap_max=0.025,
        max_board=None,
        max_broken_rate=0.398,
        min_red_rate=0.250,
        max_limit_down=22,
        max_pullback=0.096,
        threshold=0.5,
        alloc=0.96,
        sell_rule="t2_close",
    )

    rows = []
    checked = 0
    best = None
    coarse_seed_rows = []

    # Stage-1: coarse seed ranking by baseline family.
    for off in seed_offsets:
        seed = base_seed + off
        ranked_map = train_ranked_maps(spec, seed, x_train, x_all, y, full_pool, base)
        ret_pct, dd_pct, trades = simulate_one(
            rules,
            trade_dates,
            ranked_map[baseline["alpha"]],
            **baseline,
        )
        checked += 1
        rec = {
            "stage": "seed_coarse",
            "seed": seed,
            "seed_offset": off,
            **baseline,
            "ret_pct": ret_pct,
            "dd_pct": dd_pct,
            "trades": trades,
            "obj": objective(ret_pct, dd_pct),
        }
        rows.append(rec)
        coarse_seed_rows.append(rec)
        if dd_pct <= DD_LIMIT + 1e-9:
            if (best is None) or (ret_pct > best[0]):
                best = (ret_pct, dd_pct, rec)
                print(f"[seed coarse best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% seed={seed}")
        print(f"[seed coarse done] seed={seed} ret={ret_pct:.4f}% dd={dd_pct:.4f}% checked={checked}")

    coarse_seed_df = pd.DataFrame(coarse_seed_rows).sort_values(["obj", "ret_pct"], ascending=False)
    top_seed_rows = coarse_seed_df.head(12).copy()
    top_seeds = [int(s) for s in top_seed_rows["seed"].tolist()]
    print(f"top_seeds={top_seeds}")

    # Stage-2: risk family search for top seeds at fixed threshold/alloc.
    risk_families = []
    for alpha in [14, 15, 16]:
        for br in [0.394, 0.398, 0.402, 0.406]:
            for rr in [0.244, 0.247, 0.250, 0.253]:
                if rr > br - 0.01:
                    continue
                for ld in [20, 22, 24]:
                    for pb in [0.093, 0.096, 0.099, 0.102]:
                        for gap_min, gap_max in [(-0.005, 0.025), (-0.005, 0.03)]:
                            for top_k in [3, 4]:
                                risk_families.append(
                                    dict(
                                        alpha=alpha,
                                        top_k=top_k,
                                        gap_min=gap_min,
                                        gap_max=gap_max,
                                        max_board=None,
                                        max_broken_rate=br,
                                        min_red_rate=rr,
                                        max_limit_down=ld,
                                        max_pullback=pb,
                                        threshold=0.5,
                                        alloc=0.96,
                                        sell_rule="t2_close",
                                    )
                                )
    print(f"risk_families={len(risk_families)}")

    family_rows = []
    seed_ranked_cache = {}
    for seed in top_seeds:
        seed_ranked_cache[seed] = train_ranked_maps(spec, seed, x_train, x_all, y, full_pool, base)
        for fam in risk_families:
            ret_pct, dd_pct, trades = simulate_one(
                rules,
                trade_dates,
                seed_ranked_cache[seed][fam["alpha"]],
                **fam,
            )
            checked += 1
            rec = {
                "stage": "family",
                "seed": seed,
                "seed_offset": seed - base_seed,
                **fam,
                "ret_pct": ret_pct,
                "dd_pct": dd_pct,
                "trades": trades,
                "obj": objective(ret_pct, dd_pct),
            }
            rows.append(rec)
            family_rows.append(rec)
            if dd_pct <= DD_LIMIT + 1e-9:
                if (best is None) or (ret_pct > best[0]):
                    best = (ret_pct, dd_pct, rec)
                    print(f"[family best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
        print(f"[family done] seed={seed} checked={checked}")

    family_df = pd.DataFrame(family_rows).sort_values(["obj", "ret_pct"], ascending=False)
    focus = family_df.head(40).copy()
    print(f"family_focus={len(focus)}")

    # Stage-3: threshold/alloc refinement around top families.
    th_grid = [round(x, 4) for x in np.arange(0.46, 0.561, 0.002)]
    alloc_grid = [round(x, 4) for x in np.arange(0.94, 0.981, 0.001)]

    for i, row in focus.reset_index(drop=True).iterrows():
        seed = int(row["seed"])
        alpha = int(row["alpha"])
        ranked = seed_ranked_cache[seed][alpha]
        for th in th_grid:
            for alloc in alloc_grid:
                ret_pct, dd_pct, trades = simulate_one(
                    rules,
                    trade_dates,
                    ranked,
                    alpha=alpha,
                    top_k=int(row["top_k"]),
                    gap_min=float(row["gap_min"]),
                    gap_max=float(row["gap_max"]),
                    max_board=None,
                    max_broken_rate=float(row["max_broken_rate"]),
                    min_red_rate=float(row["min_red_rate"]),
                    max_limit_down=int(row["max_limit_down"]),
                    max_pullback=float(row["max_pullback"]),
                    threshold=float(th),
                    alloc=float(alloc),
                    sell_rule="t2_close",
                )
                checked += 1
                rec = {
                    "stage": "refine",
                    "seed": seed,
                    "seed_offset": seed - base_seed,
                    "alpha": alpha,
                    "sell_rule": "t2_close",
                    "top_k": int(row["top_k"]),
                    "gap_min": float(row["gap_min"]),
                    "gap_max": float(row["gap_max"]),
                    "max_board": None,
                    "max_broken_rate": float(row["max_broken_rate"]),
                    "min_red_rate": float(row["min_red_rate"]),
                    "max_limit_down": int(row["max_limit_down"]),
                    "max_pullback": float(row["max_pullback"]),
                    "threshold": float(th),
                    "alloc": float(alloc),
                    "ret_pct": ret_pct,
                    "dd_pct": dd_pct,
                    "trades": trades,
                    "obj": objective(ret_pct, dd_pct),
                }
                rows.append(rec)
                if dd_pct <= DD_LIMIT + 1e-9:
                    if (best is None) or (ret_pct > best[0]):
                        best = (ret_pct, dd_pct, rec)
                        print(f"[refine best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
        print(f"[refine done] {i+1}/{len(focus)} checked={checked}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"checked={checked}")
    print(f"best={best}")
    print(f"saved={OUT_CSV}")


if __name__ == "__main__":
    main()
