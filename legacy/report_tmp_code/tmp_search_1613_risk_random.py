import argparse
import datetime as dt
import random

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_1613_risk_random_candidates.csv"
LOG_FILE = "reports/search_1613_risk_random.txt"


def prepare():
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
    seed = int(spec["seed"]) + 388
    y = (train_df["t1_hold_ret"] > float(spec["target_gt"])).astype(np.int8).to_numpy(np.int8)
    w1, b1, w2, b2 = relay_model.train_binary_mlp(
        x=x_train,
        y=y,
        hidden_size=int(spec["hidden"]),
        epochs=int(spec["epochs"]),
        learning_rate=float(spec["lr"]),
        weight_decay=float(spec["weight_decay"]),
        seed=seed,
    )
    _, all_prob = relay_model.predict_binary_mlp(x_all, w1, b1, w2, b2)
    prob_full = pd.Series(all_prob, index=full_pool.index)
    base_prob = prob_full.reindex(base.index).to_numpy(dtype=float)

    ranked_map = {}
    for alpha in [13, 14, 15, 16]:
        scored = base.copy()
        scored["raw_prob"] = base_prob
        scored["score"] = mb._score_from_raw_prob(base_prob, alpha=alpha)
        ranked_map[alpha] = mb._build_ranked_by_date(scored)

    return rules, trade_dates, ranked_map


def sample_configs(n: int, rng: random.Random):
    sell_rules = ["t2_close", "strong_hold_t3", "loss_rebound_t3"]
    top_ks = [2, 3, 4]
    gaps = [(-0.005, 0.025), (-0.005, 0.03), (-0.01, 0.03), (None, None)]
    boards = [None, 2, 3, 4]

    out = []
    base_cfgs = [
        dict(alpha=15, sell_rule="t2_close", top_k=3, gap_min=-0.005, gap_max=0.025, max_board=None, max_broken_rate=0.45, min_red_rate=0.24, max_limit_down=20, max_pullback=0.09),
        dict(alpha=15, sell_rule="t2_close", top_k=3, gap_min=-0.005, gap_max=0.03, max_board=None, max_broken_rate=0.42, min_red_rate=0.22, max_limit_down=20, max_pullback=0.08),
        dict(alpha=13, sell_rule="strong_hold_t3", top_k=2, gap_min=-0.005, gap_max=0.03, max_board=None, max_broken_rate=0.42, min_red_rate=0.22, max_limit_down=20, max_pullback=0.08),
        dict(alpha=16, sell_rule="loss_rebound_t3", top_k=2, gap_min=-0.005, gap_max=0.025, max_board=3, max_broken_rate=0.42, min_red_rate=0.22, max_limit_down=24, max_pullback=0.08),
    ]
    out.extend(base_cfgs)

    while len(out) < n:
        alpha = rng.choice([14, 15, 16])
        sell_rule = rng.choice(sell_rules)
        top_k = rng.choice(top_ks)
        gap_min, gap_max = rng.choice(gaps)
        max_board = rng.choice(boards)

        if rng.random() < 0.12:
            br = None
            rr = None
            ld = None
            pb = None
        else:
            br = round(rng.uniform(0.36, 0.55), 3)
            rr = round(rng.uniform(0.16, 0.35), 3)
            if rr > br - 0.02 and rng.random() < 0.3:
                rr = round(max(0.10, br - rng.uniform(0.03, 0.12)), 3)
            ld = int(rng.randint(8, 30))
            pb = round(rng.uniform(0.06, 0.12), 3)

        out.append(
            dict(
                alpha=alpha,
                sell_rule=sell_rule,
                top_k=top_k,
                gap_min=gap_min,
                gap_max=gap_max,
                max_board=max_board,
                max_broken_rate=br,
                min_red_rate=rr,
                max_limit_down=ld,
                max_pullback=pb,
            )
        )

    df = pd.DataFrame(out)
    dedup_cols = [
        "alpha",
        "sell_rule",
        "top_k",
        "gap_min",
        "gap_max",
        "max_board",
        "max_broken_rate",
        "min_red_rate",
        "max_limit_down",
        "max_pullback",
    ]
    return df.drop_duplicates(subset=dedup_cols).to_dict("records")


def objective(ret_pct: float, dd_pct: float) -> float:
    over = max(0.0, dd_pct - DD_LIMIT)
    return ret_pct - 40.0 * over


def main() -> None:
    rng = random.Random(20260303)
    rules, trade_dates, ranked_map = prepare()
    print("prepared")

    candidates = sample_configs(7000, rng)
    print(f"sampled_cfg={len(candidates)}")

    coarse_rows = []
    checked = 0
    best_feasible = None

    for c in candidates:
        ranked = ranked_map[int(c["alpha"])]
        m = mb._simulate_fast(
            trade_dates=trade_dates,
            ranked_by_date=ranked,
            threshold=0.65,
            initial_capital=10000.0,
            sell_rule=c["sell_rule"],
            top_k=int(c["top_k"]),
            max_board=(None if pd.isna(c["max_board"]) else int(c["max_board"])),
            gap_min=(None if pd.isna(c["gap_min"]) else float(c["gap_min"])),
            gap_max=(None if pd.isna(c["gap_max"]) else float(c["gap_max"])),
            max_broken_rate=(None if pd.isna(c["max_broken_rate"]) else float(c["max_broken_rate"])),
            min_red_rate=(None if pd.isna(c["min_red_rate"]) else float(c["min_red_rate"])),
            max_limit_down=(None if pd.isna(c["max_limit_down"]) else int(c["max_limit_down"])),
            max_pullback=(None if pd.isna(c["max_pullback"]) else float(c["max_pullback"])),
            alloc_pct=1.0,
            rules=rules,
        )
        checked += 1
        ret_pct = float(m["ret"]) * 100.0
        dd_pct = float(m["max_dd"]) * 100.0
        rec = dict(c)
        rec.update(
            stage="coarse",
            threshold=0.65,
            alloc=1.0,
            ret_pct=ret_pct,
            dd_pct=dd_pct,
            trades=int(m["trades"]),
            obj=objective(ret_pct, dd_pct),
        )
        coarse_rows.append(rec)
        if dd_pct <= DD_LIMIT + 1e-9:
            if (best_feasible is None) or (ret_pct > best_feasible[0]):
                best_feasible = (ret_pct, dd_pct, rec)
                print(f"[coarse best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")

    coarse_df = pd.DataFrame(coarse_rows)
    coarse_df = coarse_df.sort_values(["obj", "ret_pct"], ascending=False)
    focus = coarse_df.head(80).copy()
    print(f"coarse_checked={checked} focus={len(focus)}")

    rows = list(coarse_rows)

    # Stage-2: dense around top coarse families
    th_grid = [round(x, 4) for x in np.arange(0.50, 0.801, 0.005)]
    alloc_grid = [round(x, 4) for x in np.arange(0.92, 1.0001, 0.001)]
    for i, row in focus.reset_index(drop=True).iterrows():
        ranked = ranked_map[int(row["alpha"])]
        for th in th_grid:
            for alloc in alloc_grid:
                m = mb._simulate_fast(
                    trade_dates=trade_dates,
                    ranked_by_date=ranked,
                    threshold=float(th),
                    initial_capital=10000.0,
                    sell_rule=str(row["sell_rule"]),
                    top_k=int(row["top_k"]),
                    max_board=(None if pd.isna(row["max_board"]) else int(row["max_board"])),
                    gap_min=(None if pd.isna(row["gap_min"]) else float(row["gap_min"])),
                    gap_max=(None if pd.isna(row["gap_max"]) else float(row["gap_max"])),
                    max_broken_rate=(None if pd.isna(row["max_broken_rate"]) else float(row["max_broken_rate"])),
                    min_red_rate=(None if pd.isna(row["min_red_rate"]) else float(row["min_red_rate"])),
                    max_limit_down=(None if pd.isna(row["max_limit_down"]) else int(row["max_limit_down"])),
                    max_pullback=(None if pd.isna(row["max_pullback"]) else float(row["max_pullback"])),
                    alloc_pct=float(alloc),
                    rules=rules,
                )
                checked += 1
                ret_pct = float(m["ret"]) * 100.0
                dd_pct = float(m["max_dd"]) * 100.0
                rec = {
                    "stage": "refine",
                    "alpha": int(row["alpha"]),
                    "sell_rule": str(row["sell_rule"]),
                    "top_k": int(row["top_k"]),
                    "gap_min": (None if pd.isna(row["gap_min"]) else float(row["gap_min"])),
                    "gap_max": (None if pd.isna(row["gap_max"]) else float(row["gap_max"])),
                    "max_board": (None if pd.isna(row["max_board"]) else int(row["max_board"])),
                    "max_broken_rate": (None if pd.isna(row["max_broken_rate"]) else float(row["max_broken_rate"])),
                    "min_red_rate": (None if pd.isna(row["min_red_rate"]) else float(row["min_red_rate"])),
                    "max_limit_down": (None if pd.isna(row["max_limit_down"]) else int(row["max_limit_down"])),
                    "max_pullback": (None if pd.isna(row["max_pullback"]) else float(row["max_pullback"])),
                    "threshold": float(th),
                    "alloc": float(alloc),
                    "ret_pct": ret_pct,
                    "dd_pct": dd_pct,
                    "trades": int(m["trades"]),
                    "obj": objective(ret_pct, dd_pct),
                }
                rows.append(rec)
                if dd_pct <= DD_LIMIT + 1e-9:
                    if (best_feasible is None) or (ret_pct > best_feasible[0]):
                        best_feasible = (ret_pct, dd_pct, rec)
                        print(f"[refine best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
        print(f"[refine-done] {i+1}/{len(focus)} checked={checked}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"checked={checked}")
    print(f"best_feasible={best_feasible}")
    print(f"saved={OUT_CSV}")


if __name__ == "__main__":
    main()
