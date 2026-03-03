import argparse
import datetime as dt
import random

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_1613_lossreb_random2_candidates.csv"


def objective(ret_pct: float, dd_pct: float) -> float:
    # Push strongly toward feasible high-return frontier.
    return ret_pct - 65.0 * max(0.0, dd_pct - DD_LIMIT)


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def prepare_ranked():
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
    seed = int(spec["seed"]) + 388
    print(f"variant={spec['name']} seed={seed}")

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
    for alpha in [14, 15, 16]:
        scored = base.copy()
        scored["raw_prob"] = base_prob
        scored["score"] = mb._score_from_raw_prob(base_prob, alpha=alpha)
        ranked_map[alpha] = mb._build_ranked_by_date(scored)
    return rules, trade_dates, ranked_map


def simulate(cfg, rules, trade_dates, ranked_map):
    ranked = ranked_map[int(cfg["alpha"])]
    m = mb._simulate_fast(
        trade_dates=trade_dates,
        ranked_by_date=ranked,
        threshold=float(cfg["threshold"]),
        initial_capital=10000.0,
        sell_rule="loss_rebound_t3",
        top_k=int(cfg["top_k"]),
        max_board=cfg["max_board"],
        gap_min=float(cfg["gap_min"]),
        gap_max=float(cfg["gap_max"]),
        max_broken_rate=(None if cfg["max_broken_rate"] is None else float(cfg["max_broken_rate"])),
        min_red_rate=(None if cfg["min_red_rate"] is None else float(cfg["min_red_rate"])),
        max_limit_down=(None if cfg["max_limit_down"] is None else int(cfg["max_limit_down"])),
        max_pullback=(None if cfg["max_pullback"] is None else float(cfg["max_pullback"])),
        alloc_pct=float(cfg["alloc"]),
        rules=rules,
    )
    return float(m["ret"]) * 100.0, float(m["max_dd"]) * 100.0, int(m["trades"])


def sample_cfg(rng):
    alpha = rng.choice([14, 15, 16])
    top_k = rng.choice([2, 3, 4])
    gap_min = round(rng.uniform(-0.0065, -0.0002), 4)
    gap_max = round(rng.uniform(0.0210, 0.0320), 4)
    if gap_max < gap_min + 0.017:
        gap_max = round(gap_min + 0.017, 4)

    max_board = rng.choice([None, None, None, 2, 3, 4])

    max_broken_rate = round(rng.uniform(0.37, 0.46), 3)
    min_red_rate = round(rng.uniform(0.20, 0.33), 3)
    if min_red_rate > max_broken_rate - 0.004:
        min_red_rate = round(max(0.10, max_broken_rate - 0.01), 3)
    max_limit_down = int(rng.randint(8, 24))
    max_pullback = round(rng.uniform(0.06, 0.12), 3)

    threshold = round(rng.uniform(0.50, 0.64), 4)
    alloc = round(rng.uniform(0.95, 1.00), 4)

    return {
        "alpha": alpha,
        "top_k": top_k,
        "gap_min": gap_min,
        "gap_max": gap_max,
        "max_board": max_board,
        "max_broken_rate": max_broken_rate,
        "min_red_rate": min_red_rate,
        "max_limit_down": max_limit_down,
        "max_pullback": max_pullback,
        "threshold": threshold,
        "alloc": alloc,
    }


def mutate(cfg, rng):
    c = dict(cfg)
    c["alpha"] = rng.choice([c["alpha"], 14, 15, 16])
    c["top_k"] = int(clip(int(c["top_k"]) + rng.choice([-1, 0, 1]), 1, 5))

    c["gap_min"] = round(clip(float(c["gap_min"]) + rng.uniform(-0.0015, 0.0015), -0.0100, 0.0020), 4)
    c["gap_max"] = round(clip(float(c["gap_max"]) + rng.uniform(-0.0020, 0.0020), 0.0160, 0.0400), 4)
    if c["gap_max"] < c["gap_min"] + 0.012:
        c["gap_max"] = round(c["gap_min"] + 0.012, 4)

    if c["max_board"] is None:
        if rng.random() < 0.15:
            c["max_board"] = rng.choice([2, 3, 4])
    else:
        if rng.random() < 0.18:
            c["max_board"] = None
        else:
            c["max_board"] = int(clip(int(c["max_board"]) + rng.choice([-1, 0, 1]), 1, 4))

    c["max_broken_rate"] = round(clip(float(c["max_broken_rate"]) + rng.uniform(-0.010, 0.010), 0.34, 0.50), 3)
    c["min_red_rate"] = round(clip(float(c["min_red_rate"]) + rng.uniform(-0.015, 0.015), 0.10, 0.36), 3)
    if c["min_red_rate"] > c["max_broken_rate"] - 0.004:
        c["min_red_rate"] = round(max(0.10, c["max_broken_rate"] - 0.008), 3)

    c["max_limit_down"] = int(clip(int(c["max_limit_down"]) + rng.choice([-2, -1, 0, 1, 2]), 6, 30))
    c["max_pullback"] = round(clip(float(c["max_pullback"]) + rng.uniform(-0.006, 0.006), 0.04, 0.16), 3)

    c["threshold"] = round(clip(float(c["threshold"]) + rng.uniform(-0.02, 0.02), 0.40, 0.75), 4)
    c["alloc"] = round(clip(float(c["alloc"]) + rng.uniform(-0.015, 0.015), 0.85, 1.00), 4)
    return c


def main():
    rng = random.Random(2026030317)
    rules, trade_dates, ranked_map = prepare_ranked()

    rows = []
    checked = 0
    best = None

    n_random = 160000
    print(f"random_trials={n_random}")
    pool = []

    for i in range(n_random):
        cfg = sample_cfg(rng)
        ret_pct, dd_pct, trades = simulate(cfg, rules, trade_dates, ranked_map)
        checked += 1
        rec = dict(cfg)
        rec.update(
            stage="random",
            sell_rule="loss_rebound_t3",
            ret_pct=ret_pct,
            dd_pct=dd_pct,
            trades=trades,
            obj=objective(ret_pct, dd_pct),
        )
        rows.append(rec)
        pool.append(rec)
        if dd_pct <= DD_LIMIT + 1e-9:
            if (best is None) or (ret_pct > best[0]):
                best = (ret_pct, dd_pct, rec)
                print(f"[random best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
        if (i + 1) % 10000 == 0:
            print(f"[random progress] {i+1}/{n_random} checked={checked}")

    focus = sorted(pool, key=lambda r: (r["obj"], r["ret_pct"]), reverse=True)[:320]
    print(f"mutate_parents={len(focus)}")

    per_parent = 40
    for idx, parent in enumerate(focus, 1):
        for _ in range(per_parent):
            cfg = mutate(parent, rng)
            ret_pct, dd_pct, trades = simulate(cfg, rules, trade_dates, ranked_map)
            checked += 1
            rec = dict(cfg)
            rec.update(
                stage="mutate",
                sell_rule="loss_rebound_t3",
                ret_pct=ret_pct,
                dd_pct=dd_pct,
                trades=trades,
                obj=objective(ret_pct, dd_pct),
            )
            rows.append(rec)
            if dd_pct <= DD_LIMIT + 1e-9:
                if (best is None) or (ret_pct > best[0]):
                    best = (ret_pct, dd_pct, rec)
                    print(f"[mutate best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
        if idx % 20 == 0:
            print(f"[mutate progress] {idx}/{len(focus)} checked={checked}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"checked={checked}")
    print(f"best={best}")
    print(f"saved={OUT_CSV}")


if __name__ == "__main__":
    main()
