import argparse
import datetime as dt

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_1613_seed_neighborhood2_candidates.csv"


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

    return rules, trade_dates, full_pool, base, train_df, x_train, x_all


def main() -> None:
    rules, trade_dates, full_pool, base, train_df, x_train, x_all = prepare_data()

    spec = [s for s in mb.list_variant_specs("aggressive") if s["name"] == "mlp_h88_e500_gt1_wd2e4"][0]
    base_seed = int(spec["seed"])
    y = (train_df["t1_hold_ret"] > float(spec["target_gt"])).astype(np.int8).to_numpy(np.int8)

    alphas = [14, 15, 16]
    thresholds = [round(x, 4) for x in np.arange(0.50, 0.801, 0.01)]
    allocs = [round(x, 4) for x in np.arange(0.94, 1.0001, 0.002)]
    seed_offsets = list(range(373, 404))
    print(f"seed_offsets={seed_offsets[0]}..{seed_offsets[-1]} n={len(seed_offsets)}")
    print(f"alphas={alphas} thresholds={len(thresholds)} allocs={len(allocs)}")

    rows = []
    best = None
    checked = 0

    for off in seed_offsets:
        seed = base_seed + off
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

        local_best = None
        for alpha in alphas:
            scored = base.copy()
            scored["raw_prob"] = base_prob
            scored["score"] = mb._score_from_raw_prob(base_prob, alpha=alpha)
            ranked = mb._build_ranked_by_date(scored)

            for th in thresholds:
                for alloc in allocs:
                    m = mb._simulate_fast(
                        trade_dates=trade_dates,
                        ranked_by_date=ranked,
                        threshold=float(th),
                        initial_capital=10000.0,
                        sell_rule="t2_close",
                        top_k=3,
                        max_board=None,
                        gap_min=-0.005,
                        gap_max=0.025,
                        max_broken_rate=0.45,
                        min_red_rate=0.24,
                        max_limit_down=20,
                        max_pullback=0.09,
                        alloc_pct=float(alloc),
                        rules=rules,
                    )
                    checked += 1
                    ret_pct = float(m["ret"]) * 100.0
                    dd_pct = float(m["max_dd"]) * 100.0
                    rec = {
                        "seed": seed,
                        "seed_offset": off,
                        "alpha": alpha,
                        "threshold": float(th),
                        "alloc": float(alloc),
                        "ret_pct": ret_pct,
                        "dd_pct": dd_pct,
                        "trades": int(m["trades"]),
                    }
                    rows.append(rec)
                    if (local_best is None) or (ret_pct > local_best[0]):
                        local_best = (ret_pct, dd_pct, rec)
                    if dd_pct <= DD_LIMIT + 1e-9:
                        if (best is None) or (ret_pct > best[0]):
                            best = (ret_pct, dd_pct, rec)
                            print(f"[best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% seed={seed} alpha={alpha} th={th:.3f} alloc={alloc:.3f}")

        print(
            f"[seed-done] seed={seed} local_best_ret={local_best[0]:.4f}% "
            f"local_best_dd={local_best[1]:.4f}% checked={checked}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"checked={checked}")
    print(f"best={best}")
    print(f"saved={OUT_CSV}")


if __name__ == "__main__":
    main()
