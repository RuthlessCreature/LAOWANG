import argparse
import datetime as dt

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_1613_strategy_adaptive_candidates.csv"


def main() -> None:
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
    if not trade_dates:
        raise SystemExit("no trade dates")
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
    variant_name = str(spec["name"])
    seed = int(spec["seed"]) + 388
    print(f"variant={variant_name} seed={seed}")
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

    alphas = [13, 14, 15, 16]
    ranked_map = {}
    for alpha in alphas:
        scored = base.copy()
        scored["raw_prob"] = base_prob
        scored["score"] = mb._score_from_raw_prob(base_prob, alpha=alpha)
        ranked_map[alpha] = mb._build_ranked_by_date(scored)
    print(f"alphas={alphas}")

    sell_rules = ["t2_close", "strong_hold_t3", "loss_rebound_t3"]
    top_ks = [2, 3, 4, 5]
    gaps = [(-0.005, 0.025), (-0.005, 0.03), (-0.01, 0.03), (None, None)]
    boards = [None, 2, 3, 4]

    risks = [
        ("off", None, None, None, None),
        ("balanced", 0.45, 0.24, 20, 0.09),
        ("strict", 0.40, 0.28, 15, 0.08),
        ("ultra", 0.35, 0.30, 12, 0.07),
    ]
    for br in [0.42, 0.45, 0.48]:
        for rr in [0.22, 0.24, 0.26]:
            for ld in [16, 20, 24]:
                for pb in [0.08, 0.09, 0.10]:
                    risks.append((f"custom_b{br:.2f}_r{rr:.2f}_ld{ld}_pb{pb:.2f}", br, rr, ld, pb))

    print(f"coarse space={len(alphas)*len(sell_rules)*len(top_ks)*len(gaps)*len(boards)*len(risks)}")

    records = []
    best_feasible = None
    best_any = None
    checked = 0

    for alpha in alphas:
        ranked = ranked_map[alpha]
        for rule in sell_rules:
            for top_k in top_ks:
                for gap_min, gap_max in gaps:
                    for max_board in boards:
                        for risk_name, br, rr, ld, pb in risks:
                            m = mb._simulate_fast(
                                trade_dates=trade_dates,
                                ranked_by_date=ranked,
                                threshold=0.65,
                                initial_capital=10000.0,
                                sell_rule=rule,
                                top_k=top_k,
                                max_board=max_board,
                                gap_min=gap_min,
                                gap_max=gap_max,
                                max_broken_rate=br,
                                min_red_rate=rr,
                                max_limit_down=ld,
                                max_pullback=pb,
                                alloc_pct=1.0,
                                rules=rules,
                            )
                            checked += 1
                            ret_pct = float(m["ret"]) * 100.0
                            dd_pct = float(m["max_dd"]) * 100.0
                            rec = {
                                "stage": "coarse",
                                "alpha": alpha,
                                "sell_rule": rule,
                                "top_k": top_k,
                                "gap_min": gap_min,
                                "gap_max": gap_max,
                                "max_board": max_board,
                                "risk": risk_name,
                                "max_broken_rate": br,
                                "min_red_rate": rr,
                                "max_limit_down": ld,
                                "max_pullback": pb,
                                "threshold": 0.65,
                                "alloc": 1.0,
                                "ret_pct": ret_pct,
                                "dd_pct": dd_pct,
                                "trades": int(m["trades"]),
                            }
                            records.append(rec)
                            if (best_any is None) or (ret_pct > best_any[0]):
                                best_any = (ret_pct, dd_pct, rec)
                            if dd_pct <= DD_LIMIT + 1e-9:
                                if (best_feasible is None) or (ret_pct > best_feasible[0]):
                                    best_feasible = (ret_pct, dd_pct, rec)
                                    print(f"[coarse best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
        print(f"[coarse alpha-done] a={alpha} checked={checked}")

    coarse_df = pd.DataFrame([r for r in records if r["stage"] == "coarse"])
    coarse_df["score18"] = coarse_df["ret_pct"] - 35.0 * (coarse_df["dd_pct"] - 18.0).clip(lower=0.0)
    focus = coarse_df.sort_values(["score18", "ret_pct"], ascending=False).head(50).copy()
    print(f"refine targets={len(focus)}")

    thresholds = [round(x, 3) for x in np.arange(0.55, 0.851, 0.01)]
    allocs = [round(x, 4) for x in np.arange(0.92, 1.0001, 0.002)]
    for i, row in focus.reset_index(drop=True).iterrows():
        alpha = int(row["alpha"])
        ranked = ranked_map[alpha]
        for th in thresholds:
            for alloc in allocs:
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
                    "alpha": alpha,
                    "sell_rule": str(row["sell_rule"]),
                    "top_k": int(row["top_k"]),
                    "gap_min": (None if pd.isna(row["gap_min"]) else float(row["gap_min"])),
                    "gap_max": (None if pd.isna(row["gap_max"]) else float(row["gap_max"])),
                    "max_board": (None if pd.isna(row["max_board"]) else int(row["max_board"])),
                    "risk": str(row["risk"]),
                    "max_broken_rate": (None if pd.isna(row["max_broken_rate"]) else float(row["max_broken_rate"])),
                    "min_red_rate": (None if pd.isna(row["min_red_rate"]) else float(row["min_red_rate"])),
                    "max_limit_down": (None if pd.isna(row["max_limit_down"]) else int(row["max_limit_down"])),
                    "max_pullback": (None if pd.isna(row["max_pullback"]) else float(row["max_pullback"])),
                    "threshold": float(th),
                    "alloc": float(alloc),
                    "ret_pct": ret_pct,
                    "dd_pct": dd_pct,
                    "trades": int(m["trades"]),
                }
                records.append(rec)
                if (best_any is None) or (ret_pct > best_any[0]):
                    best_any = (ret_pct, dd_pct, rec)
                if dd_pct <= DD_LIMIT + 1e-9:
                    if (best_feasible is None) or (ret_pct > best_feasible[0]):
                        best_feasible = (ret_pct, dd_pct, rec)
                        print(f"[refine best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
        print(f"[refine cfg-done] {i + 1}/{len(focus)} checked={checked}")

    if best_feasible is not None:
        bf = best_feasible[2]
        alpha = int(bf["alpha"])
        ranked = ranked_map[alpha]
        base_th = float(bf["threshold"])
        base_alloc = float(bf["alloc"])
        ths = [
            round(x, 4)
            for x in np.arange(max(0.5, base_th - 0.03), min(0.99, base_th + 0.03) + 1e-12, 0.001)
        ]
        alloc_grid = [
            round(x, 4)
            for x in np.arange(max(0.90, base_alloc - 0.03), min(1.0, base_alloc + 0.03) + 1e-12, 0.0005)
        ]
        print(f"ultra around th={base_th:.4f} alloc={base_alloc:.4f} -> ths={len(ths)} allocs={len(alloc_grid)}")
        for th in ths:
            for alloc in alloc_grid:
                m = mb._simulate_fast(
                    trade_dates=trade_dates,
                    ranked_by_date=ranked,
                    threshold=float(th),
                    initial_capital=10000.0,
                    sell_rule=str(bf["sell_rule"]),
                    top_k=int(bf["top_k"]),
                    max_board=bf["max_board"],
                    gap_min=bf["gap_min"],
                    gap_max=bf["gap_max"],
                    max_broken_rate=bf["max_broken_rate"],
                    min_red_rate=bf["min_red_rate"],
                    max_limit_down=bf["max_limit_down"],
                    max_pullback=bf["max_pullback"],
                    alloc_pct=float(alloc),
                    rules=rules,
                )
                checked += 1
                ret_pct = float(m["ret"]) * 100.0
                dd_pct = float(m["max_dd"]) * 100.0
                rec = {
                    "stage": "ultra",
                    "alpha": alpha,
                    "sell_rule": str(bf["sell_rule"]),
                    "top_k": int(bf["top_k"]),
                    "gap_min": bf["gap_min"],
                    "gap_max": bf["gap_max"],
                    "max_board": bf["max_board"],
                    "risk": str(bf["risk"]),
                    "max_broken_rate": bf["max_broken_rate"],
                    "min_red_rate": bf["min_red_rate"],
                    "max_limit_down": bf["max_limit_down"],
                    "max_pullback": bf["max_pullback"],
                    "threshold": float(th),
                    "alloc": float(alloc),
                    "ret_pct": ret_pct,
                    "dd_pct": dd_pct,
                    "trades": int(m["trades"]),
                }
                records.append(rec)
                if dd_pct <= DD_LIMIT + 1e-9 and ret_pct > best_feasible[0]:
                    best_feasible = (ret_pct, dd_pct, rec)
                    print(f"[ultra best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")

    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False)
    print(f"checked={checked}")
    print(f"best_any={best_any}")
    print(f"best_feasible={best_feasible}")
    print(f"saved={OUT_CSV}")


if __name__ == "__main__":
    main()
