import argparse
import datetime as dt

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
COARSE_CSV = "reports/search_1613_strategy_adaptive_candidates.csv"
OUT_CSV = "reports/search_1613_target_refine_candidates.csv"


def _to_opt(v):
    if pd.isna(v):
        return None
    return float(v)


def build_ranked_map():
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

    spec = [s for s in mb.list_variant_specs("aggressive") if s["name"] == "mlp_h88_e500_gt1_wd2e4"][0]
    seed = int(spec["seed"]) + 388
    variant_name = str(spec["name"])
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

    alphas = sorted({13, 14, 15, 16})
    ranked_map = {}
    for alpha in alphas:
        scored = base.copy()
        scored["raw_prob"] = base_prob
        scored["score"] = mb._score_from_raw_prob(base_prob, alpha=alpha)
        ranked_map[alpha] = mb._build_ranked_by_date(scored)
    print(f"prepared_alphas={alphas}")
    return rules, trade_dates, ranked_map


def main() -> None:
    raw = pd.read_csv(COARSE_CSV)
    coarse = raw[raw["stage"] == "coarse"].copy()
    coarse = coarse[(coarse["dd_pct"] <= 17.5) & (coarse["ret_pct"] >= 1100.0)].copy()
    coarse = coarse.sort_values(["ret_pct", "dd_pct"], ascending=[False, True])
    key_cols = [
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
    coarse = coarse.drop_duplicates(subset=key_cols).head(12).copy()
    print(f"refine_configs={len(coarse)}")

    rules, trade_dates, ranked_map = build_ranked_map()

    thresholds = [round(x, 4) for x in np.arange(0.55, 0.851, 0.002)]
    allocs = [round(x, 4) for x in np.arange(0.95, 1.0001, 0.0005)]
    print(f"grid thresholds={len(thresholds)} allocs={len(allocs)}")

    best = None
    checked = 0
    rows = []

    for i, row in coarse.reset_index(drop=True).iterrows():
        alpha = int(row["alpha"])
        ranked = ranked_map[alpha]
        cfg = {
            "alpha": alpha,
            "sell_rule": str(row["sell_rule"]),
            "top_k": int(row["top_k"]),
            "gap_min": _to_opt(row["gap_min"]),
            "gap_max": _to_opt(row["gap_max"]),
            "max_board": (None if pd.isna(row["max_board"]) else int(row["max_board"])),
            "max_broken_rate": _to_opt(row["max_broken_rate"]),
            "min_red_rate": _to_opt(row["min_red_rate"]),
            "max_limit_down": (None if pd.isna(row["max_limit_down"]) else int(row["max_limit_down"])),
            "max_pullback": _to_opt(row["max_pullback"]),
        }
        for th in thresholds:
            for alloc in allocs:
                m = mb._simulate_fast(
                    trade_dates=trade_dates,
                    ranked_by_date=ranked,
                    threshold=float(th),
                    initial_capital=10000.0,
                    sell_rule=cfg["sell_rule"],
                    top_k=cfg["top_k"],
                    max_board=cfg["max_board"],
                    gap_min=cfg["gap_min"],
                    gap_max=cfg["gap_max"],
                    max_broken_rate=cfg["max_broken_rate"],
                    min_red_rate=cfg["min_red_rate"],
                    max_limit_down=cfg["max_limit_down"],
                    max_pullback=cfg["max_pullback"],
                    alloc_pct=float(alloc),
                    rules=rules,
                )
                checked += 1
                ret_pct = float(m["ret"]) * 100.0
                dd_pct = float(m["max_dd"]) * 100.0
                rec = {
                    "stage": "refine",
                    **cfg,
                    "threshold": float(th),
                    "alloc": float(alloc),
                    "ret_pct": ret_pct,
                    "dd_pct": dd_pct,
                    "trades": int(m["trades"]),
                }
                rows.append(rec)
                if dd_pct <= DD_LIMIT + 1e-9:
                    if (best is None) or (ret_pct > best[0]):
                        best = (ret_pct, dd_pct, rec)
                        print(f"[best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
        print(f"[cfg-done] {i+1}/{len(coarse)} checked={checked}")

    if best is not None:
        b = best[2]
        alpha = int(b["alpha"])
        ranked = ranked_map[alpha]
        ths = [round(x, 4) for x in np.arange(max(0.5, b["threshold"] - 0.03), min(0.99, b["threshold"] + 0.03) + 1e-12, 0.0005)]
        alloc2 = [round(x, 4) for x in np.arange(max(0.90, b["alloc"] - 0.02), min(1.0, b["alloc"] + 0.02) + 1e-12, 0.0002)]
        print(f"ultra thresholds={len(ths)} allocs={len(alloc2)}")
        for th in ths:
            for alloc in alloc2:
                m = mb._simulate_fast(
                    trade_dates=trade_dates,
                    ranked_by_date=ranked,
                    threshold=float(th),
                    initial_capital=10000.0,
                    sell_rule=b["sell_rule"],
                    top_k=int(b["top_k"]),
                    max_board=b["max_board"],
                    gap_min=b["gap_min"],
                    gap_max=b["gap_max"],
                    max_broken_rate=b["max_broken_rate"],
                    min_red_rate=b["min_red_rate"],
                    max_limit_down=b["max_limit_down"],
                    max_pullback=b["max_pullback"],
                    alloc_pct=float(alloc),
                    rules=rules,
                )
                checked += 1
                ret_pct = float(m["ret"]) * 100.0
                dd_pct = float(m["max_dd"]) * 100.0
                rec = {
                    "stage": "ultra",
                    "alpha": alpha,
                    "sell_rule": b["sell_rule"],
                    "top_k": int(b["top_k"]),
                    "gap_min": b["gap_min"],
                    "gap_max": b["gap_max"],
                    "max_board": b["max_board"],
                    "max_broken_rate": b["max_broken_rate"],
                    "min_red_rate": b["min_red_rate"],
                    "max_limit_down": b["max_limit_down"],
                    "max_pullback": b["max_pullback"],
                    "threshold": float(th),
                    "alloc": float(alloc),
                    "ret_pct": ret_pct,
                    "dd_pct": dd_pct,
                    "trades": int(m["trades"]),
                }
                rows.append(rec)
                if dd_pct <= DD_LIMIT + 1e-9 and ret_pct > best[0]:
                    best = (ret_pct, dd_pct, rec)
                    print(f"[ultra best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"checked={checked}")
    print(f"best={best}")
    print(f"saved={OUT_CSV}")


if __name__ == "__main__":
    main()
