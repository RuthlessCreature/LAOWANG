import argparse
import datetime as dt

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_1613_custom_risk_refine_candidates.csv"


def objective(ret_pct: float, dd_pct: float) -> float:
    # Penalize only excess drawdown; prefer high return near DD boundary.
    return ret_pct - 50.0 * max(0.0, dd_pct - DD_LIMIT)


def prepare(seed_offset: int = 388):
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
    seed = int(spec["seed"]) + int(seed_offset)
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

    scored = base.copy()
    scored["raw_prob"] = base_prob
    scored["score"] = mb._score_from_raw_prob(base_prob, alpha=15)
    ranked = mb._build_ranked_by_date(scored)
    return rules, trade_dates, ranked


def run_search():
    rules, trade_dates, ranked = prepare(seed_offset=388)

    rows = []
    checked = 0
    best = None

    br_values = [round(x, 3) for x in np.arange(0.392, 0.431, 0.002)]
    rr_values = [round(x, 3) for x in np.arange(0.250, 0.311, 0.003)]
    pb_values = [round(x, 3) for x in np.arange(0.096, 0.126, 0.003)]
    ld_values = [16, 18, 20, 22, 24]
    topk_values = [3, 4]
    board_values = [None, 2, 3]
    gap_values = [(-0.005, 0.025), (-0.005, 0.03)]

    coarse_cfg = []
    for br in br_values:
        for rr in rr_values:
            if rr > br - 0.01:
                continue
            for pb in pb_values:
                for ld in ld_values:
                    for topk in topk_values:
                        for board in board_values:
                            for gap_min, gap_max in gap_values:
                                coarse_cfg.append(
                                    (br, rr, pb, ld, topk, board, gap_min, gap_max)
                                )

    print(f"coarse_cfg={len(coarse_cfg)}")

    for br, rr, pb, ld, topk, board, gap_min, gap_max in coarse_cfg:
        m = mb._simulate_fast(
            trade_dates=trade_dates,
            ranked_by_date=ranked,
            threshold=0.5,
            initial_capital=10000.0,
            sell_rule="t2_close",
            top_k=int(topk),
            max_board=board,
            gap_min=float(gap_min),
            gap_max=float(gap_max),
            max_broken_rate=float(br),
            min_red_rate=float(rr),
            max_limit_down=int(ld),
            max_pullback=float(pb),
            alloc_pct=0.96,
            rules=rules,
        )
        checked += 1
        ret_pct = float(m["ret"]) * 100.0
        dd_pct = float(m["max_dd"]) * 100.0
        rec = {
            "stage": "coarse",
            "alpha": 15,
            "sell_rule": "t2_close",
            "top_k": int(topk),
            "gap_min": float(gap_min),
            "gap_max": float(gap_max),
            "max_board": board,
            "max_broken_rate": float(br),
            "min_red_rate": float(rr),
            "max_limit_down": int(ld),
            "max_pullback": float(pb),
            "threshold": 0.5,
            "alloc": 0.96,
            "ret_pct": ret_pct,
            "dd_pct": dd_pct,
            "trades": int(m["trades"]),
            "obj": objective(ret_pct, dd_pct),
        }
        rows.append(rec)
        if dd_pct <= DD_LIMIT + 1e-9:
            if (best is None) or (ret_pct > best[0]):
                best = (ret_pct, dd_pct, rec)
                print(f"[coarse best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")

    coarse_df = pd.DataFrame(rows)
    focus = coarse_df.sort_values(["obj", "ret_pct"], ascending=False).head(80).copy()
    print(f"coarse_checked={checked} focus={len(focus)}")

    th_grid = [round(x, 4) for x in np.arange(0.45, 0.581, 0.002)]
    alloc_grid = [round(x, 4) for x in np.arange(0.93, 0.981, 0.001)]

    for i, row in focus.reset_index(drop=True).iterrows():
        for th in th_grid:
            for alloc in alloc_grid:
                m = mb._simulate_fast(
                    trade_dates=trade_dates,
                    ranked_by_date=ranked,
                    threshold=float(th),
                    initial_capital=10000.0,
                    sell_rule="t2_close",
                    top_k=int(row["top_k"]),
                    max_board=(None if pd.isna(row["max_board"]) else int(row["max_board"])),
                    gap_min=float(row["gap_min"]),
                    gap_max=float(row["gap_max"]),
                    max_broken_rate=float(row["max_broken_rate"]),
                    min_red_rate=float(row["min_red_rate"]),
                    max_limit_down=int(row["max_limit_down"]),
                    max_pullback=float(row["max_pullback"]),
                    alloc_pct=float(alloc),
                    rules=rules,
                )
                checked += 1
                ret_pct = float(m["ret"]) * 100.0
                dd_pct = float(m["max_dd"]) * 100.0
                rec = {
                    "stage": "refine",
                    "alpha": 15,
                    "sell_rule": "t2_close",
                    "top_k": int(row["top_k"]),
                    "gap_min": float(row["gap_min"]),
                    "gap_max": float(row["gap_max"]),
                    "max_board": (None if pd.isna(row["max_board"]) else int(row["max_board"])),
                    "max_broken_rate": float(row["max_broken_rate"]),
                    "min_red_rate": float(row["min_red_rate"]),
                    "max_limit_down": int(row["max_limit_down"]),
                    "max_pullback": float(row["max_pullback"]),
                    "threshold": float(th),
                    "alloc": float(alloc),
                    "ret_pct": ret_pct,
                    "dd_pct": dd_pct,
                    "trades": int(m["trades"]),
                    "obj": objective(ret_pct, dd_pct),
                }
                rows.append(rec)
                if dd_pct <= DD_LIMIT + 1e-9:
                    if (best is None) or (ret_pct > best[0]):
                        best = (ret_pct, dd_pct, rec)
                        print(f"[refine best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
        print(f"[refine-done] {i+1}/{len(focus)} checked={checked}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"checked={checked}")
    print(f"best={best}")
    print(f"saved={OUT_CSV}")


if __name__ == "__main__":
    run_search()
