import argparse
import datetime as dt

import numpy as np
import pandas as pd

import mock_backtest as mb
import relay_strategy_model as relay_model

DD_LIMIT = 16.13
RULES_FILE = "rules_1613.txt"
OUT_CSV = "reports/search_1613_lossreb_systematic_candidates.csv"


def objective(ret_pct: float, dd_pct: float) -> float:
    return ret_pct - 70.0 * max(0.0, dd_pct - DD_LIMIT)


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


def simulate(cfg, *, rules, trade_dates, ranked_map):
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
        max_broken_rate=float(cfg["max_broken_rate"]),
        min_red_rate=float(cfg["min_red_rate"]),
        max_limit_down=int(cfg["max_limit_down"]),
        max_pullback=float(cfg["max_pullback"]),
        alloc_pct=float(cfg["alloc"]),
        rules=rules,
    )
    ret_pct = float(m["ret"]) * 100.0
    dd_pct = float(m["max_dd"]) * 100.0
    return ret_pct, dd_pct, int(m["trades"])


def run_stage(cfg_iter, stage_name, *, rules, trade_dates, ranked_map, rows, checked, best):
    local_cnt = 0
    for cfg in cfg_iter:
        ret_pct, dd_pct, trades = simulate(cfg, rules=rules, trade_dates=trade_dates, ranked_map=ranked_map)
        checked += 1
        local_cnt += 1
        rec = dict(cfg)
        rec.update(
            stage=stage_name,
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
                print(f"[{stage_name} best<=16.13] ret={ret_pct:.6f}% dd={dd_pct:.6f}% cfg={rec}")
    return checked, best, local_cnt


def main():
    rules, trade_dates, ranked_map = prepare_ranked()

    rows = []
    checked = 0
    best = None

    # Current best as anchor.
    anchor = {
        "alpha": 15,
        "top_k": 3,
        "gap_min": -0.0009,
        "gap_max": 0.0255,
        "max_board": None,
        "max_broken_rate": 0.412,
        "min_red_rate": 0.287,
        "max_limit_down": 10,
        "max_pullback": 0.065,
        "threshold": 0.6141,
        "alloc": 0.9782,
    }

    # Stage A: threshold/alloc sweep under anchor risk family.
    th_grid = [round(x, 4) for x in np.arange(0.50, 0.681, 0.001)]
    alloc_grid = [round(x, 4) for x in np.arange(0.94, 1.0001, 0.001)]
    print(f"stageA th={len(th_grid)} alloc={len(alloc_grid)}")

    def gen_a():
        for th in th_grid:
            for alloc in alloc_grid:
                c = dict(anchor)
                c["threshold"] = th
                c["alloc"] = alloc
                yield c

    checked, best, cnt = run_stage(gen_a(), "stageA", rules=rules, trade_dates=trade_dates, ranked_map=ranked_map, rows=rows, checked=checked, best=best)
    print(f"stageA_done checked={checked} cnt={cnt}")

    # Stage B: risk sweep around current best th/alloc.
    ref = dict(best[2]) if best is not None else dict(anchor)
    br_values = [round(x, 3) for x in np.arange(0.380, 0.441, 0.003)]
    rr_values = [round(x, 3) for x in np.arange(0.220, 0.331, 0.005)]
    pb_values = [round(x, 3) for x in np.arange(0.050, 0.101, 0.005)]
    ld_values = [8, 9, 10, 11, 12, 14]
    print(f"stageB br={len(br_values)} rr={len(rr_values)} pb={len(pb_values)} ld={len(ld_values)}")

    def gen_b():
        for br in br_values:
            for rr in rr_values:
                if rr > br - 0.004:
                    continue
                for pb in pb_values:
                    for ld in ld_values:
                        c = dict(ref)
                        c["max_broken_rate"] = br
                        c["min_red_rate"] = rr
                        c["max_pullback"] = pb
                        c["max_limit_down"] = ld
                        yield c

    checked, best, cnt = run_stage(gen_b(), "stageB", rules=rules, trade_dates=trade_dates, ranked_map=ranked_map, rows=rows, checked=checked, best=best)
    print(f"stageB_done checked={checked} cnt={cnt}")

    # Stage C: alpha/topk/gap/board variants around best risk values.
    ref = dict(best[2]) if best is not None else dict(ref)
    alpha_values = [14, 15, 16]
    topk_values = [2, 3, 4]
    gap_min_values = [round(x, 4) for x in np.arange(-0.0045, 0.0006, 0.0005)]
    gap_max_values = [round(x, 4) for x in np.arange(0.0230, 0.0281, 0.0005)]
    board_values = [None, 2, 3]
    print(f"stageC a={len(alpha_values)} topk={len(topk_values)} gmin={len(gap_min_values)} gmax={len(gap_max_values)} board={len(board_values)}")

    def gen_c():
        for a in alpha_values:
            for k in topk_values:
                for gmin in gap_min_values:
                    for gmax in gap_max_values:
                        if gmax < gmin + 0.012:
                            continue
                        for board in board_values:
                            c = dict(ref)
                            c["alpha"] = a
                            c["top_k"] = k
                            c["gap_min"] = gmin
                            c["gap_max"] = gmax
                            c["max_board"] = board
                            yield c

    checked, best, cnt = run_stage(gen_c(), "stageC", rules=rules, trade_dates=trade_dates, ranked_map=ranked_map, rows=rows, checked=checked, best=best)
    print(f"stageC_done checked={checked} cnt={cnt}")

    # Stage D: final threshold/alloc micro-refine on top families from A/B/C.
    all_df = pd.DataFrame(rows)
    feasible = all_df[all_df["dd_pct"] <= DD_LIMIT + 1e-9].copy()
    fam_cols = [
        "alpha",
        "top_k",
        "gap_min",
        "gap_max",
        "max_board",
        "max_broken_rate",
        "min_red_rate",
        "max_limit_down",
        "max_pullback",
    ]
    fam_focus = (
        feasible.sort_values(["obj", "ret_pct"], ascending=False)
        .drop_duplicates(subset=fam_cols)
        .head(20)
        .copy()
    )
    print(f"stageD_families={len(fam_focus)}")

    def gen_d():
        for _, r in fam_focus.iterrows():
            th0 = float(r["threshold"])
            al0 = float(r["alloc"])
            ths = [round(x, 4) for x in np.arange(max(0.40, th0 - 0.020), min(0.75, th0 + 0.020) + 1e-12, 0.001)]
            als = [round(x, 4) for x in np.arange(max(0.90, al0 - 0.015), min(1.00, al0 + 0.015) + 1e-12, 0.001)]
            base = {k: (None if pd.isna(r[k]) else r[k]) for k in fam_cols}
            for th in ths:
                for al in als:
                    c = {
                        "alpha": int(base["alpha"]),
                        "top_k": int(base["top_k"]),
                        "gap_min": float(base["gap_min"]),
                        "gap_max": float(base["gap_max"]),
                        "max_board": (None if base["max_board"] is None else int(base["max_board"])),
                        "max_broken_rate": float(base["max_broken_rate"]),
                        "min_red_rate": float(base["min_red_rate"]),
                        "max_limit_down": int(base["max_limit_down"]),
                        "max_pullback": float(base["max_pullback"]),
                        "threshold": th,
                        "alloc": al,
                    }
                    yield c

    checked, best, cnt = run_stage(gen_d(), "stageD", rules=rules, trade_dates=trade_dates, ranked_map=ranked_map, rows=rows, checked=checked, best=best)
    print(f"stageD_done checked={checked} cnt={cnt}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"checked={checked}")
    print(f"best={best}")
    print(f"saved={OUT_CSV}")


if __name__ == "__main__":
    main()
