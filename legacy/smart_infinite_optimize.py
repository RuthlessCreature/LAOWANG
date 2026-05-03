#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infinite dual-objective optimizer for relay models.

This script keeps optimizing indefinitely (until Ctrl+C) and tracks:
1) aggressive champion: maximize return
2) balanced champion: maximize return with drawdown penalty
3) compromise champion: weighted middle objective

Search strategy is hybrid:
- TPE-like global sampling from top history
- CMA-like local mutation around current champions
- random restarts for exploration
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import mock_backtest as backtest_model
except Exception:  # noqa: BLE001
    import backtest_model  # type: ignore


try:
    # Keep redirected output visible in real time when running detached.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class VariantSpec:
    name: str
    hidden: int
    epochs: int
    lr: float
    weight_decay: float
    target_gt: float
    seed: int


@dataclass
class EvalRecord:
    iteration: int
    candidate_index: int
    spec: Dict[str, Any]
    status: str
    ret_pct: float
    max_dd_pct: float
    final_capital: float
    history_rows: int
    best_cfg_text: str
    best_seed_offset: int
    report_dir: str
    aggressive_score: float
    balanced_score: float
    compromise_score: float


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infinite dual-objective relay optimizer.")
    p.add_argument("--config", default="config.ini")
    p.add_argument("--db-url", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--rules-file", default="rules.txt")
    p.add_argument("--base-best-json", default="reports/iter_rounds_m5_ret_stop10/best/best_model.json")

    p.add_argument("--train-start-date", default="2020-01-01")
    p.add_argument("--backtest-start-date", default="2025-01-01")
    p.add_argument("--end-date", default="2026-02-13")
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--thresholds", default="0.75")
    p.add_argument("--alphas", default="2,5,10")

    p.add_argument("--population-size", type=int, default=10, help="Candidates per global iteration.")
    p.add_argument("--branch-max-rounds", type=int, default=80, help="Per-candidate inner optimize rounds.")
    p.add_argument("--branch-patience", type=int, default=10, help="Per-candidate no-improve stop.")
    p.add_argument("--branch-seed-step", type=int, default=97)
    p.add_argument("--samples-per-key", type=int, default=3)
    p.add_argument("--max-dd-limit", type=float, default=0.45)
    p.add_argument(
        "--eval-mode",
        choices=["leaky", "walk_forward"],
        default="walk_forward",
        help="Pass-through to optimize_selected_model --eval-mode.",
    )
    p.add_argument("--retrain-every-days", type=int, default=20, help="Pass-through walk_forward retrain cadence.")
    p.add_argument("--min-train-rows", type=int, default=80, help="Pass-through walk_forward minimum train rows.")
    p.add_argument("--train-end-date", default=None, help="Optional strict train cutoff for inner optimizer.")
    p.add_argument("--disable-minute-features", action="store_true")
    p.add_argument("--candidate-timeout-sec", type=int, default=0, help="0 disables timeout.")

    p.add_argument("--balanced-lambda", type=float, default=1.3, help="Balanced score: ret - lambda*dd.")
    p.add_argument("--balanced-dd-cap", type=float, default=15.0, help="Soft cap for balanced objective.")
    p.add_argument("--balanced-cap-penalty", type=float, default=2.0, help="Extra penalty above balanced-dd-cap.")
    p.add_argument("--compromise-weight", type=float, default=0.55, help="Compromise = w*aggressive + (1-w)*balanced.")

    p.add_argument("--fit-on-improve", action="store_true", help="Fit champion model whenever champion improves.")
    p.add_argument("--score-on-improve", action="store_true", help="Run scoring_relay after champion fit.")
    p.add_argument("--fit-model-dir", default="models")

    p.add_argument("--report-root", default=None, help="Default reports/smart_infinite_<timestamp>")
    p.add_argument("--state-file", default=None, help="Default <report-root>/state.json")
    p.add_argument("--resume", action="store_true", help="Resume from state-file if exists.")
    p.add_argument("--max-global-iterations", type=int, default=0, help="0 means infinite.")
    p.add_argument("--seed", type=int, default=20260227)
    return p.parse_args(argv)


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(v))))


def _q_hidden(v: float) -> int:
    return int(max(16, min(256, round(float(v) / 8.0) * 8)))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _choose_cfg_from_best_json(payload: Dict[str, Any]) -> str:
    metrics = payload.get("metrics_by_threshold")
    if isinstance(metrics, dict) and metrics:
        best_ret = float("-inf")
        best_cfg = ""
        for _, v in metrics.items():
            if not isinstance(v, dict):
                continue
            ret = _safe_float(v.get("ret_pct"), float("-inf"))
            cfg = str(v.get("cfg") or "").strip()
            if cfg and ret > best_ret:
                best_ret = ret
                best_cfg = cfg
        if best_cfg:
            return best_cfg
    cfg = str(payload.get("best_cfg_text") or "").strip()
    if cfg:
        return cfg
    raise ValueError("cannot parse cfg from best json")


def _load_parent_spec(best_json_path: Path) -> Tuple[VariantSpec, str]:
    payload = _load_json(best_json_path)
    cfg = _choose_cfg_from_best_json(payload)
    variant_name = cfg.split("/", 1)[0].strip()
    if not variant_name:
        raise ValueError(f"invalid cfg in {best_json_path.as_posix()}: {cfg}")
    spec = backtest_model.resolve_variant_spec(variant_name)
    if not spec:
        raise ValueError(f"variant not found in built-in specs: {variant_name}")
    return (
        VariantSpec(
            name=str(variant_name),
            hidden=int(spec["hidden"]),
            epochs=int(spec["epochs"]),
            lr=float(spec["lr"]),
            weight_decay=float(spec["weight_decay"]),
            target_gt=float(spec["target_gt"]),
            seed=int(spec["seed"]),
        ),
        cfg,
    )


def _mutate_variant(base: VariantSpec, rng: random.Random, *, strength: float, name: str) -> VariantSpec:
    hidden = _q_hidden(base.hidden * math.exp(rng.gauss(0.0, 0.28 * strength)))
    epochs = int(round(_clamp(base.epochs * math.exp(rng.gauss(0.0, 0.26 * strength)), 180, 1100)))
    lr = _clamp(base.lr * math.exp(rng.gauss(0.0, 0.35 * strength)), 0.006, 0.08)
    log_wd = math.log10(max(base.weight_decay, 1e-7))
    wd = 10 ** _clamp(log_wd + rng.gauss(0.0, 0.35 * strength), -6.5, -2.0)
    target_gt = _clamp(base.target_gt + rng.gauss(0.0, 0.012 * strength), -0.04, 0.10)
    seed = int(max(1, base.seed + rng.randint(-1200, 1200)))
    return VariantSpec(
        name=name,
        hidden=int(hidden),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(wd),
        target_gt=float(target_gt),
        seed=seed,
    )


def _random_restart(rng: random.Random, *, name: str) -> VariantSpec:
    hidden = _q_hidden(rng.uniform(24, 160))
    epochs = int(round(rng.uniform(220, 980)))
    lr = float(10 ** rng.uniform(math.log10(0.008), math.log10(0.060)))
    wd = float(10 ** rng.uniform(-6.2, -2.3))
    target_gt = float(rng.uniform(-0.03, 0.08))
    seed = int(rng.randint(1, 50000))
    return VariantSpec(
        name=name,
        hidden=int(hidden),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(wd),
        target_gt=float(target_gt),
        seed=seed,
    )


def _variant_signature(spec: VariantSpec) -> Tuple[int, int, int, int, int]:
    return (
        int(spec.hidden),
        int(spec.epochs),
        int(round(spec.lr * 1_000_000)),
        int(round(spec.weight_decay * 100_000_000)),
        int(round(spec.target_gt * 10000)),
    )


def _build_population(
    *,
    iteration: int,
    population_size: int,
    rng: random.Random,
    parent: VariantSpec,
    history: Sequence[Dict[str, Any]],
    aggressive_champion: Optional[Dict[str, Any]],
    balanced_champion: Optional[Dict[str, Any]],
) -> List[VariantSpec]:
    pop: List[VariantSpec] = []
    used = set()

    def _spec_from_dict(d: Dict[str, Any], fallback: VariantSpec) -> VariantSpec:
        return VariantSpec(
            name=str(d.get("name", fallback.name)),
            hidden=int(d.get("hidden", fallback.hidden)),
            epochs=int(d.get("epochs", fallback.epochs)),
            lr=float(d.get("lr", fallback.lr)),
            weight_decay=float(d.get("weight_decay", fallback.weight_decay)),
            target_gt=float(d.get("target_gt", fallback.target_gt)),
            seed=int(d.get("seed", fallback.seed)),
        )

    def add(spec: VariantSpec) -> None:
        sig = _variant_signature(spec)
        if sig in used:
            return
        used.add(sig)
        pop.append(spec)

    add(_mutate_variant(parent, rng, strength=0.55, name=f"i{iteration:04d}c01"))
    if aggressive_champion and isinstance(aggressive_champion.get("spec"), dict):
        add(
            _mutate_variant(
                _spec_from_dict(aggressive_champion["spec"], parent),
                rng,
                strength=0.45,
                name=f"i{iteration:04d}c02",
            )
        )
    if balanced_champion and isinstance(balanced_champion.get("spec"), dict):
        add(
            _mutate_variant(
                _spec_from_dict(balanced_champion["spec"], parent),
                rng,
                strength=0.45,
                name=f"i{iteration:04d}c03",
            )
        )

    top_hist = sorted(
        [h for h in history if str(h.get("status")) == "ok"],
        key=lambda x: _safe_float(x.get("compromise_score"), float("-inf")),
        reverse=True,
    )
    top_hist = top_hist[: max(3, int(len(top_hist) * 0.25))]

    idx = len(pop) + 1
    while len(pop) < int(population_size):
        cand_name = f"i{iteration:04d}c{idx:02d}"
        idx += 1
        roll = rng.random()
        if top_hist and roll < 0.42:
            src = rng.choice(top_hist)
            spec = _mutate_variant(
                _spec_from_dict(src["spec"], parent),
                rng,
                strength=0.62,
                name=cand_name,
            )
            add(spec)
            continue
        if roll < 0.78:
            anchors: List[VariantSpec] = [parent]
            if aggressive_champion and isinstance(aggressive_champion.get("spec"), dict):
                anchors.append(_spec_from_dict(aggressive_champion["spec"], parent))
            if balanced_champion and isinstance(balanced_champion.get("spec"), dict):
                anchors.append(_spec_from_dict(balanced_champion["spec"], parent))
            add(_mutate_variant(rng.choice(anchors), rng, strength=0.38, name=cand_name))
            continue
        add(_random_restart(rng, name=cand_name))
    return pop[: int(population_size)]


def _run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    timeout_sec: int = 0,
    stdout_log_path: Optional[Path] = None,
    stderr_log_path: Optional[Path] = None,
    stream_prefix: str = "",
) -> Tuple[int, str, str, bool]:
    timed_out = False
    out_lines: List[str] = []
    err_lines: List[str] = []

    out_fp = None
    err_fp = None
    try:
        if stdout_log_path is not None:
            stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
            out_fp = stdout_log_path.open("w", encoding="utf-8")
        if stderr_log_path is not None:
            stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
            err_fp = stderr_log_path.open("w", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def _consume_pipe(
            pipe: Optional[Any],
            sink: List[str],
            echo_stream: Any,
            file_stream: Optional[Any],
        ) -> None:
            if pipe is None:
                return
            for line in iter(pipe.readline, ""):
                sink.append(line)
                if file_stream is not None:
                    file_stream.write(line)
                    file_stream.flush()
                if stream_prefix:
                    echo_stream.write(f"{stream_prefix}{line}")
                else:
                    echo_stream.write(line)
                echo_stream.flush()
            pipe.close()

        t_out = threading.Thread(
            target=_consume_pipe,
            args=(proc.stdout, out_lines, sys.stdout, out_fp),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_consume_pipe,
            args=(proc.stderr, err_lines, sys.stderr, err_fp),
            daemon=True,
        )
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=(None if int(timeout_sec) <= 0 else int(timeout_sec)))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()

        t_out.join()
        t_err.join()
        code = 124 if timed_out else int(proc.returncode or 0)
        return code, "".join(out_lines), "".join(err_lines), timed_out
    except Exception:
        raise
    finally:
        if out_fp is not None:
            out_fp.close()
        if err_fp is not None:
            err_fp.close()


def _calc_scores(
    *,
    ret_pct: float,
    dd_pct: float,
    balanced_lambda: float,
    balanced_dd_cap: float,
    balanced_cap_penalty: float,
    compromise_weight: float,
) -> Tuple[float, float, float]:
    aggressive = float(ret_pct)
    balanced = float(ret_pct) - float(balanced_lambda) * float(dd_pct)
    over = max(0.0, float(dd_pct) - float(balanced_dd_cap))
    if over > 0:
        balanced -= float(balanced_cap_penalty) * over
    w = _clamp(float(compromise_weight), 0.0, 1.0)
    compromise = w * aggressive + (1.0 - w) * balanced
    return aggressive, balanced, compromise


def _champion_better(new_row: Dict[str, Any], old_row: Optional[Dict[str, Any]], score_key: str) -> bool:
    if old_row is None:
        return True
    ns = _safe_float(new_row.get(score_key), float("-inf"))
    os = _safe_float(old_row.get(score_key), float("-inf"))
    if ns > os + 1e-12:
        return True
    if ns < os - 1e-12:
        return False
    nr = _safe_float(new_row.get("ret_pct"), float("-inf"))
    orr = _safe_float(old_row.get("ret_pct"), float("-inf"))
    if nr > orr + 1e-12:
        return True
    if nr < orr - 1e-12:
        return False
    nd = _safe_float(new_row.get("max_dd_pct"), float("inf"))
    od = _safe_float(old_row.get("max_dd_pct"), float("inf"))
    return nd < od - 1e-12


def _write_spec_file(path: Path, spec: Dict[str, Any]) -> None:
    payload = [spec]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fit_champion(
    *,
    label: str,
    champ: Dict[str, Any],
    args: argparse.Namespace,
    report_root: Path,
    log_dir: Path,
) -> Dict[str, Any]:
    spec = dict(champ["spec"])
    spec_file = report_root / "champions" / f"{label}_variant_spec.json"
    _write_spec_file(spec_file, spec)
    fit_model_dir = Path(args.fit_model_dir)
    fit_model_dir.mkdir(parents=True, exist_ok=True)
    model_file = fit_model_dir / f"relay_model_{label}_smart.npz"

    best_cfg = str(champ.get("best_cfg_text") or "")
    seed_offset = int(_safe_float(champ.get("best_seed_offset"), 0.0))
    fit_cmd = [
        sys.executable,
        "-u",
        "fit_relay_model_file.py",
        "--config",
        str(args.config),
        "--train-start-date",
        str(args.train_start_date),
        "--score-base-start-date",
        str(args.backtest_start_date),
        "--end-date",
        str(args.end_date),
        "--variant-specs-file",
        str(spec_file.as_posix()),
        "--cfg",
        best_cfg,
        "--seed-offset",
        str(seed_offset),
        "--model-file",
        str(model_file.as_posix()),
    ]
    if args.db_url:
        fit_cmd.extend(["--db-url", str(args.db_url)])
    if args.db:
        fit_cmd.extend(["--db", str(args.db)])
    if bool(args.disable_minute_features):
        fit_cmd.append("--disable-minute-features")

    out_log = log_dir / f"{label}_fit.stdout.log"
    err_log = log_dir / f"{label}_fit.stderr.log"
    code, _out, _err, timed_out = _run_cmd(
        fit_cmd,
        cwd=Path("."),
        timeout_sec=max(0, int(args.candidate_timeout_sec)),
        stdout_log_path=out_log,
        stderr_log_path=err_log,
        stream_prefix=f"[fit:{label}] ",
    )
    resp: Dict[str, Any] = {
        "label": label,
        "return_code": int(code),
        "timed_out": bool(timed_out),
        "model_file": model_file.as_posix(),
        "spec_file": spec_file.as_posix(),
        "stdout_log": out_log.as_posix(),
        "stderr_log": err_log.as_posix(),
    }
    if int(code) == 0 and bool(args.score_on_improve):
        score_cmd = [
            sys.executable,
            "-u",
            "scoring_relay.py",
            "--config",
            str(args.config),
            "--start-date",
            str(args.backtest_start_date),
            "--end-date",
            str(args.end_date),
            "--model-file",
            str(model_file.as_posix()),
            "--full-rebuild",
        ]
        if args.db_url:
            score_cmd.extend(["--db-url", str(args.db_url)])
        if args.db:
            score_cmd.extend(["--db", str(args.db)])
        s_out_log = log_dir / f"{label}_score.stdout.log"
        s_err_log = log_dir / f"{label}_score.stderr.log"
        s_code, _s_out, _s_err, s_to = _run_cmd(
            score_cmd,
            cwd=Path("."),
            timeout_sec=max(0, int(args.candidate_timeout_sec)),
            stdout_log_path=s_out_log,
            stderr_log_path=s_err_log,
            stream_prefix=f"[score:{label}] ",
        )
        resp["scoring"] = {
            "return_code": int(s_code),
            "timed_out": bool(s_to),
            "stdout_log": s_out_log.as_posix(),
            "stderr_log": s_err_log.as_posix(),
        }
    return resp


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception:  # noqa: BLE001
        return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report_root = Path(args.report_root) if args.report_root else Path(f"reports/smart_infinite_{_now_tag()}")
    report_root.mkdir(parents=True, exist_ok=True)
    state_file = Path(args.state_file) if args.state_file else report_root / "state.json"
    iter_root = report_root / "iterations"
    log_root = report_root / "logs"
    iter_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    parent, parent_cfg = _load_parent_spec(Path(args.base_best_json))
    rng = random.Random(int(args.seed))

    state: Dict[str, Any] = {}
    if bool(args.resume):
        state = _load_state(state_file)

    history: List[Dict[str, Any]] = list(state.get("history", [])) if isinstance(state.get("history"), list) else []
    aggressive_champion: Optional[Dict[str, Any]] = (
        dict(state["aggressive_champion"]) if isinstance(state.get("aggressive_champion"), dict) else None
    )
    balanced_champion: Optional[Dict[str, Any]] = (
        dict(state["balanced_champion"]) if isinstance(state.get("balanced_champion"), dict) else None
    )
    compromise_champion: Optional[Dict[str, Any]] = (
        dict(state["compromise_champion"]) if isinstance(state.get("compromise_champion"), dict) else None
    )

    iteration = int(_safe_float(state.get("iteration"), 0.0)) + 1
    max_global = int(args.max_global_iterations)

    _log(
        f"[smart] start parent={parent.name} h={parent.hidden} e={parent.epochs} lr={parent.lr:.4f} "
        f"wd={parent.weight_decay:.1e} gt={parent.target_gt:.4f}"
    )
    _log(
        f"[smart] report_root={report_root.as_posix()} population={int(args.population_size)} "
        f"branch_patience={int(args.branch_patience)} branch_rounds={int(args.branch_max_rounds)} "
        f"eval_mode={args.eval_mode} retrain_every={int(args.retrain_every_days)}"
    )

    try:
        while True:
            if max_global > 0 and iteration > max_global:
                break

            this_iter_dir = iter_root / f"iter_{iteration:05d}"
            this_iter_dir.mkdir(parents=True, exist_ok=True)
            this_log_dir = log_root / f"iter_{iteration:05d}"
            this_log_dir.mkdir(parents=True, exist_ok=True)

            population = _build_population(
                iteration=iteration,
                population_size=int(args.population_size),
                rng=rng,
                parent=parent,
                history=history,
                aggressive_champion=aggressive_champion,
                balanced_champion=balanced_champion,
            )
            iter_records: List[Dict[str, Any]] = []
            _log(f"[smart][iter {iteration}] candidates={len(population)}")

            for idx, spec in enumerate(population, 1):
                spec_dict = asdict(spec)
                spec_path = this_iter_dir / f"candidate_{idx:02d}_spec.json"
                _write_spec_file(spec_path, spec_dict)
                cand_dir = this_iter_dir / f"candidate_{idx:02d}_{spec.name}"
                cand_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable,
                    "-u",
                    "optimize_selected_model.py",
                    "--config",
                    str(args.config),
                    "--rules-file",
                    str(args.rules_file),
                    "--train-start-date",
                    str(args.train_start_date),
                    "--backtest-start-date",
                    str(args.backtest_start_date),
                    "--end-date",
                    str(args.end_date),
                    "--initial-capital",
                    str(float(args.initial_capital)),
                    "--max-dd-limit",
                    str(float(args.max_dd_limit)),
                    "--eval-mode",
                    str(args.eval_mode),
                    "--retrain-every-days",
                    str(int(args.retrain_every_days)),
                    "--min-train-rows",
                    str(int(args.min_train_rows)),
                    "--max-rounds",
                    str(int(args.branch_max_rounds)),
                    "--patience",
                    str(int(args.branch_patience)),
                    "--seed-step",
                    str(int(args.branch_seed_step)),
                    "--samples-per-key",
                    str(int(args.samples_per_key)),
                    "--alphas",
                    str(args.alphas),
                    "--thresholds",
                    str(args.thresholds),
                    "--variant-specs-file",
                    str(spec_path.as_posix()),
                    "--report-dir",
                    str(cand_dir.as_posix()),
                ]
                if args.db_url:
                    cmd.extend(["--db-url", str(args.db_url)])
                if args.db:
                    cmd.extend(["--db", str(args.db)])
                if args.train_end_date:
                    cmd.extend(["--train-end-date", str(args.train_end_date)])
                if bool(args.disable_minute_features):
                    cmd.append("--disable-minute-features")

                _log(
                    f"[smart][iter {iteration}][{idx:02d}/{len(population)}] "
                    f"{spec.name} h={spec.hidden} e={spec.epochs} lr={spec.lr:.4f} wd={spec.weight_decay:.1e} gt={spec.target_gt:.4f}"
                )
                cand_out_log = this_log_dir / f"candidate_{idx:02d}.stdout.log"
                cand_err_log = this_log_dir / f"candidate_{idx:02d}.stderr.log"
                code, _out, _err, timed_out = _run_cmd(
                    cmd,
                    cwd=Path("."),
                    timeout_sec=max(0, int(args.candidate_timeout_sec)),
                    stdout_log_path=cand_out_log,
                    stderr_log_path=cand_err_log,
                    stream_prefix=f"[smart][iter {iteration}][{idx:02d}] ",
                )

                summary_path = cand_dir / "summary.json"
                if int(code) != 0 or timed_out or not summary_path.exists():
                    row = asdict(
                        EvalRecord(
                            iteration=iteration,
                            candidate_index=idx,
                            spec=spec_dict,
                            status="failed",
                            ret_pct=float("-inf"),
                            max_dd_pct=float("inf"),
                            final_capital=float("-inf"),
                            history_rows=0,
                            best_cfg_text="",
                            best_seed_offset=0,
                            report_dir=cand_dir.as_posix(),
                            aggressive_score=float("-inf"),
                            balanced_score=float("-inf"),
                            compromise_score=float("-inf"),
                        )
                    )
                    iter_records.append(row)
                    _log(f"[smart][iter {iteration}][{idx:02d}] FAILED code={code} timeout={timed_out}")
                    continue

                summary = _load_json(summary_path)
                best_detail = summary.get("best_detail") if isinstance(summary.get("best_detail"), dict) else {}
                ret_pct = _safe_float(best_detail.get("ret_pct"), float("-inf"))
                dd_pct = _safe_float(best_detail.get("max_dd_pct"), float("inf"))
                final_capital = _safe_float(best_detail.get("final_capital"), float("-inf"))
                history_rows = int(_safe_float(summary.get("history_rows"), 0.0))
                best_cfg = str(summary.get("best_cfg_text") or "")
                best = summary.get("best") if isinstance(summary.get("best"), dict) else {}
                best_seed_offset = int(_safe_float(best.get("seed_offset"), 0.0))
                ags, bas, com = _calc_scores(
                    ret_pct=ret_pct,
                    dd_pct=dd_pct,
                    balanced_lambda=float(args.balanced_lambda),
                    balanced_dd_cap=float(args.balanced_dd_cap),
                    balanced_cap_penalty=float(args.balanced_cap_penalty),
                    compromise_weight=float(args.compromise_weight),
                )
                row = asdict(
                    EvalRecord(
                        iteration=iteration,
                        candidate_index=idx,
                        spec=spec_dict,
                        status="ok",
                        ret_pct=ret_pct,
                        max_dd_pct=dd_pct,
                        final_capital=final_capital,
                        history_rows=history_rows,
                        best_cfg_text=best_cfg,
                        best_seed_offset=best_seed_offset,
                        report_dir=cand_dir.as_posix(),
                        aggressive_score=ags,
                        balanced_score=bas,
                        compromise_score=com,
                    )
                )
                iter_records.append(row)
                _log(
                    f"[smart][iter {iteration}][{idx:02d}] ret={ret_pct:.2f}% dd={dd_pct:.2f}% "
                    f"ag={ags:.2f} bal={bas:.2f} rounds={history_rows}"
                )

            ok_rows = [r for r in iter_records if str(r.get("status")) == "ok"]
            if ok_rows:
                best_ag = max(ok_rows, key=lambda x: _safe_float(x.get("aggressive_score"), float("-inf")))
                best_ba = max(ok_rows, key=lambda x: _safe_float(x.get("balanced_score"), float("-inf")))
                best_co = max(ok_rows, key=lambda x: _safe_float(x.get("compromise_score"), float("-inf")))
                _log(
                    f"[smart][iter {iteration}] best_ag={best_ag['spec']['name']} ret={best_ag['ret_pct']:.2f}% dd={best_ag['max_dd_pct']:.2f}%"
                )
                _log(
                    f"[smart][iter {iteration}] best_bal={best_ba['spec']['name']} ret={best_ba['ret_pct']:.2f}% dd={best_ba['max_dd_pct']:.2f}%"
                )

                ag_improved = _champion_better(best_ag, aggressive_champion, "aggressive_score")
                ba_improved = _champion_better(best_ba, balanced_champion, "balanced_score")
                co_improved = _champion_better(best_co, compromise_champion, "compromise_score")
                if ag_improved:
                    aggressive_champion = dict(best_ag)
                if ba_improved:
                    balanced_champion = dict(best_ba)
                if co_improved:
                    compromise_champion = dict(best_co)

                if bool(args.fit_on_improve):
                    fit_runs: List[Dict[str, Any]] = []
                    if ag_improved and aggressive_champion is not None:
                        fit_runs.append(
                            _fit_champion(
                                label="aggressive",
                                champ=aggressive_champion,
                                args=args,
                                report_root=report_root,
                                log_dir=this_log_dir,
                            )
                        )
                    if ba_improved and balanced_champion is not None:
                        fit_runs.append(
                            _fit_champion(
                                label="balanced",
                                champ=balanced_champion,
                                args=args,
                                report_root=report_root,
                                log_dir=this_log_dir,
                            )
                        )
                    if co_improved and compromise_champion is not None:
                        fit_runs.append(
                            _fit_champion(
                                label="compromise",
                                champ=compromise_champion,
                                args=args,
                                report_root=report_root,
                                log_dir=this_log_dir,
                            )
                        )
                    if fit_runs:
                        _log(f"[smart][iter {iteration}] fitted champions={len(fit_runs)}")

            history.extend(iter_records)
            iter_summary = {
                "iteration": iteration,
                "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "candidate_n": len(iter_records),
                "ok_n": len([r for r in iter_records if str(r.get("status")) == "ok"]),
                "records": iter_records,
                "best_aggressive": aggressive_champion,
                "best_balanced": balanced_champion,
                "best_compromise": compromise_champion,
            }
            _save_json(this_iter_dir / "summary.json", iter_summary)

            state_payload = {
                "updated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "iteration": iteration,
                "parent_spec": asdict(parent),
                "parent_cfg": parent_cfg,
                "history_count": len(history),
                "history": history,
                "aggressive_champion": aggressive_champion,
                "balanced_champion": balanced_champion,
                "compromise_champion": compromise_champion,
                "report_root": report_root.as_posix(),
            }
            _save_json(state_file, state_payload)
            iteration += 1
    except KeyboardInterrupt:
        _log("[smart] stopped by user (Ctrl+C). state saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
