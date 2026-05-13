#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central trading protocol defaults for LAOWANG strategy workflows.

This module is intentionally side-effect free: importing it must not open a
database connection or read local config. Strategy specs live here so backtests,
plan generation, and later journal writers can share one versioned source of
truth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


ORDINARY_MODEL_ORDER: Tuple[str, ...] = ("laowang", "stwg", "ywcx", "fhkq")


@dataclass(frozen=True)
class OrdinaryModelProtocol:
    model: str
    strategy_id: str
    strategy_version: str
    min_score: float
    min_gap: float
    max_gap: float
    min_amount_ma20: float
    required_tags: Tuple[str, ...]
    any_tags: Tuple[str, ...]
    blocked_tags: Tuple[str, ...]
    planned_position_pct: float
    hard_stop_pct: float
    structure_stop: str
    max_hold_days: int
    t1_buy_condition: str
    t2_sell_condition: str
    profit_arm_pct: Optional[float] = None
    profit_trailing_stop: Optional[str] = None
    paused: bool = False
    protocol_notes: Tuple[str, ...] = ()

    def to_plan_config(self) -> Dict[str, Any]:
        cfg = asdict(self)
        cfg["required_tags"] = list(self.required_tags)
        cfg["any_tags"] = list(self.any_tags)
        cfg["blocked_tags"] = list(self.blocked_tags)
        cfg["protocol_notes"] = list(self.protocol_notes)
        return cfg


ORDINARY_MODEL_PROTOCOLS: Dict[str, OrdinaryModelProtocol] = {
    "laowang": OrdinaryModelProtocol(
        model="laowang",
        strategy_id="laowang_manual",
        strategy_version="laowang_manual_v1",
        min_score=65.0,
        min_gap=-0.03,
        max_gap=0.03,
        min_amount_ma20=100_000_000.0,
        required_tags=("TREND_UP", "LOW_BASE", "SPACE_OK"),
        any_tags=(),
        blocked_tags=("RISK_FILTERED", "NEAR_RESISTANCE"),
        planned_position_pct=0.05,
        hard_stop_pct=-0.05,
        structure_stop="close_below_ma20",
        max_hold_days=20,
        t1_buy_condition=(
            "T+1 only: open gap must be -3% to +3%; buy only after price holds above "
            "signal close or reclaims it intraday, with no risk tag and acceptable liquidity."
        ),
        t2_sell_condition=(
            "T+2 or later: exit on -5% hard stop, close below MA20, or after +12% profit "
            "arm when price closes below MA10; force exit by max_hold_days."
        ),
        profit_arm_pct=0.12,
        profit_trailing_stop="close_below_ma10",
        protocol_notes=("trend_low_base", "slow_signal", "do_not_chase_gap_gt_3pct"),
    ),
    "stwg": OrdinaryModelProtocol(
        model="stwg",
        strategy_id="stwg_manual",
        strategy_version="stwg_manual_v1",
        min_score=65.0,
        min_gap=-0.03,
        max_gap=0.04,
        min_amount_ma20=80_000_000.0,
        required_tags=("STAGE_B_COMPRESSED",),
        any_tags=("BREAKOUT_R", "VOLUME_EXPANSION"),
        blocked_tags=("RISK_FILTERED",),
        planned_position_pct=0.05,
        hard_stop_pct=-0.04,
        structure_stop="close_below_ma10_or_region_high",
        max_hold_days=10,
        t1_buy_condition=(
            "T+1 only: open gap must be -3% to +4%; buy only on breakout continuation "
            "above signal close/region high with volume confirmation; no chase above +4%."
        ),
        t2_sell_condition=(
            "T+2 or later: exit on -4% hard stop, close back below MA10 or failed region "
            "high; after +10% profit arm trail with close below MA5; force exit by max_hold_days."
        ),
        profit_arm_pct=0.10,
        profit_trailing_stop="close_below_ma5",
        protocol_notes=("breakout_only", "do_not_buy_observation_state"),
    ),
    "ywcx": OrdinaryModelProtocol(
        model="ywcx",
        strategy_id="ywcx_manual",
        strategy_version="ywcx_manual_v1",
        min_score=65.0,
        min_gap=-0.03,
        max_gap=0.02,
        min_amount_ma20=50_000_000.0,
        required_tags=("BROKEN_IPO", "NEAR_IPO_LOW", "VOLUME_DRY", "JUST_ABOVE_MA5"),
        any_tags=(),
        blocked_tags=("RISK_FILTERED",),
        planned_position_pct=0.03,
        hard_stop_pct=-0.04,
        structure_stop="close_below_ma5",
        max_hold_days=5,
        t1_buy_condition=(
            "T+1 only: open gap must be -3% to +2%; buy only if price remains just above "
            "MA5 or reclaims signal close without liquidity collapse."
        ),
        t2_sell_condition=(
            "T+2 or later: exit on -4% hard stop, close below MA5, failed volume recovery, "
            "or after +8% profit target/reduction; force exit by max_hold_days."
        ),
        profit_arm_pct=0.08,
        profit_trailing_stop="profit_target_or_close_below_ma5",
        paused=False,
        protocol_notes=("small_position_until_sample_size_improves",),
    ),
    "fhkq": OrdinaryModelProtocol(
        model="fhkq",
        strategy_id="fhkq_event",
        strategy_version="fhkq_event_v1",
        min_score=80.0,
        min_gap=-0.10,
        max_gap=0.03,
        min_amount_ma20=0.0,
        required_tags=(),
        any_tags=(),
        blocked_tags=(),
        planned_position_pct=0.015,
        hard_stop_pct=-0.04,
        structure_stop="event_exit_t2",
        max_hold_days=2,
        t1_buy_condition=(
            "T+1 only: buy only if the stock is tradable, not locked limit-down, gap is "
            "-10% to +3%, and liquidity recovery is still visible; never chase event spikes."
        ),
        t2_sell_condition=(
            "T+2 or later: default event exit/reduce on first sellable day; exit immediately "
            "on -4% hard stop or liquidity recovery failure; max hold is 2 days."
        ),
        profit_arm_pct=0.06,
        profit_trailing_stop="first_sellable_day_reduce_or_profit_exit",
        protocol_notes=("event_model", "do_not_apply_trend_ma_entry", "liquidity_recovery_required"),
    ),
}


def ordinary_model_protocol(model: str) -> OrdinaryModelProtocol:
    key = str(model or "").strip().lower()
    if key not in ORDINARY_MODEL_PROTOCOLS:
        raise KeyError(f"unknown ordinary strategy model: {model}")
    return ORDINARY_MODEL_PROTOCOLS[key]


def ordinary_model_plan_config(model: str) -> Dict[str, Any]:
    return ordinary_model_protocol(model).to_plan_config()


def strategy_versions() -> Dict[str, str]:
    return {model: spec.strategy_version for model, spec in ORDINARY_MODEL_PROTOCOLS.items()}


def main() -> int:
    print(json.dumps(strategy_versions(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
