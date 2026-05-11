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


@dataclass(frozen=True)
class RelayCandidateProtocol:
    strategy_id: str = "relay_hybrid"
    strategy_version: str = "relay_hybrid_v1"
    min_score: float = 0.70
    max_rank_no: int = 2
    max_board: int = 3
    max_broken_rate: float = 0.40
    min_red_rate: float = 0.28
    max_limit_down_count: int = 15
    max_pullback: float = 0.08
    min_amount_ma20: float = 150_000_000.0


@dataclass(frozen=True)
class RelayAuctionProtocol:
    min_gap: float = -0.01
    max_gap: float = 0.04
    min_auction_amount_ratio: float = 0.08
    block_limit_up_open: bool = True
    block_limit_down_open: bool = True
    reject_obvious_price_slope_down: bool = True
    reject_cancel_risk: bool = True


@dataclass(frozen=True)
class RelayIntradayProtocol:
    min_first_15m_low_drawdown: float = -0.03
    triggers: Tuple[str, ...] = (
        "reclaim_vwap",
        "reclaim_open_price",
        "reseal",
        "divergence_absorbed",
    )
    invalidation: Tuple[str, ...] = (
        "first_15m_ret_lte_-3pct",
        "cannot_reclaim_vwap",
        "theme_ladder_broken",
        "stronger_candidate_replaces",
    )


@dataclass(frozen=True)
class RelayExitProtocol:
    weak_open_stop_pct: float = -0.03
    hard_stop_close_pct: float = -0.035
    profit_extend_pct: float = 0.05
    max_hold_days: int = 3
    default_exit: str = "t2_close_exit"
    extension_exit: str = "t3_clear"


@dataclass(frozen=True)
class RelayPositionRiskProtocol:
    initial_position_pct: float = 0.02
    validated_position_pct: float = 0.04
    max_strategy_position_pct: float = 0.15
    max_new_positions_per_day: int = 1
    halve_after_consecutive_losses: int = 3
    pause_after_consecutive_losses: int = 5
    monthly_loss_stop_pct: float = -0.06


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
        profit_arm_pct=0.10,
        profit_trailing_stop="close_below_ma5",
        protocol_notes=("breakout_only", "do_not_buy_observation_state"),
    ),
    "ywcx": OrdinaryModelProtocol(
        model="ywcx",
        strategy_id="ywcx_manual",
        strategy_version="ywcx_manual_v1_paused",
        min_score=65.0,
        min_gap=-0.03,
        max_gap=0.02,
        min_amount_ma20=50_000_000.0,
        required_tags=("BROKEN_IPO", "NEAR_IPO_LOW", "VOLUME_DRY", "JUST_ABOVE_MA5"),
        any_tags=(),
        blocked_tags=("RISK_FILTERED",),
        planned_position_pct=0.0,
        hard_stop_pct=-0.04,
        structure_stop="close_below_ma5",
        max_hold_days=5,
        profit_arm_pct=0.08,
        profit_trailing_stop="profit_target_or_close_below_ma5",
        paused=True,
        protocol_notes=("paused_until_listing_and_issue_data_fixed",),
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
        profit_arm_pct=0.06,
        profit_trailing_stop="first_sellable_day_reduce_or_profit_exit",
        protocol_notes=("event_model", "do_not_apply_trend_ma_entry", "liquidity_recovery_required"),
    ),
}


RELAY_PROTOCOLS = {
    "candidate": RelayCandidateProtocol(),
    "auction": RelayAuctionProtocol(),
    "intraday": RelayIntradayProtocol(),
    "exit": RelayExitProtocol(),
    "position_risk": RelayPositionRiskProtocol(),
}


def ordinary_model_protocol(model: str) -> OrdinaryModelProtocol:
    key = str(model or "").strip().lower()
    if key not in ORDINARY_MODEL_PROTOCOLS:
        raise KeyError(f"unknown ordinary strategy model: {model}")
    return ORDINARY_MODEL_PROTOCOLS[key]


def ordinary_model_plan_config(model: str) -> Dict[str, Any]:
    return ordinary_model_protocol(model).to_plan_config()


def relay_protocol_config() -> Dict[str, Dict[str, Any]]:
    return {name: asdict(protocol) for name, protocol in RELAY_PROTOCOLS.items()}


def strategy_versions() -> Dict[str, str]:
    versions = {model: spec.strategy_version for model, spec in ORDINARY_MODEL_PROTOCOLS.items()}
    versions["relay"] = RELAY_PROTOCOLS["candidate"].strategy_version
    return versions


def main() -> int:
    print(json.dumps(strategy_versions(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
