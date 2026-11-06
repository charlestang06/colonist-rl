"""Frozen public engine view for a direct, currently legal point win."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from colonist_rl.adapters.observation_io import view_from_dict
from colonist_rl.public_turn_planner import (
    PlannedStep,
    PublicTurnPlannerPolicy,
    TurnPlannerConfig,
)


def _captured_view():
    record = json.loads(
        (Path(__file__).parent / "fixtures/public_direct_settlement_win.json").read_text()
    )
    envelope = record["observation"]
    canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert record["provenance"]["source_commit"] == ("2914cba3ff84e5cb7fce7af70d4b603783e9190a")
    assert record["provenance"]["observation_sha256"] == (
        "c08165a99fb23f9ca65cf61a0cb4e5a83600b9b0e1a15a772a1fee2f38cf952a"
    )
    assert (
        hashlib.sha256(canonical.encode()).hexdigest() == record["provenance"]["observation_sha256"]
    )
    return view_from_dict(envelope)


def test_current_legal_settlement_wins_before_longer_winning_trade_route():
    view = _captured_view()
    assert view.context.version == 365
    assert view.player_id == 3
    assert view.hand == (1, 4, 2, 2, 1)
    assert view.own_victory_points == 9
    direct_index = next(
        i
        for i, action in enumerate(view.legal_actions)
        if action.kind == "BUILD_SETTLEMENT" and action.target == 48
    )

    plan = PublicTurnPlannerPolicy(TurnPlannerConfig(), gain_margin_micro=10_000).plan(view)

    assert plan.first_action is view.legal_actions[direct_index]
    assert plan.root_index == direct_index
    assert plan.steps == (PlannedStep("BUILD_SETTLEMENT", 48),)
    assert (plan.reason, plan.search_stop) == ("planned", "current_legal_win")
    assert plan.search_diagnostics is not None
    assert plan.search_diagnostics.root_attempts == 13
    assert plan.search_diagnostics.future_attempts == 0
    assert plan.search_diagnostics.selected_plan_depth == 1
    assert plan.nodes == 13


def test_current_win_does_not_bypass_later_root_contract_validation():
    view = _captured_view()
    end_index = next(i for i, action in enumerate(view.legal_actions) if action.kind == "END_TURN")
    altered = replace(
        view,
        legal_actions=tuple(
            replace(action, resource="WOOD") if i == end_index else action
            for i, action in enumerate(view.legal_actions)
        ),
    )

    plan = PublicTurnPlannerPolicy().plan(altered)

    assert plan.reason == "public_contract_mismatch"
    assert plan.search_diagnostics is not None
    assert plan.search_diagnostics.root_attempts == 13
    assert plan.search_diagnostics.invalid_transitions == 1
    assert plan.search_diagnostics.selected_plan_depth == 0


def test_current_win_waits_for_complete_root_budget_validation():
    view = _captured_view()

    plan = PublicTurnPlannerPolicy(TurnPlannerConfig(max_nodes=1)).plan(view)

    assert (plan.reason, plan.search_stop) == ("root_validation_limit", "node_limit")
    assert plan.search_diagnostics is not None
    assert plan.search_diagnostics.root_attempts == 1
    assert plan.search_diagnostics.selected_plan_depth == 0
