"""Opt-in public comparison of a card purchase, pass and one-trade point build.

The card credit is a broad initial-deck hypothesis, not a prediction of the
hidden draw. Only original current-menu actions can be executed. A projected
next build is an intention and must be rechecked after the selected trade.
"""

from __future__ import annotations

from dataclasses import replace

from .board import RulesConfig
from .consequences import PUBLIC_BUILD_COSTS
from .contracts import DEVELOPMENT_CARDS, RESOURCES, ZERO, Action, Observation, TradeOffer
from .public_turn_planner import (
    PlannedStep,
    PublicTurnPlannerPolicy,
    TurnPlan,
    TurnPlannerConfig,
    _context_supported,
    _Geometry,
    _GeometryCap,
    _initial_projection,
    _PublicContract,
    _root_transition,
    _score,
    _step_key,
    _transition,
    _validate_gain_margin,
    _Work,
)

# Score units are frozen in docs/public-option-purchase-design-2026-11-06.md.
_CARD_OPTION_MICRO = dict(zip(
    DEVELOPMENT_CARDS, (60_000_000, 50_000_000, 20_000_000, 30_000_000, 100_000_000)
))
_PRESERVED_RESOURCE_MICRO = 35_000_000
_POINT_BUILDS = ("BUILD_CITY", "BUILD_SETTLEMENT")


def _card_option_credit(view: Observation) -> int:
    """Broad initial-deck prior with an integer public horizon factor."""
    counts = view.rules.development_cards
    total = sum(counts)
    if total <= 0:
        raise _PublicContract("empty development deck rules")
    weighted = sum(
        count * _CARD_OPTION_MICRO[kind]
        for kind, count in zip(DEVELOPMENT_CARDS, counts)
    )
    horizon = min(6, max(0, view.rules.victory_points - view.own_victory_points - 2))
    return weighted * (6 + horizon) // (6 * total)


def _project_purchase(view: Observation, state, action: Action):
    """Project only the known resource cost; no card type or deck draw."""
    if (
        action.kind != "BUY_DEVELOPMENT_CARD"
        or action.target is not None
        or any(value is not None for value in (
            action.resource, action.second_resource, action.partner, action.offer_id
        ))
        or action.give != ZERO
        or action.receive != ZERO
    ):
        return None
    cost = PUBLIC_BUILD_COSTS["BUY_DEVELOPMENT_CARD"]
    if any(h < c for h, c in zip(state.hand, cost)):
        return None
    hand = tuple(h - c for h, c in zip(state.hand, cost))
    lower = tuple(b + c for b, c in zip(state.bank_lower, cost))
    upper = tuple(b + c for b, c in zip(state.bank_upper, cost))
    if any(not 0 <= l <= u <= supply for l, u, supply in zip(
        lower, upper, view.rules.bank_resources
    )):
        return None
    return replace(state, hand=hand, bank_lower=lower, bank_upper=upper)


def _nearest_point_deficit(state, shape) -> int | None:
    goals = (
        ("BUILD_CITY", shape.cities, state.cities_left),
        ("BUILD_SETTLEMENT", shape.sites, state.settlements_left),
    )
    deficits = [
        sum(max(0, cost - held) for cost, held in zip(PUBLIC_BUILD_COSTS[kind], state.hand))
        for kind, targets, pieces in goals if pieces > 0 and targets
    ]
    return min(deficits) if deficits else None


def _current_winning_root_valid(view: Observation, action: Action) -> bool:
    """Check obvious public invariants of an engine-certified point build."""
    if (
        action.kind not in _POINT_BUILDS
        or type(action.target) is not int
        or not 0 <= action.target < len(view.board.vertices)
        or any(value is not None for value in (
            action.resource, action.second_resource, action.partner, action.offer_id
        ))
        or action.give != ZERO
        or action.receive != ZERO
        or any(h < c for h, c in zip(view.hand, PUBLIC_BUILD_COSTS[action.kind]))
    ):
        return False
    own = next((player for player in view.players if player.player_id == view.player_id), None)
    if own is None:
        return False
    if action.kind == "BUILD_CITY":
        return own.cities_left > 0 and any(
            building.vertex == action.target
            and building.owner == view.player_id
            and building.level == 1
            for building in view.buildings
        )
    return own.settlements_left > 0 and not any(
        building.vertex == action.target for building in view.buildings
    )


class PublicOptionPurchasePolicy:
    """Retain the raw parent except at its ordinary BUY fallback."""

    def __init__(self, parent: PublicTurnPlannerPolicy, *, enabled: bool = False):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        TurnPlannerConfig(
            max_depth=parent.config.max_depth,
            max_nodes=parent.config.max_nodes,
            beam_width=parent.config.beam_width,
            max_geometry_visits=parent.config.max_geometry_visits,
            seed=parent.config.seed,
        )
        _validate_gain_margin(parent.gain_margin_micro)
        self.parent = parent
        self.enabled = enabled
        self.last_diagnostics: dict = self._diagnostics()

    def _diagnostics(self) -> dict:
        return {
            "enabled": self.enabled,
            "status": "not_called",
            "parent_reason": None,
            "parent_root_index": None,
            "parent_kind": None,
            "known_own_vp": None,
            "card_option_credit_micro": None,
            "buy_hand_after_cost": None,
            "pass_reserve_micro": None,
            "score_before_micro": None,
            "score_buy_micro": None,
            "score_pass_micro": None,
            "score_trade_build_micro": None,
            "candidate_trades": 0,
            "completed_targets": 0,
            "nodes": 0,
            "geometry_visits": 0,
            "selected_root_index": None,
            "selected_build_kind": None,
            "selected_build_target": None,
            "residual_hand": None,
        }

    def plan(self, view: Observation) -> TurnPlan:
        diagnostics = self._diagnostics()
        self.last_diagnostics = diagnostics
        diagnostics["status"] = "parent_error"
        parent_plan = self.parent.plan(view)
        diagnostics.update(
            parent_reason=parent_plan.reason,
            parent_root_index=parent_plan.root_index,
            parent_kind=parent_plan.first_action.kind,
        )

        def reject(status: str) -> TurnPlan:
            diagnostics["status"] = status
            return parent_plan

        if not self.enabled:
            return reject("disabled")
        if self.parent.evaluator is not None:
            return reject("evaluator_configured")
        if (
            parent_plan.reason != "unsupported_heuristic_action"
            or parent_plan.first_action.kind != "BUY_DEVELOPMENT_CARD"
            or any(value != 0 for value in (
                parent_plan.nodes, parent_plan.geometry_visits,
                parent_plan.evaluator_calls, parent_plan.evaluator_time_ns,
            ))
        ):
            return reject("ineligible_parent")
        buy_index = parent_plan.root_index
        if (
            type(buy_index) is not int
            or not 0 <= buy_index < len(view.legal_actions)
            or view.legal_actions[buy_index] is not parent_plan.first_action
        ):
            return reject("parent_root_mismatch")
        if not _context_supported(view):
            return reject("unsupported_context")
        if view.rules != RulesConfig.standard():
            return reject("unsupported_rules")
        if any(
            port.ratio not in (2, 3, 4) or port.resource not in (None, *RESOURCES)
            for port in view.board.ports
        ):
            return reject("public_contract_mismatch")
        try:
            known_vp = view.own_victory_points
        except StopIteration:
            return reject("missing_own_player")
        diagnostics["known_own_vp"] = known_vp
        if known_vp >= view.rules.victory_points:
            return reject("finished")
        direct = [(i, action) for i, action in enumerate(view.legal_actions)
                  if action.kind in _POINT_BUILDS]
        if direct and known_vp + 1 >= view.rules.victory_points:
            if any(not _current_winning_root_valid(view, action) for _, action in direct):
                return reject("public_contract_mismatch")
            index, action = min(
                direct,
                key=lambda pair: (_POINT_BUILDS.index(pair[1].kind), pair[1].target, pair[0]),
            )
            diagnostics.update(status="public_option_legal_win", selected_root_index=index,
                               selected_build_kind=action.kind,
                               selected_build_target=action.target)
            return TurnPlan(
                action, index, (PlannedStep(action.kind, action.target),), 0, 0,
                0, 0, "public_option_legal_win", "current_legal_win",
            )
        if direct:
            return reject("direct_build_available")
        passes = [(i, action) for i, action in enumerate(view.legal_actions)
                  if action.kind == "END_TURN"]
        if len(passes) != 1:
            return reject("missing_or_ambiguous_pass")
        if self.parent.config.max_depth < 2:
            return reject("depth_limit")
        pass_index, pass_action = passes[0]
        trades = sorted(
            ((i, action) for i, action in enumerate(view.legal_actions)
             if action.kind == "MARITIME_TRADE"),
            key=lambda pair: (
                _step_key(PlannedStep(pair[1].kind, pair[1].target,
                                      pair[1].give, pair[1].receive)), pair[0]
            ),
        )
        diagnostics["candidate_trades"] = len(trades)
        work = _Work(self.parent.config)
        diagnostics["status"] = "projection_error"
        try:
            initial = _initial_projection(view)
            geometry = _Geometry(view, work)
            initial_shape = geometry.describe(initial)
            before = _score(view, initial, initial_shape)
            diagnostics["score_before_micro"] = before
            if work.nodes >= self.parent.config.max_nodes:
                return reject("node_limit")
            work.nodes += 1
            after_buy = _project_purchase(view, initial, parent_plan.first_action)
            if after_buy is None:
                raise _PublicContract("purchase root")
            buy_shape = geometry.describe(after_buy)
            card_credit = _card_option_credit(view)
            buy_score = _score(view, after_buy, buy_shape) + card_credit
            diagnostics.update(card_option_credit_micro=card_credit,
                               buy_hand_after_cost=after_buy.hand,
                               score_buy_micro=buy_score)
            if work.nodes >= self.parent.config.max_nodes:
                return reject("node_limit")
            work.nodes += 1
            if _root_transition(view, initial, pass_action, initial_shape) is None:
                raise _PublicContract("pass root")
            before_deficit = _nearest_point_deficit(initial, initial_shape)
            after_deficit = _nearest_point_deficit(after_buy, buy_shape)
            added_deficit = (
                max(0, after_deficit - before_deficit)
                if before_deficit is not None and after_deficit is not None else 0
            )
            reserve = added_deficit * _PRESERVED_RESOURCE_MICRO
            pass_score = before + reserve
            diagnostics.update(pass_reserve_micro=reserve, score_pass_micro=pass_score)
            best = None
            for root_index, action in trades:
                if work.nodes >= self.parent.config.max_nodes:
                    return reject("node_limit")
                work.nodes += 1
                after_trade = _root_transition(view, initial, action, initial_shape)
                if after_trade is None:
                    raise _PublicContract("maritime root")
                trade_shape = geometry.describe(after_trade)
                trade_step = PlannedStep(action.kind, action.target,
                                         action.give, action.receive)
                for kind, targets, pieces in (
                    ("BUILD_CITY", trade_shape.cities, after_trade.cities_left),
                    ("BUILD_SETTLEMENT", trade_shape.sites, after_trade.settlements_left),
                ):
                    if pieces <= 0 or any(
                        have < cost for have, cost in zip(
                            after_trade.hand, PUBLIC_BUILD_COSTS[kind]
                        )
                    ):
                        continue
                    for target in sorted(targets):
                        if work.nodes >= self.parent.config.max_nodes:
                            return reject("node_limit")
                        work.nodes += 1
                        build_step = PlannedStep(kind, target)
                        completed = _transition(view, after_trade, build_step, trade_shape)
                        if completed is None:
                            raise _PublicContract("build continuation")
                        score = _score(view, completed, geometry.describe(completed))
                        diagnostics["completed_targets"] += 1
                        winning = (
                            known_vp + completed.building_point_gain
                            >= view.rules.victory_points
                        )
                        order = (
                            -winning, -score, _step_key(trade_step),
                            _step_key(build_step), root_index,
                        )
                        if best is None or order < best[0]:
                            best = (order, root_index, action, trade_step,
                                    build_step, score, completed, winning)
            if best is not None:
                _, root_index, action, trade_step, build_step, score, completed, winning = best
                diagnostics["score_trade_build_micro"] = score
                if winning or score > max(buy_score, pass_score) + self.parent.gain_margin_micro:
                    diagnostics.update(
                        status="public_option_trade_build", selected_root_index=root_index,
                        selected_build_kind=build_step.kind,
                        selected_build_target=build_step.target,
                        residual_hand=completed.hand,
                    )
                    return TurnPlan(
                        action, root_index, (trade_step, build_step), before, score,
                        work.nodes, work.geometry_visits,
                        "public_option_trade_build", "one_trade_build_complete",
                    )
            if pass_score > buy_score + self.parent.gain_margin_micro:
                diagnostics.update(status="public_option_pass", selected_root_index=pass_index)
                return TurnPlan(
                    pass_action, pass_index, (PlannedStep("END_TURN"),), before,
                    pass_score, work.nodes, work.geometry_visits,
                    "public_option_pass", "pass_complete",
                )
            return reject("buy_preferred")
        except _GeometryCap:
            return reject("geometry_limit")
        except _PublicContract:
            return reject("public_contract_mismatch")
        except StopIteration:
            return reject("missing_own_player")
        finally:
            diagnostics["nodes"] = work.nodes
            diagnostics["geometry_visits"] = work.geometry_visits

    def act(self, view: Observation) -> Action:
        return self.plan(view).first_action

    def propose(self, view: Observation) -> TradeOffer | None:
        self.last_diagnostics = self._diagnostics()
        self.last_diagnostics["status"] = "parent_proposal"
        return self.parent.propose(view)
