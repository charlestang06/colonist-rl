"""Opt-in public one-trade city preference before an ordinary BUY fallback.

Only the original current-menu trade is executable. The projected city is an
intention; every subsequent act replans from the new actual observation. The
handcraft quality floor does not compare a city's value with an unknown card.

Private planner helpers intentionally share its conservative contracts, public
scorer, canonical ordering and logical work accounting. No projection is ever
promoted to an Observation or sent to a policy or position evaluator.
"""

from __future__ import annotations

from .board import RulesConfig
from .consequences import PUBLIC_BUILD_COSTS
from .contracts import RESOURCES, Action, Observation, TradeOffer
from .public_turn_planner import (
    PlannedStep,
    PublicTurnPlannerPolicy,
    TurnPlan,
    TurnPlannerConfig,
    _Geometry,
    _GeometryCap,
    _PublicContract,
    _Work,
    _context_supported,
    _initial_projection,
    _root_transition,
    _score,
    _step_key,
    _transition,
    _validate_gain_margin,
)


class CityBeforePurchasePolicy:
    """Wrap one parent decision, with a disabled-by-default bounded treatment."""

    def __init__(self, parent: PublicTurnPlannerPolicy, *, enabled: bool = False):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        # Instrumented parents must satisfy the same configuration contract.
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
            "vp_stratum": None,
            "candidate_trades": 0,
            "funded_trades": 0,
            "completed_targets": 0,
            "nodes": 0,
            "geometry_visits": 0,
            "selected_root_index": None,
            "selected_city_target": None,
            "score_before_micro": None,
            "score_after_micro": None,
            "residual_hand": None,
        }

    def plan(self, view: Observation) -> TurnPlan:
        diagnostics = self._diagnostics()
        self.last_diagnostics = diagnostics
        diagnostics["status"] = "parent_error"
        # Parent failures propagate outside the bounded projection handler.
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
            or any(
                value != 0
                for value in (
                    parent_plan.nodes,
                    parent_plan.geometry_visits,
                    parent_plan.evaluator_calls,
                    parent_plan.evaluator_time_ns,
                )
            )
        ):
            return reject("ineligible_parent")
        index = parent_plan.root_index
        if (
            type(index) is not int
            or not 0 <= index < len(view.legal_actions)
            or view.legal_actions[index] is not parent_plan.first_action
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
        if any(a.kind in ("BUILD_CITY", "BUILD_SETTLEMENT") for a in view.legal_actions):
            return reject("direct_build_available")
        roots = tuple(
            (i, action)
            for i, action in enumerate(view.legal_actions)
            if action.kind == "MARITIME_TRADE"
        )
        diagnostics["candidate_trades"] = len(roots)
        work = _Work(self.parent.config)
        diagnostics["status"] = "projection_error"
        try:
            own = next(player for player in view.players if player.player_id == view.player_id)
            known_vp = view.own_victory_points
            threshold = view.rules.victory_points
            diagnostics["known_own_vp"] = known_vp
            diagnostics["vp_stratum"] = (
                "finished" if known_vp >= threshold else
                "near_win" if known_vp == threshold - 1 else "economic"
            )
            if known_vp >= threshold:
                return reject("finished")
            if own.cities_left <= 0 or not any(
                b.owner == view.player_id and b.level == 1 for b in view.buildings
            ):
                return reject("no_city_piece_or_target")
            if not roots:
                return reject("no_maritime_trade")
            if self.parent.config.max_depth < 2:
                return reject("depth_limit")
            initial = _initial_projection(view)
            geometry = _Geometry(view, work)
            shape = geometry.describe(initial)
            before = _score(view, initial, shape)
            diagnostics["score_before_micro"] = before
            best = None
            city_cost = PUBLIC_BUILD_COSTS["BUILD_CITY"]
            for root_index, action in roots:
                if work.nodes >= self.parent.config.max_nodes:
                    return reject("node_limit")
                work.nodes += 1
                after_trade = _root_transition(view, initial, action, shape)
                if after_trade is None:
                    # A contradictory actual menu invalidates the entire treatment.
                    raise _PublicContract("maritime root")
                if any(h < c for h, c in zip(after_trade.hand, city_cost)):
                    continue
                diagnostics["funded_trades"] += 1
                after_shape = geometry.describe(after_trade)
                trade_step = PlannedStep(action.kind, action.target, action.give, action.receive)
                for target in sorted(after_shape.cities):
                    if work.nodes >= self.parent.config.max_nodes:
                        return reject("node_limit")
                    work.nodes += 1
                    city_step = PlannedStep("BUILD_CITY", target)
                    completed = _transition(view, after_trade, city_step, after_shape)
                    if completed is None:
                        raise _PublicContract("city continuation")
                    completed_shape = geometry.describe(completed)
                    score = _score(view, completed, completed_shape)
                    diagnostics["completed_targets"] += 1
                    order = (-score, _step_key(trade_step), _step_key(city_step), root_index)
                    if best is None or order < best[0]:
                        best = (order, root_index, action, trade_step, city_step, score, completed)
            # No partial winner is accepted: all roots and targets have completed.
            if best is None:
                return reject("no_funded_city")
            _, root_index, action, trade_step, city_step, score, completed = best
            if score <= before + self.parent.gain_margin_micro:
                return reject("insufficient_gain")
            diagnostics.update(
                status="city_before_purchase",
                selected_root_index=root_index,
                selected_city_target=city_step.target,
                score_after_micro=score,
                residual_hand=completed.hand,
            )
            return TurnPlan(
                first_action=action,
                root_index=root_index,
                steps=(trade_step, city_step),
                score_before_micro=before,
                score_after_micro=score,
                nodes=work.nodes,
                geometry_visits=work.geometry_visits,
                reason="city_before_purchase",
                search_stop="one_trade_city_complete",
            )
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
