"""Opt-in BUY fallback regression tests; public projections only, never engine replay.

This external tests-first file is intended for Root to copy beside fixtures/ and run.
No closed known-win fixture, probe, outcome, model, simulator, or old test is imported.
"""

import inspect
import json
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch


FIXTURE = Path(__file__).with_name("fixtures") / "train018-seat0-own95-before.json"
FLAG = "search_build_win_before_buy"


def modules():
    from colonist_rl import public_turn_planner as planner
    from colonist_rl.adapters.observation_io import view_from_dict

    return planner, view_from_dict


def load_view():
    _, decode = modules()
    return decode(json.loads(FIXTURE.read_text()))


class BuyWinningPrefixTests(unittest.TestCase):
    def assert_buy(self, observation, plan):
        buy = next(a for a in observation.legal_actions if a.kind == "BUY_DEVELOPMENT_CARD")
        self.assertIs(plan.first_action, buy)
        self.assertEqual(plan.steps, ())
        self.assertIs(observation.legal_actions[plan.root_index], buy)

    def assert_public_build_win(self, observation, plan):
        """Check existing public transitions, without certifying future engine menus."""
        p, _ = modules()
        self.assertEqual(plan.reason, "planned")
        self.assertIs(plan.first_action, observation.legal_actions[plan.root_index])
        self.assertNotEqual(plan.first_action.kind, "BUY_DEVELOPMENT_CARD")
        self.assertTrue(plan.steps)
        first = plan.steps[0]
        action = plan.first_action
        self.assertEqual(
            (first.kind, first.target, first.give, first.receive),
            (action.kind, action.target, action.give, action.receive),
        )
        state = p._initial_projection(observation)
        geometry = p._Geometry(observation, p._Work(p.TurnPlannerConfig()))
        for index, step in enumerate(plan.steps):
            self.assertNotEqual(step.kind, "BUY_DEVELOPMENT_CARD")
            shape = geometry.describe(state)
            state = (
                p._root_transition(observation, state, action, shape)
                if index == 0
                else p._transition(observation, state, step, shape)
            )
            self.assertIsNotNone(state)
        self.assertTrue(any(s.kind in ("BUILD_CITY", "BUILD_SETTLEMENT") for s in plan.steps))
        self.assertGreaterEqual(
            observation.own_victory_points + state.building_point_gain,
            observation.rules.victory_points,
        )

    def test_01_opt_in_keyword_exists_and_defaults_to_false(self):
        # Catches missing opt-in API; this is the first intentional Root RED assertion.
        p, _ = modules()
        for function in (p.plan_turn, p.PublicTurnPlannerPolicy):
            parameters = inspect.signature(function).parameters
            self.assertIn(FLAG, parameters, "winning BUY prefix opt-in is missing")
            self.assertIs(parameters[FLAG].default, False)
            self.assertEqual(parameters[FLAG].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_02_default_and_explicit_false_preserve_zero_search_buy(self):
        # Catches accidental default behavior changes or work charged before opt-in.
        p, _ = modules()
        observation = load_view()
        self.assertEqual(observation.own_victory_points, 9)
        self.assertEqual(p.HeuristicViewPolicy().act(observation).kind, "BUY_DEVELOPMENT_CARD")
        default = p.plan_turn(observation)
        explicit = p.plan_turn(observation, search_build_win_before_buy=False)
        self.assertEqual(default, explicit)
        for plan in (default, explicit):
            self.assert_buy(observation, plan)
            self.assertEqual(plan.reason, "unsupported_heuristic_action")
            self.assertEqual((plan.nodes, plan.geometry_visits, plan.evaluator_calls), (0, 0, 0))

    def test_03_opt_in_finds_any_original_first_action_with_public_build_win(self):
        # Catches both unsupported-BUY early exit and missing-H-baseline early exit.
        p, _ = modules()
        observation = load_view()
        original_transition = p._root_transition

        def supported_root_only(root, state, action, shape):
            self.assertNotEqual(action.kind, "BUY_DEVELOPMENT_CARD", "BUY has no projection")
            return original_transition(root, state, action, shape)

        with patch.object(p, "_root_transition", side_effect=supported_root_only):
            plan = p.plan_turn(observation, search_build_win_before_buy=True)
        self.assertGreater(plan.nodes, 0)
        self.assert_public_build_win(observation, plan)

    def test_04_shallow_or_nonwinning_positive_gain_keeps_buy(self):
        # Catches ordinary evaluator gains overriding the unsupported BUY fallback.
        p, _ = modules()
        observation = load_view()
        seen = []

        def trade_gain(root, state, handcraft):
            score = 10**30 if state.hand != root.hand else 0
            seen.append(score)
            return score

        shallow = p.plan_turn(
            observation, p.TurnPlannerConfig(max_depth=1),
            evaluator=trade_gain, search_build_win_before_buy=True,
        )
        self.assertIn(10**30, seen)
        self.assert_buy(observation, shallow)

        # Manufactured one-hex public view, following basic-test construction only.
        # BUY is fixed as H fallback to isolate the branch; all search remains real.
        from colonist_rl.board import BoardSpec, RulesConfig
        from colonist_rl.contracts import Action, Building, DecisionContext, Observation, PublicPlayer

        hand = (2, 2, 1, 1, 1)
        ordinary = Observation(
            10, BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,)), RulesConfig.standard(),
            DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
            (PublicPlayer(10, 9, sum(hand), 0, settlements_left=0, cities_left=0),
             PublicPlayer(20, 2, 30, 0), PublicPlayer(30, 0, 0, 0), PublicPlayer(40, 0, 0, 0)),
            hand, (0, 0, 0, 0, 0), (Building(0, 10),), (), 0,
            (Action("BUY_DEVELOPMENT_CARD"), Action("BUILD_ROAD", 0), Action("END_TURN")),
        )
        gain_states = []

        def road_gain(root, state, handcraft):
            if state.roads:
                gain_states.append(state)
            return 10**30 * len(state.roads)

        with patch.object(p.HeuristicViewPolicy, "act", return_value=ordinary.legal_actions[0]):
            plan = p.plan_turn(
                ordinary, p.TurnPlannerConfig(max_depth=2), evaluator=road_gain,
                search_build_win_before_buy=True,
            )
        self.assertTrue(gain_states)
        self.assertTrue(all(s.building_point_gain == 0 for s in gain_states))
        self.assert_buy(ordinary, plan)

    def test_05_zero_node_or_geometry_budget_keeps_buy(self):
        # Catches cap bypasses or returning uncertified root intentions.
        p, _ = modules()
        observation = load_view()
        for config in (p.TurnPlannerConfig(max_nodes=0), p.TurnPlannerConfig(max_geometry_visits=0)):
            with self.subTest(config=config):
                plan = p.plan_turn(observation, config, search_build_win_before_buy=True)
                self.assert_buy(observation, plan)
                self.assertLessEqual(plan.nodes, config.max_nodes)
                self.assertLessEqual(plan.geometry_visits, config.max_geometry_visits)

    def test_06_context_rules_point_gate_and_other_unsupported_actions_stay_closed(self):
        # Catches widening the exception beyond standard normal last-point BUY roots.
        p, _ = modules()
        observation = load_view()
        own = next(player for player in observation.players if player.player_id == observation.player_id)
        lower_points = replace(observation, players=tuple(
            replace(player, victory_points=own.victory_points - 1) if player is own else player
            for player in observation.players
        ))
        cases = (
            (replace(observation, context=replace(observation.context, has_rolled=False)), "unsupported_context"),
            (replace(observation, context=replace(observation.context, free_roads=1)), "unsupported_context"),
            (replace(observation, context=replace(observation.context, actor=1)), "unsupported_context"),
            (replace(observation, rules=replace(observation.rules, victory_points=12)), "unsupported_rules"),
            (lower_points, "unsupported_heuristic_action"),
        )
        for altered, reason in cases:
            with self.subTest(reason=reason, observation=altered):
                buy = next(a for a in altered.legal_actions if a.kind == "BUY_DEVELOPMENT_CARD")
                with patch.object(p.HeuristicViewPolicy, "act", return_value=buy):
                    plan = p.plan_turn(altered, search_build_win_before_buy=True)
                self.assert_buy(altered, plan)
                self.assertEqual(plan.reason, reason)
                self.assertEqual((plan.nodes, plan.geometry_visits, plan.evaluator_calls), (0, 0, 0))
        from colonist_rl.contracts import Action

        knight = Action("PLAY_KNIGHT")
        altered = replace(observation, legal_actions=(knight, *observation.legal_actions))
        with patch.object(p.HeuristicViewPolicy, "act", return_value=knight):
            plan = p.plan_turn(altered, search_build_win_before_buy=True)
        self.assertIs(plan.first_action, knight)
        self.assertEqual(plan.reason, "unsupported_heuristic_action")
        self.assertEqual((plan.nodes, plan.geometry_visits), (0, 0))

    def test_07_input_is_unchanged_and_first_action_retains_original_identity(self):
        # Catches mutation, detached root actions, and nondeterministic intentions.
        p, _ = modules()
        observation = load_view()
        before = deepcopy(asdict(observation))
        action_ids = tuple(map(id, observation.legal_actions))
        first = p.plan_turn(observation, search_build_win_before_buy=True)
        second = p.plan_turn(observation, search_build_win_before_buy=True)
        self.assert_public_build_win(observation, first)
        self.assertEqual(first, second)
        self.assertEqual(asdict(observation), before)
        self.assertEqual(tuple(map(id, observation.legal_actions)), action_ids)
        self.assertIs(first.first_action, observation.legal_actions[first.root_index])

    def test_08_adverse_integer_scores_and_huge_margin_cannot_displace_build_win(self):
        # Catches score/margin ordering ahead of the existing known-building-win bit.
        p, _ = modules()
        observation = load_view()
        plan = p.plan_turn(
            observation, evaluator=lambda root, state, handcraft: -state.building_point_gain * 10**30,
            gain_margin_micro=10**100, search_build_win_before_buy=True,
        )
        self.assert_public_build_win(observation, plan)
        self.assertLess(plan.score_after_micro, plan.score_before_micro)
        self.assertGreater(plan.evaluator_calls, 0)

    def test_09_policy_propagates_opt_in_and_both_entry_points_require_strict_bool(self):
        # Catches forgotten wrapper propagation and permissive truthiness conversion.
        p, _ = modules()
        observation = load_view()
        self.assert_buy(observation, p.PublicTurnPlannerPolicy().plan(observation))
        enabled = p.PublicTurnPlannerPolicy(search_build_win_before_buy=True)
        self.assert_public_build_win(observation, enabled.plan(observation))
        self.assertIs(enabled.act(observation), enabled.plan(observation).first_action)
        for invalid in (0, 1, None, "true", [], object()):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    p.PublicTurnPlannerPolicy(search_build_win_before_buy=invalid)
                with self.assertRaises(ValueError):
                    p.plan_turn(observation, search_build_win_before_buy=invalid)
