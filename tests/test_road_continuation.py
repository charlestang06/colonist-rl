"""Hand-computable public-geometry examples for the early road selector."""

import importlib
import importlib.util
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

from colonist_rl.board import BoardSpec, RulesConfig, Tile
from colonist_rl.contracts import Action, Building, DecisionContext, Observation, PublicPlayer, Road, TradeOffer


def graph(edges, production=()):
    """Each synthetic tile touches only its named site; no engine is involved."""
    size = 1 + max(vertex for edge in edges for vertex in edge)
    return BoardSpec(
        tuple(Tile(i, i, 0, resource, number) for i, (_, resource, number) in enumerate(production)),
        tuple((vertex, 0) for vertex in range(size)),
        tuple(edges),
        tuple((vertex,) for vertex, _, _ in production),
        tuple(() for _ in production),
    )


def view(board=None, *, actions=None, roads=None, buildings=None, roads_left=14, settlements_left=4):
    # Own building 0 excludes vertices 0 and 1. Road 1 opens productive site 2;
    # road 2 reaches barren site 3. A WOOD/6 site has quality 5 + 1.4 = 6.4.
    return Observation(
        player_id=10,
        board=board or graph(((0, 1), (1, 2), (1, 3)), ((2, "WOOD", 6),)),
        rules=RulesConfig.standard(),
        context=DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        players=(PublicPlayer(10, 2, 8, 0, roads_left, settlements_left), PublicPlayer(20, 2, 0, 0)),
        hand=(2, 2, 2, 2, 2),
        development=(0, 0, 0, 0, 0),
        buildings=(Building(0, 10),) if buildings is None else buildings,
        roads=(Road(0, 10),) if roads is None else roads,
        robber_tile=0,
        legal_actions=(Action("BUILD_ROAD", 2), Action("BUILD_ROAD", 1)) if actions is None else actions,
    )


class RoadContinuationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(
            importlib.util.find_spec("colonist_rl.road_continuation"),
            "pure road continuation selector is missing",
        )
        return importlib.import_module("colonist_rl.road_continuation")

    def gains(self, api, observation):
        scores = api.road_access_scores(observation)
        self.assertIsInstance(scores, tuple)
        self.assertTrue(all(isinstance(score, api.RoadAccessScore) for score in scores))
        return {score.edge: score.gain for score in scores}

    def test_zero_gain_selected_is_replaced_by_original_positive_menu_object(self):
        api = self.api()
        observation = view()
        selected, alternative = observation.legal_actions
        before = deepcopy(observation)
        decision = api.choose_road_continuation(observation, selected)
        self.assertIsInstance(decision, api.RoadContinuationDecision)
        self.assertIs(decision.action, alternative)
        self.assertTrue(decision.changed)
        self.assertTrue(decision.reason)
        self.assertEqual(decision.scores, api.road_access_scores(observation))
        self.assertEqual(self.gains(api, observation)[2], 0.0)
        self.assertAlmostEqual(self.gains(api, observation)[1], 3.2)
        self.assertEqual(observation, before)
        with self.assertRaises(FrozenInstanceError):
            decision.changed = False
        with self.assertRaises(FrozenInstanceError):
            decision.scores[0].gain = 999

    def test_selected_positive_is_preserved_even_if_another_road_scores_higher(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (1, 3)), ((2, "WOOD", 2), (3, "WHEAT", 6)))
        observation = view(board)
        selected = observation.legal_actions[1]
        self.assertGreater(self.gains(api, observation)[2], self.gains(api, observation)[1])
        decision = api.choose_road_continuation(observation, selected)
        self.assertIs(decision.action, selected)
        self.assertFalse(decision.changed)

    def test_no_positive_alternative_preserves_selected_and_distinct_fallback_reason(self):
        api = self.api()
        observation = view(graph(((0, 1), (1, 2), (1, 3))))
        selected = observation.legal_actions[0]
        decision = api.choose_road_continuation(observation, selected)
        self.assertIs(decision.action, selected)
        self.assertFalse(decision.changed)
        positive = view()
        retained_positive = api.choose_road_continuation(positive, positive.legal_actions[1])
        self.assertNotEqual(decision.reason, retained_positive.reason)
        singleton = replace(observation, legal_actions=(selected,))
        self.assertIs(api.choose_road_continuation(singleton, selected).action, selected)

    def test_selected_must_be_an_original_legal_object_even_for_fallback(self):
        api = self.api()
        observation = view()
        for foreign in (Action("BUILD_ROAD", 2), Action("END_TURN")):
            with self.subTest(action=foreign), self.assertRaises(ValueError):
                api.choose_road_continuation(observation, foreign)

    def test_enemy_edge_is_impassable_to_high_value_bait(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (2, 3), (3, 4)), ((4, "WHEAT", 6),))
        observation = view(board, actions=(Action("BUILD_ROAD", 1),), roads=(Road(0, 10), Road(2, 20)))
        self.assertEqual(self.gains(api, observation), {1: 0.0})

    def test_enemy_building_stops_transit_along_candidate_route(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (2, 3), (3, 4)), ((4, "WHEAT", 6),))
        observation = view(
            board, actions=(Action("BUILD_ROAD", 1),),
            roads=(Road(0, 10),), buildings=(Building(0, 10), Building(2, 20)),
        )
        self.assertEqual(self.gains(api, observation), {1: 0.0})

    def test_own_building_allows_continuation_through_its_network(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (2, 3), (3, 4)), ((4, "WOOD", 6),))
        observation = view(
            board, actions=(Action("BUILD_ROAD", 2),),
            roads=(Road(0, 10), Road(1, 10)), buildings=(Building(0, 10), Building(2, 10)),
        )
        # Site 4 moves from two empty roads away to one: 6.4 * (1/2 - 1/4).
        self.assertAlmostEqual(self.gains(api, observation)[2], 1.6)

    def test_spacing_excludes_occupied_and_neighboring_sites_for_either_owner(self):
        api = self.api()
        for owner in (10, 20):
            with self.subTest(owner=owner):
                board = graph(((0, 1), (1, 2), (2, 3)), ((2, "WHEAT", 6), (3, "ORE", 6)))
                observation = view(board, actions=(Action("BUILD_ROAD", 1),), buildings=(Building(0, 10), Building(3, owner)))
                self.assertEqual(self.gains(api, observation), {1: 0.0})

    def test_already_connected_settlement_site_does_not_count_as_new_access(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (1, 3)), ((2, "WOOD", 6),))
        observation = view(board, actions=(Action("BUILD_ROAD", 2),), roads=(Road(0, 10), Road(1, 10)))
        self.assertEqual(self.gains(api, observation), {2: 0.0})

    def test_zero_cost_own_roads_reduce_empty_edge_distance(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (2, 3), (3, 4)), ((4, "WOOD", 6),))
        observation = view(board, actions=(Action("BUILD_ROAD", 2),), roads=(Road(0, 10), Road(1, 10)))
        self.assertAlmostEqual(self.gains(api, observation)[2], 1.6)

    def test_three_before_two_after_horizon_and_remaining_road_stock(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)), ((4, "WOOD", 6), (5, "WHEAT", 6)))
        for stock, expected in ((1, 0.0), (2, 0.0), (3, 0.8), (14, 0.8)):
            with self.subTest(roads_left=stock):
                observation = view(board, actions=(Action("BUILD_ROAD", 1),), roads_left=stock)
                # Site 4 moves 3 -> 2 (6.4/8); site 5 stays outside both horizons.
                self.assertAlmostEqual(self.gains(api, observation)[1], expected)

    def test_score_sums_only_three_largest_site_gains(self):
        api = self.api()
        board = graph(
            ((0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (2, 6)),
            ((3, "WOOD", 6), (4, "WOOD", 5), (5, "WOOD", 4), (6, "WOOD", 2)),
        )
        observation = view(board, actions=(Action("BUILD_ROAD", 1),), settlements_left=1)
        # Quality 6.4, 5.4, 4.4, 2.4; each improves 2 -> 1. Stock merely
        # permits a future settlement; the specified score still sums top three.
        self.assertAlmostEqual(self.gains(api, observation)[1], 4.05)

    def test_nominal_quality_uses_weights_distinct_resources_and_robber_tile(self):
        api = self.api()
        board = graph(
            ((0, 1), (1, 2)),
            ((2, "WOOD", 6), (2, "WOOD", 2), (2, "BRICK", 6), (2, "SHEEP", 6),
             (2, "WHEAT", 6), (2, "ORE", 6), (2, "ORE", 0), (2, "DESERT", 0)),
        )
        observation = view(board, actions=(Action("BUILD_ROAD", 1),))
        # 6 wood pips + 5.25 brick + 4.25 sheep + 7 wheat + 6.25 ore + 7 diversity.
        self.assertAlmostEqual(self.gains(api, observation)[1], 17.875)

    def test_equal_positive_gains_choose_numeric_edge_then_original_menu_order(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (1, 3), (1, 4)), ((2, "WOOD", 6), (3, "WOOD", 6)))
        selected, high_id, first_low_id, duplicate = (Action("BUILD_ROAD", edge) for edge in (3, 2, 1, 1))
        observation = view(board, actions=(selected, high_id, first_low_id, duplicate))
        decision = api.choose_road_continuation(observation, selected)
        self.assertTrue(decision.changed)
        self.assertIs(decision.action, first_low_id)

    def test_ineligible_context_stock_network_and_near_win_preserve_original_action(self):
        api = self.api()
        base = view()
        cases = {
            "setup": replace(base, context=replace(base.context, kind="SETUP_ROAD")),
            "free_road": replace(base, context=replace(base.context, free_roads=1)),
            "other_actor": replace(base, context=replace(base.context, actor=20)),
            "other_turn_owner": replace(base, context=replace(base.context, turn_owner=20)),
            "unrolled": replace(base, context=replace(base.context, has_rolled=False)),
            "no_roads": view(roads_left=0),
            "no_settlements": view(settlements_left=0),
            "four_roads_award_proximity": view(
                graph(((0, 1), (1, 2), (1, 3), (0, 4), (0, 5), (0, 6)), ((2, "WOOD", 6),)),
                roads=(Road(0, 10), Road(3, 10), Road(4, 10), Road(5, 10)),
            ),
            "near_win_opponent": replace(base, players=(base.players[0], replace(base.players[1], victory_points=8))),
            "custom_target_near_win": replace(base, rules=replace(base.rules, victory_points=12), players=(base.players[0], replace(base.players[1], victory_points=10))),
            "own_public_near_win": replace(base, players=(replace(base.players[0], victory_points=8), base.players[1])),
            "own_hidden_near_win": replace(
                base, players=(replace(base.players[0], victory_points=6), base.players[1]),
                development=(0, 0, 0, 0, 2),
            ),
            "own_custom_target_near_win": replace(
                base, rules=replace(base.rules, victory_points=12),
                players=(replace(base.players[0], victory_points=10), base.players[1]),
            ),
            "active_offer": replace(
                base, context=replace(
                    base.context,
                    offer=TradeOffer("offer-1", 10, 20, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
                ),
            ),
            "pending_discard": replace(base, context=replace(base.context, discard_remaining=1)),
        }
        for name, observation in cases.items():
            with self.subTest(case=name):
                selected = observation.legal_actions[0]
                decision = api.choose_road_continuation(observation, selected)
                self.assertIs(decision.action, selected)
                self.assertFalse(decision.changed)
                self.assertTrue(decision.reason)
        nonroad = Action("END_TURN")
        observation = replace(base, legal_actions=(nonroad,) + base.legal_actions)
        self.assertIs(api.choose_road_continuation(observation, nonroad).action, nonroad)

    def test_turn_alias_and_three_own_roads_remain_eligible(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (1, 3), (0, 4), (0, 5)), ((2, "WOOD", 6),))
        observation = view(board, roads=(Road(0, 10), Road(3, 10), Road(4, 10)))
        observation = replace(observation, context=replace(observation.context, kind="TURN"))
        self.assertIs(api.choose_road_continuation(observation, observation.legal_actions[0]).action, observation.legal_actions[1])

    def test_invalid_legal_road_claims_are_skipped_or_rejected_without_synthesis(self):
        api = self.api()
        board = graph(((0, 1), (1, 2), (1, 3), (4, 5)), ((2, "WOOD", 6), (5, "WHEAT", 6)))
        for target in (None, -1, 999, 0, 3):
            with self.subTest(target=target):
                invalid = Action("BUILD_ROAD", target)
                observation = view(board)
                observation = replace(observation, legal_actions=observation.legal_actions + (invalid,))
                try:
                    scores = api.road_access_scores(observation)
                    decision = api.choose_road_continuation(observation, observation.legal_actions[0])
                except ValueError:
                    continue  # Explicit malformed-contract rejection is permitted.
                self.assertNotIn(target, {score.edge for score in scores})
                self.assertIs(decision.action, observation.legal_actions[1])


if __name__ == "__main__":
    unittest.main()
