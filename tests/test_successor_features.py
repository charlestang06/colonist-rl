"""Known public successors are candidate features, never future legal menus."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from colonist_rl.board import BoardSpec, Port, RulesConfig
from colonist_rl.contracts import Action, Building, DecisionContext, Observation, PublicPlayer, Road
from colonist_rl.settlement_sites import connectable_road_edges, connected_settlement_sites
from colonist_rl.successor_features import (
    SUCCESSOR_COLUMNS,
    SUCCESSOR_DIM,
    SUCCESSOR_SCHEMA,
    successor_feature_values,
    successor_features_for_menu,
)


def view(*, hand=(4, 1, 1, 3, 3), buildings=(), roads=None, robber=99,
         context=None, actions=None):
    board = BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,))
    board = replace(board, ports=(Port((0, 1), "WOOD", 2),))
    roads = (Road(0, 10),) if roads is None else roads
    actions = actions or (
        Action("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 1, 0, 0)),
        Action("BUILD_SETTLEMENT", target=0),
        Action("BUY_DEVELOPMENT_CARD"),
        Action("END_TURN"),
    )
    return Observation(
        10, board, RulesConfig.standard(),
        context or DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        (PublicPlayer(10, 2, sum(hand), 0), *(PublicPlayer(i, 0, 0, 0) for i in (20, 30, 40))),
        hand, (0, 0, 0, 0, 0), buildings, roads, robber, actions,
    )


def _native_topology():
    # The pinned engine's standard map visits tiles in this order. Only public
    # topology matters for these synthetic road and settlement observations.
    return BoardSpec.from_hexes((
        (0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1), (-2, 0),
        (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1),
    ))


def _settlement_cut_view(*, actor_roads=(70, 68, 28, 10, 0),
                         opponent_roads=(25, 19, 5, 24, 29, 64),
                         actor_building=53, opponent_building=21,
                         actor_length=5, opponent_length=6, actor_vp=7,
                         holder=20, third_length=0, third_roads=(),
                         third_building=None, extra_opponent_building=None):
    board = _native_topology()
    buildings = (Building(actor_building, 10), Building(opponent_building, 20))
    if third_building is not None:
        buildings += (Building(third_building, 30),)
    if extra_opponent_building is not None:
        buildings += (Building(extra_opponent_building, 20),)
    roads = tuple(Road(edge, 10) for edge in actor_roads) + tuple(
        Road(edge, 20) for edge in opponent_roads
    ) + tuple(Road(edge, 30) for edge in third_roads)
    observation = replace(
        view(hand=(1, 1, 1, 1, 0), actions=(Action("BUILD_SETTLEMENT", target=0),)),
        board=board, buildings=buildings, roads=roads,
        players=(
            PublicPlayer(10, actor_vp, 4, 0, roads_left=15 - len(actor_roads),
                         settlements_left=4, longest_road=actor_length,
                         has_longest_road=holder == 10),
            PublicPlayer(20, 3, 0, 0, roads_left=15 - len(opponent_roads),
                         settlements_left=4 - int(extra_opponent_building is not None),
                         longest_road=opponent_length,
                         has_longest_road=holder == 20),
            PublicPlayer(30, 0, 0, 0, roads_left=15 - len(third_roads),
                         settlements_left=5 - int(third_building is not None),
                         longest_road=third_length),
            PublicPlayer(40, 0, 0, 0),
        ),
    )
    assert 0 in connected_settlement_sites(board, buildings, roads, 10)
    return observation


def test_trade_build_buy_and_end_encode_known_hand_and_recipe_consequences():
    observation = view()
    trade, build, buy, end = (
        successor_feature_values(observation, action) for action in observation.legal_actions
    )
    assert trade["hand_known"] == 1
    assert [trade[f"hand_after_{r}"] for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")] == [
        0, 0.1, 0.2, 0.3, 0.3
    ]
    assert trade["settlement_deficit_after_WOOD"] == 0.25
    assert trade["bank_trade_ratio"] == 1
    assert build["point_gain_lower_bound"] == 1
    assert build["own_vp_after"] == 0.3
    assert build["settlement_pieces_after"] == 0.8
    assert buy["hand_after_SHEEP"] == 0
    assert buy["hand_after_WHEAT"] == 0.2
    assert buy["hand_after_ORE"] == 0.2
    assert buy["unknown_dev_draw"] == 1
    assert buy["point_gain_lower_bound"] == 0
    assert buy["point_outcome_uncertain"] == 1
    assert end["hand_after_WOOD"] == end["hand_before_WOOD"] == 0.4
    assert end["unknown_dev_draw"] == 0


def test_settlement_and_city_production_port_and_piece_effects_are_public():
    observation = view(actions=(Action("BUILD_SETTLEMENT", target=0),))
    got = successor_feature_values(observation, observation.legal_actions[0])
    assert got["production_nominal_delta_WHEAT"] == pytest.approx(5 / 36)
    assert got["production_unblocked_delta_WHEAT"] == pytest.approx(5 / 36)
    assert got["port_ratio_after_WOOD"] == 0.5
    assert got["connected_sites_after"] < got["connected_sites_before"]
    blocked = replace(observation, robber_tile=0)
    blocked_got = successor_feature_values(blocked, blocked.legal_actions[0])
    assert blocked_got["production_nominal_delta_WHEAT"] == pytest.approx(5 / 36)
    assert blocked_got["production_unblocked_delta_WHEAT"] == 0

    city = view(hand=(0, 0, 0, 2, 3), buildings=(Building(3, 10),), roads=(),
                actions=(Action("BUILD_CITY", target=3),))
    upgraded = successor_feature_values(city, city.legal_actions[0])
    assert upgraded["point_gain_lower_bound"] == 1
    assert upgraded["production_nominal_delta_WHEAT"] == pytest.approx(5 / 36)
    assert upgraded["settlement_pieces_after"] == 1
    assert upgraded["city_pieces_after"] == 0.75
    assert upgraded["hand_after_WHEAT"] == upgraded["hand_after_ORE"] == 0


@pytest.mark.parametrize("phase", ["PLAY_TURN", "BUILD_INITIAL_ROAD"])
def test_free_or_setup_road_uses_a_piece_without_spending_hand(phase):
    context = DecisionContext(phase, 10, 10, 1, has_rolled=phase == "PLAY_TURN",
                              free_roads=int(phase == "PLAY_TURN"))
    observation = view(hand=(0, 0, 0, 0, 0), context=context,
                       actions=(Action("BUILD_ROAD", target=1),))
    got = successor_feature_values(observation, observation.legal_actions[0])
    assert got["hand_known"] == 1
    assert all(got[f"hand_delta_{r}"] == 0 for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"))
    assert got["road_pieces_after"] == pytest.approx(14 / 15)


def test_second_setup_settlement_grants_its_public_adjacent_starting_resources():
    board = BoardSpec.from_hexes(((0, 0), (2, 0)), ("WHEAT", "BRICK"), (6, 5))
    target = board.tile_vertices[0][0]
    first_vertex = board.tile_vertices[1][0]
    context = DecisionContext("BUILD_INITIAL_SETTLEMENT", 10, 10, 1)
    first = replace(
        view(hand=(0, 0, 0, 0, 0), buildings=(), roads=(), context=context,
             actions=(Action("BUILD_SETTLEMENT", target=target),)),
        board=board,
        players=(PublicPlayer(10, 0, 0, 0), *(PublicPlayer(i, 0, 0, 0) for i in (20, 30, 40))),
    )
    second = replace(
        first, buildings=(Building(first_vertex, 10),),
        players=(PublicPlayer(10, 1, 0, 0), *(PublicPlayer(i, 0, 0, 0) for i in (20, 30, 40))),
    )
    first_features = successor_feature_values(first, first.legal_actions[0])
    second_features = successor_feature_values(second, second.legal_actions[0])
    assert first_features["hand_after_WHEAT"] == 0
    assert second_features["hand_known"] == 1
    assert second_features["hand_after_WHEAT"] == 0.1
    assert second_features["hand_after_BRICK"] == 0


def test_uncertain_robber_result_has_an_explicit_mask_without_invented_hand():
    observation = view(context=DecisionContext("MOVE_ROBBER", 10, 10, 2),
                       actions=(Action("MOVE_ROBBER", target=0, partner=20),))
    got = successor_feature_values(observation, observation.legal_actions[0])
    assert got["hand_known"] == 0
    assert got["uncertain_resource_result"] == 1
    assert all(got[f"hand_after_{r}"] == 0 for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"))
    assert all(got[f"settlement_deficit_after_{r}"] == 0
               for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"))

    no_victim = replace(observation, legal_actions=(Action("MOVE_ROBBER", target=0),))
    safe = successor_feature_values(no_victim, no_victim.legal_actions[0])
    assert safe["hand_known"] == 1
    assert safe["uncertain_resource_result"] == 0
    assert safe["hand_after_WOOD"] == safe["hand_before_WOOD"] == 0.4


def test_dev_resource_choice_and_robber_geometry_keep_uncertainty_separate():
    plenty = view(hand=(0, 0, 0, 0, 0), actions=(
        Action("PLAY_YEAR_OF_PLENTY", resource="ORE", second_resource="ORE"),
        Action("PLAY_YEAR_OF_PLENTY", resource="ORE"),
        Action("PLAY_MONOPOLY", resource="ORE"),
    ))
    chosen = successor_feature_values(plenty, plenty.legal_actions[0])
    one_card = successor_feature_values(plenty, plenty.legal_actions[1])
    unknown = successor_feature_values(plenty, plenty.legal_actions[2])
    assert chosen["hand_known"] == 1
    assert chosen["hand_after_ORE"] == 0.2
    assert one_card["hand_known"] == 1
    assert one_card["hand_after_ORE"] == 0.1
    assert unknown["hand_known"] == 0
    assert unknown["uncertain_resource_result"] == 1
    assert unknown["hand_after_ORE"] == 0

    board = BoardSpec.from_hexes(((0, 0), (1, 0)), ("WHEAT", "BRICK"), (6, 3))
    own_vertex = next(v for v in board.tile_vertices[0] if v not in board.tile_vertices[1])
    robber = replace(
        plenty, board=board, buildings=(Building(own_vertex, 10),), robber_tile=0,
        legal_actions=(Action("MOVE_ROBBER", target=1, partner=20),),
    )
    moved = successor_feature_values(robber, robber.legal_actions[0])
    assert moved["hand_known"] == 0
    assert moved["production_nominal_delta_WHEAT"] == 0
    assert moved["production_unblocked_delta_WHEAT"] == pytest.approx(5 / 36)


def test_public_largest_army_and_longest_road_awards_can_win_immediately():
    knight = replace(
        view(actions=(Action("PLAY_KNIGHT_CARD"),)),
        players=(
            PublicPlayer(10, 9, 0, 1, played_knights=2),
            *(PublicPlayer(i, 0, 0, 0) for i in (20, 30, 40)),
        ),
        development=(1, 0, 0, 0, 0),
    )
    army = successor_feature_values(knight, knight.legal_actions[0])
    assert army["point_gain_lower_bound"] == 2
    assert army["point_outcome_uncertain"] == 0
    assert army["wins_now"] == 1
    assert army["own_vp_after"] == 1.1
    held = replace(
        knight,
        players=(
            knight.players[0],
            PublicPlayer(20, 4, 0, 0, played_knights=4, has_largest_army=True),
            *knight.players[2:],
        ),
    )
    assert successor_feature_values(held, held.legal_actions[0])["point_gain_lower_bound"] == 0

    road = replace(
        view(hand=(1, 1, 0, 0, 0), roads=tuple(Road(i, 10) for i in range(4)),
             actions=(Action("BUILD_ROAD", target=4),)),
        players=(
            PublicPlayer(10, 9, 2, 0, roads_left=11, longest_road=4),
            *(PublicPlayer(i, 0, 0, 0) for i in (20, 30, 40)),
        ),
    )
    longest = successor_feature_values(road, road.legal_actions[0])
    assert longest["point_gain_lower_bound"] == 2
    assert longest["wins_now"] == 1
    assert longest["own_vp_after"] == 1.1
    tied = replace(
        road,
        players=(
            road.players[0],
            PublicPlayer(20, 7, 0, 0, longest_road=5, has_longest_road=True),
            *road.players[2:],
        ),
    )
    assert successor_feature_values(tied, tied.legal_actions[0])["point_gain_lower_bound"] == 0


def test_enemy_endpoint_can_count_for_road_award_but_is_not_guaranteed():
    base = view(
        hand=(1, 1, 0, 0, 0), roads=tuple(Road(i, 10) for i in range(4)),
        actions=(Action("BUILD_ROAD", target=4),),
    )
    base = replace(base, players=(
        PublicPlayer(10, 8, 2, 0, roads_left=11, longest_road=4),
        *(PublicPlayer(i, 0, 0, 0) for i in (20, 30, 40)),
    ))
    at_end = replace(base, buildings=(Building(0, 20),))
    endpoint = successor_feature_values(at_end, at_end.legal_actions[0])
    assert endpoint["point_gain_lower_bound"] == 0
    assert endpoint["point_outcome_uncertain"] == 1
    in_middle = replace(base, buildings=(Building(2, 20),))
    blocked = successor_feature_values(in_middle, in_middle.legal_actions[0])
    assert blocked["point_gain_lower_bound"] == 0
    assert blocked["point_outcome_uncertain"] == 0


def test_ordinary_settlement_cut_transfers_longest_road_and_guarantees_win():
    observation = _settlement_cut_view()
    features = successor_feature_values(observation, observation.legal_actions[0])
    assert features["point_gain_lower_bound"] == 3
    assert features["point_outcome_uncertain"] == 0
    assert features["own_vp_after"] == 1
    assert features["wins_now"] == 1


def test_settlement_cut_can_award_longest_road_below_five_roads():
    observation = _settlement_cut_view(
        actor_roads=(28, 10, 0), opponent_roads=(19, 5, 24),
        actor_building=23, opponent_building=16,
        actor_length=3, opponent_length=3, holder=None,
    )
    features = successor_feature_values(observation, observation.legal_actions[0])
    assert features["point_gain_lower_bound"] == 3
    assert features["point_outcome_uncertain"] == 0


def test_settlement_cut_uses_cached_length_for_unaffected_players():
    observation = _settlement_cut_view(
        third_length=6, third_roads=(2, 3, 18, 17, 16), third_building=2,
    )
    features = successor_feature_values(observation, observation.legal_actions[0])
    assert features["point_gain_lower_bound"] == 1
    assert features["point_outcome_uncertain"] == 0


def test_settlement_cut_tie_marks_possible_actor_award_as_uncertain():
    observation = _settlement_cut_view(
        actor_roads=(28, 10, 0), actor_building=23, actor_length=3,
    )
    features = successor_feature_values(observation, observation.legal_actions[0])
    assert features["point_gain_lower_bound"] == 1
    assert features["point_outcome_uncertain"] == 1
    assert features["wins_now"] == 0


def test_settlement_cut_tie_can_remove_actors_existing_road_award():
    observation = _settlement_cut_view(
        opponent_roads=(25, 19, 5, 24, 29), opponent_length=5,
        holder=10, third_length=5, third_roads=(2, 3, 18, 17, 16),
        third_building=2, actor_vp=9,
    )
    features = successor_feature_values(observation, observation.legal_actions[0])
    assert features["point_gain_lower_bound"] == -1
    assert features["point_outcome_uncertain"] == 1
    assert features["own_vp_after"] == 0.8
    assert features["wins_now"] == 0


def test_second_enemy_endpoint_makes_settlement_cut_road_award_uncertain():
    # Orange's second route ends at Blue's older building 53. Its visible edge
    # trail is 6, while the pinned backend can cache only 5 for that component.
    observation = _settlement_cut_view(
        opponent_roads=(25, 19, 5, 24, 29, 2, 9, 8, 7, 34, 71),
        opponent_length=5, extra_opponent_building=3,
        holder=10, actor_vp=9,
    )
    features = successor_feature_values(observation, observation.legal_actions[0])
    assert features["point_gain_lower_bound"] == -1
    assert features["point_outcome_uncertain"] == 1
    assert features["wins_now"] == 0


def test_road_to_existing_enemy_endpoint_cannot_claim_a_guaranteed_award():
    board = _native_topology()
    buildings = (Building(52, 10), Building(0, 20))
    roads = tuple(Road(edge, 10) for edge in (68, 28, 10, 0))
    observation = replace(
        view(hand=(1, 1, 0, 0, 0), actions=(Action("BUILD_ROAD", target=70),)),
        board=board, buildings=buildings, roads=roads,
        players=(
            PublicPlayer(10, 9, 2, 0, roads_left=11, settlements_left=4, longest_road=3),
            PublicPlayer(20, 1, 0, 0, settlements_left=4),
            PublicPlayer(30, 0, 0, 0), PublicPlayer(40, 0, 0, 0),
        ),
    )
    assert 70 in connectable_road_edges(board, buildings, roads, 10)
    features = successor_feature_values(observation, observation.legal_actions[0])
    assert features["point_gain_lower_bound"] == 0
    assert features["point_outcome_uncertain"] == 1
    assert features["wins_now"] == 0


def test_exact_legal_identity_menu_parity_finite_values_and_no_mutation():
    observation = view()
    original = deepcopy(observation)
    assert SUCCESSOR_SCHEMA == "public-candidate-successor-v3"
    assert SUCCESSOR_DIM == len(SUCCESSOR_COLUMNS)
    assert len(set(SUCCESSOR_COLUMNS)) == SUCCESSOR_DIM
    with pytest.raises(ValueError, match="exact current legal action"):
        successor_feature_values(observation, replace(observation.legal_actions[0]))
    rows = successor_features_for_menu(observation)
    assert rows.shape == (len(observation.legal_actions), SUCCESSOR_DIM)
    assert torch.isfinite(rows).all()
    for i, action in enumerate(observation.legal_actions):
        expected = successor_feature_values(observation, action)
        torch.testing.assert_close(rows[i], torch.tensor([expected[key] for key in SUCCESSOR_COLUMNS]))
    assert observation == original


def test_full_menu_computes_each_candidate_once(monkeypatch):
    from colonist_rl import successor_features as module

    observation = view()
    original = module._values
    calls = []

    def record(prepared, action):
        calls.append(action)
        return original(prepared, action)

    monkeypatch.setattr(module, "_values", record)
    module.successor_features_for_menu(observation)
    assert len(calls) == len(observation.legal_actions)
