"""Known public successors are candidate features, never future legal menus."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from colonist_rl.board import BoardSpec, Port, RulesConfig
from colonist_rl.contracts import Action, Building, DecisionContext, Observation, PublicPlayer, Road
from colonist_rl.successor_features import (
    SUCCESSOR_COLUMNS,
    SUCCESSOR_DIM,
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
    assert build["known_point_gain"] == 1
    assert build["own_vp_after"] == 0.3
    assert build["settlement_pieces_after"] == 0.8
    assert buy["hand_after_SHEEP"] == 0
    assert buy["hand_after_WHEAT"] == 0.2
    assert buy["hand_after_ORE"] == 0.2
    assert buy["unknown_dev_draw"] == 1
    assert buy["known_point_gain"] == 0
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
    assert upgraded["known_point_gain"] == 1
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
                       actions=(Action("MOVE_ROBBER", target=0),))
    got = successor_feature_values(observation, observation.legal_actions[0])
    assert got["hand_known"] == 0
    assert got["uncertain_resource_result"] == 1
    assert all(got[f"hand_after_{r}"] == 0 for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"))
    assert all(got[f"settlement_deficit_after_{r}"] == 0
               for r in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"))


def test_dev_resource_choice_and_robber_geometry_keep_uncertainty_separate():
    plenty = view(hand=(0, 0, 0, 0, 0), actions=(
        Action("PLAY_YEAR_OF_PLENTY", resource="ORE", second_resource="ORE"),
        Action("PLAY_MONOPOLY", resource="ORE"),
    ))
    chosen = successor_feature_values(plenty, plenty.legal_actions[0])
    unknown = successor_feature_values(plenty, plenty.legal_actions[1])
    assert chosen["hand_known"] == 1
    assert chosen["hand_after_ORE"] == 0.2
    assert unknown["hand_known"] == 0
    assert unknown["uncertain_resource_result"] == 1
    assert unknown["hand_after_ORE"] == 0

    board = BoardSpec.from_hexes(((0, 0), (1, 0)), ("WHEAT", "BRICK"), (6, 3))
    own_vertex = next(v for v in board.tile_vertices[0] if v not in board.tile_vertices[1])
    robber = replace(
        plenty, board=board, buildings=(Building(own_vertex, 10),), robber_tile=0,
        legal_actions=(Action("MOVE_ROBBER", target=1),),
    )
    moved = successor_feature_values(robber, robber.legal_actions[0])
    assert moved["hand_known"] == 0
    assert moved["production_nominal_delta_WHEAT"] == 0
    assert moved["production_unblocked_delta_WHEAT"] == pytest.approx(5 / 36)


def test_exact_legal_identity_menu_parity_finite_values_and_no_mutation():
    observation = view()
    original = deepcopy(observation)
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
