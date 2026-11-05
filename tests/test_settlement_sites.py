"""Shared public geometry keeps both frozen feature schemas numerically exact."""

import importlib.util
from copy import deepcopy

import pytest
import torch

from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.consequences import consequence_features
from colonist_rl.contracts import Action, Building, DecisionContext, Observation, PublicPlayer, Road
from colonist_rl.trading_context import trading_context_features


@pytest.mark.parametrize(
    "roads,buildings,count,normalized,city",
    [
        ((), (), 0, 0, 0),
        ((Road(0, 20),), (), 0, 0, 0),
        ((Road(0, 10),), (), 2, 0.25, 0),
        ((Road(0, 10),), (Building(0, 10),), 0, 0, 1),
        ((Road(0, 10),), (Building(2, 20),), 1, 0.125, 0),
        ((Road(1, 10),), (Building(0, 10),), 1, 0.125, 1),
        ((Road(0, 10), Road(1, 10)), (), 3, 0.375, 0),
        ((Road(0, 10), Road(1, 20)), (Building(4, 20),), 2, 0.25, 0),
    ],
)
def test_shared_geometry_matches_literal_count_and_entire_feature_vectors(
    roads, buildings, count, normalized, city
):
    assert importlib.util.find_spec("colonist_rl.settlement_sites") is not None, (
        "shared public settlement-site geometry helper missing"
    )
    from colonist_rl.settlement_sites import connected_settlement_site_count

    observation = Observation(
        10,
        BoardSpec.from_hexes(((0, 0),)),
        RulesConfig.standard(),
        DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        (PublicPlayer(10, 0, 12, 0), PublicPlayer(20, 0, 0, 0)),
        (5, 1, 1, 2, 3),
        (0, 0, 0, 0, 0),
        buildings,
        roads,
        0,
        (Action("END_TURN"),),
    )
    before = deepcopy(observation)
    assert connected_settlement_site_count(observation) == count
    expected34 = torch.tensor(
        [1, 0.5, 0.1, 0.1, 0.2, 0.3] + [0] * 20 + [1] * 4 + [normalized, int(count > 0), city, city]
    )
    expected24 = torch.tensor(
        [1, 1, 1, 0.2 * city, 0, normalized, city, 1, 0, 1, 1, city, 1, 0] + [0] * 10
    )
    assert torch.equal(consequence_features(observation, observation.legal_actions[0]), expected34)
    assert torch.equal(trading_context_features(observation), expected24)
    assert observation == before


def test_shared_geometry_returns_unclipped_count_and_both_callers_keep_their_clipping():
    assert importlib.util.find_spec("colonist_rl.settlement_sites") is not None, (
        "shared public settlement-site geometry helper missing"
    )
    from colonist_rl.settlement_sites import connected_settlement_site_count

    board = BoardSpec.standard()
    observation = Observation(
        10,
        board,
        RulesConfig.standard(),
        DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        (PublicPlayer(10, 0, 0, 0),),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (),
        tuple(Road(i, 10) for i in range(len(board.edges))),
        0,
        (Action("END_TURN"),),
    )
    assert connected_settlement_site_count(observation) == 54
    assert consequence_features(observation, observation.legal_actions[0])[30].item() == 1
    assert trading_context_features(observation)[5].item() == 1
