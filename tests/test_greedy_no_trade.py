"""Frozen plain-greedy behavior, using real immutable observation contracts."""

import random
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from colonist_rl.board import BoardSpec, Port, RulesConfig
from colonist_rl.contracts import (
    RESOURCES,
    Action,
    Building,
    DecisionContext,
    Observation,
    PublicPlayer,
    ResourceBelief,
    TradeOffer,
    VisibleEvent,
)
from colonist_rl.evaluation import evaluate_blocks


def policy():
    from colonist_rl.greedy_no_trade import GreedyNoTrade

    return GreedyNoTrade()


def view(*actions, **changes):
    # Disjoint productive hexes: vertex 0 has 1 pip; vertices 6 and 12 have 5.
    board = BoardSpec.from_hexes(
        [(0, 0), (3, 0), (6, 0), (9, 0)],
        ["ORE", "WOOD", "SHEEP", "DESERT"],
        [2, 6, 8, 0],
    )
    observation = Observation(
        player_id=20,
        board=board,
        rules=RulesConfig.standard(),
        context=DecisionContext("PLAY_TURN", 20, 20, 7, has_rolled=True),
        players=tuple(PublicPlayer(p, 2, 4, 0) for p in (10, 20, 30, 40)),
        hand=(4, 1, 1, 1, 1),
        development=(0, 0, 0, 0, 0),
        buildings=(),
        roads=(),
        robber_tile=1,
        legal_actions=tuple(actions),
    )
    return replace(observation, **changes)


def test_rejects_even_a_free_incoming_offer_before_every_other_action():
    offer = TradeOffer("gift", 10, 20, (0, 0, 0, 0, 5), (0, 0, 0, 0, 0))
    reject = Action("REJECT_TRADE", offer_id="gift")
    observation = view(
        Action("ACCEPT_TRADE", offer_id="gift"), Action("BUILD_CITY", target=6), reject
    )
    observation = replace(observation, context=replace(observation.context, offer=offer))
    assert policy().select_action(observation) is reject


def test_refuses_legal_four_to_one_bank_trade_without_an_owned_port():
    end = Action("END_TURN")
    trade = Action("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0))
    # Four wood can legally buy a wheat at the bank without owning any port.
    observation = view(trade, end, hand=(4, 1, 1, 0, 1))
    assert observation.board.ports == observation.buildings == ()
    assert policy().select_action(observation) is end


@pytest.mark.parametrize(
    "actions",
    [
        (),
        (Action("ACCEPT_TRADE"),),
        (Action("MARITIME_TRADE"),),
        (Action("PROPOSE_TRADE"),),
        (Action("UNKNOWN"),),
        (Action("ACCEPT_TRADE"), Action("MARITIME_TRADE"), Action("UNKNOWN")),
    ],
)
def test_empty_or_unsupported_only_menu_raises(actions):
    with pytest.raises(ValueError, match="supported|safe|legal"):
        policy().select_action(view(*actions))


@pytest.mark.parametrize("kind", ["BUILD_CITY", "BUILD_SETTLEMENT", "BUILD_ROAD", "END_TURN"])
def test_never_proposes_even_with_a_profitable_offer_available(kind):
    observation = view(Action(kind, target=6))
    assert policy().propose(observation) is None


@pytest.mark.parametrize(
    "kinds, expected",
    [
        (
            [
                "END_TURN",
                "PLAY_ROAD_BUILDING",
                "BUY_DEVELOPMENT_CARD",
                "BUILD_ROAD",
                "BUILD_SETTLEMENT",
                "BUILD_CITY",
            ],
            "BUILD_CITY",
        ),
        (
            [
                "END_TURN",
                "PLAY_ROAD_BUILDING",
                "BUY_DEVELOPMENT_CARD",
                "BUILD_ROAD",
                "BUILD_SETTLEMENT",
            ],
            "BUILD_SETTLEMENT",
        ),
        (["END_TURN", "PLAY_ROAD_BUILDING", "BUY_DEVELOPMENT_CARD", "BUILD_ROAD"], "BUILD_ROAD"),
        (["END_TURN", "PLAY_ROAD_BUILDING", "BUY_DEVELOPMENT_CARD"], "BUY_DEVELOPMENT_CARD"),
        (["BUILD_CITY", "PLAY_KNIGHT_CARD", "ROLL"], "ROLL"),
    ],
)
def test_literal_purchase_and_roll_priority(kinds, expected):
    observation = view(*(Action(kind, target=6) for kind in kinds))
    assert policy().select_action(observation).kind == expected


@pytest.mark.parametrize("kind", ["BUILD_CITY", "BUILD_SETTLEMENT"])
@pytest.mark.parametrize("phase", ["PLAY_TURN", "BUILD_INITIAL_SETTLEMENT"])
def test_city_and_settlement_maximize_raw_pips_then_smallest_vertex(kind, phase):
    chosen = Action(kind, target=6)
    observation = view(Action(kind, target=12), Action(kind, target=0), chosen)
    observation = replace(observation, context=replace(observation.context, kind=phase))
    assert policy().select_action(observation) is chosen


def test_raw_pips_add_all_adjacent_productive_numbered_tiles():
    board = BoardSpec.from_hexes([(0, 0), (1, 0)], ["WOOD", "WOOD"], [6, 2])
    # Vertex 1 touches both tiles (5 + 1 pips); vertex 0 touches only the six.
    chosen = Action("BUILD_SETTLEMENT", target=1)
    assert (
        policy().select_action(view(Action("BUILD_SETTLEMENT", target=0), chosen, board=board))
        is chosen
    )


def test_desert_and_unnumbered_tiles_do_not_score():
    board = BoardSpec.from_hexes([(0, 0), (3, 0)], ["WOOD", "ORE"], [0, 2])
    chosen = Action("BUILD_SETTLEMENT", target=6)
    assert (
        policy().select_action(view(Action("BUILD_SETTLEMENT", target=0), chosen, board=board))
        is chosen
    )


def test_site_choices_ignore_resource_identity_ports_robber_and_existing_production():
    chosen = Action("BUILD_SETTLEMENT", target=6)
    original = view(Action("BUILD_SETTLEMENT", target=12), chosen)
    board = replace(
        original.board,
        tiles=tuple(
            replace(t, resource="ORE") if t.resource != "DESERT" else t
            for t in original.board.tiles
        ),
        ports=(Port((12, 13), "ORE", 2),),
    )
    altered = replace(
        original, board=board, robber_tile=2, hand=(0, 0, 0, 0, 19), buildings=(Building(6, 20, 2),)
    )
    assert policy().select_action(original) is chosen
    assert policy().select_action(altered) is chosen


@pytest.mark.parametrize("free_roads", [0, 2])
def test_roads_always_use_lowest_edge_even_toward_a_poor_site(free_roads):
    chosen = Action("BUILD_ROAD", target=0)  # One-pip tile, versus five-pip edge 6.
    observation = view(Action("BUILD_ROAD", target=6), chosen)
    observation = replace(observation, context=replace(observation.context, free_roads=free_roads))
    assert policy().select_action(observation) is chosen


@pytest.mark.parametrize(
    "hand, expected",
    [((2, 7, 1, 3, 7), "BRICK"), ((4, 4, 4, 4, 4), "WOOD"), ((0, 0, 2, 8, 8), "WHEAT")],
)
def test_discard_uses_largest_own_count_then_shared_resource_order(hand, expected):
    observation = view(
        *(Action("DISCARD_RESOURCE", resource=r) for r in reversed(RESOURCES)), hand=hand
    )
    assert policy().select_action(observation).resource == expected


def test_discard_only_compares_resources_present_in_legal_menu():
    chosen = Action("DISCARD_RESOURCE", resource="SHEEP")
    assert (
        policy().select_action(
            view(Action("DISCARD_RESOURCE", resource="ORE"), chosen, hand=(19, 0, 3, 0, 2))
        )
        is chosen
    )


def test_robber_uses_smallest_tile_then_real_victim_id_before_none():
    chosen = Action("MOVE_ROBBER", target=0, partner=10)
    observation = view(
        Action("MOVE_ROBBER", target=1, partner=1),
        Action("MOVE_ROBBER", target=0),
        Action("MOVE_ROBBER", target=0, partner=30),
        chosen,
    )
    assert policy().select_action(observation) is chosen


def test_legacy_discard_uses_canonical_target_without_hand_inference():
    chosen = Action("DISCARD", target=1, partner=10)
    observation = view(Action("DISCARD", target=2), Action("DISCARD", target=1), chosen)
    assert policy().select_action(observation) is chosen


@pytest.mark.parametrize(
    "kinds, expected",
    [
        (
            ["PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY", "PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING"],
            "PLAY_ROAD_BUILDING",
        ),
        (["PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY", "PLAY_KNIGHT_CARD"], "PLAY_KNIGHT_CARD"),
        (["PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY"], "PLAY_YEAR_OF_PLENTY"),
        (["PLAY_MONOPOLY"], "PLAY_MONOPOLY"),
    ],
)
def test_development_plays_follow_fixed_kind_order(kinds, expected):
    observation = view(Action("END_TURN"), *(Action(k) for k in kinds))
    assert policy().select_action(observation).kind == expected


@pytest.mark.parametrize("kind", ["PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY"])
def test_development_resources_use_shared_lexicographic_order_not_hand_utility(kind):
    chosen = Action(kind, resource="WOOD", second_resource="BRICK")
    observation = view(
        Action(kind, resource="BRICK", second_resource="WOOD"),
        Action(kind, resource="WOOD", second_resource="ORE"),
        chosen,
        hand=(19, 0, 0, 0, 0),
    )
    assert policy().select_action(observation) is chosen


def test_exact_development_ties_keep_original_menu_member():
    first = Action("PLAY_KNIGHT_CARD", target=2, partner=10)
    equal = replace(first)
    observation = view(
        Action("PLAY_KNIGHT_CARD", target=2),
        first,
        equal,
        Action("PLAY_KNIGHT_CARD", target=3, partner=1),
    )
    assert policy().select_action(observation) is first


def test_end_turn_is_safe_fallback_and_unknown_kinds_are_ignored():
    end = Action("END_TURN")
    assert policy().select_action(view(Action("UNKNOWN"), Action("ACCEPT_TRADE"), end)) is end


def test_input_rng_and_actions_are_unchanged_and_memory_is_dropped():
    agent = policy()
    chosen = Action("BUILD_CITY", target=6)
    observation = view(
        Action("BUILD_CITY", target=12),
        chosen,
        beliefs=(ResourceBelief(10, (1.0,) * 5, (0,) * 5, (4,) * 5, (1.0,) * 5),),
        events=(VisibleEvent(1, "ROLL", 10),),
    )
    snapshot = deepcopy(observation)
    memory = torch.tensor([9.0])
    for seed in (1, 77, 908):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        for deterministic in (False, True):
            action, returned_memory = agent.act(observation, memory, deterministic=deterministic)
            assert action is chosen and returned_memory is None
            assert agent.propose(observation) is None
        assert random.getstate() == states[0]
        restored = np.random.get_state()
        assert restored[0] == states[1][0] and restored[2:] == states[1][2:]
        np.testing.assert_array_equal(restored[1], states[1][1])
        assert torch.equal(torch.get_rng_state(), states[2])
    assert observation == snapshot
    assert torch.equal(memory, torch.tensor([9.0]))
    assert isinstance(agent, torch.nn.Module)
    assert list(agent.parameters()) == list(agent.buffers()) == []
    assert agent.state_dict() == {}
    assert agent.identity == "plain-greedy-no-trade-v1"
    with pytest.raises(AttributeError):
        agent.identity = "changed"
    assert policy().select_action(observation) is chosen


def test_evaluator_uses_act_resets_memory_and_restores_rng_and_mode(monkeypatch):
    from colonist_rl import evaluation

    seen = []
    base = type(policy())

    class RecordingGreedy(base):
        def act(self, observation, memory=None, *, deterministic=False):
            seen.append((memory, deterministic, self.training))
            return super().act(observation, memory, deterministic=deterministic)

    # Script just the slow environment boundary, retaining the real evaluator and
    # match driver. The ordinary real-engine suite covers the simulator itself.
    created = []

    class ScriptedGame:
        def __init__(self, *, max_steps):
            self.version = 0
            self.current_player = len(created)
            created.append(self)

        def reset(self, *, seed):
            self.version = 0

        def observe(self):
            return view(
                Action("ACCEPT_TRADE"),
                Action("MARITIME_TRADE"),
                Action("REJECT_TRADE"),
                player_id=self.current_player,
                context=DecisionContext("TRADE_RESPONSE", self.current_player, 1, self.version),
            )

        def step(self, action):
            assert action.kind == "REJECT_TRADE"
            self.version += 1
            return SimpleNamespace(terminated=self.version == 2, truncated=False, winner=0)

        def _actual_vp(self):
            return [10, 2, 2, 2]

    monkeypatch.setattr(evaluation, "FourPlayerGame", ScriptedGame)
    agent = RecordingGreedy().train()
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = evaluate_blocks(
        agent,
        board_seeds=[99],
        candidate="graph",
        baseline="heuristic",
        deterministic=True,
        max_game_steps=8,
    )
    assert len(seen) == 8 and all(item == (None, True, False) for item in seen)
    assert result["action_kind_counts"] == {"REJECT_TRADE": 8}
    assert result["games"] == 4 and agent.training
    assert random.getstate() == states[0]
    restored = np.random.get_state()
    assert restored[0] == states[1][0] and restored[2:] == states[1][2:]
    np.testing.assert_array_equal(restored[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
