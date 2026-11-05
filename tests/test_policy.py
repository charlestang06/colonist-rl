from dataclasses import replace

import torch

from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, DecisionContext, Observation, PublicPlayer, TradeOffer
from colonist_rl.policy import GraphPolicy


def view(players=4):
    board = BoardSpec.standard() if players == 4 else BoardSpec.extended()
    rules = RulesConfig.standard() if players == 4 else RulesConfig.extended()
    return Observation(
        0,
        board,
        rules,
        DecisionContext("TURN", 0, 0, 0, has_rolled=True),
        tuple(PublicPlayer(i, 2, 0, 0) for i in range(players)),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (),
        (),
        0,
        (Action("END_TURN"), Action("ROLL")),
    )


def test_same_model_handles_both_sizes_and_memory_has_gradients():
    policy = GraphPolicy(hidden_size=32)
    for count in (4, 6):
        result = policy(view(count))
        assert result.logits.shape == (2,)
        assert torch.isfinite(result.value)
        result.value.backward()
        assert policy.gru.weight_ih.grad is not None


def test_trade_response_has_only_legal_choices():
    policy = GraphPolicy(hidden_size=32)
    obs = view()
    offer = TradeOffer("x", 1, 0, (1, 0, 0, 0, 0), (0, 0, 0, 1, 0))
    obs = replace(
        obs,
        context=DecisionContext("TRADE_RESPONSE", 0, 1, 1, offer=offer),
        legal_actions=(Action("REJECT_TRADE", offer_id="x"),),
    )
    assert policy.act(obs)[0] == obs.legal_actions[0]


def test_arbitrary_player_ids_preserve_seat_order_and_policy_outputs():
    from colonist_rl.policy import graph_features

    original = replace(
        view(),
        player_id=2,
        context=DecisionContext("ROBBER", 2, 1, 0, has_rolled=True),
        legal_actions=(
            Action("MOVE_ROBBER", target=0, partner=0),
            Action("MOVE_ROBBER", target=0, partner=3),
        ),
    )
    ids = (10, 20, 30, 40)
    relabeled = replace(
        original,
        player_id=30,
        players=tuple(
            replace(player, player_id=ids[player.player_id]) for player in original.players
        ),
        context=replace(original.context, actor=30, turn_owner=20),
        legal_actions=tuple(
            replace(action, partner=ids[action.partner]) for action in original.legal_actions
        ),
    )
    features, _, (_, _, nodes) = graph_features(relabeled, "cpu")
    assert [features[nodes[player], 36].item() for player in ids] == [0.5, 0.75, 0.0, 0.25]
    policy = GraphPolicy(hidden_size=32)
    expected = policy(original)
    actual = policy(relabeled)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.value, expected.value)
    torch.testing.assert_close(actual.memory, expected.memory)


def test_off_turn_view_can_be_valued_without_legal_actions():
    import pytest

    ids = (10, 20, 30, 40)
    observation = replace(
        view(),
        player_id=30,
        players=tuple(
            replace(player, player_id=ids[player.player_id]) for player in view().players
        ),
        context=DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        legal_actions=(),
    )
    policy = GraphPolicy(hidden_size=32)
    output = policy(observation)
    assert output.logits.shape == (0,)
    assert torch.isfinite(output.value)
    assert output.memory.shape == (32,)
    output.value.backward()
    assert policy.gru.weight_ih.grad is not None
    with pytest.raises(ValueError, match="legal action"):
        policy.act(observation)


def test_value_only_foreign_trade_offer_uses_no_own_trade_vectors():
    from colonist_rl.policy import context_features

    observation = replace(
        view(),
        player_id=2,
        context=DecisionContext(
            "TRADE_RESPONSE",
            1,
            0,
            2,
            offer=TradeOffer("foreign", 0, 1, (1, 0, 0, 0, 0), (0, 0, 0, 1, 0)),
        ),
        legal_actions=(),
    )
    encoded = context_features(observation, "cpu")
    assert torch.equal(encoded[38:48], torch.zeros(10))
    result = GraphPolicy(hidden_size=32)(observation)
    assert result.logits.shape == (0,)
    assert torch.isfinite(result.value)
