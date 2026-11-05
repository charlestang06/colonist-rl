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


def test_forward_many_isolates_mixed_graph_sizes_memories_and_variable_legal_menus():
    torch.set_num_threads(1)
    policy = GraphPolicy(8)
    small, large = view(), view(6)
    trade = replace(
        small,
        context=replace(small.context, kind="TRADE_RESPONSE"),
        legal_actions=(Action("REJECT_TRADE"), Action("ACCEPT_TRADE")),
    )
    robber = replace(
        large,
        context=replace(large.context, kind="ROBBER"),
        legal_actions=(Action("MOVE_ROBBER", target=2, partner=1),),
    )
    off_turn = replace(small, legal_actions=())
    views = (small, robber, trade, off_turn)
    memories = (None, torch.linspace(-0.2, 0.2, 8), None, torch.ones(8))
    batched = policy.forward_many(views, memories)
    for observation, memory, got in zip(views, memories, batched):
        want = policy(observation, memory)
        for field in ("logits", "value", "memory"):
            torch.testing.assert_close(getattr(got, field), getattr(want, field))
    assert [len(output.logits) for output in batched] == [2, 1, 2, 0]
    assert policy.forward_many((), ()) == ()
    import pytest

    with pytest.raises(ValueError, match="length"):
        policy.forward_many(views, (None,))


def test_recurrent_only_routes_never_invoke_unused_actor_or_value_heads():
    policy = GraphPolicy(8)
    calls = []
    handles = [
        module.register_forward_hook(lambda *args: calls.append(1))
        for module in (policy.action_in, policy.heads, policy.trade_response, policy.value)
    ]
    try:
        policy.recurrent_step(view())
        policy.recurrent_many((view(), view(6)), (None, torch.zeros(8)))
        assert calls == []
        policy(view())
        assert calls
    finally:
        for handle in handles:
            handle.remove()
