from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import (
    Action,
    Building,
    DecisionContext,
    Observation,
    PublicPlayer,
    Road,
    TradeOffer,
)
from colonist_rl.policy import GraphPolicy


def exchange_view():
    return Observation(
        10,
        BoardSpec.from_hexes(((0, 0),)),
        RulesConfig.standard(),
        DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        tuple(PublicPlayer(i, 0, 0, 0) for i in (10, 20, 30, 40)),
        (5, 1, 1, 0, 0),
        (0, 0, 0, 0, 0),
        (),
        (),
        0,
        (
            Action("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0)),
            Action("END_TURN"),
        ),
    )


def features(view, index=0):
    from colonist_rl.consequences import consequence_features

    return consequence_features(view, view.legal_actions[index])


def test_exact_exchange_hand_deficits_and_recipe_flags_without_mutating_view():
    observation = exchange_view()
    original = deepcopy(observation)
    got = features(observation)
    assert got.shape == (34,)
    torch.testing.assert_close(
        got,
        torch.tensor(
            [1, 0.1, 0.1, 0.1, 0.1, 0]
            + [0] * 10  # road and settlement deficits
            + [0, 0, 0, 0.25, 0.75]  # city deficits
            + [0, 0, 0, 0, 0.25]  # development deficits
            + [1, 1, 0, 0, 0, 0, 0, 0]
        ),
    )
    assert observation == original
    assert features(observation, 1)[14:16].tolist() == [0.25, 0.0]
    # Unknown bank does not suppress an already legal first exchange or invent stock.
    torch.testing.assert_close(got, features(replace(observation, bank=(0, 0, 0, 0, 0))))


@pytest.mark.parametrize("proposer", [True, False])
def test_accept_uses_offer_perspective_and_reject_keeps_hand(proposer):
    observation = exchange_view()
    offer = TradeOffer(
        "x",
        10 if proposer else 20,
        20 if proposer else 10,
        (4, 0, 0, 0, 0) if proposer else (0, 0, 0, 1, 0),
        (0, 0, 0, 1, 0) if proposer else (4, 0, 0, 0, 0),
    )
    observation = replace(
        observation,
        context=replace(observation.context, kind="TRADE_RESPONSE", offer=offer),
        legal_actions=(Action("ACCEPT_TRADE", offer_id="x"), Action("REJECT_TRADE", offer_id="x")),
    )
    torch.testing.assert_close(features(observation)[:30], features(exchange_view())[:30])
    torch.testing.assert_close(features(observation, 1)[1:6], torch.tensor([0.5, 0.1, 0.1, 0, 0]))
    assert features(observation)[31] == 0


@pytest.mark.parametrize(
    "roads,buildings,want",
    [
        ((), (), 0),
        ((Road(0, 20),), (), 0),
        ((Road(0, 10),), (), 2),
        ((Road(0, 10),), (Building(0, 20),), 0),
        ((Road(0, 10),), (Building(2, 20),), 1),
        ((Road(1, 10),), (Building(0, 10),), 1),
        ((Road(0, 10),), (Building(4, 20),), 2),
    ],
)
def test_settlement_readiness_requires_connected_distance_legal_vacant_site(roads, buildings, want):
    observation = replace(exchange_view(), roads=roads, buildings=buildings)
    got = features(observation)
    assert got[30].item() == want / 8
    assert got[31].item() == bool(want)
    assert got[27] == 1  # recipe sufficiency alone never implies a site


def test_connectivity_count_clips_and_readiness_respects_phase_and_piece_limits():
    observation = exchange_view()
    board = BoardSpec.standard()
    crowded = replace(
        observation, board=board, roads=tuple(Road(i, 10) for i in range(len(board.edges)))
    )
    assert features(crowded)[30] == 1
    connected = replace(observation, roads=(Road(0, 10),))
    assert features(connected)[31] == 1
    for context in (
        replace(connected.context, has_rolled=False),
        replace(connected.context, kind="SETUP_SETTLEMENT"),
        replace(connected.context, kind="TRADE_RESPONSE"),
        replace(connected.context, actor=20),
        replace(connected.context, turn_owner=20),
    ):
        assert features(replace(connected, context=context))[31] == 0
    exhausted = replace(
        connected,
        players=(replace(connected.players[0], settlements_left=0), *connected.players[1:]),
    )
    assert features(exhausted)[31] == 0


@pytest.mark.parametrize(
    "buildings,cities_left,want",
    [
        ((), 4, [0, 0]),
        ((Building(0, 20),), 4, [0, 0]),
        ((Building(0, 10, 2),), 4, [0, 0]),
        ((Building(0, 10),), 0, [0, 0]),
        ((Building(0, 10),), 1, [1, 1]),
    ],
)
def test_city_readiness_requires_own_settlement_available_piece_and_recipe(
    buildings, cities_left, want
):
    observation = replace(exchange_view(), hand=(0, 0, 0, 2, 3), buildings=buildings)
    observation = replace(
        observation,
        players=(
            replace(observation.players[0], cities_left=cities_left),
            *observation.players[1:],
        ),
    )
    got = features(observation, 1)
    assert got[32:34].tolist() == want
    before_roll = replace(observation, context=replace(observation.context, has_rolled=False))
    assert features(before_roll, 1)[33] == 0
    poor = replace(observation, hand=(0, 0, 0, 1, 3))
    assert features(poor, 1)[33] == 0
    assert features(poor, 1)[32] == want[0]


def test_unsupported_present_actions_are_zero_and_absent_actions_are_rejected():
    from colonist_rl.consequences import consequence_features

    observation = replace(
        exchange_view(), legal_actions=(Action("BUILD_ROAD", target=0), Action("ROLL"))
    )
    for action in observation.legal_actions:
        assert torch.equal(consequence_features(observation, action), torch.zeros(34))
    with pytest.raises(ValueError, match="legal"):
        consequence_features(observation, Action("END_TURN"))


def test_zero_branch_copies_base_exactly_preserves_rng_outputs_and_has_first_gradient():
    from colonist_rl.policy import consequence_from_legacy, policy_architecture

    torch.manual_seed(19)
    legacy = GraphPolicy(32).eval()
    rng = torch.get_rng_state().clone()
    variant = consequence_from_legacy(legacy)
    assert torch.equal(rng, torch.get_rng_state())
    assert not variant.training
    assert policy_architecture(legacy) == {"hidden_size": 32}
    assert policy_architecture(variant) == {
        "kind": "graph_consequence_v1",
        "hidden_size": 32,
        "feature_schema": "public-consequence-v1",
        "feature_set": "hand_recipe",
    }
    for name, parameter in legacy.named_parameters():
        assert torch.equal(parameter, dict(variant.named_parameters())[name])
        assert parameter.data_ptr() != dict(variant.named_parameters())[name].data_ptr()
    assert variant.consequence.bias is None
    assert torch.count_nonzero(variant.consequence.weight) == 0
    observation = exchange_view()
    before, after = legacy(observation), variant(observation)
    for field in ("logits", "memory", "value"):
        assert torch.equal(getattr(before, field), getattr(after, field))
    torch.nn.functional.cross_entropy(after.logits[None], torch.tensor([0])).backward()
    assert variant.consequence.weight.grad.abs().sum().item() > 0
    with pytest.raises(ValueError, match="legacy"):
        consequence_from_legacy(variant)


def test_additive_scorer_changes_scalar_and_batched_trade_logits_only_and_skips_prefix():
    from colonist_rl.policy import consequence_from_legacy

    legacy = GraphPolicy(16)
    variant = consequence_from_legacy(legacy)
    normal = exchange_view()
    response = replace(
        normal,
        context=replace(
            normal.context,
            kind="TRADE_RESPONSE",
            offer=TradeOffer("x", 10, 20, (4, 0, 0, 0, 0), (0, 0, 0, 1, 0)),
        ),
        legal_actions=(Action("ACCEPT_TRADE", offer_id="x"), Action("REJECT_TRADE", offer_id="x")),
    )
    off_turn = replace(normal, legal_actions=())
    with torch.no_grad():
        variant.consequence.weight[0, 4] = 2  # post-action wheat/10
    views, memories = (normal, response, off_turn), (torch.ones(16), None, None)
    for before, after, observation, memory in zip(
        legacy.forward_many(views, memories), variant.forward_many(views, memories), views, memories
    ):
        want = torch.tensor([0.2, 0]) if observation.legal_actions else torch.empty(0)
        torch.testing.assert_close(after.logits - before.logits, want)
        assert torch.equal(before.value, after.value)
        assert torch.equal(before.memory, after.memory)
        scalar = variant(observation, memory)
        torch.testing.assert_close(after.logits, scalar.logits)
    calls = []
    handle = variant.consequence.register_forward_hook(lambda *args: calls.append(1))
    variant.recurrent_step(normal)
    variant.recurrent_many(views, memories)
    handle.remove()
    assert calls == []


def test_feature_sets_are_explicit_and_connectivity_coefficients_start_zero():
    from colonist_rl.policy import consequence_from_legacy

    legacy = GraphPolicy(16)
    recipe = consequence_from_legacy(legacy)
    connected = consequence_from_legacy(legacy, feature_set="hand_recipe_and_connectivity")
    observation = replace(exchange_view(), roads=(Road(0, 10),))
    assert torch.equal(recipe(observation).logits, connected(observation).logits)
    with torch.no_grad():
        recipe.consequence.weight[0, 31] = 1
        connected.consequence.weight[0, 31] = 1
    assert torch.equal(recipe(observation).logits, legacy(observation).logits)
    torch.testing.assert_close(
        connected(observation).logits - legacy(observation).logits, torch.tensor([1.0, 0])
    )
    with pytest.raises(AttributeError):
        recipe.feature_set = "hand_recipe_and_connectivity"
