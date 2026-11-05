"""Literal public context and matched residual-head contracts."""

import importlib
import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

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
from colonist_rl.policy import ConsequenceGraphPolicy, GraphPolicy, consequence_from_legacy


def api():
    from colonist_rl import policy

    assert hasattr(policy, "trading_residual_from_consequence"), "trading residual upgrade missing"
    return policy


def context(view):
    assert importlib.util.find_spec("colonist_rl.trading_context") is not None, (
        "public trading context extractor missing"
    )
    from colonist_rl.trading_context import trading_context_features

    return trading_context_features(view)


def view():
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


def own(observation, **changes):
    return replace(
        observation, players=(replace(observation.players[0], **changes), *observation.players[1:])
    )


def response():
    observation = view()
    return replace(
        observation,
        context=replace(
            observation.context,
            kind="TRADE_RESPONSE",
            offer=TradeOffer("x", 10, 20, (4, 0, 0, 0, 0), (0, 0, 0, 1, 0)),
        ),
        legal_actions=(Action("ACCEPT_TRADE", offer_id="x"), Action("REJECT_TRADE", offer_id="x")),
    )


def trained_parent():
    torch.manual_seed(31)
    parent = consequence_from_legacy(GraphPolicy(16)).eval()
    with torch.no_grad():
        parent.consequence.weight.copy_(torch.linspace(-0.7, 0.9, 34)[None])
    return parent


def test_literal_all_context_slots_resource_order_and_no_teacher_calls(monkeypatch):
    from colonist_rl.opponents import HeuristicPolicy

    def forbidden(*args, **kwargs):
        raise AssertionError("context must not call teacher methods")

    for method in ("_utility", "_goals", "_production", "_road_score"):
        monkeypatch.setattr(HeuristicPolicy, method, forbidden)
    board = BoardSpec.from_hexes(
        tuple((3 * i, 0) for i in range(8)),
        ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", "WOOD", "DESERT", "ORE"),
        (5, 6, 8, 9, 10, 11, 0, 0),
    )
    observation = own(
        replace(
            view(),
            board=board,
            buildings=(
                Building(0, 10),
                Building(6, 10, 2),
                Building(12, 10),
                Building(18, 10, 2),
                Building(24, 10),
                Building(30, 10),
                Building(36, 20),
                Building(42, 20),
            ),
            roads=(Road(1, 10), Road(2, 20)),
        ),
        roads_left=9,
        settlements_left=2,
        cities_left=2,
        development_count=10,
    )
    before = deepcopy(observation)
    expected = torch.tensor(
        [
            0.6,
            0.4,
            0.5,
            0.8,
            1.25,
            0.125,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            6 / 30,
            10 / 30,
            5 / 30,
            8 / 30,
            3 / 30,
            2 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
            1 / 6,
        ]
    )
    torch.testing.assert_close(context(observation), expected, rtol=0, atol=0)
    assert observation == before
    assert observation.board.ports == ()  # Explicit legal no-port 4:1 menu.
    assert torch.equal(context(observation), context(replace(observation, bank=(0, 0, 0, 0, 0))))
    assert torch.equal(context(observation), context(replace(observation, robber_tile=1)))


@pytest.mark.parametrize(
    "changes,buildings,want",
    [
        ({}, (), [1, 1, 0, 1, 0]),
        ({"roads_left": 0}, (), [0, 0, 0, 1, 0]),
        ({"settlements_left": 0}, (), [1, 0, 0, 1, 0]),
        ({"cities_left": 0}, (Building(0, 10),), [1, 1, 1, 1, 0]),
        ({"roads_left": 0, "settlements_left": 0, "development_count": 8}, (), [0, 0, 0, 0, 1]),
        ({"roads_left": 0, "development_count": 8}, (Building(0, 10, 2),), [0, 0, 0, 0, 1]),
    ],
)
def test_teacher_gates_and_fallback_distinguish_real_city_stock(changes, buildings, want):
    observation = own(replace(view(), buildings=buildings), **changes)
    got = context(observation)
    assert got[9:14].tolist() == want
    assert got[6].item() == bool(
        buildings and buildings[0].level == 1 and changes.get("cities_left", 4) > 0
    )


@pytest.mark.parametrize(
    "roads,buildings,want",
    [
        ((), (), 0),
        ((Road(0, 20),), (), 0),
        ((Road(0, 10),), (), 2 / 8),
        ((Road(0, 10),), (Building(0, 20),), 0),
        ((Road(0, 10),), (Building(2, 20),), 1 / 8),
        ((Road(1, 10),), (Building(0, 10),), 1 / 8),
    ],
)
def test_connected_vacant_sites_exclude_occupied_neighbors_and_enemy_roads(roads, buildings, want):
    assert context(replace(view(), roads=roads, buildings=buildings))[5].item() == want


def test_phase_clip_empty_production_and_coverage_per_vertex():
    observation = view()
    assert context(observation)[14:].tolist() == [0] * 10
    for changes in (
        {"has_rolled": False},
        {"kind": "SETUP_ROAD"},
        {"actor": 20},
        {"turn_owner": 20},
    ):
        assert (
            context(replace(observation, context=replace(observation.context, **changes)))[7] == 0
        )
    assert context(replace(observation, context=replace(observation.context, kind="TURN")))[7] == 1
    assert context(response())[7:9].tolist() == [0, 1]
    board = BoardSpec.standard()
    assert (
        context(
            replace(
                observation, board=board, roads=tuple(Road(i, 10) for i in range(len(board.edges)))
            )
        )[5]
        == 1
    )
    board = BoardSpec.from_hexes(((0, 0), (1, 0)), ("WOOD", "WOOD"), (6, 8))
    # Shared vertex 1 sees two wood tiles: production adds, coverage counts once.
    got = context(replace(observation, board=board, buildings=(Building(1, 10, 2),)))
    assert got[14].item() == pytest.approx(20 / 30)
    assert got[19:].tolist() == [1, 0, 0, 0, 0]
    zero_board = BoardSpec.from_hexes(((0, 0),), ("WOOD",), (0,))
    assert (
        context(replace(observation, board=zero_board, buildings=(Building(0, 10),)))[14:].tolist()
        == [0] * 10
    )


@pytest.mark.parametrize("kind", ["context_linear", "nonlinear"])
def test_upgrade_trained_parent_exact_independent_rng_mode_and_scalar_batch_prefix(kind):
    policy = api()
    parent = trained_parent()
    parent.heads["setup"].train()  # Preserve mixed module modes too.
    rng = torch.get_rng_state().clone()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    child = policy.trading_residual_from_consequence(parent, kind=kind, seed=23)
    assert torch.equal(rng, torch.get_rng_state())
    assert py_rng == random.getstate()
    assert np.array_equal(np_rng[1], np.random.get_state()[1])
    assert [m.training for m in parent.modules()] == [
        m.training for name, m in child.named_modules() if name in dict(parent.named_modules())
    ]
    for name, tensor in parent.state_dict().items():
        assert torch.equal(tensor, child.state_dict()[name])
        assert tensor.data_ptr() != child.state_dict()[name].data_ptr()
    if kind == "context_linear":
        assert child.context_linear.bias is None
        assert torch.count_nonzero(child.context_linear.weight) == 0
    else:
        assert torch.count_nonzero(child.residual[0].weight) > 0
        assert torch.count_nonzero(child.residual[2].weight) == 0
        assert child.residual[2].bias is None
        duplicate = policy.trading_residual_from_consequence(parent, kind=kind, seed=23)
        assert torch.equal(child.residual[0].weight, duplicate.residual[0].weight)
        another = policy.trading_residual_from_consequence(parent, kind=kind, seed=24)
        assert not torch.equal(child.residual[0].weight, another.residual[0].weight)
    views = (view(), response(), replace(view(), legal_actions=()))
    memories = (torch.linspace(-1, 1, 16), None, None)
    for observation, memory in zip(views, memories):
        previous = memory.clone() if memory is not None else None
        before, after = parent(observation, memory), child(observation, memory)
        for field in ("logits", "value", "memory"):
            assert torch.equal(getattr(before, field), getattr(after, field))
        if memory is not None:
            assert torch.equal(memory, previous)
            assert after.memory.data_ptr() != memory.data_ptr()
    for before, after in zip(
        parent.forward_many(views, memories), child.forward_many(views, memories)
    ):
        for field in ("logits", "value", "memory"):
            assert torch.equal(getattr(before, field), getattr(after, field))
    calls = []
    residual = child.context_linear if kind == "context_linear" else child.residual
    handle = residual.register_forward_hook(lambda *args: calls.append(1))
    assert torch.equal(parent.recurrent_step(view()), child.recurrent_step(view()))
    for before, after in zip(
        parent.recurrent_many(views, memories), child.recurrent_many(views, memories)
    ):
        assert torch.equal(before, after)
    handle.remove()
    assert calls == []
    with pytest.raises(ValueError, match="parent|upgrade|consequence"):
        policy.trading_residual_from_consequence(child, kind=kind)
    double = policy.trading_residual_from_consequence(parent.double(), kind=kind)
    assert all(p.dtype == torch.float64 for p in double.parameters())
    assert all(p.device == next(parent.parameters()).device for p in double.parameters())


def test_upgrade_refuses_wrong_parents_and_unknown_kind():
    policy = api()
    for parent in (
        GraphPolicy(16),
        ConsequenceGraphPolicy(16, feature_set="hand_recipe_and_connectivity"),
    ):
        with pytest.raises(ValueError):
            policy.trading_residual_from_consequence(parent, kind="nonlinear")
    with pytest.raises(ValueError):
        policy.trading_residual_from_consequence(trained_parent(), kind="unknown")
    for width in (True, 0, 15, 17, 16.0):
        with pytest.raises(ValueError):
            policy.NonlinearConsequencePolicy(16, residual_hidden_size=width)


@pytest.mark.parametrize("kind", ["context_linear", "nonlinear"])
def test_nonzero_residual_masks_unsupported_actions_and_covers_both_response_choices(kind):
    policy = api()
    parent = trained_parent()
    child = policy.trading_residual_from_consequence(parent, kind=kind)
    with torch.no_grad():
        if kind == "context_linear":
            child.context_linear.weight.fill_(0.25)
        else:
            child.residual[0].weight.fill_(0.05)
            child.residual[0].bias.fill_(0.2)
            child.residual[2].weight.fill_(0.5)
    normal = view()
    normal = replace(
        normal,
        legal_actions=(
            *normal.legal_actions,
            Action("BUILD_ROAD", target=0),
            Action("ROLL"),
            Action("BUY_DEVELOPMENT_CARD"),
        ),
    )
    for observation in (normal, response()):
        before, after = parent(observation), child(observation)
        delta = after.logits - before.logits
        assert torch.all(delta[:2] != 0)
        assert torch.equal(delta[2:], torch.zeros_like(delta[2:]))
        assert torch.equal(before.memory, after.memory)
        assert torch.equal(before.value, after.value)
        torch.testing.assert_close(
            child.forward_many((observation,), (None,))[0].logits, after.logits, rtol=0, atol=0
        )


def test_nonlinear_output_gradient_then_hidden_gradient_after_one_output_update():
    child = api().trading_residual_from_consequence(trained_parent(), kind="nonlinear", seed=3)
    loss = F.cross_entropy(child(view()).logits[None], torch.tensor([0]))
    loss.backward()
    assert child.residual[2].weight.grad.abs().sum() > 0
    assert torch.count_nonzero(child.residual[0].weight.grad) == 0
    assert torch.count_nonzero(child.residual[0].bias.grad) == 0
    with torch.no_grad():
        child.residual[2].weight.sub_(0.2 * child.residual[2].weight.grad)
    child.zero_grad(set_to_none=True)
    F.cross_entropy(child(view()).logits[None], torch.tensor([0])).backward()
    assert child.residual[0].weight.grad.abs().sum() > 0
    assert child.residual[0].bias.grad.abs().sum() > 0


def test_same_hand_opposite_goal_states_nonlinear_can_reverse_trade_preference_linear_cannot():
    policy = api()
    parent = trained_parent()
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.zero_()
    settlement = own(view(), roads_left=15, settlements_left=5)
    no_settlement = own(
        replace(view(), buildings=(Building(0, 10),)), roads_left=0, settlements_left=0
    )
    assert settlement.hand == no_settlement.hand
    assert context(settlement)[10:12].tolist() == [1, 0]
    assert context(no_settlement)[10:12].tolist() == [0, 1]
    linear = policy.trading_residual_from_consequence(parent, kind="context_linear")
    nonlinear = policy.trading_residual_from_consequence(parent, kind="nonlinear")
    with torch.no_grad():
        linear.context_linear.weight.copy_(torch.linspace(-2, 3, 24)[None])
        nonlinear.residual[0].weight.zero_()
        nonlinear.residual[0].bias.zero_()
        nonlinear.residual[2].weight.zero_()
        # f(x,g)=tanh(10*x+2*g)-tanh(10*x-2+2*g).
        # x=post-trade wheat/10: 0.1 for trade, 0 for END; g=settlement gate.
        nonlinear.residual[0].weight[0:2, 4] = 10
        nonlinear.residual[0].weight[0:2, 34 + 10] = 2
        nonlinear.residual[0].bias[1] = -2
        nonlinear.residual[2].weight[0, 0:2] = torch.tensor([1.0, -1.0])
    preferences = []
    for observation in (settlement, no_settlement):
        before = parent(observation).logits
        logits = linear(observation).logits
        assert (logits[0] - logits[1]).item() == pytest.approx(
            (before[0] - before[1]).item(), abs=1e-6
        )
        nonlinear_logits = nonlinear(observation).logits
        preferences.append((nonlinear_logits[0] - nonlinear_logits[1]).item())
    assert preferences[0] < -0.5  # settlement gate changes the hand-dependent response
    assert preferences[1] > 0.5


@pytest.mark.parametrize("kind", ["context_linear", "nonlinear"])
def test_shared_hard_label_fitter_two_updates_with_frozen_graph_and_owned_adam(kind):
    from colonist_rl.supervised_fit import fit_sequences

    policy = api()
    parent = trained_parent()
    arms = [policy.trading_residual_from_consequence(parent, kind=kind, seed=3) for _ in range(2)]
    optimizers = []
    forced = replace(view(), legal_actions=(Action("END_TURN"),))
    choice1 = replace(view(), context=replace(view().context, version=2))
    choice2 = replace(response(), context=replace(response().context, version=3))
    episodes = (((forced, 0), (choice1, 0), (choice2, 1)),)
    for arm in arms:
        for name, parameter in arm.named_parameters():
            parameter.requires_grad_(
                name.startswith(("consequence.", "context_linear.", "residual."))
            )
        frozen = {n: p.detach().clone() for n, p in arm.named_parameters() if not p.requires_grad}
        before = arm.consequence.weight.detach().clone()
        optimizer = torch.optim.Adam(arm.parameters(), lr=0.02)
        optimizers.append(optimizer)
        result = fit_sequences(
            arm,
            episodes,
            epochs=1,
            sequence_length=1,
            optimizer=optimizer,
            batch_size=1,
            profile=True,
        )
        assert result["updates"] == 2
        assert result["profile"]["prefix_forwards"] == 3
        assert result["supervised_presentations"] == 2
        assert not torch.equal(before, arm.consequence.weight)
        assert not arm.training
        for name, tensor in frozen.items():
            assert torch.equal(tensor, dict(arm.named_parameters())[name])
            assert dict(arm.named_parameters())[name].grad is None
        assert all(p in optimizer.state for p in arm.parameters() if p.requires_grad)
        assert all(p not in optimizer.state for p in arm.parameters() if not p.requires_grad)
    for left, right in zip(arms[0].parameters(), arms[1].parameters()):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr()
        if left in optimizers[0].state:
            a, b = optimizers[0].state[left], optimizers[1].state[right]
            assert a["exp_avg"].data_ptr() != b["exp_avg"].data_ptr()
    incomplete = torch.optim.Adam(p for p in arms[0].parameters() if p.requires_grad)
    with pytest.raises(ValueError, match="every policy parameter"):
        fit_sequences(arms[0], episodes, epochs=1, optimizer=incomplete)


def test_nonlinear_input_masks_parent_connectivity_slots_even_when_all_are_available():
    from colonist_rl.consequences import consequence_features

    parent = trained_parent()
    child = api().trading_residual_from_consequence(parent, kind="nonlinear")
    observation = replace(
        view(),
        hand=(1, 1, 1, 2, 3),
        roads=(Road(1, 10),),
        buildings=(Building(0, 10),),
        legal_actions=(Action("END_TURN"),),
    )
    assert consequence_features(observation, observation.legal_actions[0])[30:].tolist() == [
        0.125,
        1,
        1,
        1,
    ]
    with torch.no_grad():
        child.residual[0].weight.zero_()
        child.residual[0].bias.zero_()
        child.residual[0].weight[0, 30:34] = 1
        child.residual[2].weight.zero_()
        child.residual[2].weight[0, 0] = 1
    assert torch.equal(child(observation).logits, parent(observation).logits)
