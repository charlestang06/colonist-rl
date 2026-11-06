"""The opt-in successor policy starts at its consequence parent's exact logits."""

from copy import deepcopy

import pytest
import torch

from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, DecisionContext, Observation, PublicPlayer, Road
from colonist_rl.policy import GraphPolicy, consequence_from_legacy
from colonist_rl.successor_features import SUCCESSOR_DIM, SUCCESSOR_SCHEMA
from colonist_rl.training import load_checkpoint, save_checkpoint


def _view():
    return Observation(
        10, BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,)), RulesConfig.standard(),
        DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        tuple(PublicPlayer(i, 2 if i == 10 else 0, 0, 0) for i in (10, 20, 30, 40)),
        (4, 1, 1, 3, 3), (0, 0, 0, 0, 0), (), (Road(0, 10),), 99,
        (
            Action("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 1, 0, 0)),
            Action("BUILD_SETTLEMENT", target=0), Action("BUY_DEVELOPMENT_CARD"),
            Action("END_TURN"),
        ),
    )


def _parent():
    parent = consequence_from_legacy(GraphPolicy(16)).eval()
    with torch.no_grad():
        parent.consequence.weight[0, 1] = 0.37
    parent.node_in.weight.requires_grad_(False)
    parent.heads["turn"].train(True)
    return parent


def test_conversion_preserves_exact_parent_outputs_rng_parameters_and_mixed_modes():
    from colonist_rl.policy import candidate_successor_from_consequence

    torch.manual_seed(19)
    parent = _parent()
    rng = torch.get_rng_state().clone()
    child = candidate_successor_from_consequence(parent, seed=7)
    assert torch.equal(rng, torch.get_rng_state())
    assert child.training == parent.training
    for name, parameter in parent.named_parameters():
        copied = dict(child.named_parameters())[name]
        assert torch.equal(parameter, copied)
        assert parameter.data_ptr() != copied.data_ptr()
        assert parameter.requires_grad == copied.requires_grad
    assert child.heads["turn"].training == parent.heads["turn"].training
    assert child.heads["setup"].training == parent.heads["setup"].training
    assert torch.count_nonzero(child.successor_linear.weight) == 0
    assert torch.count_nonzero(child.successor_out.weight) == 0

    observation = _view()
    original = deepcopy(observation)
    before, after = parent(observation), child(observation)
    for field in ("logits", "memory", "value"):
        assert torch.equal(getattr(before, field), getattr(after, field))
    assert observation == original
    with pytest.raises(ValueError, match="hand_recipe"):
        candidate_successor_from_consequence(
            consequence_from_legacy(GraphPolicy(16), feature_set="hand_recipe_and_connectivity")
        )
    with pytest.raises(ValueError, match="parent"):
        candidate_successor_from_consequence(child)


def test_candidate_features_receive_first_update_gradients_while_value_stays_inherited():
    from colonist_rl.policy import candidate_successor_from_consequence

    child = candidate_successor_from_consequence(_parent())
    output = child(_view())
    torch.nn.functional.cross_entropy(output.logits[None], torch.tensor([1])).backward()
    assert child.successor_linear.weight.grad.abs().sum().item() > 0
    assert child.successor_out.weight.grad.abs().sum().item() > 0
    assert child.value.weight.grad is None


def test_conversion_preserves_mixed_parent_dtypes_and_exact_logits():
    from colonist_rl.policy import candidate_successor_from_consequence

    parent = _parent()
    parent.consequence.double()
    child = candidate_successor_from_consequence(parent, seed=41)
    for name, parameter in parent.named_parameters():
        copied = dict(child.named_parameters())[name]
        assert copied.dtype == parameter.dtype
        assert torch.equal(copied, parameter)
        assert copied.data_ptr() != parameter.data_ptr()
    observation = _view()
    before, after = parent(observation), child(observation)
    for field in ("logits", "memory", "value"):
        assert torch.equal(getattr(before, field), getattr(after, field))


def test_conversion_constructs_on_cpu_under_ambient_meta_default():
    from colonist_rl.policy import candidate_successor_from_consequence

    parent = _parent()
    with torch.device("meta"):
        first = candidate_successor_from_consequence(parent, seed=47)
        second = candidate_successor_from_consequence(parent, seed=47)
    assert all(parameter.device.type == "cpu" for parameter in first.parameters())
    assert torch.equal(first.successor_hidden.weight, second.successor_hidden.weight)
    assert torch.equal(parent(_view()).logits, first(_view()).logits)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_conversion_does_not_advance_ambient_mps_rng():
    from colonist_rl.policy import candidate_successor_from_consequence

    parent = _parent()
    before = torch.mps.get_rng_state().clone()
    with torch.device("mps"):
        first = candidate_successor_from_consequence(parent, seed=53)
        second = candidate_successor_from_consequence(parent, seed=53)
    assert torch.equal(torch.mps.get_rng_state(), before)
    assert torch.equal(first.successor_hidden.weight, second.successor_hidden.weight)
    assert torch.equal(parent(_view()).logits, first(_view()).logits)


def test_new_checkpoint_has_strict_identity_and_roundtrips(tmp_path):
    from colonist_rl.policy import (
        CandidateSuccessorPolicy,
        candidate_successor_from_consequence,
        policy_architecture,
        policy_from_architecture,
    )

    child = candidate_successor_from_consequence(_parent())
    architecture = policy_architecture(child)
    assert architecture == {
        "kind": "graph_candidate_successor_v3",
        "hidden_size": 16,
        "feature_schema": "public-consequence-v1",
        "feature_set": "hand_recipe",
        "successor_schema": SUCCESSOR_SCHEMA,
        "successor_dim": SUCCESSOR_DIM,
        "residual_hidden_size": 16,
    }
    assert type(policy_from_architecture(architecture)) is CandidateSuccessorPolicy
    path = tmp_path / "candidate.pt"
    save_checkpoint(path, child, steps=4, metadata={"purpose": "representation_preflight"})
    loaded, state = load_checkpoint(path)
    assert type(loaded) is CandidateSuccessorPolicy
    assert state["architecture"] == architecture
    assert state["steps"] == 4
    assert all(torch.equal(t, loaded.state_dict()[name]) for name, t in child.state_dict().items())
    for wrong in (
        {**architecture, "kind": "graph_candidate_successor_v1"},
        {**architecture, "kind": "graph_candidate_successor_v2"},
        {**architecture, "successor_schema": "future"},
        {**architecture, "successor_dim": SUCCESSOR_DIM + 1},
        {**architecture, "feature_set": "hand_recipe_and_connectivity"},
        {**architecture, "extra": True},
        {key: value for key, value in architecture.items() if key != "successor_schema"},
    ):
        with pytest.raises(ValueError, match="architecture|schema|feature"):
            policy_from_architecture(wrong)
    damaged = dict(state)
    damaged["policy_state"] = {
        key: value for key, value in state["policy_state"].items()
        if key != "successor_linear.weight"
    }
    torch.save(damaged, path)
    with pytest.raises(RuntimeError, match="successor_linear.weight"):
        load_checkpoint(path)
