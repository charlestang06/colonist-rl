import numpy as np
import torch

from colonist_rl.policy import GraphPolicy
from colonist_rl.training import advantages, load_checkpoint, save_checkpoint


def test_gae_bootstraps_cutoff_and_stops_at_episode_boundary():
    adv, returns = advantages([0.0, 1.0], [0.2, 0.3], [0.3, 0.0], [False, True], lam=1.0)
    np.testing.assert_allclose(returns, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(adv, [0.8, 0.7], atol=1e-6)
    adv, returns = advantages([0.0], [0.2], [0.7], [True], lam=1.0)
    np.testing.assert_allclose(returns, [0.7])
    np.testing.assert_allclose(adv, [0.5])


def test_checkpoint_roundtrip_and_resume_optimizer(tmp_path):
    original = GraphPolicy(hidden_size=32)
    optimizer = torch.optim.Adam(original.parameters(), lr=3e-4)
    path = tmp_path / "model.pt"
    save_checkpoint(path, original, optimizer, steps=12)
    loaded, state = load_checkpoint(path)
    assert state["steps"] == 12
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[key])


def test_opt_in_checkpoint_identities_roundtrip_with_optimizer_and_strict_parameters(tmp_path):
    from colonist_rl.policy import consequence_from_legacy, policy_architecture

    for feature_set in ("hand_recipe", "hand_recipe_and_connectivity"):
        policy = consequence_from_legacy(GraphPolicy(16), feature_set=feature_set)
        with torch.no_grad():
            policy.consequence.weight[0, 4] = 0.75
        optimizer = torch.optim.Adam(policy.parameters())
        policy.consequence.weight.sum().backward()
        optimizer.step()
        path = tmp_path / (feature_set + ".pt")
        save_checkpoint(path, policy, optimizer, steps=0, metadata={"purpose": "opt-in"})
        loaded, state = load_checkpoint(path)
        assert state["schema_version"] == 1
        assert state["steps"] == 0
        assert state["metadata"] == {"purpose": "opt-in"}
        assert policy_architecture(loaded) == policy_architecture(policy) == state["architecture"]
        assert all(
            torch.equal(value, loaded.state_dict()[key])
            for key, value in policy.state_dict().items()
        )
        resumed = torch.optim.Adam(loaded.parameters())
        resumed.load_state_dict(state["optimizer_state"])
        assert len(resumed.state) == 1
        del state["policy_state"]["consequence.weight"]
        torch.save(state, path)
        import pytest

        with pytest.raises(RuntimeError, match="consequence.weight"):
            load_checkpoint(path)


def test_architecture_dispatch_rejects_unknown_shapes_schema_feature_set_and_sizes():
    import pytest

    from colonist_rl.policy import policy_from_architecture

    tagged = {
        "kind": "graph_consequence_v1",
        "hidden_size": 16,
        "feature_schema": "public-consequence-v1",
        "feature_set": "hand_recipe",
    }
    for architecture in (
        None,
        [],
        {},
        {"hidden_size": 0},
        {"hidden_size": True},
        {"hidden_size": 16, "extra": 0},
        {**tagged, "kind": "unknown"},
        {**tagged, "feature_schema": "future"},
        {**tagged, "feature_set": "future"},
        {**tagged, "extra": 0},
        {key: value for key, value in tagged.items() if key != "feature_set"},
    ):
        with pytest.raises(ValueError, match="architecture|feature"):
            policy_from_architecture(architecture)
    assert type(policy_from_architecture({"hidden_size": 16})) is GraphPolicy


def test_curriculum_dispatch_and_frozen_history_preserve_zero_step_opt_in_checkpoint(tmp_path):
    from colonist_rl import curriculum
    from colonist_rl.policy import consequence_from_legacy, policy_architecture
    from colonist_rl.training import RolloutCollector

    original = consequence_from_legacy(GraphPolicy(16), feature_set="hand_recipe_and_connectivity")
    path = tmp_path / "initial.pt"
    save_checkpoint(path, original, steps=0)
    before = path.read_bytes()
    loaded, state = curriculum.load_checkpoint(path)
    collector = RolloutCollector(loaded, historical_policies=[loaded], deadline=0)
    frozen = collector.historical_policies[0]
    assert state["steps"] == 0
    assert policy_architecture(frozen) == policy_architecture(original)
    assert not frozen.training
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    assert all(
        torch.equal(value, frozen.state_dict()[key]) for key, value in original.state_dict().items()
    )
    with torch.no_grad():
        loaded.consequence.weight.add_(1)
    assert torch.count_nonzero(frozen.consequence.weight) == 0
    assert frozen.consequence.weight.data_ptr() != loaded.consequence.weight.data_ptr()
    assert path.read_bytes() == before
