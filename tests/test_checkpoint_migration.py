"""Disposable, explicitly legacy-labelled fixtures; no active checkpoints are read."""

import importlib
import pickle
import random
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch

from colonist_rl.engine_identity import BACKEND_VERSION
from colonist_rl.policy import (
    KINDS,
    ConsequenceGraphPolicy,
    GraphPolicy,
    policy_architecture,
)
from colonist_rl.training import load_checkpoint, save_checkpoint


def bridge():
    # A missing feature should fail a test, rather than prevent collection.
    assert importlib.util.find_spec("colonist_rl.checkpoint_migration") is not None
    return importlib.import_module("colonist_rl.checkpoint_migration")


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(expected):
    actual = rng_state()
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])


@pytest.fixture(autouse=True)
def preserve_test_rng():
    before = rng_state()
    yield
    random.setstate(before[0])
    np.random.set_state(before[1])
    torch.set_rng_state(before[2])


@pytest.fixture
def legacy(tmp_path):
    """Create only a disposable synthetic 3.2.1-labelled checkpoint."""
    return make_legacy(tmp_path)


def make_legacy(tmp_path, feature_set=None):
    random.seed(991)
    np.random.seed(991)
    torch.manual_seed(991)
    policy = (
        GraphPolicy(16)
        if feature_set is None
        else ConsequenceGraphPolicy(16, feature_set=feature_set)
    )
    optimizer = torch.optim.Adam(policy.parameters())
    sum(parameter.sum() for parameter in policy.parameters()).backward()
    optimizer.step()
    path = tmp_path / "disposable-legacy-3.2.1.pt"
    save_checkpoint(path, policy, optimizer, steps=321, metadata={"old_metrics": {"win_rate": 0.9}})
    state = torch.load(path, weights_only=True)
    state["backend_version"] = "3.2.1"
    state["evaluation_seeds"] = [9001]
    torch.save(state, path)
    return path, state


def test_default_checkpoint_loader_still_rejects_legacy(legacy):
    path, _ = legacy
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(path)


def test_inspection_reports_labels_and_hash_without_rng_or_source_mutation(legacy):
    path, state = legacy
    original = path.read_bytes()
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    before = rng_state()
    inspected = bridge().inspect_checkpoint(path)
    assert inspected == {
        "source_sha256": sha256(original).hexdigest(),
        "schema_version": 1,
        "backend_version": "3.2.1",
        "action_kinds": list(KINDS),
        "architecture": {"hidden_size": 16},
        "steps": 321,
    }
    assert state["optimizer_state"]["state"]
    assert_rng_equal(before)
    assert path.read_bytes() == original


class UnsafePayload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "unsafe pickle executed")


@pytest.mark.parametrize("operation", ["inspect", "migrate"])
def test_restricted_loading_never_executes_pickle(tmp_path, operation):
    marker = tmp_path / "unsafe-pickle-marker"
    source = tmp_path / "unsafe.pt"
    destination = tmp_path / "new.pt"
    torch.save(UnsafePayload(marker), source)
    before = rng_state()
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        if operation == "inspect":
            bridge().inspect_checkpoint(source)
        else:
            bridge().migrate_legacy_weights(source, destination)
    assert not marker.exists()
    assert not destination.exists()
    assert_rng_equal(before)


@pytest.mark.parametrize("feature_set", [None, "hand_recipe", "hand_recipe_and_connectivity"])
def test_transfer_copies_exact_policy_only_and_records_fresh_seeded_state(tmp_path, feature_set):
    source, legacy_state = make_legacy(tmp_path, feature_set)
    original = source.read_bytes()
    before = rng_state()
    destination = tmp_path / "new-cohort" / "weights.pt"
    provenance = bridge().migrate_legacy_weights(source, destination, seed=17)
    assert_rng_equal(before)
    assert provenance == {
        "kind": "weights_only_cross_engine",
        "source_sha256": sha256(original).hexdigest(),
        "source_backend_version": "3.2.1",
        "target_backend_version": BACKEND_VERSION,
        "source_steps": 321,
        "seed": 17,
    }
    assert source.read_bytes() == original
    assert sha256(source.read_bytes()).hexdigest() == provenance["source_sha256"]
    policy, state = load_checkpoint(destination)
    assert state["backend_version"] == BACKEND_VERSION
    assert state["steps"] == 0
    assert state["optimizer_state"] is None
    assert state["metadata"] == {"migration": provenance}
    assert "evaluation_seeds" not in state
    assert policy_architecture(policy) == legacy_state["architecture"] == state["architecture"]
    assert set(policy.state_dict()) == set(legacy_state["policy_state"])
    for name, tensor in legacy_state["policy_state"].items():
        copied = policy.state_dict()[name]
        assert copied.dtype == tensor.dtype
        assert torch.equal(copied, tensor)
    # These are the independently seeded generators, with no legacy RNG restored.
    assert random.getstate() == random.Random(17).getstate()
    np.testing.assert_array_equal(
        np.random.get_state()[1], np.random.RandomState(17).get_state()[1]
    )
    assert np.random.get_state()[2:] == np.random.RandomState(17).get_state()[2:]
    assert torch.equal(torch.get_rng_state(), torch.Generator().manual_seed(17).get_state())
    assert state["rng"]["python"] != legacy_state["rng"]["python"]
    assert not torch.equal(state["rng"]["torch"], legacy_state["rng"]["torch"])


def test_legacy_optimizer_rng_and_metrics_are_not_consumed(legacy, tmp_path):
    source, state = legacy
    state["optimizer_state"] = "not an optimizer"
    state["rng"] = "not a restorable RNG"
    state["metadata"] = {"metrics": "do not copy", "migration": "do not inherit"}
    torch.save(state, source)
    destination = tmp_path / "new.pt"
    bridge().migrate_legacy_weights(source, destination)
    _, migrated = load_checkpoint(destination)
    assert migrated["steps"] == 0 and migrated["optimizer_state"] is None
    assert set(migrated["metadata"]) == {"migration"}


def test_cpu_transfer_ignores_ambient_default_device(legacy, tmp_path):
    source, legacy_state = legacy
    destination = tmp_path / "new.pt"
    before = rng_state()
    # Meta is available without a GPU and exposes accidental non-CPU construction.
    with torch.device("meta"):
        bridge().migrate_legacy_weights(source, destination)
    assert_rng_equal(before)
    policy, _ = load_checkpoint(destination)
    assert all(tensor.device.type == "cpu" for tensor in policy.state_dict().values())
    assert all(
        torch.equal(tensor, policy.state_dict()[name])
        for name, tensor in legacy_state["policy_state"].items()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", "1"),
        ("backend_version", "unknown"),
        ("backend_version", BACKEND_VERSION),
        ("backend_version", "3.3.0"),
        ("action_kinds", list(reversed(KINDS))),
        ("action_kinds", list(KINDS) + ["OFFER_TRADE"]),
        ("action_kinds", tuple(KINDS)),
        ("architecture", None),
        ("architecture", {"hidden_size": True}),
        ("architecture", {"hidden_size": 0}),
        ("architecture", {"hidden_size": 16, "extra": 0}),
        ("architecture", {"kind": "unknown", "hidden_size": 16}),
        ("steps", True),
        ("steps", -1),
        ("steps", "321"),
        ("policy_state", None),
    ],
)
def test_invalid_identity_or_contract_is_rejected_without_output_or_rng_changes(
    legacy, tmp_path, field, value
):
    source, state = legacy
    state[field] = value
    torch.save(state, source)
    original = source.read_bytes()
    before = rng_state()
    destination = tmp_path / "new.pt"
    with pytest.raises(ValueError):
        bridge().migrate_legacy_weights(source, destination, seed=17)
    assert not destination.exists()
    assert source.read_bytes() == original
    assert_rng_equal(before)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "non_tensor"])
def test_state_dict_requires_exact_tensor_keys_shapes_and_dtypes(legacy, tmp_path, mutation):
    source, state = legacy
    tensors = state["policy_state"]
    name = next(iter(tensors))
    if mutation == "missing":
        del tensors[name]
    elif mutation == "extra":
        tensors["unrecognized.weight"] = torch.zeros(1)
    elif mutation == "shape":
        tensors[name] = tensors[name][:1]
    elif mutation == "dtype":
        tensors[name] = tensors[name].double()
    else:
        tensors[name] = [0]
    torch.save(state, source)
    before = rng_state()
    with pytest.raises(ValueError, match="policy|tensor|state"):
        bridge().migrate_legacy_weights(source, tmp_path / "new.pt")
    assert not (tmp_path / "new.pt").exists()
    assert_rng_equal(before)


@pytest.mark.parametrize("kind", ["same", "existing", "symlink", "dangling", "hardlink", "alias"])
def test_destination_never_clobbers_source_existing_files_or_aliases(legacy, tmp_path, kind):
    source, _ = legacy
    destination = tmp_path / "destination.pt"
    if kind == "same":
        destination = source
    elif kind == "existing":
        destination.write_bytes(b"existing destination")
    elif kind == "symlink":
        destination.symlink_to(source)
    elif kind == "dangling":
        destination.symlink_to(tmp_path / "absent.pt")
    elif kind == "hardlink":
        destination.hardlink_to(source)
    else:
        alias = tmp_path / "directory-alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        destination = alias / source.name
    original = source.read_bytes()
    existing = destination.read_bytes() if destination.exists() else None
    before = rng_state()
    with pytest.raises((ValueError, FileExistsError)):
        bridge().migrate_legacy_weights(source, destination)
    assert source.read_bytes() == original
    assert_rng_equal(before)
    if existing is not None:
        assert destination.read_bytes() == existing
    else:
        assert destination.is_symlink() and not destination.exists()


@pytest.mark.parametrize("rival", ["file", "source_symlink"])
def test_concurrent_destination_creation_wins_without_replacement(
    legacy, tmp_path, monkeypatch, rival
):
    module = bridge()
    source, _ = legacy
    destination = tmp_path / "new.pt"
    original = source.read_bytes()
    real_save = module.save_checkpoint

    def create_rival_after_staging(*args, **kwargs):
        real_save(*args, **kwargs)
        if rival == "file":
            destination.write_bytes(b"concurrent owner")
        else:
            destination.symlink_to(source)

    monkeypatch.setattr(module, "save_checkpoint", create_rival_after_staging)
    before = rng_state()
    with pytest.raises(FileExistsError):
        module.migrate_legacy_weights(source, destination)
    assert source.read_bytes() == original
    assert_rng_equal(before)
    assert destination.read_bytes() == (b"concurrent owner" if rival == "file" else original)
    assert set(tmp_path.iterdir()) == {source, destination}


def test_target_installation_must_verify_before_publication(legacy, tmp_path, monkeypatch):
    module = bridge()
    source, _ = legacy
    destination = tmp_path / "new.pt"

    def invalid_installation():
        raise RuntimeError("wrong target commit provenance")

    monkeypatch.setattr(module, "verify_engine_installation", invalid_installation)
    before = rng_state()
    with pytest.raises(RuntimeError, match="target commit"):
        module.migrate_legacy_weights(source, destination)
    assert_rng_equal(before)
    assert not destination.exists()
