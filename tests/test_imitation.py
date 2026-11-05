"""Observation-only collection, persistence and current-weight recurrent replay."""

import gzip
import hashlib
import importlib
import json
import math
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
from torch import nn

from colonist_rl.policy import GraphPolicy, PolicyOutput
from colonist_rl.simulator import FourPlayerGame


def _imitation():
    # A missing interface is an explicit assertion failure in the initial RED run.
    assert importlib.util.find_spec("colonist_rl.imitation") is not None, (
        "persisted observation-only imitation interfaces are missing"
    )
    return importlib.import_module("colonist_rl.imitation")


def test_collector_keeps_actual_board_seeds_legal_choices_and_hard_caps():
    imitation = _imitation()
    dataset = imitation.collect_demonstrations(
        board_seeds=[31, 57, 93, 108], max_decisions=3, max_game_steps=1
    )
    assert dataset["collection"]["actual_board_seeds"] == [31, 57, 93, 108]
    assert [e["seat"] for e in dataset["episodes"]] == [0, 1, 2, 3]
    assert dataset["coverage"]["decisions"] == 1
    for episode in dataset["episodes"]:
        assert not episode["complete"]
        assert episode["incomplete_reason"] == "step_cap"
        assert episode["steps"] <= 1
        env = FourPlayerGame()
        env.reset(seed=episode["board_seed"])
        for sample in episode["samples"]:
            view = sample["observation"]
            assert view.board == env.observe().board
            assert view.player_id == episode["seat"]
            assert 0 <= sample["action_index"] < len(view.legal_actions)
            with pytest.raises(FrozenInstanceError):
                view.player_id = 9
    expected_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert dataset["source"]["revision"] == expected_revision
    assert dataset["source"]["dirty"] == bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    capped = imitation.collect_demonstrations(board_seeds=[31, 57], max_decisions=2)
    assert capped["coverage"]["decisions"] == 2
    assert len(capped["episodes"]) == 1
    assert capped["episodes"][-1]["incomplete_reason"] == "decision_cap"
    assert capped["episodes"][-1]["complete"] is False


def test_gzip_roundtrip_and_rejects_incompatible_or_illegal_labels(tmp_path):
    imitation = _imitation()
    dataset = imitation.collect_demonstrations(board_seeds=[91], max_decisions=5)
    path = tmp_path / "teacher.json.gz"
    digest = imitation.save_demonstrations(path, dataset)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert imitation.load_demonstrations(path) == dataset
    raw = json.loads(gzip.decompress(path.read_bytes()))
    assert set(raw["episodes"][0]["samples"][0]) == {"observation", "action_index"}
    assert "memory" not in json.dumps(raw)
    for key, value in (("backend_version", "wrong"), ("action_kinds", []), ("version", 99)):
        bad = deepcopy(raw)
        bad[key] = value
        path.write_bytes(gzip.compress(json.dumps(bad).encode()))
        with pytest.raises(ValueError):
            imitation.load_demonstrations(path)
    for index in (-1, 1000000, True):
        bad = deepcopy(raw)
        bad["episodes"][0]["samples"][0]["action_index"] = index
        path.write_bytes(gzip.compress(json.dumps(bad).encode()))
        with pytest.raises(ValueError):
            imitation.load_demonstrations(path)


def test_expired_deadline_does_not_start_an_episode():
    imitation = _imitation()
    data = imitation.collect_demonstrations(
        board_seeds=[1], max_decisions=5, deadline=time.monotonic() - 1
    )
    assert data["episodes"] == []
    assert data["collection"]["stop_reason"] == "deadline"
    assert data["coverage"]["decisions"] == 0


class StateSensitivePolicy(nn.Module):
    """Small recurrent actor: a wrong prefix directly changes subsequent logits."""

    def __init__(self, consume_rng=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.child = nn.Dropout()
        self.consume_rng = consume_rng
        self.calls = []

    def forward(self, view, memory=None):
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(1)
        incoming = None if memory is None else memory.detach().item()
        self.calls.append(
            (view.context.version, self.weight.item(), incoming, torch.is_grad_enabled())
        )
        memory = self.weight * (view.context.version + 1) + (0 if memory is None else memory)
        logits = torch.stack([memory * (i + 1) for i in range(len(view.legal_actions))])
        return PolicyOutput(logits, memory * 0, memory)


def _sequence_dataset(imitation, forced=False):
    data = imitation.collect_demonstrations(board_seeds=[18], max_decisions=4)
    original = data["episodes"][0]["samples"]
    for i, sample in enumerate(original):
        view = sample["observation"]
        sample["observation"] = replace(
            view,
            context=replace(view.context, version=i),
            legal_actions=view.legal_actions[: 1 if forced or i in (0, 2) else 2],
        )
        sample["action_index"] = 0
    # Synthetic coverage describes exactly the retained observation sequence.
    phases, kinds = {}, {}
    for sample in original:
        view = sample["observation"]
        phases[view.context.kind] = phases.get(view.context.kind, 0) + 1
        kind = view.legal_actions[0].kind
        kinds[kind] = kinds.get(kind, 0) + 1
    data["coverage"] = {
        "decisions": 4,
        "non_forced_decisions": 0 if forced else 2,
        "phase_counts": phases,
        "action_kind_counts": kinds,
    }
    return data


def test_validation_preserves_parameters_module_modes_and_all_rng():
    imitation = _imitation()
    torch.set_num_threads(1)
    data = _sequence_dataset(imitation)
    policy = StateSensitivePolicy(consume_rng=True).train()
    policy.child.eval()
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    metrics = imitation.imitation_metrics(policy, data)
    assert metrics["non_forced"]["count"] == 2
    assert metrics["all"]["count"] == 4
    assert metrics["non_forced"]["cross_entropy"] > 0
    assert metrics["all"]["cross_entropy"] == pytest.approx(
        metrics["non_forced"]["cross_entropy"] / 2
    )
    assert policy.training and not policy.child.training
    assert all(torch.equal(v, policy.state_dict()[k]) for k, v in before.items())
    assert random.getstate() == rng[0]
    current = np.random.get_state()
    assert current[0] == rng[1][0] and np.array_equal(current[1], rng[1][1])
    assert current[2:] == rng[1][2:]
    assert torch.equal(torch.get_rng_state(), rng[2])
    json.dumps(metrics, allow_nan=False)


def test_all_forced_has_no_supervision_and_no_update():
    imitation = _imitation()
    data = _sequence_dataset(imitation, forced=True)
    policy = StateSensitivePolicy()
    metrics = imitation.imitation_metrics(policy, data)
    assert metrics["non_forced"]["count"] == 0
    assert metrics["non_forced"]["accuracy"] is None
    assert metrics["non_forced"]["cross_entropy"] is None
    before = policy.weight.detach().clone()
    fitted = imitation.fit_imitation(policy, data, epochs=2, sequence_length=2)
    assert fitted["updates"] == 0
    assert fitted["supervised_decisions"] == 0
    assert fitted["loss"] is None
    assert torch.equal(before, policy.weight)


@pytest.mark.parametrize("phase_weight", [None, 0.25])
def test_recurrent_chunks_replay_current_weight_prefixes_and_reset_episodes(phase_weight):
    imitation = _imitation()
    data = _sequence_dataset(imitation)
    second = deepcopy(data["episodes"][0])
    second.update(board_seed=19, backend_seed=20, seat=1, teacher_seed=119)
    second["samples"] = [
        {
            **s,
            "observation": replace(
                s["observation"], player_id=1, context=replace(s["observation"].context, actor=1)
            ),
        }
        for s in second["samples"]
    ]
    data["episodes"].append(second)
    data["collection"]["actual_board_seeds"] = [18, 19]
    data["collection"]["max_decisions"] = 8
    data["coverage"] = {
        k: {a: b * 2 for a, b in v.items()} if isinstance(v, dict) else v * 2
        for k, v in data["coverage"].items()
    }
    policy = StateSensitivePolicy()
    weights = (
        None
        if phase_weight is None
        else {
            s["observation"].context.kind: phase_weight
            for e in data["episodes"]
            for s in e["samples"]
            if len(s["observation"].legal_actions) > 1
        }
    )
    result = imitation.fit_imitation(
        policy, data, epochs=2, sequence_length=2, learning_rate=0.02, phase_weights=weights
    )
    assert result["updates"] == 8
    assert result["supervised_decisions"] == 4
    assert result["supervised_presentations"] == 8
    assert math.isfinite(result["loss"])
    # Each episode has chunks [0,1], then replay [0,1] and train [2,3].
    assert [c[0] for c in policy.calls] == [0, 1, 0, 1, 2, 3] * 4
    for start in range(0, len(policy.calls), 6):
        calls = policy.calls[start : start + 6]
        assert calls[0][2] is None and calls[2][2] is None
        assert calls[2][1] != calls[0][1]
        assert calls[2][3] is False and calls[3][3] is False
        assert calls[4][3] is True
        assert calls[4][2] == pytest.approx(calls[4][1] * 3)
    assert all(
        set(s) == {"observation", "action_index"} for e in data["episodes"] for s in e["samples"]
    )


def test_graph_policy_small_dataset_optimization_and_legacy_warmstart():
    imitation = _imitation()
    from colonist_rl.training import imitation_warmstart

    torch.set_num_threads(1)
    torch.manual_seed(22)
    data = imitation.collect_demonstrations(board_seeds=[3], max_decisions=4)
    policy = GraphPolicy(hidden_size=8)
    before = imitation.imitation_metrics(policy, data)["non_forced"]["cross_entropy"]
    result = imitation.fit_imitation(policy, data, epochs=3, learning_rate=0.01)
    after = imitation.imitation_metrics(policy, data)["non_forced"]["cross_entropy"]
    assert after < before
    assert result["updates"] == 3
    legacy = imitation_warmstart(policy, decisions=3, epochs=1, seed=3)
    assert legacy["decisions"] == 3 and legacy["epochs"] == 1
    assert math.isfinite(legacy["loss"])


def test_training_deadline_after_last_chunk_forward_skips_update(monkeypatch):
    imitation = _imitation()
    data = _sequence_dataset(imitation)
    now = [0.0]
    monkeypatch.setattr(imitation.time, "monotonic", lambda: now[0])

    class DeadlinePolicy(StateSensitivePolicy):
        def forward(self, view, memory=None):
            result = super().forward(view, memory)
            if view.context.version == 1:
                now[0] = 2.0
            return result

    policy = DeadlinePolicy()
    before = policy.weight.detach().clone()
    result = imitation.fit_imitation(policy, data, epochs=1, sequence_length=2, deadline=1.0)
    assert result["status"] == "incomplete"
    assert result["epochs"] == 0 and result["updates"] == 0
    assert torch.equal(policy.weight, before)


def test_loader_rejects_nonfinite_observations_and_extra_private_labels(tmp_path):
    imitation = _imitation()
    data = imitation.collect_demonstrations(board_seeds=[8], max_decisions=2)
    path = tmp_path / "data.gz"
    imitation.save_demonstrations(path, data)
    original = json.loads(gzip.decompress(path.read_bytes()))
    bad = deepcopy(original)
    bad["episodes"][0]["samples"][0]["observation"]["observation"]["players"][0][
        "victory_points"
    ] = math.nan
    path.write_bytes(gzip.compress(json.dumps(bad).encode()))
    with pytest.raises(ValueError):
        imitation.load_demonstrations(path)
    bad = deepcopy(original)
    bad["episodes"][0]["samples"][0]["engine_return"] = 1.0
    path.write_bytes(gzip.compress(json.dumps(bad).encode()))
    with pytest.raises(ValueError):
        imitation.load_demonstrations(path)


def test_forced_only_chunk_is_replayed_for_later_supervision():
    imitation = _imitation()
    data = _sequence_dataset(imitation)
    policy = StateSensitivePolicy()
    result = imitation.fit_imitation(policy, data, epochs=1, sequence_length=1)
    assert result["updates"] == 2 and result["supervised_presentations"] == 2
    assert [call[0] for call in policy.calls] == [0, 1, 0, 1, 2, 3]
    assert [call[3] for call in policy.calls] == [False, True, False, False, False, True]
    assert policy.calls[-1][2] == pytest.approx(policy.calls[-1][1] * 6)


def test_collection_deadline_after_actor_prevents_later_step(monkeypatch):
    imitation = _imitation()
    now = [0.0]
    monkeypatch.setattr(imitation.time, "monotonic", lambda: now[0])
    original = imitation.HeuristicPolicy

    class ExpiringTeacher(original):
        def act(self, view):
            action = super().act(view)
            now[0] = 2.0
            return action

    monkeypatch.setattr(imitation, "HeuristicPolicy", ExpiringTeacher)
    data = imitation.collect_demonstrations(board_seeds=[6], max_decisions=5, deadline=1.0)
    assert data["collection"]["stop_reason"] == "deadline"
    episode = data["episodes"][0]
    assert episode["steps"] == 0
    assert not episode["complete"] and episode["incomplete_reason"] == "deadline"


def test_truncated_gzip_is_reported_as_invalid_dataset(tmp_path):
    imitation = _imitation()
    path = tmp_path / "truncated.gz"
    path.write_bytes(gzip.compress(b"{}")[:-3])
    with pytest.raises(ValueError):
        imitation.load_demonstrations(path)


def test_collection_deadline_after_proposal_prevents_offer_mutation(monkeypatch):
    imitation = _imitation()
    now = [0.0]
    expired_proposals, offers_after_deadline = [], []
    propose = imitation.HeuristicPolicy.propose
    offer_trade = FourPlayerGame.offer_trade

    def expiring_proposal(self, view):
        offer = propose(self, view)
        if offer is not None:
            expired_proposals.append(view.context.version)
            now[0] = 2.0
        return offer

    def observed_offer(self, offer):
        if now[0] >= 1.0:
            offers_after_deadline.append(self.version)
        return offer_trade(self, offer)

    monkeypatch.setattr(imitation.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(imitation.HeuristicPolicy, "propose", expiring_proposal)
    monkeypatch.setattr(FourPlayerGame, "offer_trade", observed_offer)
    data = imitation.collect_demonstrations(
        board_seeds=[31], max_decisions=128, max_game_steps=500, deadline=1.0
    )
    assert expired_proposals
    assert offers_after_deadline == []
    assert data["collection"]["stop_reason"] == "deadline"
    assert data["episodes"][0]["steps"] == expired_proposals[0]
    assert data["episodes"][0]["incomplete_reason"] == "deadline"


@pytest.mark.parametrize("unavailable", ["installed_layout", "missing_git"])
def test_legacy_warmstart_runs_with_explicit_unavailable_provenance(
    monkeypatch, tmp_path, unavailable
):
    from colonist_rl import experiments, training

    imitation = _imitation()
    if unavailable == "installed_layout":
        monkeypatch.setattr(
            imitation, "__file__", str(tmp_path / "site-packages" / "colonist_rl" / "imitation.py")
        )
    else:

        def missing_git(*args, **kwargs):
            raise FileNotFoundError("git executable is unavailable")

        monkeypatch.setattr(imitation.subprocess, "check_output", missing_git)
        # Legacy fallback must not turn strict experiment identity into unknown/clean.
        with pytest.raises(FileNotFoundError):
            experiments.source_identity()
    torch.set_num_threads(1)
    result = training.imitation_warmstart(GraphPolicy(hidden_size=8), decisions=2, epochs=1)
    assert result["decisions"] == 2 and result["epochs"] == 1
    assert math.isfinite(result["loss"])
    assert result["updates"] == 1 and result["supervised_decisions"] == 2
    assert result["source"] == {"revision": None, "dirty": None}
    assert result["episode_summary"]["incomplete"] == 1
    json.dumps(result, allow_nan=False)


def test_unavailable_source_roundtrips_but_mixed_provenance_is_rejected(monkeypatch, tmp_path):
    imitation = _imitation()
    monkeypatch.setattr(
        imitation, "__file__", str(tmp_path / "site-packages" / "colonist_rl" / "imitation.py")
    )
    data = imitation.collect_demonstrations(board_seeds=[3], max_decisions=2)
    assert data["source"] == {"revision": None, "dirty": None}
    path = tmp_path / "installed-demonstrations.json.gz"
    imitation.save_demonstrations(path, data)
    assert imitation.load_demonstrations(path) == data
    for source in (
        {"revision": None, "dirty": False},
        {"revision": None, "dirty": True},
        {"revision": "known-revision", "dirty": None},
    ):
        invalid = deepcopy(data)
        invalid["source"] = source
        with pytest.raises(ValueError, match="provenance"):
            imitation.save_demonstrations(path, invalid)


def test_legacy_warmstart_reports_completed_and_cap_interrupted_episodes(monkeypatch):
    from colonist_rl.board import RulesConfig
    from colonist_rl.training import imitation_warmstart

    imitation = _imitation()
    created = [0]

    def bounded_games(*, max_steps):
        created[0] += 1
        if created[0] == 1:
            return FourPlayerGame(
                replace(RulesConfig.standard(), victory_points=1), max_steps=max_steps
            )
        return FourPlayerGame(max_steps=1 if created[0] == 2 else max_steps)

    monkeypatch.setattr(imitation, "FourPlayerGame", bounded_games)
    data = imitation.collect_demonstrations(
        board_seeds=[3, 4, 5], max_decisions=2, max_game_steps=10
    )
    assert [episode["incomplete_reason"] for episode in data["episodes"]] == [
        None,
        "step_cap",
        "decision_cap",
    ]
    # Feed the bounded real sequence to the legacy wrapper; keep the real fitter.
    monkeypatch.setattr(imitation, "collect_demonstrations", lambda **kwargs: data)
    torch.set_num_threads(1)
    result = imitation_warmstart(GraphPolicy(hidden_size=8), decisions=2, epochs=1)
    assert math.isfinite(result["loss"]) and result["decisions"] == 2 and result["epochs"] == 1
    assert result["episode_summary"] == {
        "total": 3,
        "completed": 1,
        "incomplete": 2,
        "terminated": 1,
        "truncated": 1,
        "incomplete_reason_counts": {"step_cap": 1, "decision_cap": 1},
    }
    assert result["source"] == data["source"]
    assert result["collection"]["stop_reason"] == "decision_cap"
    json.dumps(result, allow_nan=False)


def test_installed_layout_does_not_claim_an_ancestor_repository_revision(monkeypatch, tmp_path):
    imitation = _imitation()
    outer = tmp_path / "unrelated-checkout"
    installed = outer / ".venv" / "lib" / "python3.13" / "site-packages" / "colonist_rl"
    installed.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(outer),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.setattr(imitation, "__file__", str(installed / "imitation.py"))
    assert imitation._source() == {"revision": None, "dirty": None}


def test_phase_weights_change_single_phase_chunk_gradients_without_normalizing_away():
    imitation = _imitation()
    data = _sequence_dataset(imitation)
    phase = data["episodes"][0]["samples"][1]["observation"].context.kind
    # Capture pre-clipping gradients: Adam's first update is scale invariant.
    gradients = []
    losses = []
    for weight in (1.0, 0.25):
        policy = StateSensitivePolicy()
        policy.weight.register_hook(lambda grad: gradients.append(grad.item()))
        result = imitation.fit_imitation(
            policy, data, epochs=1, sequence_length=4, phase_weights={phase: weight}
        )
        losses.append(result["loss"])
        assert [call[0] for call in policy.calls] == [0, 1, 2, 3]
        assert result["supervised_presentations"] == 2
    assert gradients[1] == pytest.approx(gradients[0] * 0.25)
    assert losses[1] == pytest.approx(losses[0] * 0.25)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_phase_weights_fail_before_policy_forward(weight):
    imitation = _imitation()
    data = _sequence_dataset(imitation)
    phase = data["episodes"][0]["samples"][1]["observation"].context.kind
    policy = StateSensitivePolicy()
    with pytest.raises(ValueError, match="phase_weights"):
        imitation.fit_imitation(policy, data, phase_weights={phase: weight})
    assert policy.calls == []


def test_missing_phase_weights_fail_instead_of_dropping_supervision():
    imitation = _imitation()
    data = _sequence_dataset(imitation)
    policy = StateSensitivePolicy()
    with pytest.raises(ValueError, match="phase_weights"):
        imitation.fit_imitation(policy, data, phase_weights={})
    assert policy.calls == []
    # No phase requires a weight when the dataset has no non-forced choices.
    forced = _sequence_dataset(imitation, forced=True)
    result = imitation.fit_imitation(policy, forced, phase_weights={})
    assert result["updates"] == 0 and policy.calls == []


def test_default_none_and_unit_phase_weights_preserve_uniform_training_exactly():
    imitation = _imitation()
    data = _sequence_dataset(imitation)
    phase = data["episodes"][0]["samples"][1]["observation"].context.kind
    policies = [StateSensitivePolicy() for _ in range(3)]
    results = [
        imitation.fit_imitation(policies[0], data, epochs=2, sequence_length=2),
        imitation.fit_imitation(policies[1], data, epochs=2, sequence_length=2, phase_weights=None),
        imitation.fit_imitation(
            policies[2], data, epochs=2, sequence_length=2, phase_weights={phase: 1.0}
        ),
    ]
    assert results[0] == results[1] == results[2]
    assert all(torch.equal(policies[0].weight, p.weight) for p in policies[1:])
    assert policies[0].calls == policies[1].calls == policies[2].calls
