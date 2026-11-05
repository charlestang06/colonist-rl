"""Experiment safety and statistical eligibility, without benchmark compute."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from colonist_rl import experiments


def evaluation(boards=8):
    return {
        "status": "complete",
        "partial_outcomes": [],
        "outcomes": [
            {
                "board_seed": b,
                "seat": s,
                "won": s == 0,
                "victory_points": 8,
                "terminated": True,
                "truncated": False,
                "winner": 0,
            }
            for b in range(90000, 90000 + boards)
            for s in range(4)
        ],
    }


def summary(boards=8):
    score = {"count": 100, "correct": 60, "accuracy": 0.60, "cross_entropy": 1.0}
    return {
        "teacher": evaluation(boards),
        "variants": {"teacher": evaluation(boards)},
        "students": [
            {
                "seed": seed,
                "untrained": evaluation(boards),
                "imitated": evaluation(boards),
                "validation_metrics": {
                    "non_forced": score.copy(),
                    "by_phase": {"TURN": {"non_forced": score.copy()}},
                },
            }
            for seed in (11, 29)
        ],
    }


def test_gate_accepts_exact_boundaries_and_rejects_inadequate_samples():
    assert experiments.baseline_gate(summary())["eligible"]
    assert not experiments.baseline_gate(summary(1))["eligible"]
    evidence = summary()
    evidence["students"][0]["validation_metrics"]["non_forced"]["count"] = 99
    assert not experiments.baseline_gate(evidence)["eligible"]


def test_gate_uses_phase_non_forced_accuracy_and_rejects_forced_only():
    evidence = summary()
    evidence["students"][0]["validation_metrics"]["by_phase"]["TURN"]["non_forced"] = {
        "count": 20,
        "accuracy": 0.59,
    }
    assert not experiments.baseline_gate(evidence)["eligible"]
    evidence = summary()
    evidence["students"][0]["validation_metrics"]["non_forced"] = {
        "count": 0,
        "accuracy": None,
    }
    assert not experiments.baseline_gate(evidence)["eligible"]


@pytest.mark.parametrize("mutation", ["truncated", "incomplete", "missing", "different"])
def test_gate_rejects_unresolved_or_not_exactly_paired_blocks(mutation):
    evidence = summary()
    candidate = evidence["students"][0]["imitated"]
    if mutation == "truncated":
        candidate["outcomes"][0].update(truncated=True, won=None, winner=None)
    elif mutation == "incomplete":
        candidate["status"] = "incomplete"
    elif mutation == "missing":
        candidate["outcomes"].pop()
    else:
        for row in candidate["outcomes"]:
            row["board_seed"] += 100
    assert not experiments.baseline_gate(evidence)["eligible"]


def test_gate_vp_and_win_tolerances_use_same_teacher_blocks():
    evidence = summary()
    candidate = evidence["students"][0]["imitated"]
    for row in candidate["outcomes"]:
        row["victory_points"] = 7
    assert experiments.baseline_gate(evidence)["eligible"]
    candidate["outcomes"][0]["victory_points"] = 6.9
    assert not experiments.baseline_gate(evidence)["eligible"]
    evidence = summary()
    for row in evidence["students"][0]["imitated"]["outcomes"]:
        row["won"] = False
    assert not experiments.baseline_gate(evidence)["eligible"]


def test_split_collision_rejected_before_source_or_simulator(tmp_path, monkeypatch):
    def forbidden():
        pytest.fail("expensive stage reached before split validation")

    monkeypatch.setattr(experiments, "source_identity", forbidden)
    with pytest.raises(ValueError, match="disjoint"):
        experiments.run_experiment(tmp_path, validation_demo_seeds=(11000,))
    assert not list(tmp_path.iterdir())


def test_canonical_digest_is_order_independent_and_changes_with_content():
    assert experiments.manifest_digest({"a": 1, "b": 2}) == experiments.manifest_digest(
        {"b": 2, "a": 1}
    )
    assert experiments.manifest_digest({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()
    assert experiments.manifest_digest({"a": 1}) != experiments.manifest_digest({"a": 2})


def test_expired_runner_persists_manifest_and_partial_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiments,
        "source_identity",
        lambda: {"revision": "abc", "dirty": False, "content_identity": "git:abc"},
    )
    monkeypatch.setattr(experiments.time, "monotonic", iter([0.0, 2.0] + [2.0] * 100).__next__)
    result = experiments.run_experiment(tmp_path, wall_seconds=1)
    assert result["status"] == "partial"
    assert result["ppo_status"] == "deferred_insufficient_baseline_evidence"
    assert result["stages"]["probe"]["status"] == "incomplete"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["seed_splits"]["final_test_reserved"] == [70000017, 70000019]
    assert manifest["seed_splits"]["training_candidates"]["11"] == list(range(11000, 11064))
    assert manifest["source"]["content_identity"] == "git:abc"
    assert (tmp_path / "results.json").is_file()


def test_stage_exception_is_saved_without_restarting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiments,
        "source_identity",
        lambda: {"revision": "abc", "dirty": False, "content_identity": "git:abc"},
    )

    def failure(**kwargs):
        raise RuntimeError("broken probe")

    monkeypatch.setattr(experiments, "evaluate_blocks", failure)
    result = experiments.run_experiment(tmp_path, decisions=1, epochs=1, wall_seconds=1)
    assert result["status"] == "failed"
    assert result["stages"]["probe"]["error"] == "RuntimeError: broken probe"
    assert json.loads((tmp_path / "results.json").read_text())["status"] == "failed"


def test_dirty_source_refused_before_any_computation(tmp_path, monkeypatch):
    identity = {"revision": "abc", "dirty": True, "content_identity": "git:abc"}
    monkeypatch.setattr(experiments, "source_identity", lambda: copy.deepcopy(identity))
    with pytest.raises(ValueError, match="clean"):
        experiments.run_experiment(tmp_path)


def test_incomplete_variant_suppresses_all_apparently_paired_comparisons():
    evidence = summary()
    evidence["variants"]["other_opponent"] = evaluation(8)
    evidence["variants"]["other_opponent"]["status"] = "incomplete"
    gate = experiments.baseline_gate(evidence)
    assert not gate["eligible"]
    assert gate["teacher_comparisons"] == {}


def test_tiny_real_stages_preserve_artifacts_and_independent_initializations(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(
        experiments,
        "source_identity",
        lambda: {"revision": "abc", "dirty": False, "content_identity": "git:abc"},
    )
    result = experiments.run_experiment(
        tmp_path, decisions=2, epochs=1, hidden_size=8, max_game_steps=8, wall_seconds=5
    )
    assert len(result["students"]) == 2
    assert result["gate"]["eligible"] is False
    assert result["budget_decision"]["demonstrations_per_seed"] <= 2
    weights = []
    for seed in (11, 29):
        original = torch.load(tmp_path / f"seed-{seed}-untrained.pt", weights_only=True)
        initial = torch.load(tmp_path / f"seed-{seed}-initial_student.pt", weights_only=True)
        assert all(
            torch.equal(value, initial["policy_state"][name])
            for name, value in original["policy_state"].items()
        )
        weights.append(original["policy_state"])
        item = result["artifacts"][f"{seed}/imitated"]
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert any(not torch.equal(value, weights[1][name]) for name, value in weights[0].items())


def test_changed_source_marks_preserved_run_failed(tmp_path, monkeypatch):
    identities = iter(
        [
            {"revision": "abc", "dirty": False, "content_identity": "git:abc"},
            {"revision": "abc", "dirty": True, "content_identity": "git:abc"},
        ]
    )
    monkeypatch.setattr(experiments, "source_identity", lambda: next(identities))
    monkeypatch.setattr(experiments.time, "monotonic", iter([0.0, 2.0] + [2.0] * 100).__next__)
    result = experiments.run_experiment(tmp_path, wall_seconds=1)
    assert result["status"] == "failed"
    assert result["source_error"] == "source changed during experiment"
    assert json.loads((tmp_path / "results.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("accuracy", [float("nan"), float("inf"), 1.1])
def test_gate_rejects_invalid_non_forced_statistics(accuracy):
    evidence = summary()
    evidence["students"][0]["validation_metrics"]["non_forced"]["accuracy"] = accuracy
    assert not experiments.baseline_gate(evidence)["eligible"]


def test_missing_stage_results_reject_without_losing_partial_evidence():
    evidence = summary()
    evidence["students"][0]["imitated"] = None
    evidence["variants"]["unknown"] = None
    assert experiments.baseline_gate(evidence)["eligible"] is False
