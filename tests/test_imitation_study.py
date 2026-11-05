"""Controlled imitation arms, immutable inputs and conservative completion."""

import hashlib
import importlib
import json
import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from colonist_rl import imitation
from colonist_rl.policy import GraphPolicy
from colonist_rl.training import load_checkpoint, save_checkpoint


def study():
    assert importlib.util.find_spec("colonist_rl.imitation_study") is not None, (
        "controlled imitation study is missing"
    )
    return importlib.import_module("colonist_rl.imitation_study")


def data_with_phases(seed=18):
    data = imitation.collect_demonstrations(board_seeds=[seed], max_decisions=5)
    actions = data["episodes"][0]["samples"][0]["observation"].legal_actions[:2]
    for i, sample in enumerate(data["episodes"][0]["samples"]):
        view = sample["observation"]
        sample["observation"] = replace(
            view,
            context=replace(view.context, kind="PLAY_TURN" if i < 4 else "TRADE_RESPONSE"),
            legal_actions=actions[: 1 if i == 0 else 2],
        )
        sample["action_index"] = 0
    data["coverage"] = imitation._coverage(data["episodes"])
    return data


def inputs(tmp_path):
    datasets, checkpoints = {}, {}
    for seed in (11, 29):
        data = imitation.collect_demonstrations(board_seeds=[seed * 1000], max_decisions=2)
        datasets[seed] = tmp_path / f"{seed}.json.gz"
        imitation.save_demonstrations(datasets[seed], data)
        torch.manual_seed(seed)
        checkpoints[seed] = tmp_path / f"{seed}.pt"
        save_checkpoint(checkpoints[seed], GraphPolicy(32))
    return datasets, checkpoints


def clean_source(monkeypatch, module):
    monkeypatch.setattr(
        module,
        "source_identity",
        lambda: {"revision": "abc", "dirty": False, "content_identity": "git:abc"},
    )


def test_weights_are_training_nonforced_counts_with_global_mean_one():
    module = study()
    data = data_with_phases()
    before = deepcopy(data)
    weights = module.phase_weights(data)
    assert weights == {"PLAY_TURN": pytest.approx(2 / 3), "TRADE_RESPONSE": 2.0}
    assert 3 * weights["PLAY_TURN"] + weights["TRADE_RESPONSE"] == pytest.approx(4)
    assert data == before
    forced = deepcopy(data)
    for sample in forced["episodes"][0]["samples"]:
        sample["observation"] = replace(
            sample["observation"], legal_actions=sample["observation"].legal_actions[:1]
        )
    forced["coverage"] = imitation._coverage(forced["episodes"])
    assert module.phase_weights(forced) == {}


def test_runner_rejects_malformed_mapping_before_source_or_output(tmp_path, monkeypatch):
    module = study()
    monkeypatch.setattr(module, "source_identity", lambda: pytest.fail("source reached"))
    for datasets, checkpoints in [({}, {}), ({True: "x"}, {True: "y"}), ({11: "x"}, {29: "y"})]:
        with pytest.raises(ValueError):
            module.run_imitation_study(
                tmp_path / "out", datasets=datasets, initial_checkpoints=checkpoints
            )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["training", "validation", "reserved"])
def test_seed_overlap_is_rejected_before_output(tmp_path, monkeypatch, mutation):
    module = study()
    datasets, checkpoints = inputs(tmp_path)
    clean_source(monkeypatch, module)
    kwargs = {}
    if mutation == "training":
        datasets[29] = datasets[11]
    elif mutation == "validation":
        kwargs["validation_demo_seeds"] = (62100000,)
    else:
        kwargs["validation_board_seeds"] = (70000017,)
    with pytest.raises(ValueError, match="disjoint|unique"):
        module.run_imitation_study(
            tmp_path / "out", datasets=datasets, initial_checkpoints=checkpoints, **kwargs
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "source",
    [
        {"revision": "abc", "dirty": True, "content_identity": "git:abc"},
        {"revision": None, "dirty": None, "content_identity": None},
    ],
)
def test_study_requires_known_clean_source(tmp_path, monkeypatch, source):
    module = study()
    datasets, checkpoints = inputs(tmp_path)
    monkeypatch.setattr(module, "source_identity", lambda: source)
    with pytest.raises(ValueError, match="clean"):
        module.run_imitation_study(
            tmp_path / "out", datasets=datasets, initial_checkpoints=checkpoints
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("resolved", [False, True])
def test_tiny_real_study_preserves_inputs_rng_validation_and_complete_budget(
    tmp_path, monkeypatch, resolved
):
    module = study()
    datasets, checkpoints = inputs(tmp_path)
    clean_source(monkeypatch, module)
    # Bound the existing simulator rather than replacing training/evaluation logic.
    from colonist_rl import evaluation
    from colonist_rl.board import RulesConfig
    from colonist_rl.simulator import FourPlayerGame

    rules = (
        replace(RulesConfig.standard(), victory_points=1) if resolved else RulesConfig.standard()
    )
    monkeypatch.setattr(
        evaluation, "FourPlayerGame", lambda **kw: FourPlayerGame(rules, max_steps=8)
    )
    before = {str(p): p.read_bytes() for p in [*datasets.values(), *checkpoints.values()]}
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    result = module.run_imitation_study(
        tmp_path / "out",
        datasets=datasets,
        initial_checkpoints=checkpoints,
        epochs=1,
        wall_seconds=30,
        validation_board_seeds=(62100000,),
        validation_demo_seeds=(62110000,),
    )
    assert result["status"] == "complete"
    assert result["equal_training_budget_complete"] is True
    assert len(result["comparisons"]) == (6 if resolved else 0)
    if resolved:
        assert all(c["paired_boards"] == 1 for c in result["comparisons"].values())
    assert random.getstate() == rng[0]
    assert np.array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    assert torch.get_num_threads() == threads
    for path, payload in before.items():
        assert Path(path).read_bytes() == payload
    for seed in (11, 29):
        arms = [result["students"][str(seed)][arm] for arm in ("uniform", "phase_balanced")]
        assert arms[0]["initial_state_sha256"] == arms[1]["initial_state_sha256"]
        assert arms[0]["fit"] == arms[1]["fit"]  # Tiny single-phase data has unit weight.
        for arm in arms:
            assert arm["fit"]["epochs"] == 1
            assert arm["validation_metrics"]["non_forced"]["count"] > 0
            assert arm["state_before_validation_sha256"] == arm["state_after_validation_sha256"]
            path = arm["checkpoint"]["path"]
            loaded, _ = load_checkpoint(path)
            assert loaded.hidden_size == 32
            assert (
                arm["checkpoint"]["sha256"] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
            )
    journals = list((tmp_path / "out" / "progress").glob("*.json"))
    assert len(journals) == 5
    for path in journals:
        progress = json.loads(path.read_text())
        assert progress["finished_games"] == 4
        assert len(progress["completed_outcomes"]) == 4
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["model"]["sequence_length"] == 16
    assert manifest["model"]["learning_rate"] == 3e-4
    assert result["actual_play_diagnostics"]["status"] == "external_analysis_required"


def test_expired_study_persists_partial_and_suppresses_equal_budget(tmp_path, monkeypatch):
    module = study()
    datasets, checkpoints = inputs(tmp_path)
    clean_source(monkeypatch, module)
    monkeypatch.setattr(module.time, "monotonic", iter([0.0] + [2.0] * 500).__next__)
    result = module.run_imitation_study(
        tmp_path / "out", datasets=datasets, initial_checkpoints=checkpoints, wall_seconds=1
    )
    assert result["status"] == "partial"
    assert result["equal_training_budget_complete"] is False
    assert result["comparisons"] == {}
    assert json.loads((tmp_path / "out" / "results.json").read_text())["status"] == "partial"


def test_source_or_input_change_preserves_evidence_and_marks_failed(tmp_path, monkeypatch):
    module = study()
    datasets, checkpoints = inputs(tmp_path)
    count = [0]

    def source():
        count[0] += 1
        return {"revision": "abc", "dirty": count[0] > 1, "content_identity": "git:abc"}

    monkeypatch.setattr(module, "source_identity", source)
    monkeypatch.setattr(module.time, "monotonic", iter([0.0] + [2.0] * 500).__next__)
    result = module.run_imitation_study(
        tmp_path / "out", datasets=datasets, initial_checkpoints=checkpoints, wall_seconds=1
    )
    assert result["status"] == "failed"
    assert result["source_error"]
    assert (tmp_path / "out" / "manifest.json").exists()


def test_arms_reload_identical_initial_parameters_and_rng_with_independent_optimizers(
    tmp_path, monkeypatch
):
    module = study()
    datasets, checkpoints = inputs(tmp_path)
    clean_source(monkeypatch, module)
    observed = []
    real_fit = module.fit_imitation

    def observe_fit(policy, dataset, **kwargs):
        before = {name: value.clone() for name, value in policy.state_dict().items()}
        rng = (random.random(), np.random.random(), torch.rand(1).item())
        fitted = real_fit(policy, dataset, **kwargs)
        observed.append((policy, before, rng))
        return fitted

    monkeypatch.setattr(module, "fit_imitation", observe_fit)
    # Fit and metrics remain real; bound actual evaluation to one resolved setup.
    from colonist_rl import evaluation
    from colonist_rl.board import RulesConfig
    from colonist_rl.simulator import FourPlayerGame

    rules = replace(RulesConfig.standard(), victory_points=1)
    monkeypatch.setattr(
        evaluation, "FourPlayerGame", lambda **kw: FourPlayerGame(rules, max_steps=8)
    )
    result = module.run_imitation_study(
        tmp_path / "out",
        datasets=datasets,
        initial_checkpoints=checkpoints,
        epochs=1,
        wall_seconds=30,
        validation_board_seeds=(62100000,),
        validation_demo_seeds=(62110000,),
    )
    assert result["status"] == "complete"
    assert len(observed) == 4
    for start in (0, 2):
        first, second = observed[start : start + 2]
        assert first[0] is not second[0]
        assert first[2] == second[2]
        assert all(torch.equal(value, second[1][key]) for key, value in first[1].items())
        # Identical uniform/unit-weight arms finish identically only with fresh Adam.
        assert all(
            torch.equal(value, second[0].state_dict()[key])
            for key, value in first[0].state_dict().items()
        )
        assert any(
            not torch.equal(value, first[0].state_dict()[key]) for key, value in first[1].items()
        )
    assert observed[0][2] != observed[2][2]


def test_one_incomplete_fit_suppresses_all_equal_budget_comparisons(tmp_path, monkeypatch):
    module = study()
    datasets, checkpoints = inputs(tmp_path)
    clean_source(monkeypatch, module)
    real_fit = module.fit_imitation
    calls = [0]

    def interrupted_fit(policy, dataset, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            kwargs["deadline"] = -1.0
        return real_fit(policy, dataset, **kwargs)

    monkeypatch.setattr(module, "fit_imitation", interrupted_fit)
    from colonist_rl import evaluation
    from colonist_rl.board import RulesConfig
    from colonist_rl.simulator import FourPlayerGame

    rules = replace(RulesConfig.standard(), victory_points=1)
    monkeypatch.setattr(
        evaluation, "FourPlayerGame", lambda **kw: FourPlayerGame(rules, max_steps=8)
    )
    result = module.run_imitation_study(
        tmp_path / "out",
        datasets=datasets,
        initial_checkpoints=checkpoints,
        epochs=1,
        wall_seconds=30,
        validation_board_seeds=(62100000,),
        validation_demo_seeds=(62110000,),
    )
    assert result["status"] == "partial"
    assert result["students"]["11"]["uniform"]["fit"]["epochs"] == 1
    assert result["students"]["11"]["phase_balanced"]["fit"]["epochs"] == 0
    assert not result["equal_training_budget_complete"]
    assert result["comparisons"] == {}
    assert result["variants"]["heuristic"]["status"] == "complete"


def test_input_mutation_is_detected_and_saved_results_are_failed(tmp_path, monkeypatch):
    module = study()
    datasets, checkpoints = inputs(tmp_path)
    clean_source(monkeypatch, module)
    real_write = module._write

    def mutate_after_manifest(path, value):
        real_write(path, value)
        if path.name == "manifest.json":
            datasets[11].write_bytes(datasets[11].read_bytes() + b"changed")

    monkeypatch.setattr(module, "_write", mutate_after_manifest)
    monkeypatch.setattr(module.time, "monotonic", iter([0.0] + [2.0] * 500).__next__)
    result = module.run_imitation_study(
        tmp_path / "out", datasets=datasets, initial_checkpoints=checkpoints, wall_seconds=1
    )
    assert result["status"] == "failed" and result["input_error"]
    assert result["comparisons"] == {}
    assert json.loads((tmp_path / "out" / "results.json").read_text())["status"] == "failed"


def test_existing_output_is_refused_and_wrong_initial_architecture_rejected(tmp_path, monkeypatch):
    module = study()
    datasets, checkpoints = inputs(tmp_path)
    clean_source(monkeypatch, module)
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        module.run_imitation_study(output, datasets=datasets, initial_checkpoints=checkpoints)
    assert (output / "keep.txt").read_text() == "preserve"
    save_checkpoint(checkpoints[11], GraphPolicy(8))
    with pytest.raises(ValueError, match="hidden32"):
        module.run_imitation_study(
            tmp_path / "other", datasets=datasets, initial_checkpoints=checkpoints
        )
    assert not (tmp_path / "other").exists()
