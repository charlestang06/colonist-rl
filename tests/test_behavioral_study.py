"""Full-history selection and an audited, deliberately narrow acquisition driver."""

import hashlib
import importlib
import json
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from colonist_rl.imitation import collect_demonstrations, save_demonstrations
from colonist_rl.policy import GraphPolicy
from colonist_rl.simulator import FourPlayerGame
from colonist_rl.training import load_checkpoint, save_checkpoint


def study():
    assert importlib.util.find_spec("colonist_rl.behavioral_study") is not None, (
        "minimal behavioral acquisition driver and pure prefix selector are missing"
    )
    return importlib.import_module("colonist_rl.behavioral_study")


@pytest.fixture
def histories():
    env = FourPlayerGame()
    env.reset(seed=181)
    base = env.observe()
    raw = tuple(
        (
            replace(
                base, context=replace(base.context, version=i), legal_actions=base.legal_actions[:2]
            ),
            0,
        )
        for i in range(6)
    )
    first = tuple(
        (replace(v, legal_actions=v.legal_actions[:1]), 0) if i in (0, 2) else (v, t)
        for i, (v, t) in enumerate(raw[:4])
    )
    return (first, raw[4:])


def test_choice_prefix_retains_forced_history_boundaries_and_unused_suffixes(histories):
    module = study()
    before = deepcopy(histories)
    selected, manifest = module.select_choice_prefix(histories, choices=3)
    assert selected == (histories[0], histories[1][:1])
    assert selected[0] is histories[0]
    assert manifest["retained"] == [
        {"episode_index": 0, "sample_stop": 4},
        {"episode_index": 1, "sample_stop": 1},
    ]
    assert manifest["counts"] == {"all": 5, "forced": 2, "choices": 3}
    assert manifest["unused"] == [{"episode_index": 1, "sample_start": 1, "sample_stop": 2}]
    assert manifest["unused_counts"] == {"all": 1, "forced": 0, "choices": 1}
    assert len(manifest["actual_boards"]) == 2
    assert histories == before


@pytest.mark.parametrize("choices", [0, -1, True, 1.5, 5])
def test_choice_prefix_rejects_invalid_or_insufficient_requests(histories, choices):
    with pytest.raises(ValueError):
        study().select_choice_prefix(histories, choices=choices)


@pytest.mark.parametrize("target", [-1, True, 10000])
def test_choice_prefix_validates_even_unused_targets(histories, target):
    invalid = histories[:-1] + (histories[-1][:-1] + ((histories[-1][-1][0], target),),)
    with pytest.raises(ValueError, match="legal|index|observation"):
        study().select_choice_prefix(invalid, choices=1)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    paths = {}
    for name, board in (("train", 181), ("validation", 182)):
        path = tmp_path / f"{name}.gz"
        save_demonstrations(path, collect_demonstrations(board_seeds=[board], max_decisions=4))
        paths[name] = str(path)
    checkpoint = tmp_path / "initial.pt"
    save_checkpoint(checkpoint, GraphPolicy(hidden_size=8))
    evidence = tmp_path / "control.json"
    evidence.write_text('{"frozen_control": true}')
    source = "a" * 40
    protocol = {
        "version": 1,
        "stage": "student_labels",
        "seeds": [11],
        "source_revision": source,
        "partitions": {
            "training": [181, 183, 184],
            "validation_labels": [182],
            "validation_games": [185],
            "audit": [186],
            "final_test": [70000017, 70000019],
        },
        "acquisition_boards": {"11": {"teacher_state": [183], "student_state": [184]}},
        "input_digests": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (*paths.values(), str(checkpoint), str(evidence))
        },
        "consumed_evidence": [str(evidence)],
        "validation_labels_path": paths["validation"],
        "query_budget": 4,
        "max_game_steps": 200,
        "baseline": "heuristic",
        "deterministic": True,
        "epochs": 2,
        "segment_epochs": 1,
        "sequence_length": 2,
        "batch_size": 1,
        "learning_rate": 0.0003,
        "weighting": "original_phase_balanced",
        "collection_seconds": 5,
        "fit_seconds": 5,
        "evaluation_seconds": 5,
        "deadline": time.monotonic() + 20,
    }
    return {
        "output_dir": tmp_path / "out",
        "initial_checkpoints": {11: str(checkpoint)},
        "training_components": {11: (paths["train"],)},
        "protocol": protocol,
    }


def source_stub(monkeypatch, module, setup):
    import colonist_rl.student_labels as labels
    from colonist_rl import imitation

    source = lambda: {"revision": "a" * 40, "dirty": False}
    for target in (module, imitation, labels):
        monkeypatch.setattr(target, "_source", source)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(extra=True),
        lambda p: p.pop("query_budget"),
        lambda p: p.update(version=True),
        lambda p: p.update(query_budget=True),
        lambda p: p.update(learning_rate=True),
        lambda p: p.update(deadline=float("inf")),
        lambda p: p.update(seeds=[True]),
        lambda p: p["acquisition_boards"]["11"].update(extra=[187]),
        lambda p: p["acquisition_boards"].update({"29": p["acquisition_boards"]["11"]}),
        lambda p: p["acquisition_boards"]["11"].update(student_state=[183]),
        lambda p: p["acquisition_boards"]["11"].update(student_state=[185]),
        lambda p: p["input_digests"].pop(p["validation_labels_path"]),
        lambda p: p["input_digests"].update({"/unlisted": "0" * 64}),
    ],
)
def test_protocol_rejects_ambiguous_matrix_leakage_and_input_lists(setup, monkeypatch, change):
    module = study()
    source_stub(monkeypatch, module, setup)
    change(setup["protocol"])
    with pytest.raises(ValueError):
        module.run_behavioral_study(**setup)
    assert not setup["output_dir"].exists()


def test_dirty_source_and_input_drift_fail_before_compute(setup, monkeypatch):
    module = study()
    monkeypatch.setattr(module, "_source", lambda: {"revision": "a" * 40, "dirty": True})
    with pytest.raises(ValueError, match="source"):
        module.run_behavioral_study(**setup)
    source_stub(monkeypatch, module, setup)
    with open(setup["protocol"]["consumed_evidence"][0], "a") as output:
        output.write(" ")
    with pytest.raises(ValueError, match="digest|input"):
        module.run_behavioral_study(**setup)


def injected_collection(monkeypatch, module, *, teacher_count=4, student_count=4):
    """Inject only the acquisition boundary; use real accepted collection formats."""
    real_teacher = module.collect_demonstrations
    real_student = module.collect_student_labels

    def teacher(**kwargs):
        kwargs["max_decisions"] = teacher_count
        kwargs["max_game_steps"] = 200
        return real_teacher(**kwargs)

    def student(policy, **kwargs):
        kwargs["max_decisions"] = student_count
        kwargs["max_game_steps"] = 200
        return real_student(policy, **kwargs)

    monkeypatch.setattr(module, "collect_demonstrations", teacher)
    monkeypatch.setattr(module, "collect_student_labels", student)


def fake_evaluator(monkeypatch, module, *, censor=False, omit=False):
    def evaluate(policy=None, *, board_seeds, on_game_end, **kwargs):
        rows = [
            {
                "board_seed": b,
                "seat": s,
                "won": False if not censor else None,
                "victory_points": 2,
                "terminated": not censor,
                "truncated": censor,
                "winner": 1 if not censor else None,
            }
            for b in board_seeds
            for s in range(4)
        ]
        if omit:
            rows.pop()
        for row in rows:
            on_game_end(row)
        return {"status": "complete", "outcomes": rows, "partial_outcomes": []}

    monkeypatch.setattr(module, "evaluate_blocks", evaluate)


def test_driver_persists_independent_optimizers_exact_work_and_immutable_inputs(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    injected_collection(monkeypatch, module)
    fake_evaluator(monkeypatch, module)
    before = {p: Path(p).read_bytes() for p in setup["protocol"]["input_digests"]}
    actual_fit = module.fit_sequences
    optimizer_calls = []

    def fit(policy, episodes, *, optimizer, **kwargs):
        optimizer_calls.append((policy, optimizer, len(optimizer.state)))
        return actual_fit(policy, episodes, optimizer=optimizer, **kwargs)

    monkeypatch.setattr(module, "fit_sequences", fit)
    result = module.run_behavioral_study(**setup)
    assert result["status"] == "complete"
    assert set(result["variants"]) == {"heuristic", "teacher_state/11", "student_state/11"}
    assert result["comparisons"]
    teacher = result["students"]["11"]["teacher_state"]
    student = result["students"]["11"]["student_state"]
    assert (
        teacher["selection"]["counts"]["choices"] == student["selection"]["counts"]["choices"] == 4
    )
    assert teacher["acquisition"]["queries"] == student["acquisition"]["queries"] == 4
    assert teacher["initial_state_sha256"] == student["initial_state_sha256"]
    assert len(optimizer_calls) == 4
    assert optimizer_calls[0][1] is optimizer_calls[1][1]
    assert optimizer_calls[2][1] is optimizer_calls[3][1]
    assert optimizer_calls[0][1] is not optimizer_calls[2][1]
    assert optimizer_calls[0][2] == optimizer_calls[2][2] == 0
    assert optimizer_calls[1][2] > 0 and optimizer_calls[3][2] > 0
    for arm in (teacher, student):
        assert arm["actual_work"]["updates"] == arm["expected_work"]["updates"]
        assert (
            arm["actual_work"]["supervised_presentations"]
            == arm["expected_work"]["supervised_presentations"]
        )
        assert arm["actual_work"]["prefix_forwards"] == arm["expected_work"]["prefix_forwards"]
        assert len(arm["checkpoints"]) == 2
        assert arm["validation_metrics"]["non_forced"]["count"] == 4
    assert all(Path(p).read_bytes() == content for p, content in before.items())
    assert len(list((setup["output_dir"] / "games").glob("*/*.json"))) == 12
    assert (setup["output_dir"] / "manifest.json").exists()
    assert result["final_integrity"]["verified"] is True


@pytest.mark.parametrize("censor,omit", [(True, False), (False, True)])
def test_partial_or_censored_panel_suppresses_every_comparison(setup, monkeypatch, censor, omit):
    module = study()
    source_stub(monkeypatch, module, setup)
    injected_collection(monkeypatch, module)
    fake_evaluator(monkeypatch, module, censor=censor, omit=omit)
    result = module.run_behavioral_study(**setup)
    assert result["status"] == "partial" and result["comparisons"] == {}
    assert result["comparison_unavailable"]


def test_expired_collection_keeps_exact_matrix_and_no_zero_supervision_fit(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    setup["protocol"]["deadline"] = time.monotonic() - 1
    with pytest.raises(ValueError, match="zero|supervision|choices"):
        module.run_behavioral_study(**setup)
    result = json.loads((setup["output_dir"] / "results.json").read_text())
    assert result["status"] == "failed" and result["comparisons"] == {}
    assert (setup["output_dir"] / "manifest.json").exists()
    assert len(list(setup["output_dir"].glob("*-labels.gz"))) == 2


def test_missing_seed_and_nonlegacy_start_fail_before_output(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    setup["initial_checkpoints"][29] = setup["initial_checkpoints"][11]
    with pytest.raises(ValueError, match="matrix"):
        module.run_behavioral_study(**setup)
    setup["initial_checkpoints"].pop(29)
    path = setup["initial_checkpoints"][11]
    from colonist_rl.policy import consequence_from_legacy

    save_checkpoint(
        path, consequence_from_legacy(GraphPolicy(hidden_size=8), feature_set="hand_recipe")
    )
    setup["protocol"]["input_digests"][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="legacy"):
        module.run_behavioral_study(**setup)
    assert not setup["output_dir"].exists()


def test_partial_fit_checkpoints_actual_updates_and_optimizer_state(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    injected_collection(monkeypatch, module)
    fake_evaluator(monkeypatch, module)
    real_fit = module.fit_sequences

    def interrupted(policy, episodes, **kwargs):
        data = real_fit(policy, episodes, **kwargs)
        data.update(status="incomplete", incomplete_reason="deadline", epochs=0)
        return data

    monkeypatch.setattr(module, "fit_sequences", interrupted)
    result = module.run_behavioral_study(**setup)
    assert result["status"] == "partial" and not result["comparisons"]
    for arm in ("teacher_state", "student_state"):
        record = result["students"]["11"][arm]
        assert len(record["fit_segments"]) == 1
        assert record["actual_work"]["updates"] > 0 and record["actual_work"]["epochs"] == 0
        _, checkpoint = load_checkpoint(record["checkpoints"][0]["path"])
        assert checkpoint["optimizer_state"]["state"]
        assert checkpoint["metadata"]["partial"] is True


def test_fit_exception_keeps_changed_parameters_optimizer_and_failure_journal(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    injected_collection(monkeypatch, module)
    real_fit = module.fit_sequences

    def crash(policy, episodes, **kwargs):
        real_fit(policy, episodes, **kwargs)
        raise RuntimeError("injected fit failure after update")

    monkeypatch.setattr(module, "fit_sequences", crash)
    with pytest.raises(RuntimeError, match="injected"):
        module.run_behavioral_study(**setup)
    result = json.loads((setup["output_dir"] / "results.json").read_text())
    record = result["students"]["11"]["teacher_state"]
    assert result["status"] == "failed" and result["comparisons"] == {}
    assert "unknown" in record["work_after_exception"]
    policy, checkpoint = load_checkpoint(record["checkpoints"][0]["path"])
    assert checkpoint["optimizer_state"]["state"]
    assert module._state_digest(policy) != record["initial_state_sha256"]


def test_evaluation_exception_preserves_authoritative_completed_game(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    injected_collection(monkeypatch, module)

    def crash(policy=None, *, on_game_end, **kwargs):
        on_game_end({"board_seed": 185, "seat": 0, "won": True, "victory_points": 10})
        raise RuntimeError("injected game failure")

    monkeypatch.setattr(module, "evaluate_blocks", crash)
    with pytest.raises(RuntimeError, match="game failure"):
        module.run_behavioral_study(**setup)
    result = json.loads((setup["output_dir"] / "results.json").read_text())
    assert result["game_progress"]["heuristic"][0]["won"] is True
    assert result["status"] == "failed" and result["comparisons"] == {}
    assert (setup["output_dir"] / "games" / "heuristic" / "185-0.json").exists()


def test_final_input_and_raw_component_drift_suppresses_results(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    injected_collection(monkeypatch, module)
    fake_evaluator(monkeypatch, module)
    real_eval = module.evaluate_blocks

    def drift(*args, **kwargs):
        result = real_eval(*args, **kwargs)
        if args[0] is not None:
            raw = setup["output_dir"] / "teacher_state-11-labels.gz"
            raw.write_bytes(raw.read_bytes() + b" ")
        return result

    monkeypatch.setattr(module, "evaluate_blocks", drift)
    with pytest.raises(ValueError, match="component|digest"):
        module.run_behavioral_study(**setup)
    result = json.loads((setup["output_dir"] / "results.json").read_text())
    assert result["status"] == "failed" and result["comparisons"] == {}
    assert result["final_integrity"]["verified"] is False


def test_collector_settings_and_actual_board_schedule_are_verified(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    real_teacher = module.collect_demonstrations

    def wrong_board(**kwargs):
        kwargs["board_seeds"] = [181]
        return real_teacher(**kwargs)

    monkeypatch.setattr(module, "collect_demonstrations", wrong_board)
    with pytest.raises(ValueError, match="board|schedule"):
        module.run_behavioral_study(**setup)
    assert (setup["output_dir"] / "teacher_state-11-labels.gz").exists()


def test_unmatched_teacher_call_counts_as_query_without_fabricated_pair(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    fake_evaluator(monkeypatch, module)
    real_student = module.collect_student_labels

    def unmatched(policy, **kwargs):
        cap = kwargs["max_decisions"]
        kwargs["max_decisions"] = cap - 1
        data = real_student(policy, **kwargs)
        data["collection"].update(max_decisions=cap, status="incomplete", stop_reason="deadline")
        episode = data["episodes"][-1]
        episode["incomplete_reason"] = "deadline"
        episode["teacher_calls"].append(
            {
                "ordinal": cap - 1,
                "version": episode["steps"],
                "action_index": 0,
            }
        )
        return data

    monkeypatch.setattr(module, "collect_student_labels", unmatched)
    result = module.run_behavioral_study(**setup)
    record = result["students"]["11"]["student_state"]
    assert record["acquisition"]["queries"] == 4
    assert record["acquisition"]["paired_labels"] == 3
    assert record["acquisition"]["unmatched_teacher_calls"] == 1
    assert record["selection"]["counts"]["choices"] == 3
    assert result["students"]["11"]["teacher_state"]["selection"]["counts"]["choices"] == 3
    assert result["status"] == "partial" and result["comparisons"] == {}


def test_evaluation_has_one_global_ceiling(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    injected_collection(monkeypatch, module)
    fake_evaluator(monkeypatch, module)
    real_eval = module.evaluate_blocks
    deadlines = []

    def evaluate(*args, **kwargs):
        deadlines.append(kwargs["deadline"])
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate_blocks", evaluate)
    module.run_behavioral_study(**setup)
    assert len(deadlines) == 3 and len(set(deadlines)) == 1
    assert deadlines[0] <= setup["protocol"]["deadline"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"seed": 29},
        {"seed": True},
        {
            "seed": 11,
            "completed_epochs": 2,
            "updates": 1,
            "supervised_presentations": 4,
            "fit_calls": [
                {
                    "status": "incomplete",
                    "epochs": 2,
                    "requested_epochs": 3,
                    "updates": 1,
                    "supervised_presentations": 4,
                }
            ],
        },
    ],
)
def test_mismatched_or_partial_starting_checkpoint_is_not_completed_evidence(
    setup, monkeypatch, metadata
):
    module = study()
    source_stub(monkeypatch, module, setup)
    path = setup["initial_checkpoints"][11]
    save_checkpoint(path, GraphPolicy(hidden_size=8), metadata=metadata)
    setup["protocol"]["input_digests"][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="checkpoint|seed|completed"):
        module.run_behavioral_study(**setup)
    assert not setup["output_dir"].exists()


def test_actual_weight_mass_matches_the_declared_complete_schedule(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    injected_collection(monkeypatch, module)
    fake_evaluator(monkeypatch, module)
    result = module.run_behavioral_study(**setup)
    for arm in ("teacher_state", "student_state"):
        record = result["students"]["11"][arm]
        assert (
            record["actual_work"]["weighted_loss_mass"]
            == record["expected_work"]["weighted_loss_mass"]
        )
        assert record["actual_work"]["weighted_loss_mass"] == 16


def test_uncovered_new_phase_fails_before_any_fit(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    path = setup["training_components"][11][0]
    data = module.load_demonstrations(path)
    data["episodes"][0]["samples"] = data["episodes"][0]["samples"][:1]
    from colonist_rl.imitation import _coverage

    data["coverage"] = _coverage(data["episodes"])
    save_demonstrations(path, data)
    setup["protocol"]["input_digests"][path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    injected_collection(monkeypatch, module)
    with pytest.raises(ValueError, match="uncovered"):
        module.run_behavioral_study(**setup)
    assert not list(setup["output_dir"].glob("*.pt"))


def test_partial_last_segment_cannot_reuse_earlier_checkpoint_score(setup, monkeypatch):
    module = study()
    source_stub(monkeypatch, module, setup)
    injected_collection(monkeypatch, module)
    fake_evaluator(monkeypatch, module)
    real_fit, real_clock = module.fit_sequences, time.monotonic
    calls, offset = [0], [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: real_clock() + offset[0])

    def interrupted_last(policy, episodes, **kwargs):
        data = real_fit(policy, episodes, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            offset[0] = 10.0
            data.update(status="incomplete", epochs=0, incomplete_reason="deadline")
        return data

    monkeypatch.setattr(module, "fit_sequences", interrupted_last)
    result = module.run_behavioral_study(**setup)
    record = result["students"]["11"]["teacher_state"]
    assert "validation_metrics" not in record
    assert len(record["validation_scores"]) == 1
    assert record["validation_scores"][0]["checkpoint"] == record["checkpoints"][0]
    assert record["actual_work"]["weighted_loss_mass"] == 16
    assert result["status"] == "partial" and not result["comparisons"]


def test_teacher_save_cannot_truncate_an_input_through_legacy_staging_hardlink(setup, monkeypatch):
    import os

    module = study()
    source_stub(monkeypatch, module, setup)
    fake_evaluator(monkeypatch, module)
    real_teacher = module.collect_demonstrations
    original = Path(setup["training_components"][11][0])
    before, inode = original.read_bytes(), original.stat().st_ino
    alias = setup["output_dir"] / "teacher_state-11-labels.gz.tmp"

    def linked_stage(**kwargs):
        data = real_teacher(**kwargs)
        os.link(original, alias)
        return data

    monkeypatch.setattr(module, "collect_demonstrations", linked_stage)
    result = None
    try:
        result = module.run_behavioral_study(**setup)
    except ValueError:
        pass  # The old saver corrupts the input, then the study detects drift.
    assert original.read_bytes() == before
    assert original.stat().st_ino == alias.stat().st_ino == inode
    assert result["status"] == "complete"
