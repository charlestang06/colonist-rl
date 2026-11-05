"""Scaling preserves immutable histories, canonical Adam order and durable boundaries."""

import importlib
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from colonist_rl.imitation import collect_demonstrations
from colonist_rl.policy import GraphPolicy, consequence_from_legacy
from colonist_rl.supervised_fit import fit_sequences


def scaling():
    assert importlib.util.find_spec("colonist_rl.scaling_study") is not None, (
        "component scaling driver is missing"
    )
    return importlib.import_module("colonist_rl.scaling_study")


@pytest.fixture
def sequences():
    torch.set_num_threads(1)
    original = collect_demonstrations(board_seeds=[18], max_decisions=3)["episodes"][0]["samples"]
    components = []
    for offset in (0, 1):
        episode = []
        for i in range(21 + offset):
            view = original[i % len(original)]["observation"]
            view = replace(
                view,
                context=replace(view.context, version=i),
                legal_actions=view.legal_actions[:1] if i % 3 == 0 else view.legal_actions,
            )
            episode.append((view, 0))
        components.append((tuple(episode),))
    return tuple(components)


def equal_tree(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal_tree(left, right)
    else:
        assert a == b


def test_component_fit_is_bitwise_adam_and_counter_parity_with_negative_controls(sequences):
    module = scaling()
    torch.manual_seed(11)
    a = consequence_from_legacy(GraphPolicy(32))
    b, reset, stripped = deepcopy(a), deepcopy(a), deepcopy(a)
    adam_a, adam_b = (torch.optim.Adam(p.parameters(), lr=0.0003) for p in (a, b))
    weights = {v.context.kind: 1.0 for c in sequences for e in c for v, _ in e}
    full = fit_sequences(
        a,
        sequences[0] + sequences[1],
        epochs=2,
        sequence_length=16,
        batch_size=1,
        optimizer=adam_a,
        phase_weights=weights,
        profile=True,
    )
    totals = module._empty_fit()
    for _ in range(2):
        for episodes in sequences:
            row = module._fit_component(b, episodes, adam_b, weights, time.monotonic() + 60)
            module._add_fit(totals, row, weights, episodes)
            fit_sequences(reset, episodes, epochs=1, phase_weights=weights, profile=True)
            fit_sequences(
                stripped,
                tuple(tuple((v, t) for v, t in e if len(v.legal_actions) > 1) for e in episodes),
                epochs=1,
                phase_weights=weights,
                profile=True,
            )
    equal_tree(a.state_dict(), b.state_dict())
    equal_tree(adam_a.state_dict(), adam_b.state_dict())
    for key in ("updates", "supervised_presentations"):
        assert full[key] == totals[key]
        assert full[key] > 0
    for key in ("prefix_forwards", "chunk_forwards", "batches", "batch_chunks"):
        assert full["profile"][key] == totals["profile"][key]
    assert math.isclose(
        full["loss"] * full["supervised_presentations"],
        totals["loss_sum"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert any(not torch.equal(v, reset.state_dict()[k]) for k, v in a.state_dict().items())
    assert any(not torch.equal(v, stripped.state_dict()[k]) for k, v in a.state_dict().items())


def test_protocol_rejects_unknown_keys_and_wrong_types_before_runtime(tmp_path):
    module = scaling()
    with pytest.raises(ValueError, match="protocol"):
        module.collect_scaling_corpus(tmp_path, protocol={"version": True})
    assert not list(tmp_path.iterdir())


def test_publication_is_exclusive_and_fixed_tmp_aliases_cannot_damage_inputs(tmp_path):
    module = scaling()
    target, original = tmp_path / "evidence.json", tmp_path / "input"
    original.write_text("preserved")
    target.with_name(target.name + ".tmp").hardlink_to(original)
    module._publish_json(target, {"value": 1})
    assert original.read_text() == "preserved"
    with pytest.raises(FileExistsError):
        module._publish_json(target, {"value": 2})
    assert json.loads(target.read_text()) == {"value": 1}


def test_selection_requires_exact_matrix_and_uses_only_complete_counters():
    module = scaling()
    rows = {key: {"milestones": {}} for key in module.ARMS}
    with pytest.raises(ValueError, match="matrix"):
        module._select_milestone({})
    for row in rows.values():
        row["milestones"] = {"1": {"complete": True}, "3": {"complete": True}}
    rows[module.ARMS[0]]["milestones"]["5"] = {"complete": True}
    assert module._select_milestone(rows) == {"epoch": 3, "primary_five_complete": False}
    rows[module.ARMS[-1]]["milestones"] = {}
    assert module._select_milestone(rows)["epoch"] is None


@pytest.fixture
def launch(tmp_path, monkeypatch):
    module = scaling()
    import sys

    from colonist_rl.engine_identity import verify_engine_installation
    from colonist_rl.training import save_checkpoint

    source = {"revision": "a" * 40, "dirty": False}
    monkeypatch.setattr(module, "_source", lambda: deepcopy(source))
    monkeypatch.setattr(module, "EVALUATION_BOARDS", 1)
    monkeypatch.setattr(module, "ENDPOINTS", (2, 4))
    monkeypatch.setattr(module, "MILESTONES", (2, 3, 4))
    monkeypatch.setattr(module, "PHASES", ("SETUP_SETTLEMENT", "SETUP_ROAD"))
    starts = {}
    for seed in (11, 29):
        parent, bridge = tmp_path / f"parent-{seed}", tmp_path / f"bridge-{seed}"
        parent.write_bytes(b"legacy-parent")
        bridge.write_bytes(b"new-engine-parent")
        metadata = {
            "migration": {
                "kind": "weights_only_cross_engine",
                "source_sha256": module._ref(parent)["sha256"],
                "source_backend_version": "3.2.1",
                "target_backend_version": module.BACKEND_VERSION,
                "source_steps": 0,
                "seed": seed,
            },
            "feature_upgrade": {
                "kind": "zero_additive_consequence",
                "parent_sha256": module._ref(bridge)["sha256"],
                "feature_set": "hand_recipe",
            },
            "seed": seed,
            "purpose": "untrained_initialization",
        }
        path = tmp_path / f"start-{seed}.pt"
        save_checkpoint(path, consequence_from_legacy(GraphPolicy(32)), metadata=metadata)
        starts[str(seed)] = {
            "checkpoint": module._ref(path),
            "parent": module._ref(parent),
            "bridge": module._ref(bridge),
            "metadata": metadata,
        }
    until = time.monotonic() + 600
    protocol = {
        "format": "colonist-rl.scaling-protocol",
        "version": 1,
        "source": source,
        "runtime": {
            "engine": verify_engine_installation(),
            "python": sys.executable,
            "torch": str(torch.__version__),
            "threads": 1,
        },
        "partitions": {
            "training": list(range(100, 108)),
            "validation_labels": list(range(200, 204)),
            "validation_games": [300],
            "audit": [400],
            "final_test": [500],
        },
        "prior_reservations": {"artifacts": [], "excluded_boards": [600]},
        "inputs": {"starts": starts, "corpus": None, "results": None, "resume": None},
        "collection": {
            "milestones": [2, 3, 4],
            "max_decisions": 80004,
            "max_game_steps": 20000,
            "batch_seconds": 120,
        },
        "training": {
            "epochs": 5,
            "sequence_length": 16,
            "batch_size": 1,
            "learning_rate": 0.0003,
            "hidden_size": 32,
            "feature_set": "hand_recipe",
            "score_epochs": [0, 1, 3, 5],
            "required_phases": ["SETUP_SETTLEMENT", "SETUP_ROAD"],
        },
        "evaluation": {"max_game_steps": 20000, "deterministic": True, "baseline": "heuristic"},
        "limits": {
            "global_deadline": until,
            "collection_deadline": until,
            "fit_deadlines": {arm: until for arm in module.ARMS},
            "evaluation_deadline": until,
            "rss_bytes": 2 * 1024**3,
        },
    }
    template = collect_demonstrations(board_seeds=[18], max_decisions=3)
    calls = []

    def collector(*, board_seeds, **kwargs):
        boards = list(board_seeds)
        calls.append(boards)
        data = deepcopy(template)
        data["source"] = deepcopy(source)
        data["collection"].update(
            baseline="heuristic",
            max_decisions=80004,
            max_game_steps=20000,
            actual_board_seeds=boards,
            stop_reason="seeds_exhausted",
        )
        data["episodes"] = []
        for seat, board in enumerate(boards):
            episode = deepcopy(template["episodes"][0])
            episode.update(
                board_seed=board,
                backend_seed=board + 1,
                teacher_seed=board + 100,
                seat=seat,
                complete=True,
                terminated=True,
                truncated=False,
                incomplete_reason=None,
            )
            for sample in episode["samples"]:
                view = sample["observation"]
                sample["observation"] = replace(
                    view, player_id=seat, context=replace(view.context, actor=seat)
                )
            # Preserve a forced history element, followed by real legal menus.
            sample = episode["samples"][0]
            sample["observation"] = replace(
                sample["observation"], legal_actions=sample["observation"].legal_actions[:1]
            )
            sample["action_index"] = 0
            data["episodes"].append(episode)
        from colonist_rl.imitation import _coverage

        data["coverage"] = _coverage(data["episodes"])
        return data

    phases = list(template["coverage"]["phase_counts"])
    protocol["training"]["required_phases"] = phases
    monkeypatch.setattr(module, "PHASES", tuple(phases))

    def replay(data, protocol, deadline):
        expected = collector(board_seeds=data["collection"]["actual_board_seeds"])
        if module._json(module._episode_payload(data)) != module._json(
            module._episode_payload(expected)
        ):
            raise ValueError("teacher/menu/history replay mismatch")

    monkeypatch.setattr(module, "_replay_component", replay)
    monkeypatch.setattr(module, "collect_demonstrations", collector)
    return module, protocol, calls, collector


def test_collection_finishes_four_board_batch_overshoots_and_freezes_nested_refs(launch, tmp_path):
    module, protocol, calls, _ = launch
    result = module.collect_scaling_corpus(tmp_path / "corpus", protocol=protocol)
    assert result["status"] == "complete"
    assert result["endpoints"]["2"] == result["endpoints"]["4"] == 1
    assert result["training_non_forced"] == 8
    assert [ref["boards"] for ref in result["training"]] == [[100, 101, 102, 103]]
    assert calls[0] == [100, 101, 102, 103]
    assert result["phase_weights"]
    assert result["validation"][0]["boards"] == [200, 201, 202, 203]
    assert result["first_gate"]["non_forced"] == 8
    raw = module.load_demonstrations(result["training"][0]["artifact"]["path"])
    assert [e["seat"] for e in raw["episodes"]] == [0, 1, 2, 3]
    assert len(raw["episodes"][0]["samples"][0]["observation"].legal_actions) == 1
    assert json.loads((tmp_path / "corpus" / "protocol.json").read_bytes()) == protocol


def test_incomplete_component_is_saved_stops_without_replacement_and_never_counts(
    launch, tmp_path, monkeypatch
):
    module, protocol, calls, collector = launch

    def truncated(**kwargs):
        data = collector(**kwargs)
        data["episodes"][-1].update(
            complete=False, terminated=False, truncated=True, incomplete_reason="step_cap"
        )
        return data

    monkeypatch.setattr(module, "collect_demonstrations", truncated)
    result = module.collect_scaling_corpus(tmp_path / "corpus", protocol=protocol)
    assert result["status"] == "partial"
    assert result["training"] == []
    assert result["phase_weights"] is None
    assert len(result["raw_components"]) == 1
    assert calls == [[100, 101, 102, 103]]
    assert Path(result["raw_components"][0]["artifact"]["path"]).exists()


@pytest.mark.parametrize(
    "mutation", ["overlap", "reserved", "bool", "unknown", "hash", "runtime", "source", "cap"]
)
def test_protocol_failures_precede_collection(launch, tmp_path, mutation):
    module, protocol, calls, _ = launch
    if mutation == "overlap":
        protocol["partitions"]["audit"] = [100]
    elif mutation == "reserved":
        protocol["partitions"]["audit"] = [600]
    elif mutation == "bool":
        protocol["training"]["batch_size"] = True
    elif mutation == "unknown":
        protocol["evaluation"]["extra"] = 2
    elif mutation == "hash":
        protocol["inputs"]["starts"]["11"]["checkpoint"]["sha256"] = "x" * 64
    elif mutation == "runtime":
        protocol["runtime"]["engine"]["number_placement"] = "legacy"
    elif mutation == "source":
        protocol["source"]["dirty"] = True
    else:
        protocol["limits"]["rss_bytes"] = 2 * 1024**3 + 1
    with pytest.raises((ValueError, RuntimeError)):
        module.collect_scaling_corpus(tmp_path / "bad", protocol=protocol)
    assert calls == []
    assert not (tmp_path / "bad").exists()


def test_component_loader_rejects_digest_typed_metadata_and_teacher_history(launch, tmp_path):
    module, protocol, _, collector = launch
    data = collector(board_seeds=[100, 101, 102, 103])
    ref = module._publish(tmp_path / "component.gz", lambda p: module.save_demonstrations(p, data))
    module._load_component(ref, [100, 101, 102, 103], protocol)
    for change in (
        "digest",
        "typed",
        "teacher",
        "source",
        "backend",
        "unfinished",
        "missing_history",
    ):
        bad = deepcopy(data)
        target = tmp_path / (change + ".gz")
        if change == "typed":
            bad["episodes"][0]["backend_seed"] = 101.0
        elif change == "teacher":
            bad["episodes"][0]["samples"][1]["action_index"] = (
                bad["episodes"][0]["samples"][1]["action_index"] + 1
            ) % len(bad["episodes"][0]["samples"][1]["observation"].legal_actions)
        elif change == "source":
            bad["source"]["revision"] = "b" * 40
        elif change == "backend":
            bad["backend_version"] = "old"
        elif change == "unfinished":
            bad["episodes"][0].update(
                complete=False, terminated=False, incomplete_reason="deadline"
            )
        elif change == "missing_history":
            bad["episodes"][0]["samples"].pop(0)
            from colonist_rl.imitation import _coverage

            bad["coverage"] = _coverage(bad["episodes"])
        if change == "backend":
            import gzip

            from colonist_rl.adapters.observation_io import observation_to_dict

            raw = deepcopy(bad)
            for episode in raw["episodes"]:
                for sample in episode["samples"]:
                    sample["observation"] = observation_to_dict(sample["observation"])
            target.write_bytes(gzip.compress(json.dumps(raw).encode()))
        else:
            module.save_demonstrations(target, bad)
        artifact = module._ref(target)
        if change == "digest":
            artifact["sha256"] = "0" * 64
        with pytest.raises(ValueError):
            module._load_component(artifact, [100, 101, 102, 103], protocol)


def corpus_launch(launch, tmp_path):
    module, protocol, _, _ = launch
    corpus = module.collect_scaling_corpus(tmp_path / "corpus", protocol=protocol)
    assert corpus["status"] == "complete"
    protocol = deepcopy(protocol)
    protocol["inputs"]["corpus"] = corpus["manifest"]
    return module, protocol, corpus


def fit_arm(module, protocol, output, resume=None):
    return module.fit_scaling_arm(
        output,
        corpus_manifest=protocol["inputs"]["corpus"]["path"],
        initial_checkpoint=protocol["inputs"]["starts"]["11"]["checkpoint"]["path"],
        endpoint=2,
        seed=11,
        protocol=protocol,
        resume_cursor=resume,
    )


def test_streamed_fit_releases_every_component_and_preserves_rng(launch, tmp_path, monkeypatch):
    import weakref

    module, protocol, corpus = corpus_launch(launch, tmp_path)
    original = module._load_component
    active = []

    class Tracked(dict):
        pass

    def load(*args, **kwargs):
        assert all(ref() is None for ref in active), "previous component retained"
        obj = Tracked(original(*args, **kwargs))
        active.append(weakref.ref(obj))
        return obj

    monkeypatch.setattr(module, "_load_component", load)
    # Forbidden whole-corpus helper names cannot be reached through another adapter.
    import colonist_rl.student_labels as corpus_api

    for name in ("write_corpus", "load_corpus", "corpus_sequences"):
        monkeypatch.setattr(
            corpus_api, name, lambda *a, **k: pytest.fail("whole corpus helper called")
        )
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    result = fit_arm(module, protocol, tmp_path / "arm")
    assert result["status"] == "complete"
    assert result["completed_epochs"] == 5
    assert result["stats"]["supervised_presentations"] == 5 * corpus["training_non_forced"]
    assert result["stats"]["updates"] == 20
    assert set(result["milestones"]) == {"1", "3", "5"}
    assert set(result["scores"]) == {"0", "1", "3", "5"}
    assert all(ref() is None for ref in active)
    assert random.getstate() == state[0]
    assert np.array_equal(np.random.get_state()[1], state[1][1])
    assert torch.equal(torch.get_rng_state(), state[2])


@pytest.mark.parametrize("failure", ["expiry", "exception", "cursor"])
def test_interruption_keeps_boundary_and_explicit_resume_has_no_duplicate_updates(
    launch, tmp_path, monkeypatch, failure
):
    module, protocol, _ = corpus_launch(launch, tmp_path)
    original_fit, original_json = module._fit_component, module._publish_json
    calls = 0

    def interrupted(policy, episodes, optimizer, weights, deadline):
        nonlocal calls
        calls += 1
        row = original_fit(policy, episodes, optimizer, weights, deadline)
        if calls == 2:
            if failure == "exception":
                raise ArithmeticError("after actual updates")
            if failure == "expiry":
                row["status"], row["epochs"] = "incomplete", 0
        return row

    def publication(path, value):
        if failure == "cursor" and Path(path).name.startswith("cursor-") and calls == 2:
            raise OSError("cursor publication failed")
        return original_json(path, value)

    monkeypatch.setattr(module, "_fit_component", interrupted)
    monkeypatch.setattr(module, "_publish_json", publication)
    rng = torch.get_rng_state().clone()
    partial = fit_arm(module, protocol, tmp_path / "partial")
    assert torch.equal(torch.get_rng_state(), rng)
    assert partial["status"] == "partial"
    assert partial["completed_epochs"] == 1
    assert partial["partial_last"]
    assert partial["attempted"]["updates_observed_adam"] == 4
    assert partial["stats"]["updates"] == 4
    monkeypatch.setattr(module, "_fit_component", original_fit)
    monkeypatch.setattr(module, "_publish_json", original_json)
    protocol["inputs"]["resume"] = partial["cursor"]
    resumed = fit_arm(module, protocol, tmp_path / "resumed", resume=partial["cursor"]["path"])
    assert resumed["status"] == "complete"
    assert resumed["stats"]["updates"] == 20
    assert resumed["completed_epochs"] == 5
    uninterrupted_protocol = deepcopy(protocol)
    uninterrupted_protocol["inputs"]["resume"] = None
    full = fit_arm(module, uninterrupted_protocol, tmp_path / "full")
    left = torch.load(resumed["milestones"]["5"]["checkpoint"]["path"], weights_only=True)
    right = torch.load(full["milestones"]["5"]["checkpoint"]["path"], weights_only=True)
    equal_tree(left["policy_state"], right["policy_state"])
    equal_tree(left["optimizer_state"], right["optimizer_state"])
    assert resumed["stats"]["supervised_presentations"] == full["stats"]["supervised_presentations"]
    bad = deepcopy(protocol)
    bad["limits"]["global_deadline"] += 1
    with pytest.raises(ValueError, match="identity"):
        fit_arm(module, bad, tmp_path / "wrong-resume", resume=partial["cursor"]["path"])


def test_rss_guard_refuses_before_loading_next_component(launch, tmp_path, monkeypatch):
    module, protocol, _ = corpus_launch(launch, tmp_path)
    loaded = []
    original = module.load_demonstrations
    monkeypatch.setattr(module, "load_demonstrations", lambda p: loaded.append(p) or original(p))
    monkeypatch.setattr(module, "_rss", lambda: protocol["limits"]["rss_bytes"] + 1)
    with pytest.raises(MemoryError):
        fit_arm(module, protocol, tmp_path / "over-rss")
    assert loaded == []


def test_corpus_manifest_hashes_summaries_and_partial_primary_are_not_trusted(launch, tmp_path):
    module, protocol, corpus = corpus_launch(launch, tmp_path)
    for change in ("counts", "weights", "boards", "partial"):
        raw = json.loads(Path(corpus["manifest"]["path"]).read_bytes())
        if change == "counts":
            raw["training"][0]["summary"]["coverage"]["non_forced_decisions"] += 1
        elif change == "weights":
            raw["phase_weights"][next(iter(raw["phase_weights"]))] *= 2
        elif change == "boards":
            raw["training"][0]["boards"][0] = 600
        else:
            raw["status"] = "partial"
        ref = module._publish_json(tmp_path / f"bad-{change}.json", raw)
        bad = deepcopy(protocol)
        bad["inputs"]["corpus"] = ref
        with pytest.raises(ValueError):
            fit_arm(module, bad, tmp_path / f"fit-{change}")


@pytest.mark.parametrize("journal_mode", ["complete", "missing", "censored", "fallback"])
def test_evaluation_frozen_matrix_uses_validation_boards_and_journal_suppresses_pairs(
    launch, tmp_path, monkeypatch, journal_mode
):
    module, protocol, _ = corpus_launch(launch, tmp_path)
    results = {}
    original_fit = module._fit_component
    calls = 0

    def interrupted_fourth(*args):
        nonlocal calls
        calls += 1
        result = original_fit(*args)
        if calls == 4:
            result["status"], result["epochs"] = "incomplete", 0
        return result

    for index, arm in enumerate(module.ARMS):
        if journal_mode == "fallback" and index == 0:
            monkeypatch.setattr(module, "_fit_component", interrupted_fourth)
        else:
            monkeypatch.setattr(module, "_fit_component", original_fit)
        endpoint, seed = (2 if index < 2 else 4), int(arm.split("/")[1])
        fitted = module.fit_scaling_arm(
            tmp_path / f"arm-{index}",
            corpus_manifest=protocol["inputs"]["corpus"]["path"],
            initial_checkpoint=protocol["inputs"]["starts"][str(seed)]["checkpoint"]["path"],
            endpoint=endpoint,
            seed=seed,
            protocol=protocol,
        )
        assert fitted["status"] == (
            "partial" if journal_mode == "fallback" and index == 0 else "complete"
        )
        results[arm] = fitted["result"]
    protocol["inputs"]["results"] = results
    boards_played = []

    def evaluate(**kwargs):
        assert (tmp_path / "evaluation" / "selection.json").exists()
        assert kwargs["baseline"] == "heuristic" and kwargs["deterministic"] is True
        boards = kwargs["board_seeds"]
        boards_played.extend(boards)
        rows = [
            {
                "board_seed": board,
                "seat": seat,
                "won": False,
                "victory_points": 4,
                "terminated": True,
                "truncated": False,
                "winner": 2,
            }
            for board in boards
            for seat in range(4)
        ]
        if journal_mode == "censored":
            rows[0].update(truncated=True, terminated=False, won=None, winner=None)
        for index, row in enumerate(rows):
            if journal_mode != "missing" or index != 0:
                kwargs["on_game_end"](row)
        return {"status": "complete", "outcomes": rows, "partial_outcomes": []}

    monkeypatch.setattr(module, "evaluate_blocks", evaluate)
    result = module.evaluate_scaling_study(
        tmp_path / "evaluation",
        arm_results={k: v["path"] for k, v in results.items()},
        corpus_manifest=protocol["inputs"]["corpus"]["path"],
        protocol=protocol,
    )
    assert (
        result["selection"] == {"epoch": 3, "primary_five_complete": False}
        if journal_mode == "fallback"
        else result["selection"] == {"epoch": 5, "primary_five_complete": True}
    )
    assert boards_played == [300] * 5
    if journal_mode in ("complete", "fallback"):
        assert len(result["comparisons"]) == 6
    else:
        assert result["comparisons"] == {}
        assert result["comparison_unavailable"]
    assert set(result["journals"]) == {"heuristic", *module.ARMS}


def test_single_board_collector_restarts_at_seat_zero_but_four_board_call_rotates():
    alone = collect_demonstrations(board_seeds=[31], max_decisions=80004, max_game_steps=1)
    four = collect_demonstrations(
        board_seeds=[31, 32, 33, 34], max_decisions=80004, max_game_steps=1
    )
    assert [e["seat"] for e in alone["episodes"]] == [0]
    assert [e["seat"] for e in four["episodes"]] == [0, 1, 2, 3]


def test_real_expiry_during_component_keeps_prior_cursor_and_records_attempt(
    launch, tmp_path, monkeypatch
):
    module, protocol, _ = corpus_launch(launch, tmp_path)
    import colonist_rl.supervised_fit as fitter

    original_expired = fitter._expired
    original_step = torch.optim.Adam.step
    seen = 0

    def step(self, *args, **kwargs):
        nonlocal seen
        result = original_step(self, *args, **kwargs)
        seen += 1
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(
        fitter, "_expired", lambda deadline: seen >= 1 or original_expired(deadline)
    )
    result = fit_arm(module, protocol, tmp_path / "expired")
    assert result["status"] == "partial"
    assert result["completed_epochs"] == 0
    assert result["stats"]["updates"] == 0
    assert result["attempted"]["updates_observed_adam"] == 1
    assert result["attempted"]["reported"]["status"] == "incomplete"
    assert result["attempted"]["reported"]["epochs"] == 0


def test_resume_recovers_pending_score_before_next_component(launch, tmp_path, monkeypatch):
    module, protocol, _ = corpus_launch(launch, tmp_path)
    score = module._score
    scored = 0

    def interrupted_score(*args):
        nonlocal scored
        scored += 1
        if scored == 2:
            raise TimeoutError("epoch-one scoring interrupted")
        return score(*args)

    monkeypatch.setattr(module, "_score", interrupted_score)
    partial = fit_arm(module, protocol, tmp_path / "interrupted-score")
    assert partial["completed_epochs"] == 1
    assert "1" not in partial["scores"]
    monkeypatch.setattr(module, "_score", score)
    protocol["inputs"]["resume"] = partial["cursor"]
    result = fit_arm(module, protocol, tmp_path / "resume-score", partial["cursor"]["path"])
    assert result["status"] == "complete"
    assert set(result["scores"]) == {"0", "1", "3", "5"}


def test_required_phase_and_input_bytes_fail_before_fitting(launch, tmp_path):
    module, protocol, corpus = corpus_launch(launch, tmp_path)
    original = Path(corpus["training"][0]["artifact"]["path"]).read_bytes()
    artifact = corpus["training"][0]["artifact"]
    Path(artifact["path"]).write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="digest"):
        fit_arm(module, protocol, tmp_path / "bad-bytes")
    assert not (tmp_path / "bad-bytes").exists()


def test_two_complete_components_use_shared_nested_prefix_and_global_weights(
    launch, tmp_path, monkeypatch
):
    module, protocol, _, _ = launch
    monkeypatch.setattr(module, "ENDPOINTS", (8, 16))
    monkeypatch.setattr(module, "MILESTONES", (8, 16))
    protocol["collection"]["milestones"] = [8, 16]
    result = module.collect_scaling_corpus(tmp_path / "two-components", protocol=protocol)
    assert result["status"] == "complete"
    assert result["endpoints"] == {"8": 1, "16": 2}
    assert result["training_non_forced"] == 16
    assert [r["boards"] for r in result["training"]] == [
        list(range(100, 104)),
        list(range(104, 108)),
    ]
    assert sum(r["summary"]["coverage"]["non_forced_decisions"] for r in result["validation"]) == 8


def test_collection_refuses_missing_declared_supervised_phase(launch, tmp_path, monkeypatch):
    module, protocol, _, _ = launch
    monkeypatch.setattr(module, "PHASES", (*module.PHASES, "TRADE_RESPONSE"))
    protocol["training"]["required_phases"].append("TRADE_RESPONSE")
    result = module.collect_scaling_corpus(tmp_path / "missing-phase", protocol=protocol)
    assert result["status"] == "partial"
    assert result["phase_weights"] is None
    assert "phase" in result["stop_reason"]


def test_checkpoint_publication_isolates_fixed_tmp_hardlink_alias(launch, tmp_path):
    module, protocol, _, _ = launch
    source = Path(protocol["inputs"]["starts"]["11"]["checkpoint"]["path"])
    original = source.read_bytes()
    target = tmp_path / "new.pt"
    target.with_name(target.name + ".tmp").hardlink_to(source)
    policy = module._check_start(protocol["inputs"]["starts"]["11"], 11)
    module._publish(target, lambda p: module.save_checkpoint(p, policy))
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        module._publish(source, lambda p: module.save_checkpoint(p, policy))
    assert source.read_bytes() == original


def test_loader_rejects_hidden_observation_envelope_fields(launch, tmp_path):
    import gzip

    module, protocol, _, collector = launch
    data = collector(board_seeds=[100, 101, 102, 103])
    path = tmp_path / "hidden.json.gz"
    module.save_demonstrations(path, data)
    raw = json.loads(gzip.decompress(path.read_bytes()))
    raw["episodes"][0]["samples"][0]["observation"]["hidden_memory"] = [1.0]
    path.write_bytes(gzip.compress(json.dumps(raw).encode()))
    with pytest.raises(ValueError, match="observation"):
        module._load_component(module._ref(path), [100, 101, 102, 103], protocol)
