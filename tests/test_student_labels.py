"""Student behavior, deadline boundaries, and immutable component provenance."""

import gzip
import hashlib
import importlib
import json
import os
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from colonist_rl.imitation import collect_demonstrations, save_demonstrations
from colonist_rl.opponents import HeuristicPolicy
from colonist_rl.policy import GraphPolicy, policy_architecture
from colonist_rl.simulator import FourPlayerGame
from colonist_rl.training import save_checkpoint


def labels():
    assert importlib.util.find_spec("colonist_rl.student_labels") is not None, (
        "student-state labeling and immutable component corpus interfaces are missing"
    )
    return importlib.import_module("colonist_rl.student_labels")


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    policy = GraphPolicy(hidden_size=8).train()
    policy.node_in.eval()
    path = tmp_path / "student.pt"
    save_checkpoint(path, policy)
    identity = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "architecture": policy_architecture(policy),
    }
    return policy, identity


def scripted(monkeypatch, module, *, expire=None, illegal=False, forced=False):
    """Replace the simulator boundary; keep immutable real Observation contracts."""
    env = FourPlayerGame()
    base = env.reset(seed=19)
    if base is None:
        base = env.observe()
    views = [
        replace(
            base,
            context=replace(base.context, version=i),
            legal_actions=base.legal_actions[: 1 if forced or i == 0 else 2],
        )
        for i in range(3)
    ]
    executed, calls, teachers = [], [], []
    now = [0.0]

    class Game:
        def __init__(self, max_steps):
            self.version = 0

        def reset(self, seed):
            self.version = 0

    class Driver:
        def __init__(self, env, seat, seed, *, baseline):
            self.env = env
            self.seat = seat

        @property
        def view(self):
            if expire == "observation":
                now[0] = 2.0
            view = views[self.env.version]
            return (
                view
                if self.seat == 0
                else replace(
                    view, player_id=self.seat, context=replace(view.context, actor=self.seat)
                )
            )

        def maybe_offer(self, *, deadline):
            pass

        def opponent_action(self, *, deadline):
            raise AssertionError("all scripted views belong to the learner")

        def step(self, action):
            assert now[0] < 1.0
            executed.append(action)
            self.env.version += 1
            return SimpleNamespace(terminated=self.env.version == 3, truncated=False)

    original_act = GraphPolicy.act

    def act(self, view, memory=None, *, deterministic):
        calls.append((view, None if memory is None else memory.item(), self.training))
        random.random()
        np.random.random()
        torch.rand(1)
        action = view.legal_actions[-1]
        if expire == "student":
            now[0] = 2.0
        return action, torch.tensor(float(len(calls)))

    class Teacher:
        def __init__(self, seed):
            self.seed = seed

        def act(self, view):
            teachers.append((self.seed, view))
            if expire == "teacher":
                now[0] = 2.0
            return (
                replace(view.legal_actions[0], kind="ILLEGAL") if illegal else view.legal_actions[0]
            )

    monkeypatch.setattr(module, "FourPlayerGame", Game)
    monkeypatch.setattr(module, "_MatchDriver", Driver)
    monkeypatch.setattr(module, "HeuristicPolicy", Teacher)
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(GraphPolicy, "act", act)
    return views, executed, calls, teachers, original_act


def test_student_executes_disagreements_with_full_forced_history_and_rng_restore(
    monkeypatch, checkpoint, tmp_path
):
    module = labels()
    policy, identity = checkpoint
    views, executed, calls, teachers, _ = scripted(monkeypatch, module)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    modes = [m.training for m in policy.modules()]
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    path = tmp_path / "progress.gz"
    data = module.collect_student_labels(
        policy,
        checkpoint_identity=identity,
        board_seeds=[19],
        max_decisions=4,
        deadline=1.0,
        progress_path=path,
    )
    assert executed == [
        views[0].legal_actions[0],
        views[1].legal_actions[1],
        views[2].legal_actions[1],
    ]
    assert [c[1] for c in calls] == [None, 1.0, 2.0]
    assert all(c[2] is False for c in calls)
    assert [t[0] for t in teachers] == [119, 119, 119]
    assert [t[1] for t in teachers] == views
    samples = data["episodes"][0]["samples"]
    assert [s["teacher_action_index"] for s in samples] == [0, 0, 0]
    assert [s["behavior_action_index"] for s in samples] == [0, 1, 1]
    assert all(s["behavior_executed"] for s in samples)
    assert all(
        set(s)
        == {"observation", "teacher_action_index", "behavior_action_index", "behavior_executed"}
        for s in samples
    )
    assert data["episodes"][0]["complete"] is True
    assert module.load_student_labels(path) == data
    assert modes == [m.training for m in policy.modules()]
    assert all(torch.equal(v, policy.state_dict()[k]) for k, v in before.items())
    assert random.getstate() == rng[0]
    current = np.random.get_state()
    assert current[0] == rng[1][0] and np.array_equal(current[1], rng[1][1])
    assert current[2:] == rng[1][2:]
    assert torch.equal(torch.get_rng_state(), rng[2])


@pytest.mark.parametrize("expire, paired", [("student", 1), ("teacher", 0), ("observation", 0)])
def test_deadline_after_expensive_calls_never_applies_or_retries(
    monkeypatch, checkpoint, expire, paired
):
    module = labels()
    policy, identity = checkpoint
    _, executed, calls, teachers, _ = scripted(monkeypatch, module, expire=expire)
    data = module.collect_student_labels(
        policy,
        checkpoint_identity=identity,
        board_seeds=[19, 20],
        max_decisions=9,
        deadline=1.0,
    )
    assert executed == []
    assert len(calls) == paired
    assert len(teachers) == int(expire != "observation")
    assert len(data["episodes"]) == 1
    episode = data["episodes"][0]
    assert episode["steps"] == 0 and episode["incomplete_reason"] == "deadline"
    assert len(episode["samples"]) == paired
    assert all(s["behavior_executed"] is False for s in episode["samples"])


def test_illegal_teacher_raises_before_student_or_game_mutation(monkeypatch, checkpoint):
    module = labels()
    policy, identity = checkpoint
    _, executed, calls, _, _ = scripted(monkeypatch, module, illegal=True)
    with pytest.raises(ValueError, match="teacher"):
        module.collect_student_labels(
            policy, checkpoint_identity=identity, board_seeds=[19], max_decisions=4
        )
    assert executed == [] and calls == []


def test_checkpoint_bytes_architecture_and_parameters_must_match(checkpoint):
    module = labels()
    policy, identity = checkpoint
    for change in ({"sha256": "0" * 64}, {"architecture": {"hidden_size": 16}}):
        with pytest.raises(ValueError, match="checkpoint"):
            module.collect_student_labels(
                policy, checkpoint_identity=identity | change, board_seeds=[19], max_decisions=1
            )
    with torch.no_grad():
        next(policy.parameters()).add_(1)
    with pytest.raises(ValueError, match="checkpoint"):
        module.collect_student_labels(
            policy, checkpoint_identity=identity, board_seeds=[19], max_decisions=1
        )


def partitions(*boards):
    return {
        "training": list(boards),
        "validation_labels": [800001],
        "validation_games": [800002],
        "audit": [800003],
        "final_test": [70000017, 70000019],
    }


def components(monkeypatch, checkpoint, tmp_path):
    module = labels()
    # Real legacy data keeps its own real collection identity. No schema rewriting.
    teacher = collect_demonstrations(board_seeds=[23], max_decisions=2)
    teacher_path = tmp_path / "teacher.gz"
    first = save_demonstrations(teacher_path, teacher)
    policy, identity = checkpoint
    views, _, _, _, _ = scripted(monkeypatch, module)
    previous_revision = subprocess.check_output(["git", "rev-parse", "HEAD^"], text=True).strip()
    monkeypatch.setattr(module, "_source", lambda: {"revision": previous_revision, "dirty": False})
    student = module.collect_student_labels(
        policy, checkpoint_identity=identity, board_seeds=[19], max_decisions=4
    )
    student_path = tmp_path / "student.gz"
    second = module.save_student_labels(student_path, student)
    return module, (teacher_path, student_path), (first, second), student, views, teacher


def test_corpus_retains_component_sources_order_and_forced_prefixes(
    monkeypatch, checkpoint, tmp_path
):
    module, paths, digests, data, views, teacher = components(monkeypatch, checkpoint, tmp_path)
    manifest = tmp_path / "corpus.json"
    module.write_corpus(manifest, components=tuple(map(str, paths)), partitions=partitions(23, 19))
    corpus = module.load_corpus(manifest)
    assert [c["sha256"] for c in corpus["components"]] == list(digests)
    assert corpus["components"][0]["source"] == teacher["source"]
    assert corpus["components"][1]["source"] == data["source"]
    assert corpus["components"][0]["source"] != corpus["components"][1]["source"]
    seq = module.corpus_sequences(corpus)
    assert seq[1][0][0] == views[0]
    assert seq[1][1] == (views[1], 0)
    assert data["episodes"][0]["samples"][1]["behavior_action_index"] == 1
    assert paths[0].read_bytes() and paths[1].read_bytes()
    reordered = deepcopy(corpus)
    reordered["components"].reverse()
    with pytest.raises(ValueError):
        module.corpus_sequences(reordered)
    paths[1].write_bytes(paths[1].read_bytes() + b"changed")
    with pytest.raises(ValueError):
        module.load_corpus(manifest)
    paths[1].unlink()
    with pytest.raises(ValueError):
        module.corpus_sequences(corpus)


@pytest.mark.parametrize("bad", ["duplicate", "leakage", "overlap", "reserved", "zero"])
def test_corpus_rejects_duplicate_components_split_leaks_and_zero_supervision(
    monkeypatch, checkpoint, tmp_path, bad
):
    module, paths, _, _, _, _ = components(monkeypatch, checkpoint, tmp_path)
    split = partitions(23, 19)
    selected = tuple(map(str, paths))
    if bad == "duplicate":
        selected = (str(paths[0]), str(paths[0]))
    elif bad == "leakage":
        split["training"].remove(19)
    elif bad == "overlap":
        split["audit"] = [23]
    elif bad == "reserved":
        split["training"].append(70000017)
        split["final_test"] = []
    else:
        selected = ()
    with pytest.raises(ValueError):
        module.write_corpus(tmp_path / "bad.json", components=selected, partitions=split)


@pytest.mark.parametrize(
    "bad", ["index", "missing", "actor", "terminal", "execution", "private", "calls"]
)
def test_component_rejects_malformed_history_flags_or_private_fields(
    monkeypatch, checkpoint, tmp_path, bad
):
    module, _, _, data, _, _ = components(monkeypatch, checkpoint, tmp_path)
    episode = data["episodes"][0]
    sample = episode["samples"][0]
    if bad == "index":
        sample["teacher_action_index"] = True
    elif bad == "missing":
        episode["samples"].pop(0)
    elif bad == "actor":
        sample["observation"] = replace(sample["observation"], player_id=1)
    elif bad == "terminal":
        episode["truncated"] = True
    elif bad == "execution":
        sample["behavior_executed"] = False
    elif bad == "private":
        sample["reward"] = 1.0
    else:
        episode["teacher_calls"][0]["ordinal"] = 2
    with pytest.raises(ValueError):
        module.save_student_labels(tmp_path / "invalid.gz", data)


def test_real_teacher_rng_replays_all_calls_including_forced_and_gzip_is_strict(
    checkpoint, tmp_path
):
    module = labels()
    policy, identity = checkpoint
    data = module.collect_student_labels(
        policy,
        checkpoint_identity=identity,
        board_seeds=[37],
        max_decisions=5,
        deadline=time.monotonic() + 5,
    )
    for episode in data["episodes"]:
        teacher = HeuristicPolicy(episode["teacher_seed"])
        for sample in episode["samples"]:
            view = sample["observation"]
            assert view.legal_actions.index(teacher.act(view)) == sample["teacher_action_index"]
    path = tmp_path / "data.gz"
    digest = module.save_student_labels(path, data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert module.load_student_labels(path) == data
    payload = json.loads(gzip.decompress(path.read_bytes()))
    payload["episodes"][0]["samples"][0]["observation"]["observation"]["hand"][0] = float("nan")
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError):
        module.load_student_labels(path)


def test_private_observation_envelope_is_rejected_instead_of_silently_dropped(
    monkeypatch, checkpoint, tmp_path
):
    module, paths, _, _, _, _ = components(monkeypatch, checkpoint, tmp_path)
    payload = json.loads(gzip.decompress(paths[1].read_bytes()))
    payload["episodes"][0]["samples"][0]["observation"]["engine_state"] = {"opponent_hand": [4]}
    paths[1].write_bytes(gzip.compress(json.dumps(payload).encode()))
    with pytest.raises(ValueError):
        module.load_student_labels(paths[1])


def test_integer_coverage_cannot_be_replaced_by_equal_boolean(monkeypatch, checkpoint, tmp_path):
    module = labels()
    policy, identity = checkpoint
    scripted(monkeypatch, module)
    data = module.collect_student_labels(
        policy, checkpoint_identity=identity, board_seeds=[19], max_decisions=1
    )
    data["coverage"]["decisions"] = True
    with pytest.raises(ValueError):
        module.save_student_labels(tmp_path / "invalid.gz", data)


def test_only_forced_component_is_known_zero_supervision(monkeypatch, checkpoint, tmp_path):
    module = labels()
    policy, identity = checkpoint
    scripted(monkeypatch, module, forced=True)
    data = module.collect_student_labels(
        policy, checkpoint_identity=identity, board_seeds=[19], max_decisions=4
    )
    path = tmp_path / "forced.gz"
    module.save_student_labels(path, data)
    with pytest.raises(ValueError, match="zero"):
        module.write_corpus(
            tmp_path / "invalid.json", components=(str(path),), partitions=partitions(19)
        )


def test_each_episode_resets_memory_and_preserves_atomic_completed_progress(
    monkeypatch, checkpoint, tmp_path
):
    module = labels()
    policy, identity = checkpoint
    _, executed, calls, teachers, _ = scripted(monkeypatch, module)
    progress = tmp_path / "progress.gz"
    writes = []
    original_save = module.save_student_labels

    def record(path, data):
        digest = original_save(path, data)
        writes.append(module.load_student_labels(path))
        return digest

    monkeypatch.setattr(module, "save_student_labels", record)
    data = module.collect_student_labels(
        policy,
        checkpoint_identity=identity,
        board_seeds=[19, 20],
        max_decisions=9,
        progress_path=progress,
    )
    assert len(executed) == 6
    assert [calls[i][1] for i in (0, 3)] == [None, None]
    assert [t[0] for t in teachers] == [119, 119, 119, 120, 120, 120]
    assert [len(w["episodes"]) for w in writes] == [1, 2, 2]
    assert writes[-1] == data
    assert data["collection"]["stop_reason"] == "seeds_exhausted"
    assert not progress.with_name(progress.name + ".tmp").exists()


@pytest.mark.parametrize("alias", [False, True])
def test_corpus_output_cannot_overwrite_immutable_component(
    monkeypatch, checkpoint, tmp_path, alias
):
    module, paths, _, _, _, _ = components(monkeypatch, checkpoint, tmp_path)
    before = paths[0].read_bytes()
    output = paths[0]
    if alias:
        output = tmp_path / "corpus.json"
        output.with_name(output.name + ".tmp").symlink_to(paths[0])
    with pytest.raises(ValueError):
        module.write_corpus(
            output, components=tuple(map(str, paths)), partitions=partitions(19, 23)
        )
    assert paths[0].read_bytes() == before


def test_missing_component_write_fails_as_invalid_corpus(tmp_path):
    module = labels()
    with pytest.raises(ValueError):
        module.write_corpus(
            tmp_path / "bad.json",
            components=(str(tmp_path / "missing.gz"),),
            partitions=partitions(19),
        )


@pytest.mark.parametrize("writer", ["corpus", "student"])
def test_staging_hardlink_never_changes_component_bytes(monkeypatch, checkpoint, tmp_path, writer):
    module, paths, _, data, _, _ = components(monkeypatch, checkpoint, tmp_path)
    output = tmp_path / ("linked.json" if writer == "corpus" else "linked.gz")
    staging = output.with_name(output.name + ".tmp")
    original = paths[1 if writer == "corpus" else 0]
    before = original.read_bytes()
    os.link(original, staging)
    if writer == "corpus":
        digest = module.write_corpus(
            output, components=tuple(map(str, paths)), partitions=partitions(19, 23)
        )
    else:
        digest = module.save_student_labels(output, data)
    assert original.read_bytes() == before
    assert staging.read_bytes() == before
    assert original.samefile(staging)
    assert not original.samefile(output)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    if writer == "corpus":
        assert module.corpus_sequences(module.load_corpus(output))[1][1][1] == 0
    else:
        assert module.load_student_labels(output) == data
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize("changed", ["version_bool", "dirty_integer", "board_float"])
def test_corpus_metadata_types_must_match_components_despite_valid_digest(
    monkeypatch, checkpoint, tmp_path, changed
):
    module, paths, digests, _, _, _ = components(monkeypatch, checkpoint, tmp_path)
    manifest = tmp_path / "corpus.json"
    module.write_corpus(manifest, components=tuple(map(str, paths)), partitions=partitions(19, 23))
    corpus = json.loads(manifest.read_bytes())
    original_sources = [json.loads(gzip.decompress(path.read_bytes()))["source"] for path in paths]
    entry = corpus["components"][0]
    if changed == "version_bool":
        entry["version"] = True
    elif changed == "dirty_integer":
        entry["source"]["dirty"] = int(entry["source"]["dirty"])
    else:
        entry["actual_board_seeds"][0] = float(entry["actual_board_seeds"][0])
    corpus["component_order_sha256"] = hashlib.sha256(
        json.dumps(
            corpus["components"], allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    manifest.write_text(json.dumps(corpus))
    with pytest.raises(ValueError):
        module.load_corpus(manifest)
    with pytest.raises(ValueError):
        module.corpus_sequences(corpus)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == list(digests)
    assert [
        json.loads(gzip.decompress(path.read_bytes()))["source"] for path in paths
    ] == original_sources
