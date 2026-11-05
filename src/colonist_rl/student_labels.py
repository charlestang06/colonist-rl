"""Student-only DAgger histories and immutable, partitioned supervision corpora.

The engine driver owns rules and opponents. This module persists only immutable
learner observations, legal indices and primitive provenance, never actor memory.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .adapters.observation_io import observation_to_dict, view_from_dict
from .contracts import Observation
from .imitation import _deadline, _expired, _positive, _preserve_rng, _source, load_demonstrations
from .opponents import HeuristicPolicy
from .policy import KINDS, policy_architecture, policy_from_architecture
from .simulator import BACKEND_VERSION, FourPlayerGame
from .supervised_fit import EpisodeSequence
from .training import _MatchDriver, load_checkpoint

_FORMAT = "colonist-rl.student-labels"
_CORPUS = "colonist-rl.supervision-corpus"
_PARTITIONS = {"training", "validation_labels", "validation_games", "audit", "final_test"}
_RESERVED = {70000017, 70000019, *range(71001000, 71001008)}
_TEACHER = {
    "name": "HeuristicPolicy",
    "seed_rule": "board_seed + 100",
    "input": "immutable Observation",
    "labels": "legal action index",
    "call_order": "once before every student call, including forced decisions",
}
_SAMPLE_KEYS = {"observation", "teacher_action_index", "behavior_action_index", "behavior_executed"}
_EPISODE_KEYS = {
    "board_seed",
    "backend_seed",
    "seat",
    "teacher_seed",
    "sampling_seed",
    "student_sha256",
    "architecture_sha256",
    "complete",
    "terminated",
    "truncated",
    "incomplete_reason",
    "steps",
    "teacher_calls",
    "history_sha256",
    "samples",
}
_COLLECTION_KEYS = {
    "baseline",
    "max_decisions",
    "max_game_steps",
    "deadline",
    "seed",
    "deterministic",
    "seat_rule",
    "sampling_seed_rule",
    "actual_board_seeds",
    "status",
    "stop_reason",
    "started_unix_seconds",
    "elapsed_seconds",
}


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _fields(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid {name} fields")


def _primitive(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _primitive(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _primitive(item)
        return
    raise ValueError("metadata must contain finite JSON primitives only")


def _validate_source(source, *, known=False):
    _fields(source, {"revision", "dirty"}, "source provenance")
    if not known and source == {"revision": None, "dirty": None}:
        return
    if (
        not isinstance(source["revision"], str)
        or not source["revision"]
        or type(source["dirty"]) is not bool
    ):
        raise ValueError("invalid source provenance")


def _identity(identity):
    _fields(identity, {"path", "sha256", "architecture"}, "checkpoint identity")
    if (
        not isinstance(identity["path"], str)
        or not identity["path"]
        or not _sha(identity["sha256"])
    ):
        raise ValueError("invalid checkpoint identity")
    # Use the accepted architecture decoder, preserving the caller's RNG even
    # during load-time schema checks. Never silently drop a policy branch.
    with _preserve_rng():
        try:
            architecture = policy_architecture(policy_from_architecture(identity["architecture"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid checkpoint architecture") from error
    if architecture != identity["architecture"]:
        raise ValueError("invalid checkpoint architecture")
    _primitive(identity)


def _verified_student(policy, identity):
    _identity(identity)
    try:
        path = Path(identity["path"]).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError("checkpoint bytes do not match supplied digest")
        loaded, _ = load_checkpoint(path)
        if (
            policy_architecture(policy) != identity["architecture"]
            or policy_architecture(loaded) != identity["architecture"]
        ):
            raise ValueError("checkpoint architecture does not match student")
        actual, saved = policy.state_dict(), loaded.state_dict()
        if actual.keys() != saved.keys() or any(
            not torch.equal(value.detach().cpu(), saved[key]) for key, value in actual.items()
        ):
            raise ValueError("checkpoint parameters do not match supplied student")
    except (OSError, RuntimeError, KeyError) as error:
        raise ValueError("invalid student checkpoint") from error
    return deepcopy(policy).eval().requires_grad_(False), {**deepcopy(identity), "path": str(path)}


def _sample_json(sample):
    return {**sample, "observation": observation_to_dict(sample["observation"])}


def _history(samples):
    return _digest([_sample_json(s) for s in samples])


def _coverage(episodes):
    phases, teacher_kinds, behavior_kinds = Counter(), Counter(), Counter()
    decisions = non_forced = executed = disagreements = 0
    for episode in episodes:
        for sample in episode["samples"]:
            view = sample["observation"]
            decisions += 1
            non_forced += int(len(view.legal_actions) > 1)
            executed += int(sample["behavior_executed"])
            disagreements += int(sample["teacher_action_index"] != sample["behavior_action_index"])
            phases[view.context.kind] += 1
            teacher_kinds[view.legal_actions[sample["teacher_action_index"]].kind] += 1
            behavior_kinds[view.legal_actions[sample["behavior_action_index"]].kind] += 1
    return {
        "decisions": decisions,
        "non_forced_decisions": non_forced,
        "behavior_executed": executed,
        "disagreements": disagreements,
        "phase_counts": dict(phases),
        "teacher_action_kind_counts": dict(teacher_kinds),
        "behavior_action_kind_counts": dict(behavior_kinds),
    }


def collect_student_labels(
    policy,
    *,
    checkpoint_identity: dict,
    board_seeds,
    max_decisions,
    max_game_steps=20000,
    baseline="heuristic",
    seed=0,
    deterministic=False,
    deadline=None,
    progress_path=None,
) -> dict:
    """Freeze one verified checkpoint and label its full own-observation histories.

    Teacher-first ordering permits legality checks before behavior selection. If
    the teacher call exhausts the deadline, its final ordinal/version is retained
    but no student call is started. A returned student choice is recorded before
    the final deadline guard and may remain explicitly unexecuted.
    """
    _positive(max_decisions, "max_decisions")
    _positive(max_game_steps, "max_game_steps")
    _deadline(deadline)
    if baseline not in ("mixed", "heuristic", "random"):
        raise ValueError("unknown opponent baseline")
    if type(seed) is not int or type(deterministic) is not bool:
        raise ValueError("seed must be an integer and deterministic a boolean")
    if isinstance(board_seeds, (str, bytes, set, dict)):
        raise ValueError("board_seeds must be an ordered iterable of unique integers")  # noqa: TRY004
    seeds = iter(board_seeds)
    source = _source()
    _validate_source(source, known=True)
    start, started = time.monotonic(), time.time()
    data = {
        "format": _FORMAT,
        "version": 1,
        "backend_version": BACKEND_VERSION,
        "action_kinds": list(KINDS),
        "source": source,
        "teacher": dict(_TEACHER),
        "student": None,
        "collection": {
            "baseline": baseline,
            "max_decisions": max_decisions,
            "max_game_steps": max_game_steps,
            "deadline": deadline,
            "seed": seed,
            "deterministic": deterministic,
            "seat_rule": "board_index % 4",
            "sampling_seed_rule": "(seed + board_index) % 2**63",
            "actual_board_seeds": [],
            "status": "collecting",
            "stop_reason": "collecting",
            "started_unix_seconds": started,
            "elapsed_seconds": 0.0,
        },
        "coverage": {},
        "episodes": [],
    }
    collection, episodes = data["collection"], data["episodes"]
    reason, count, seen = "seeds_exhausted", 0, set()

    def persist():
        collection["elapsed_seconds"] = time.monotonic() - start
        data["coverage"] = _coverage(episodes)
        if _source() != source:
            raise RuntimeError("source revision/dirty status changed during student collection")
        if progress_path is not None:
            save_student_labels(progress_path, data)

    with _preserve_rng(), torch.no_grad():
        student, identity = _verified_student(policy, checkpoint_identity)
        data["student"] = {"checkpoint": identity, "behavior": "student_only", "mode": "eval"}
        while True:
            if _expired(deadline):
                reason = "deadline"
                break
            if count >= max_decisions:
                reason = "decision_cap"
                break
            try:
                board = next(seeds)
            except StopIteration:
                break
            if type(board) is not int or board in seen or board in _RESERVED:
                raise ValueError(
                    "board seeds must be unique training integers, excluding reserved finals"
                )
            seen.add(board)
            index, seat = len(episodes), len(episodes) % 4
            sampling_seed = (seed + index) % (2**63)
            random.seed(sampling_seed)
            np.random.seed(sampling_seed % (2**32))
            torch.manual_seed(sampling_seed)
            env = FourPlayerGame(max_steps=max_game_steps)
            if _expired(deadline):
                reason = "deadline"
                break
            env.reset(seed=board)
            episode = {
                "board_seed": board,
                "backend_seed": board + 1,
                "seat": seat,
                "teacher_seed": board + 100,
                "sampling_seed": sampling_seed,
                "student_sha256": identity["sha256"],
                "architecture_sha256": _digest(identity["architecture"]),
                "complete": False,
                "terminated": False,
                "truncated": False,
                "incomplete_reason": None,
                "steps": 0,
                "teacher_calls": [],
                "history_sha256": None,
                "samples": [],
            }
            episodes.append(episode)
            collection["actual_board_seeds"].append(board)
            driver = _MatchDriver(env, seat, board, baseline=baseline)
            teacher, memory, result = HeuristicPolicy(board + 100), None, None
            while True:
                if result is not None and result.terminated:
                    episode.update(complete=True, terminated=True)
                    break
                if (result is not None and result.truncated) or env.version >= max_game_steps:
                    episode.update(truncated=True, incomplete_reason="step_cap")
                    break
                if _expired(deadline):
                    episode["incomplete_reason"] = reason = "deadline"
                    break
                if count >= max_decisions:
                    episode["incomplete_reason"] = reason = "decision_cap"
                    break
                driver.maybe_offer(deadline=deadline)
                if _expired(deadline):
                    episode["incomplete_reason"] = reason = "deadline"
                    break
                if env.version >= max_game_steps:
                    episode.update(truncated=True, incomplete_reason="step_cap")
                    break
                view = driver.view
                if _expired(deadline):
                    episode["incomplete_reason"] = reason = "deadline"
                    break
                sample = None
                if view.player_id == seat:
                    if view.context.actor != seat:
                        raise ValueError("learner observation actor does not match student")
                    if _expired(deadline):
                        episode["incomplete_reason"] = reason = "deadline"
                        break
                    label = teacher.act(view)
                    try:
                        teacher_index = view.legal_actions.index(label)
                    except ValueError as error:
                        raise ValueError("teacher returned an illegal action") from error
                    episode["teacher_calls"].append(
                        {
                            "ordinal": len(episode["teacher_calls"]),
                            "version": view.context.version,
                            "action_index": teacher_index,
                        }
                    )
                    if _expired(deadline):
                        episode["incomplete_reason"] = reason = "deadline"
                        break
                    action, memory = student.act(view, memory, deterministic=deterministic)
                    try:
                        behavior_index = view.legal_actions.index(action)
                    except ValueError as error:
                        raise ValueError("student returned an illegal action") from error
                    sample = {
                        "observation": view,
                        "teacher_action_index": teacher_index,
                        "behavior_action_index": behavior_index,
                        "behavior_executed": False,
                    }
                    episode["samples"].append(sample)
                    count += 1
                    if memory is not None:
                        memory = memory.detach()
                else:
                    action = driver.opponent_action(deadline=deadline)
                if _expired(deadline):
                    episode["incomplete_reason"] = reason = "deadline"
                    break
                result = driver.step(action)
                if sample is not None:
                    sample["behavior_executed"] = True
            episode["steps"] = env.version
            episode["history_sha256"] = _history(episode["samples"])
            if reason in ("deadline", "decision_cap"):
                break
            persist()
        collection.update(
            status="incomplete" if reason in ("deadline", "decision_cap") else "complete",
            stop_reason=reason,
        )
        # Check checkpoint bytes again so a mutable source cannot silently change
        # its identity while a round is in progress.
        if hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest() != identity["sha256"]:
            raise RuntimeError("student checkpoint changed during collection")
        persist()
        _validate(data)
    return data


def _validate(data, *, export=False):
    _fields(
        data,
        {
            "format",
            "version",
            "backend_version",
            "action_kinds",
            "source",
            "teacher",
            "student",
            "collection",
            "coverage",
            "episodes",
        },
        "student-label",
    )
    if data["format"] != _FORMAT or type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("unsupported student-label schema")
    if data["backend_version"] != BACKEND_VERSION or data["action_kinds"] != list(KINDS):
        raise ValueError("incompatible engine/action schema")
    _validate_source(data["source"], known=True)
    if data["teacher"] != _TEACHER:
        raise ValueError("unsupported teacher provenance")
    student = data["student"]
    _fields(student, {"checkpoint", "behavior", "mode"}, "student")
    _identity(student["checkpoint"])
    if student["behavior"] != "student_only" or student["mode"] != "eval":
        raise ValueError("unsupported student behavior")
    collection = data["collection"]
    _fields(collection, _COLLECTION_KEYS, "collection")
    _positive(collection["max_decisions"], "max_decisions")
    _positive(collection["max_game_steps"], "max_game_steps")
    _deadline(collection["deadline"])
    reason = collection["stop_reason"]
    expected_status = {
        "collecting": "collecting",
        "seeds_exhausted": "complete",
        "decision_cap": "incomplete",
        "deadline": "incomplete",
    }.get(reason)
    if (
        collection["baseline"] not in ("mixed", "heuristic", "random")
        or type(collection["seed"]) is not int
        or type(collection["deterministic"]) is not bool
        or collection["seat_rule"] != "board_index % 4"
        or collection["sampling_seed_rule"] != "(seed + board_index) % 2**63"
        or expected_status is None
        or collection["status"] != expected_status
        or (reason == "deadline" and collection["deadline"] is None)
    ):
        raise ValueError("invalid collection settings/status")
    for key in ("started_unix_seconds", "elapsed_seconds"):
        value = collection[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid collection timing")
    if not isinstance(data["episodes"], list):
        raise ValueError("episodes must be an ordered list")  # noqa: TRY004
    seeds = []
    exported_episodes = [] if export else None
    for i, episode in enumerate(data["episodes"]):
        _fields(episode, _EPISODE_KEYS, "episode")
        board = episode["board_seed"]
        if (
            type(board) is not int
            or board in seeds
            or board in _RESERVED
            or type(episode["seat"]) is not int
            or episode["seat"] != i % 4
            or type(episode["backend_seed"]) is not int
            or episode["backend_seed"] != board + 1
            or type(episode["teacher_seed"]) is not int
            or episode["teacher_seed"] != board + 100
            or type(episode["sampling_seed"]) is not int
            or episode["sampling_seed"] != (collection["seed"] + i) % (2**63)
            or episode["student_sha256"] != student["checkpoint"]["sha256"]
            or episode["architecture_sha256"] != _digest(student["checkpoint"]["architecture"])
        ):
            raise ValueError("invalid episode student/seed/seat provenance")
        seeds.append(board)
        boundary = episode["incomplete_reason"]
        if (
            any(type(episode[k]) is not bool for k in ("complete", "terminated", "truncated"))
            or episode["complete"] != episode["terminated"]
            or (episode["terminated"] and episode["truncated"])
            or (episode["complete"] and boundary is not None)
            or (
                not episode["complete"] and boundary not in ("step_cap", "decision_cap", "deadline")
            )
            or episode["truncated"] != (boundary == "step_cap")
            or (
                boundary in ("decision_cap", "deadline")
                and (i != len(data["episodes"]) - 1 or reason != boundary)
            )
        ):
            raise ValueError("inconsistent episode boundary flags")
        steps = episode["steps"]
        if (
            type(steps) is not int
            or not 0 <= steps <= collection["max_game_steps"]
            or (boundary == "step_cap" and steps != collection["max_game_steps"])
            or (episode["terminated"] and steps == 0)
            or not isinstance(episode["samples"], list)
            or not isinstance(episode["teacher_calls"], list)
        ):
            raise ValueError("invalid episode steps/history")
        previous = -1
        serialized_samples = []
        for j, sample in enumerate(episode["samples"]):
            _fields(sample, _SAMPLE_KEYS, "sample")
            view = sample["observation"]
            if (
                not isinstance(view, Observation)
                or type(view.schema_version) is not int
                or view.schema_version != 1
                or type(view.player_id) is not int
                or type(view.context.actor) is not int
                or view.player_id != episode["seat"]
                or view.context.actor != view.player_id
                or type(view.context.version) is not int
                or view.context.version <= previous
                or not view.legal_actions
                or any(a.kind not in KINDS for a in view.legal_actions)
                or any(
                    type(sample[k]) is not int or not 0 <= sample[k] < len(view.legal_actions)
                    for k in ("teacher_action_index", "behavior_action_index")
                )
                or type(sample["behavior_executed"]) is not bool
            ):
                raise ValueError("invalid observation or legal teacher/behavior index")
            if sample["behavior_executed"]:
                if view.context.version >= steps:
                    raise ValueError("executed sample must precede endpoint")
            elif (
                boundary != "deadline"
                or j != len(episode["samples"]) - 1
                or view.context.version != steps
            ):
                raise ValueError("only final deadline sample may remain unexecuted")
            previous = view.context.version
            # The parser builds new records without mutating this detached DTO.
            # Reuse it for the strict roundtrip, history digest and optional save.
            serialized = _sample_json(sample)
            view_from_dict(serialized["observation"])
            serialized_samples.append(serialized)
        calls, samples = episode["teacher_calls"], episode["samples"]
        if len(calls) not in (len(samples), len(samples) + int(boundary == "deadline")):
            raise ValueError("teacher call order does not cover full student history")
        previous_call = -1
        for j, call in enumerate(calls):
            _fields(call, {"ordinal", "version", "action_index"}, "teacher call")
            if (
                type(call["ordinal"]) is not int
                or call["ordinal"] != j
                or type(call["version"]) is not int
                or call["version"] <= previous_call
                or type(call["action_index"]) is not int
                or call["action_index"] < 0
                or (
                    j < len(samples)
                    and (
                        call["version"] != samples[j]["observation"].context.version
                        or call["action_index"] != samples[j]["teacher_action_index"]
                    )
                )
                or (j == len(samples) and call["version"] != steps)
            ):
                raise ValueError("invalid teacher call ordinal/version/index")
            previous_call = call["version"]
        if len(calls) > len(samples) and samples and not samples[-1]["behavior_executed"]:
            raise ValueError("no teacher calls may follow an unexecuted student")
        if episode["history_sha256"] != _digest(serialized_samples):
            raise ValueError("student history digest does not match ordered samples")
        if export:
            exported_episodes.append({**episode, "samples": serialized_samples})
    if seeds != collection["actual_board_seeds"] or any(
        type(v) is not int for v in collection["actual_board_seeds"]
    ):
        raise ValueError("actual board schedule does not match episodes")
    coverage = _coverage(data["episodes"])
    _primitive(data["coverage"])
    if (
        _json(data["coverage"]) != _json(coverage)
        or coverage["decisions"] > collection["max_decisions"]
        or (reason == "decision_cap" and coverage["decisions"] != collection["max_decisions"])
    ):
        raise ValueError("inconsistent coverage/decision cap")
    _primitive({k: v for k, v in data.items() if k != "episodes"})
    _primitive([{k: v for k, v in e.items() if k != "samples"} for e in data["episodes"]])
    if export:
        return {**data, "episodes": exported_episodes}


def _atomic(path, encoded):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def save_student_labels(path, component: dict) -> str:
    """Atomically persist finite versioned gzip JSON; return exact byte digest."""
    payload = _validate(component, export=True)
    return _atomic(path, gzip.compress(_json(payload), mtime=0))


def load_student_labels(path) -> dict:
    """Load only the documented narrow schema and immutable own observations."""
    try:
        data = json.loads(gzip.decompress(Path(path).read_bytes()))
        _primitive(data)
        for episode in data["episodes"]:
            for sample in episode["samples"]:
                _fields(
                    sample["observation"],
                    {"format", "version", "observation"},
                    "observation export",
                )
                if type(sample["observation"]["version"]) is not int:
                    raise ValueError("invalid observation export version")
                sample["observation"] = view_from_dict(sample["observation"])
        _validate(data)
        return data
    except (KeyError, TypeError, AttributeError, OSError, EOFError, ValueError) as error:
        raise ValueError("invalid or incompatible student-label component") from error


def _partitions(partitions):
    _fields(partitions, _PARTITIONS, "partitions")
    used = set()
    for name, seeds in partitions.items():
        if (
            type(seeds) is not list
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or used.intersection(seeds)
            or (name != "final_test" and _RESERVED.intersection(seeds))
        ):
            raise ValueError(
                "partitions require disjoint explicit unique integer lists; finals are reserved"
            )
        used.update(seeds)


def _component(path):
    path = Path(path).resolve(strict=True)
    encoded = path.read_bytes()
    header = json.loads(gzip.decompress(encoded))
    for episode in header["episodes"]:
        for sample in episode["samples"]:
            _fields(
                sample["observation"], {"format", "version", "observation"}, "observation export"
            )
    if header.get("format") == _FORMAT:
        data = load_student_labels(path)
        if data["collection"]["status"] == "collecting":
            raise ValueError("active progress is not a finalized component")
    elif header.get("format") == "colonist-rl.demonstrations":
        data = load_demonstrations(path)
    else:
        raise ValueError("unsupported component format")
    # Ensure the loader validated the bytes whose digest we report.
    if path.read_bytes() != encoded:
        raise ValueError("component changed while loading")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "format": data["format"],
        "version": data["version"],
        "source": deepcopy(data["source"]),
        "actual_board_seeds": list(data["collection"]["actual_board_seeds"]),
    }, data


def _corpus_data(corpus):
    _fields(
        corpus,
        {"format", "version", "source", "partitions", "components", "component_order_sha256"},
        "corpus",
    )
    _primitive(corpus)
    if corpus["format"] != _CORPUS or type(corpus["version"]) is not int or corpus["version"] != 1:
        raise ValueError("unsupported corpus schema")
    _validate_source(corpus["source"], known=True)
    _partitions(corpus["partitions"])
    entries = corpus["components"]
    if type(entries) is not list or not entries:
        raise ValueError("corpus requires nonempty ordered immutable components")
    if corpus["component_order_sha256"] != _digest(entries):
        raise ValueError("component ordering/identity digest mismatch")
    digests, sources = set(), []
    for order, entry in enumerate(entries):
        _fields(
            entry,
            {"order", "path", "sha256", "format", "version", "source", "actual_board_seeds"},
            "corpus component",
        )
        if type(entry["order"]) is not int or entry["order"] != order or entry["sha256"] in digests:
            raise ValueError("duplicate components or inconsistent ordering")
        actual, data = _component(entry["path"])
        if _json({k: v for k, v in entry.items() if k != "order"}) != _json(actual):
            raise ValueError("component bytes/provenance changed")
        if not set(actual["actual_board_seeds"]).issubset(corpus["partitions"]["training"]):
            raise ValueError("component boards must be declared training boards")
        # Apply additional endpoint checks without weakening legacy validation.
        for episode in data["episodes"]:
            if episode["terminated"] and (episode["steps"] == 0 or episode["truncated"]):
                raise ValueError("contradictory terminal episode")
            for sample in episode["samples"]:
                view = sample["observation"]
                executed = sample.get("behavior_executed", True)
                if view.context.version > episode["steps"] or (
                    executed and view.context.version == episode["steps"]
                ):
                    raise ValueError("sample history exceeds executed endpoint")
        digests.add(entry["sha256"])
        sources.append(data)
    if not any(
        len(s["observation"].legal_actions) > 1
        for data in sources
        for episode in data["episodes"]
        for s in episode["samples"]
    ):
        raise ValueError("corpus has zero non-forced supervision")
    return sources


def write_corpus(path, *, components: tuple[str, ...], partitions: dict) -> str:
    """Reference original bytes and provenance; never merge into a fake v1 source."""
    if type(components) is not tuple:
        raise ValueError("components must be an ordered tuple of paths")
    try:
        entries = [{"order": i, **_component(p)[0]} for i, p in enumerate(components)]
        output = Path(path)
        output_paths = {
            str(output.resolve()),
            str(output.with_name(output.name + ".tmp").resolve()),
        }
        if output_paths.intersection(entry["path"] for entry in entries):
            raise ValueError("corpus output cannot overwrite an immutable component")
        corpus = {
            "format": _CORPUS,
            "version": 1,
            "source": _source(),
            "partitions": deepcopy(partitions),
            "components": entries,
            "component_order_sha256": _digest(entries),
        }
        _corpus_data(corpus)
    except (KeyError, TypeError, AttributeError, OSError, EOFError) as error:
        raise ValueError("invalid or missing corpus component") from error
    return _atomic(path, _json(corpus))


def load_corpus(path) -> dict:
    """Verify paths, exact bytes, order, partitions and source metadata on load."""
    try:
        corpus = json.loads(Path(path).read_bytes())
        _corpus_data(corpus)
        return corpus
    except (KeyError, TypeError, AttributeError, OSError, EOFError, ValueError) as error:
        raise ValueError("invalid or changed supervision corpus") from error


def corpus_sequences(corpus: dict) -> tuple[EpisodeSequence, ...]:
    """Project every full ordered history to shared-fitter supervision tuples."""
    try:
        sources = _corpus_data(corpus)
        return tuple(
            tuple(
                (
                    s["observation"],
                    s["teacher_action_index"] if data["format"] == _FORMAT else s["action_index"],
                )
                for s in episode["samples"]
            )
            for data in sources
            for episode in data["episodes"]
        )
    except (KeyError, TypeError, AttributeError, OSError, EOFError, ValueError) as error:
        raise ValueError("invalid or changed supervision corpus") from error
