"""Persisted observation-only demonstrations and current-weight recurrent imitation.

Datasets contain no recurrent tensors, rewards, hidden engine state or return labels.
Incomplete episodes remain useful supervised sequences, never terminal evidence.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .adapters.observation_io import observation_to_dict, view_from_dict
from .contracts import Observation
from .opponents import HeuristicPolicy
from .policy import KINDS
from .simulator import BACKEND_VERSION, FourPlayerGame
from .training import _MatchDriver

_FORMAT = "colonist-rl.demonstrations"
_KEYS = {
    "format",
    "version",
    "backend_version",
    "action_kinds",
    "teacher",
    "source",
    "collection",
    "coverage",
    "episodes",
}
_EPISODE_KEYS = {
    "board_seed",
    "backend_seed",
    "seat",
    "teacher_seed",
    "complete",
    "terminated",
    "truncated",
    "incomplete_reason",
    "steps",
    "samples",
}


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _deadline(value):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
    ):
        raise ValueError("deadline must be a finite absolute monotonic time")


def _expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


def _source():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _coverage(episodes):
    phases, kinds = Counter(), Counter()
    decisions = non_forced = 0
    for episode in episodes:
        for sample in episode["samples"]:
            view = sample["observation"]
            decisions += 1
            non_forced += int(len(view.legal_actions) > 1)
            phases[view.context.kind] += 1
            kinds[view.legal_actions[sample["action_index"]].kind] += 1
    return {
        "decisions": decisions,
        "non_forced_decisions": non_forced,
        "phase_counts": dict(phases),
        "action_kind_counts": dict(kinds),
    }


@contextmanager
def _preserve_rng():
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])


def collect_demonstrations(
    *, board_seeds, max_decisions, max_game_steps=20000, baseline="mixed", deadline=None
) -> dict:
    """Collect ordered learner sequences; iterable seeds are consumed lazily.

    Seat is board index modulo four. The backend receives board_seed + 1;
    the heuristic teacher receives board_seed + 100, and driver board_seed.
    ``deadline`` is absolute monotonic time, checked between bounded actor steps.
    """
    _positive(max_decisions, "max_decisions")
    _positive(max_game_steps, "max_game_steps")
    _deadline(deadline)
    if baseline not in ("mixed", "heuristic", "random"):
        raise ValueError("unknown opponent baseline")
    if isinstance(board_seeds, (str, bytes, set, dict)):
        raise ValueError("board_seeds must be an ordered iterable of unique integers")  # noqa: TRY004
    seeds = iter(board_seeds)
    source = _source()
    started = time.time()
    start = time.monotonic()
    episodes, actual, seen = [], [], set()
    count = 0
    reason = "seeds_exhausted"
    with _preserve_rng():
        while True:
            if _expired(deadline):
                reason = "deadline"
                break
            if count >= max_decisions:
                reason = "decision_cap"
                break
            try:
                board_seed = next(seeds)
            except StopIteration:
                break
            if type(board_seed) is not int or board_seed in seen:
                raise ValueError("board seeds must be unique integers")
            seen.add(board_seed)
            seat = len(episodes) % 4
            actual.append(board_seed)
            env = FourPlayerGame(max_steps=max_game_steps)
            env.reset(seed=board_seed)
            driver = _MatchDriver(env, seat, board_seed, baseline=baseline)
            teacher = HeuristicPolicy(board_seed + 100)
            episode = {
                "board_seed": board_seed,
                "backend_seed": board_seed + 1,
                "seat": seat,
                "teacher_seed": board_seed + 100,
                "complete": False,
                "terminated": False,
                "truncated": False,
                "incomplete_reason": None,
                "steps": 0,
                "samples": [],
            }
            episodes.append(episode)
            while True:
                if _expired(deadline):
                    episode["incomplete_reason"] = reason = "deadline"
                    break
                if env._terminated:
                    episode.update(complete=True, terminated=True)
                    break
                if env._truncated or env.version >= max_game_steps:
                    episode.update(truncated=True, incomplete_reason="step_cap")
                    break
                if count >= max_decisions:
                    episode["incomplete_reason"] = reason = "decision_cap"
                    break
                # Offers increment simulator version too. Stop before an ensuing
                # action would exceed the exact game-step cap.
                driver.maybe_offer(deadline=deadline)
                if env.version >= max_game_steps:
                    episode.update(truncated=True, incomplete_reason="step_cap")
                    break
                if _expired(deadline):
                    episode["incomplete_reason"] = reason = "deadline"
                    break
                view = driver.view
                if view.player_id == seat:
                    action = teacher.act(view)
                    index = view.legal_actions.index(action)
                    episode["samples"].append({"observation": view, "action_index": index})
                    count += 1
                else:
                    action = driver.opponent_action()
                if _expired(deadline):
                    episode["incomplete_reason"] = reason = "deadline"
                    break
                driver.step(action)
            episode["steps"] = env.version
            if reason in ("deadline", "decision_cap"):
                break
    if _source() != source:
        raise RuntimeError("source revision/dirty status changed during demonstration collection")
    dataset = {
        "format": _FORMAT,
        "version": 1,
        "backend_version": BACKEND_VERSION,
        "action_kinds": list(KINDS),
        "teacher": {
            "name": "HeuristicPolicy",
            "seed_rule": "board_seed + 100",
            "input": "immutable Observation",
            "labels": "legal action index",
        },
        "source": source,
        "collection": {
            "baseline": baseline,
            "max_decisions": max_decisions,
            "max_game_steps": max_game_steps,
            "deadline": deadline,
            "actual_board_seeds": actual,
            "stop_reason": reason,
            "started_unix_seconds": started,
            "elapsed_seconds": time.monotonic() - start,
        },
        "coverage": _coverage(episodes),
        "episodes": episodes,
    }
    _validate(dataset)
    return dataset


def _validate(dataset):
    """Validate the actor-data boundary before serialization or any policy call."""
    if not isinstance(dataset, dict) or set(dataset) != _KEYS:
        raise ValueError("invalid demonstration fields")
    if (
        dataset["format"] != _FORMAT
        or type(dataset["version"]) is not int
        or dataset["version"] != 1
    ):
        raise ValueError("unsupported demonstration schema")
    if dataset["backend_version"] != BACKEND_VERSION or dataset["action_kinds"] != list(KINDS):
        raise ValueError("demonstration engine/action schema is incompatible")
    source = dataset["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"revision", "dirty"}
        or not isinstance(source["revision"], str)
        or not source["revision"]
        or type(source["dirty"]) is not bool
    ):
        raise ValueError("invalid source provenance")
    teacher = dataset["teacher"]
    if teacher != {
        "name": "HeuristicPolicy",
        "seed_rule": "board_seed + 100",
        "input": "immutable Observation",
        "labels": "legal action index",
    }:
        raise ValueError("unsupported teacher provenance")
    collection = dataset["collection"]
    if not isinstance(collection, dict) or set(collection) != {
        "baseline",
        "max_decisions",
        "max_game_steps",
        "deadline",
        "actual_board_seeds",
        "stop_reason",
        "started_unix_seconds",
        "elapsed_seconds",
    }:
        raise ValueError("invalid collection metadata")
    _positive(collection["max_decisions"], "max_decisions")
    _positive(collection["max_game_steps"], "max_game_steps")
    _deadline(collection["deadline"])
    if collection["baseline"] not in ("mixed", "random", "heuristic") or collection[
        "stop_reason"
    ] not in ("seeds_exhausted", "decision_cap", "deadline"):
        raise ValueError("invalid collection settings")
    for key in ("started_unix_seconds", "elapsed_seconds"):
        value = collection[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid collection timing")
    episodes = dataset["episodes"]
    if not isinstance(episodes, list):
        raise ValueError("episodes must be an ordered list")  # noqa: TRY004
    seeds = []
    for i, episode in enumerate(episodes):
        if not isinstance(episode, dict) or set(episode) != _EPISODE_KEYS:
            raise ValueError("invalid episode fields")
        seed = episode["board_seed"]
        if (
            type(seed) is not int
            or seed in seeds
            or type(episode["seat"]) is not int
            or episode["seat"] != i % 4
            or episode["backend_seed"] != seed + 1
            or episode["teacher_seed"] != seed + 100
        ):
            raise ValueError("invalid episode seed/seat provenance")
        seeds.append(seed)
        if any(type(episode[k]) is not bool for k in ("complete", "terminated", "truncated")):
            raise ValueError("invalid episode boundary flags")
        reason = episode["incomplete_reason"]
        if (
            episode["complete"] != episode["terminated"]
            or (episode["terminated"] and episode["truncated"])
            or (episode["complete"] and reason is not None)
            or (not episode["complete"] and reason not in ("step_cap", "decision_cap", "deadline"))
            or episode["truncated"] != (reason == "step_cap")
        ):
            raise ValueError("inconsistent episode completeness")
        if (
            type(episode["steps"]) is not int
            or not 0 <= episode["steps"] <= collection["max_game_steps"]
            or not isinstance(episode["samples"], list)
        ):
            raise ValueError("invalid episode steps/samples")
        previous = -1
        for sample in episode["samples"]:
            if not isinstance(sample, dict) or set(sample) != {"observation", "action_index"}:
                raise ValueError("samples must contain only observation and action_index")
            view, index = sample["observation"], sample["action_index"]
            if (
                not isinstance(view, Observation)
                or view.player_id != episode["seat"]
                or view.context.actor != view.player_id
                or view.schema_version != 1
                or view.context.version <= previous
                or type(index) is not int
                or not 0 <= index < len(view.legal_actions)
                or any(a.kind not in KINDS for a in view.legal_actions)
            ):
                raise ValueError("invalid observation or legal teacher index")
            previous = view.context.version
    if seeds != collection["actual_board_seeds"]:
        raise ValueError("actual board seed provenance does not match episodes")
    coverage = _coverage(episodes)
    if dataset["coverage"] != coverage or coverage["decisions"] > collection["max_decisions"]:
        raise ValueError("inconsistent demonstration coverage or decision cap")
    # Reject NaN/Infinity or arbitrary objects anywhere in primitive metadata.
    json.dumps({k: v for k, v in dataset.items() if k != "episodes"}, allow_nan=False)


def save_demonstrations(path, dataset) -> str:
    """Write validated gzip JSON atomically; return SHA-256 of persisted bytes."""
    _validate(dataset)
    payload = {
        **dataset,
        "episodes": [
            {
                **episode,
                "samples": [
                    {
                        "observation": observation_to_dict(s["observation"]),
                        "action_index": s["action_index"],
                    }
                    for s in episode["samples"]
                ],
            }
            for episode in dataset["episodes"]
        ],
    }
    encoded = gzip.compress(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(),
        mtime=0,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def load_demonstrations(path) -> dict:
    """Load and validate a dataset, rebuilding immutable observation tuples."""
    try:
        dataset = json.loads(gzip.decompress(Path(path).read_bytes()))
        json.dumps(dataset, allow_nan=False)  # Reject nonstandard NaN/Infinity before decoding.
        for episode in dataset["episodes"]:
            for sample in episode["samples"]:
                sample["observation"] = view_from_dict(sample["observation"])
        _validate(dataset)
        return dataset
    except (KeyError, TypeError, AttributeError, OSError, EOFError, ValueError) as error:
        raise ValueError("invalid or incompatible demonstration dataset") from error


def _empty_score():
    return {"count": 0, "correct": 0, "accuracy": None, "cross_entropy": None}


def imitation_metrics(policy, dataset) -> dict:
    """Replay complete ordered sequences without updating mode, weights or RNG."""
    _validate(dataset)
    modes = [(module, module.training) for module in policy.modules()]
    totals = {"all": [], "non_forced": []}
    phases, kinds = {}, {}
    try:
        with _preserve_rng(), torch.no_grad():
            policy.eval()
            for episode in dataset["episodes"]:
                memory = None
                for sample in episode["samples"]:
                    view, target = sample["observation"], sample["action_index"]
                    output = policy(view, memory)
                    memory = output.memory.detach()
                    ce = F.cross_entropy(
                        output.logits.unsqueeze(0),
                        output.logits.new_tensor([target], dtype=torch.long),
                    ).item()
                    if not math.isfinite(ce):
                        raise FloatingPointError("imitation validation CE is not finite")
                    score = (int(output.logits.argmax().item() == target), ce)
                    totals["all"].append(score)
                    phase, kind = view.context.kind, view.legal_actions[target].kind
                    phases.setdefault(phase, {"all": [], "non_forced": []})["all"].append(score)
                    kinds.setdefault(kind, {"all": [], "non_forced": []})["all"].append(score)
                    if len(view.legal_actions) > 1:
                        totals["non_forced"].append(score)
                        phases[phase]["non_forced"].append(score)
                        kinds[kind]["non_forced"].append(score)
    finally:
        for module, training in modes:
            module.training = training

    def summarize(rows):
        if not rows:
            return _empty_score()
        correct = sum(r[0] for r in rows)
        return {
            "count": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "cross_entropy": sum(r[1] for r in rows) / len(rows),
        }

    return {
        **{name: summarize(rows) for name, rows in totals.items()},
        "by_phase": {
            key: {n: summarize(r) for n, r in value.items()} for key, value in phases.items()
        },
        "by_action_kind": {
            key: {n: summarize(r) for n, r in value.items()} for key, value in kinds.items()
        },
        "coverage": dataset["coverage"],
        "note": "Teacher agreement is supervised fit, not playing competence; forced choices are excluded from non_forced metrics.",
    }


def fit_imitation(
    policy, dataset, *, epochs=2, sequence_length=16, learning_rate=3e-4, deadline=None
) -> dict:
    """Ordered truncated BPTT with fresh no-grad prefix replay before every update."""
    _validate(dataset)
    _positive(epochs, "epochs")
    _positive(sequence_length, "sequence_length")
    _deadline(deadline)
    if (
        type(learning_rate) not in (float, int)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    modes = [(module, module.training) for module in policy.modules()]
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    updates = presentations = completed_epochs = 0
    total = 0.0
    expired = False
    try:
        policy.train()
        for _ in range(epochs):
            for episode in dataset["episodes"]:
                samples = episode["samples"]
                for start in range(0, len(samples), sequence_length):
                    if _expired(deadline):
                        expired = True
                        break
                    chunk = samples[start : start + sequence_length]
                    if not any(len(s["observation"].legal_actions) > 1 for s in chunk):
                        continue
                    # Never reuse hidden states produced before an optimizer step.
                    memory = None
                    with torch.no_grad():
                        for sample in samples[:start]:
                            if _expired(deadline):
                                expired = True
                                break
                            memory = policy(sample["observation"], memory).memory.detach()
                    if expired:
                        break
                    memory = memory.detach() if memory is not None else None
                    losses = []
                    for sample in chunk:
                        if _expired(deadline):
                            expired = True
                            break
                        view, target = sample["observation"], sample["action_index"]
                        output = policy(view, memory)
                        memory = output.memory
                        if len(view.legal_actions) > 1:
                            losses.append(
                                F.cross_entropy(
                                    output.logits.unsqueeze(0),
                                    output.logits.new_tensor([target], dtype=torch.long),
                                )
                            )
                    if expired or _expired(deadline):
                        expired = True
                        break  # A partial chunk never updates parameters.
                    loss = torch.stack(losses).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("imitation training loss is not finite")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), 0.5, error_if_nonfinite=True
                    )
                    if _expired(deadline):
                        expired = True
                        optimizer.zero_grad(set_to_none=True)
                        break
                    optimizer.step()
                    total += loss.item() * len(losses)
                    presentations += len(losses)
                    updates += 1
                if expired:
                    break
            if expired:
                break
            completed_epochs += 1
    finally:
        for module, training in modes:
            module.training = training
    return {
        "loss": total / presentations if presentations else None,
        "decisions": dataset["coverage"]["decisions"],
        "epochs": completed_epochs,
        "requested_epochs": epochs,
        "updates": updates,
        "supervised_decisions": dataset["coverage"]["non_forced_decisions"],
        "supervised_presentations": presentations,
        "status": "incomplete" if expired else "complete",
        "incomplete_reason": "deadline" if expired else None,
    }
