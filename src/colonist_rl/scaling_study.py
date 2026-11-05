"""One-component, immutable nested-corpus scaling orchestration.

No embeddings, recurrent memories or corpus-sized observation lists are retained.
See docs/scaling-study.md for the strict version-one launch dictionaries.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import resource
import sys
import tempfile
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .adapters.observation_io import observation_to_dict
from .contracts import Observation
from .engine_identity import BACKEND_VERSION, verify_engine_installation
from .evaluation import evaluate_blocks, paired_block_summary
from .imitation import (
    _preserve_rng,
    _source,
    collect_demonstrations,
    load_demonstrations,
    save_demonstrations,
)
from .policy import PHASES, policy_architecture
from .supervised_fit import fit_sequences, score_sequences
from .training import load_checkpoint, save_checkpoint

ARMS = ("10000/11", "10000/29", "100000/11", "100000/29")
ENDPOINTS = (10000, 100000)
MILESTONES = (10000, 25000, 50000, 100000)
EVALUATION_BOARDS = 16


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _keys(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"invalid {name} fields")


def _hex(value, length=64):
    if (
        type(value) is not str
        or len(value) != length
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("malformed hash")


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _typed_equal(a, b):
    return type(a) is type(b) and _json(a) == _json(b)


def _ref(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _check_ref(ref):
    _keys(ref, ("path", "sha256"), "artifact reference")
    _hex(ref["sha256"])
    if type(ref["path"]) is not str or not Path(ref["path"]).is_absolute():
        raise ValueError("artifact path must be absolute")
    if _ref(ref["path"]) != ref:
        raise ValueError("artifact digest/path changed")
    return Path(ref["path"])


def _read(ref):
    return json.loads(_check_ref(ref).read_bytes())


def _publish(path, writer):
    """Unique private fixed.tmp namespace, then exclusive same-filesystem link.

    A pre-existing destination (including an alias) is never opened for writing.
    fsync both file and directory before a subsequent cursor can refer to it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scaling-stage-", dir=path.parent) as stage:
        temporary = Path(stage) / "artifact"
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return _ref(path)


def _publish_json(path, value):
    encoded = _json(value)
    return _publish(path, lambda temporary: temporary.write_bytes(encoded))


def _empty_fit():
    return {
        "updates": 0,
        "supervised_presentations": 0,
        "loss_sum": 0.0,
        "weighted_presentation_mass": 0.0,
        "profile": dict.fromkeys(
            ("prefix_forwards", "chunk_forwards", "batches", "batch_chunks"), 0
        ),
        "scope_seconds": dict.fromkeys(("encoding", "prefix", "chunk", "backward", "step"), 0.0),
        "fit_seconds": 0.0,
    }


def _fit_component(policy, episodes, optimizer, weights, deadline):
    return fit_sequences(
        policy,
        episodes,
        epochs=1,
        sequence_length=16,
        batch_size=1,
        learning_rate=0.0003,
        optimizer=optimizer,
        phase_weights=weights,
        deadline=deadline,
        profile=True,
    )


def _add_fit(totals, row, weights, episodes):
    for key in ("updates", "supervised_presentations"):
        totals[key] += row[key]
    totals["loss_sum"] += (row["loss"] or 0.0) * row["supervised_presentations"]
    for key in totals["profile"]:
        totals["profile"][key] += row["profile"][key]
    for key in totals["scope_seconds"]:
        totals["scope_seconds"][key] += row["profile"]["scope_seconds"][key]
    totals["fit_seconds"] += row["profile"]["wall_seconds"]
    # Complete components have known exact phase denominators. Partial calls do
    # not expose phase-level presentation counts in the existing fitter API.
    if row["status"] == "complete":
        totals["weighted_presentation_mass"] += sum(
            weights[v.context.kind] for e in episodes for v, _ in e if len(v.legal_actions) > 1
        )


def _select_milestone(rows):
    if type(rows) is not dict or set(rows) != set(ARMS):
        raise ValueError("exact four-arm matrix required")
    common = [
        epoch
        for epoch in (1, 3, 5)
        if all(
            row["milestones"].get(str(epoch), {}).get("complete") is True for row in rows.values()
        )
    ]
    return {"epoch": max(common) if common else None, "primary_five_complete": 5 in common}


def _seed_list(value, name, *, empty=False):
    if (
        type(value) is not list
        or (not value and not empty)
        or any(type(x) is not int for x in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"invalid {name} boards")


def _expected_metadata(seed, start):
    return {
        "migration": {
            "kind": "weights_only_cross_engine",
            "source_sha256": start["parent"]["sha256"],
            "source_backend_version": "3.2.1",
            "target_backend_version": BACKEND_VERSION,
            "source_steps": 0,
            "seed": seed,
        },
        "feature_upgrade": {
            "kind": "zero_additive_consequence",
            "parent_sha256": start["bridge"]["sha256"],
            "feature_set": "hand_recipe",
        },
        "seed": seed,
        "purpose": "untrained_initialization",
    }


def _check_start(start, seed):
    _keys(start, ("checkpoint", "parent", "bridge", "metadata"), "initial input")
    for key in ("checkpoint", "parent", "bridge"):
        _check_ref(start[key])
    if not _typed_equal(start["metadata"], _expected_metadata(seed, start)):
        raise ValueError("initial checkpoint lineage metadata differs")
    policy, state = load_checkpoint(start["checkpoint"]["path"])
    architecture = policy_architecture(policy)
    if (
        architecture.get("kind") != "graph_consequence_v1"
        or architecture.get("hidden_size") != 32
        or architecture.get("feature_set") != "hand_recipe"
        or type(state["steps"]) is not int
        or state["steps"] != 0
        or state["optimizer_state"] is not None
        or not _typed_equal(state["metadata"], start["metadata"])
        or any(not torch.isfinite(v).all() for v in policy.state_dict().values())
        or torch.count_nonzero(policy.consequence.weight).item() != 0
    ):
        raise ValueError(
            "initial checkpoint must be finite zero-additive untrained hidden32 hand_recipe"
        )
    return policy


def _protocol(protocol, stage):
    _keys(
        protocol,
        (
            "format",
            "version",
            "source",
            "runtime",
            "partitions",
            "prior_reservations",
            "inputs",
            "collection",
            "training",
            "evaluation",
            "limits",
        ),
        "protocol",
    )
    if (
        protocol["format"] != "colonist-rl.scaling-protocol"
        or type(protocol["version"]) is not int
        or protocol["version"] != 1
    ):
        raise ValueError("unsupported protocol")
    _keys(protocol["source"], ("revision", "dirty"), "source")
    _hex(protocol["source"]["revision"], 40)
    if protocol["source"]["dirty"] is not False or _source() != protocol["source"]:
        raise ValueError("known clean imported source required")
    runtime = protocol["runtime"]
    _keys(runtime, ("engine", "python", "torch", "threads"), "runtime")
    expected = {
        "engine": verify_engine_installation(),
        "python": sys.executable,
        "torch": str(torch.__version__),
        "threads": 1,
    }
    if not _typed_equal(runtime, expected):
        raise ValueError("runtime identity changed")
    partitions = protocol["partitions"]
    _keys(
        partitions,
        ("training", "validation_labels", "validation_games", "audit", "final_test"),
        "partitions",
    )
    seen = set()
    for name, seeds in partitions.items():
        _seed_list(seeds, name)
        if seen.intersection(seeds):
            raise ValueError("overlapping partitions")
        seen.update(seeds)
        if name in ("training", "validation_labels") and len(seeds) % 4:
            raise ValueError("component partitions require four-board batches")
    if len(partitions["validation_games"]) != EVALUATION_BOARDS:
        raise ValueError("evaluation requires exactly sixteen boards")
    prior = protocol["prior_reservations"]
    _keys(prior, ("artifacts", "excluded_boards"), "prior reservations")
    _seed_list(prior["excluded_boards"], "prior exclusions", empty=True)
    if type(prior["artifacts"]) is not list:
        raise ValueError("prior artifacts must be a list")
    for ref in prior["artifacts"]:
        value = _read(ref)
        _keys(value, ("reserved_boards",), "prior reservation artifact")
        _seed_list(value["reserved_boards"], "prior reservation artifact", empty=True)
        if not set(value["reserved_boards"]).issubset(prior["excluded_boards"]):
            raise ValueError("prior reservations omitted from exclusions")
    if seen.intersection(prior["excluded_boards"]):
        raise ValueError("partition uses previously reserved boards")
    collection = {
        "milestones": list(MILESTONES),
        "max_decisions": 80004,
        "max_game_steps": 20000,
        "batch_seconds": 120,
    }
    if not _typed_equal(protocol["collection"], collection):
        raise ValueError("invalid collection settings")
    training = protocol["training"]
    _keys(
        training,
        (
            "epochs",
            "sequence_length",
            "batch_size",
            "learning_rate",
            "hidden_size",
            "feature_set",
            "score_epochs",
            "required_phases",
        ),
        "training",
    )
    fixed = {
        "epochs": 5,
        "sequence_length": 16,
        "batch_size": 1,
        "learning_rate": 0.0003,
        "hidden_size": 32,
        "feature_set": "hand_recipe",
        "score_epochs": [0, 1, 3, 5],
    }
    if not _typed_equal({k: training[k] for k in fixed}, fixed):
        raise ValueError("invalid training settings")
    phases = training["required_phases"]
    if (
        type(phases) is not list
        or not phases
        or any(type(p) is not str or p not in PHASES for p in phases)
        or len(set(phases)) != len(phases)
    ):
        raise ValueError("invalid required supervised phases")
    if not _typed_equal(
        protocol["evaluation"],
        {"max_game_steps": 20000, "deterministic": True, "baseline": "heuristic"},
    ):
        raise ValueError("invalid evaluation settings")
    limits = protocol["limits"]
    _keys(
        limits,
        (
            "global_deadline",
            "collection_deadline",
            "fit_deadlines",
            "evaluation_deadline",
            "rss_bytes",
        ),
        "limits",
    )
    _keys(limits["fit_deadlines"], ARMS, "fit deadline matrix")
    for value in [
        limits[k] for k in ("global_deadline", "collection_deadline", "evaluation_deadline")
    ] + list(limits["fit_deadlines"].values()):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("deadlines must be finite absolute monotonic times")
    _positive(limits["rss_bytes"], "rss_bytes")
    if limits["rss_bytes"] > 2 * 1024**3:
        raise ValueError("RSS ceiling exceeds two GiB")
    inputs = protocol["inputs"]
    _keys(inputs, ("starts", "corpus", "results", "resume"), "inputs")
    _keys(inputs["starts"], ("11", "29"), "initial seed matrix")
    for seed, start in inputs["starts"].items():
        policy = _check_start(start, int(seed))
        del policy
    if inputs["corpus"] is not None:
        _check_ref(inputs["corpus"])
    if inputs["results"] is not None:
        _keys(inputs["results"], ARMS, "result matrix")
        for ref in inputs["results"].values():
            _check_ref(ref)
    if inputs["resume"] is not None:
        _check_ref(inputs["resume"])
    if stage != "collection" and inputs["corpus"] is None:
        raise ValueError("corpus path/SHA expectation required")
    if stage == "fit" and inputs["results"] is not None:
        raise ValueError("fit requires unbound result inputs")
    if stage != "fit" and inputs["resume"] is not None:
        raise ValueError("resume is only valid for fit")
    if stage == "evaluation" and inputs["results"] is None:
        raise ValueError("arm result path/SHA expectations required")
    _json(protocol)


def _identity(protocol):
    # Later stages bind newly published artifacts without changing original
    # collection/training caps, starts, partition reservations or settings.
    return _digest(
        {
            **protocol,
            "inputs": {**protocol["inputs"], "corpus": None, "results": None, "resume": None},
        }
    )


def _rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _deadline(protocol, stage):
    limits = protocol["limits"]
    end = limits["fit_deadlines"][stage] if stage in ARMS else limits[stage + "_deadline"]
    return min(end, limits["global_deadline"])


def _guard(protocol, deadline):
    if _source() != protocol["source"] or not _typed_equal(
        verify_engine_installation(), protocol["runtime"]["engine"]
    ):
        raise RuntimeError("source/runtime drift")
    if _rss() > protocol["limits"]["rss_bytes"]:
        raise MemoryError("RSS ceiling reached")
    if time.monotonic() >= deadline:
        raise TimeoutError("deadline")


@contextmanager
def _execution():
    threads = torch.get_num_threads()
    try:
        with _preserve_rng():
            torch.set_num_threads(1)
            yield
    finally:
        torch.set_num_threads(threads)


def _output(output_dir, protocol):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _publish_json(output / "protocol.json", protocol)
    return output


def _episode_payload(data):
    return [
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
        for episode in data["episodes"]
    ]


def _replay_component(data, protocol, deadline):
    """Compare every learner view and teacher call without a second dataset."""
    from .opponents import HeuristicPolicy
    from .simulator import FourPlayerGame
    from .training import _MatchDriver

    with _preserve_rng():
        for episode in data["episodes"]:
            board, seat = episode["board_seed"], episode["seat"]
            env = FourPlayerGame(max_steps=20000)
            env.reset(seed=board)
            driver = _MatchDriver(env, seat, board, baseline="heuristic")
            teacher = HeuristicPolicy(board + 100)
            index = 0
            while not env._terminated:
                if _rss() > protocol["limits"]["rss_bytes"]:
                    raise MemoryError("RSS ceiling reached")
                if time.monotonic() >= deadline:
                    raise TimeoutError("deadline during teacher replay")
                if env._truncated or env.version >= 20000:
                    raise ValueError("replay did not terminate naturally")
                driver.maybe_offer(deadline=deadline)
                view = driver.view
                if view.player_id == seat:
                    action = teacher.act(view)  # Forced choices also advance teacher call order.
                    if index >= len(episode["samples"]):
                        raise ValueError("missing own observation history")
                    sample = episode["samples"][index]
                    if (
                        _json(observation_to_dict(view))
                        != _json(observation_to_dict(sample["observation"]))
                        or view.legal_actions.index(action) != sample["action_index"]
                    ):
                        raise ValueError("teacher/menu/history replay mismatch")
                    index += 1
                else:
                    action = driver.opponent_action()
                driver.step(action)
            if index != len(episode["samples"]) or env.version != episode["steps"]:
                raise ValueError("terminal history mismatch")


def _load_component(ref, boards, protocol, deadline=None):
    deadline = protocol["limits"]["global_deadline"] if deadline is None else deadline
    _guard(protocol, deadline)
    path = _check_ref(ref)
    # The general decoder accepts envelope extensions/defaulted fields. Scaling
    # requires original full observations, so reject ignored hidden additions
    # before decoding. Release the raw representation before the official load.
    try:
        with gzip.open(path, "rt") as stream:
            raw = json.load(stream)
        for episode in raw["episodes"]:
            for sample in episode["samples"]:
                view = sample["observation"]
                _keys(view, ("format", "version", "observation"), "observation envelope")
                _keys(
                    view["observation"], (f.name for f in fields(Observation)), "full observation"
                )
        del raw, episode, sample, view
    except (KeyError, TypeError, UnboundLocalError) as error:
        raise ValueError("invalid full observation history") from error
    _guard(protocol, deadline)
    data = load_demonstrations(path)
    if not _typed_equal(data["source"], protocol["source"]):
        raise ValueError("component source differs")
    settings = data["collection"]
    for key in ("baseline", "max_decisions", "max_game_steps"):
        expected = "heuristic" if key == "baseline" else protocol["collection"][key]
        if not _typed_equal(settings[key], expected):
            raise ValueError("component collection settings differ")
    if not _typed_equal(settings["actual_board_seeds"], boards) or len(data["episodes"]) != 4:
        raise ValueError("component board order differs")
    for seat, episode in enumerate(data["episodes"]):
        for key in ("board_seed", "backend_seed", "teacher_seed", "seat", "steps"):
            if type(episode[key]) is not int:
                raise ValueError("component integer metadata type differs")
        if (
            episode["seat"] != seat
            or episode["complete"] is not True
            or episode["terminated"] is not True
            or episode["truncated"] is not False
        ):
            raise ValueError("component is unfinished")
        for sample in episode["samples"]:
            if type(sample["observation"].context.version) is not int:
                raise ValueError("observation version must be integer")
    # Exact primitive count typing, avoiding bool/int equality slippage.
    summary = _summary(data)
    if not _typed_equal(data["coverage"], summary["coverage"]):
        raise ValueError("component count metadata type differs")
    _replay_component(data, protocol, deadline)
    _check_ref(ref)
    return data


def _sequences(data):
    return tuple(
        tuple((s["observation"], s["action_index"]) for s in e["samples"]) for e in data["episodes"]
    )


def _summary(data):
    phase, kind, supervised = Counter(), Counter(), Counter()
    lengths = []
    for episode in data["episodes"]:
        lengths.append(len(episode["samples"]))
        for sample in episode["samples"]:
            view = sample["observation"]
            phase[view.context.kind] += 1
            kind[view.legal_actions[sample["action_index"]].kind] += 1
            if len(view.legal_actions) > 1:
                supervised[view.context.kind] += 1
    return {
        "coverage": {
            "decisions": sum(lengths),
            "non_forced_decisions": sum(supervised.values()),
            "phase_counts": dict(phase),
            "action_kind_counts": dict(kind),
        },
        "supervised_phases": dict(supervised),
        "episode_lengths": lengths,
        "prefix_forwards_per_epoch": sum(
            start
            for e in data["episodes"]
            for start in range(0, len(e["samples"]), 16)
            if any(
                len(s["observation"].legal_actions) > 1 for s in e["samples"][start : start + 16]
            )
        ),
        "updates_per_epoch": sum(
            1
            for e in data["episodes"]
            for start in range(0, len(e["samples"]), 16)
            if any(
                len(s["observation"].legal_actions) > 1 for s in e["samples"][start : start + 16]
            )
        ),
    }


def collect_scaling_corpus(output_dir, *, protocol: dict) -> dict:
    """Collect whole four-board batches; incomplete raw evidence stops the prefix."""
    with _execution():
        _protocol(protocol, "collection")
        if protocol["inputs"]["corpus"] is not None or protocol["inputs"]["results"] is not None:
            raise ValueError("collection requires unbound corpus/results")
        deadline = _deadline(protocol, "collection")
        _guard(protocol, deadline)
        output = _output(output_dir, protocol)
        result = {
            "format": "colonist-rl.scaling-corpus",
            "version": 1,
            "protocol_identity": _identity(protocol),
            "protocol": protocol,
            "training": [],
            "validation": [],
            "raw_components": [],
            "endpoints": {},
            "training_non_forced": 0,
            "phase_weights": None,
            "first_gate": None,
            "status": "partial",
            "stop_reason": None,
        }
        phases = Counter()
        started = time.perf_counter()
        try:
            for split, destination in (
                ("training", "training"),
                ("validation_labels", "validation"),
            ):
                boards = protocol["partitions"][split]
                for offset in range(0, len(boards), 4):
                    _guard(protocol, deadline)
                    batch = boards[offset : offset + 4]
                    data = collect_demonstrations(
                        board_seeds=batch,
                        max_decisions=80004,
                        max_game_steps=20000,
                        baseline="heuristic",
                        deadline=min(deadline, time.monotonic() + 120),
                    )
                    ref = _publish(
                        output / f"{split}-{offset // 4:05d}.json.gz",
                        lambda p, data=data: save_demonstrations(p, data),
                    )
                    complete = len(data["episodes"]) == 4 and all(
                        e["complete"] is True
                        and e["terminated"] is True
                        and e["truncated"] is False
                        for e in data["episodes"]
                    )
                    summary = _summary(data)
                    row = {
                        "artifact": ref,
                        "boards": batch,
                        "summary": summary,
                        "eligible": complete,
                        "split": split,
                    }
                    result["raw_components"].append(row)
                    del data
                    if not complete:
                        result["stop_reason"] = "incomplete_component"
                        _publish_json(
                            output / f"progress-{len(result['raw_components']):05d}.json", result
                        )
                        raise TimeoutError("incomplete component; no replacement")
                    begin = time.perf_counter()
                    data = _load_component(ref, batch, protocol, deadline)
                    del data
                    row["load_replay_seconds"] = time.perf_counter() - begin
                    result[destination].append(row)
                    if split == "training":
                        phases.update(summary["supervised_phases"])
                        result["training_non_forced"] += summary["coverage"]["non_forced_decisions"]
                        for threshold in MILESTONES:
                            if result["training_non_forced"] >= threshold:
                                result["endpoints"].setdefault(
                                    str(threshold), len(result["training"])
                                )
                        if (
                            result["first_gate"] is None
                            and str(ENDPOINTS[0]) in result["endpoints"]
                        ):
                            lengths = sorted(
                                n
                                for r in result["training"]
                                for n in r["summary"]["episode_lengths"]
                            )
                            result["first_gate"] = {
                                "non_forced": result["training_non_forced"],
                                "components": len(result["training"]),
                                "compressed_bytes": sum(
                                    Path(r["artifact"]["path"]).stat().st_size
                                    for r in result["training"]
                                ),
                                "elapsed_seconds": time.perf_counter() - started,
                                "load_replay_seconds": sum(
                                    r["load_replay_seconds"] for r in result["training"]
                                ),
                                "history_quantiles": {
                                    str(q): lengths[int(q * (len(lengths) - 1))]
                                    for q in (0, 0.5, 0.9, 1)
                                },
                                "prefix_forwards_per_epoch": sum(
                                    r["summary"]["prefix_forwards_per_epoch"]
                                    for r in result["training"]
                                ),
                                "peak_rss_bytes": _rss(),
                                "remaining_collection_seconds_linear_estimate": (
                                    time.perf_counter() - started
                                )
                                * max(0, ENDPOINTS[-1] / result["training_non_forced"] - 1),
                            }
                    _publish_json(
                        output / f"progress-{len(result['raw_components']):05d}.json", result
                    )
                    _guard(protocol, deadline)
                    if split == "training" and str(ENDPOINTS[-1]) in result["endpoints"]:
                        break
                if split == "training" and str(ENDPOINTS[-1]) not in result["endpoints"]:
                    raise TimeoutError("training boards exhausted before final endpoint")
            if set(phases) != set(protocol["training"]["required_phases"]):
                raise ValueError("missing or undeclared supervised training phase")
            total = sum(phases.values())
            result["phase_weights"] = {p: total / (len(phases) * n) for p, n in phases.items()}
            result["status"] = "complete"
        except (TimeoutError, MemoryError, RuntimeError, ValueError) as error:
            result["stop_reason"] = result["stop_reason"] or f"{type(error).__name__}: {error}"
        result["peak_rss_bytes"] = _rss()
        result["manifest"] = _publish_json(output / "corpus.json", result)
        return result


def _scan_corpus(protocol, deadline):
    corpus = _read(protocol["inputs"]["corpus"])
    _keys(
        corpus,
        (
            "format",
            "version",
            "protocol_identity",
            "protocol",
            "training",
            "validation",
            "raw_components",
            "endpoints",
            "training_non_forced",
            "phase_weights",
            "first_gate",
            "status",
            "stop_reason",
            "peak_rss_bytes",
        ),
        "corpus",
    )
    if (
        corpus["format"] != "colonist-rl.scaling-corpus"
        or type(corpus["version"]) is not int
        or corpus["version"] != 1
        or corpus["status"] != "complete"
        or corpus["protocol_identity"] != _identity(protocol)
        or _identity(corpus["protocol"]) != _identity(protocol)
    ):
        raise ValueError("corpus is incomplete or protocol identity differs")
    phases, endpoints, count = Counter(), {}, 0
    for split, partition in (("training", "training"), ("validation", "validation_labels")):
        rows = corpus[split]
        if type(rows) is not list or not rows:
            raise ValueError("corpus requires complete training and validation components")
        boards = protocol["partitions"][partition]
        if len(rows) * 4 > len(boards) or (split == "validation" and len(rows) * 4 != len(boards)):
            raise ValueError("component partition length mismatch")
        for index, row in enumerate(rows):
            _keys(
                row,
                ("artifact", "boards", "summary", "eligible", "split", "load_replay_seconds"),
                "component reference",
            )
            if (
                row["eligible"] is not True
                or row["split"] != partition
                or not _typed_equal(row["boards"], boards[index * 4 : index * 4 + 4])
            ):
                raise ValueError("component partition order mismatch")
            _guard(protocol, deadline)
            data = _load_component(row["artifact"], row["boards"], protocol, deadline)
            try:
                summary = _summary(data)
                if not _typed_equal(summary, row["summary"]):
                    raise ValueError("component summary differs from immutable observations")
            finally:
                del data
            if split == "training":
                if str(ENDPOINTS[-1]) in endpoints:
                    raise ValueError("training contains components after final threshold")
                phases.update(summary["supervised_phases"])
                count += summary["coverage"]["non_forced_decisions"]
                for threshold in MILESTONES:
                    if count >= threshold:
                        endpoints.setdefault(str(threshold), index + 1)
            _guard(protocol, deadline)
    if (
        str(ENDPOINTS[-1]) not in endpoints
        or not _typed_equal(endpoints, corpus["endpoints"])
        or not _typed_equal(count, corpus["training_non_forced"])
    ):
        raise ValueError("nested endpoints/counts differ")
    if set(phases) != set(protocol["training"]["required_phases"]):
        raise ValueError("missing or undeclared supervised training phase")
    weights = {p: count / (len(phases) * n) for p, n in phases.items()}
    if not _typed_equal(weights, corpus["phase_weights"]) or not _typed_equal(
        corpus["raw_components"], corpus["training"] + corpus["validation"]
    ):
        raise ValueError("shared training weights or raw component prefix differs")
    return corpus


def _merge_score(totals, row):
    for name in ("all", "non_forced"):
        target = totals.setdefault(name, {"count": 0, "correct": 0, "cross_entropy_sum": 0.0})
        target["count"] += row[name]["count"]
        target["correct"] += row[name]["correct"]
        target["cross_entropy_sum"] += (row[name]["cross_entropy"] or 0.0) * row[name]["count"]
    for name in ("by_phase", "by_action_kind"):
        target = totals.setdefault(name, {})
        for key, scores in row[name].items():
            bucket = target.setdefault(key, {})
            _merge_score(bucket, {**scores, "by_phase": {}, "by_action_kind": {}})


def _finish_score(totals):
    for name in ("all", "non_forced"):
        value = totals[name]
        count = value["count"]
        value["accuracy"] = value["correct"] / count if count else None
        value["cross_entropy"] = value["cross_entropy_sum"] / count if count else None
    for name in ("by_phase", "by_action_kind"):
        for bucket in totals[name].values():
            _finish_score(bucket)
    return totals


def _score(policy, rows, protocol, deadline):
    totals = {}
    for row in rows:
        _guard(protocol, deadline)
        data = _load_component(row["artifact"], row["boards"], protocol, deadline)
        try:
            _merge_score(totals, score_sequences(policy, _sequences(data)))
        finally:
            del data
        _guard(protocol, deadline)
    return _finish_score(totals)


def _arm(endpoint, seed):
    if (
        type(endpoint) is not int
        or endpoint not in ENDPOINTS
        or type(seed) is not int
        or seed not in (11, 29)
    ):
        raise ValueError("invalid endpoint/seed matrix member")
    return ARMS[ENDPOINTS.index(endpoint) * 2 + (seed == 29)]


def _core(identity, epoch, next_component, stats):
    return {"identity": identity, "epoch": epoch, "next_component": next_component, "stats": stats}


def _boundary(
    output, policy, optimizer, identity, epoch, next_component, stats, milestones, scores, timings
):
    token = uuid.uuid4().hex
    begin = time.perf_counter()
    core = _core(identity, epoch, next_component, stats)
    checkpoint = _publish(
        output / f"boundary-{token}.pt",
        lambda p: save_checkpoint(
            p, policy, optimizer, steps=stats["updates"], metadata={"scaling_boundary": core}
        ),
    )
    # Milestones contain only completed component/epoch checkpoints, never scores.
    selected = dict(milestones)
    if next_component == 0 and epoch in (1, 3, 5):
        selected[str(epoch)] = {
            "complete": True,
            "checkpoint": checkpoint,
            "updates": stats["updates"],
            "supervised_presentations": stats["supervised_presentations"],
        }
    updated_timings = {
        **timings,
        "checkpoint_seconds": timings["checkpoint_seconds"] + time.perf_counter() - begin,
    }
    cursor = {
        "format": "colonist-rl.scaling-cursor",
        "version": 1,
        **core,
        "checkpoint": checkpoint,
        "milestones": selected,
        "scores": scores,
        "timings": updated_timings,
    }
    reference = _publish_json(output / f"cursor-{token}.json", cursor)
    return reference, cursor


def _adam_steps(optimizer):
    return max((int(state.get("step", 0)) for state in optimizer.state.values()), default=0)


def _resume(ref, identity, rows):
    selected_count = len(rows)
    cursor = _read(ref)
    _keys(
        cursor,
        (
            "format",
            "version",
            "identity",
            "epoch",
            "next_component",
            "stats",
            "checkpoint",
            "milestones",
            "scores",
            "timings",
        ),
        "cursor",
    )
    if (
        cursor["format"] != "colonist-rl.scaling-cursor"
        or type(cursor["version"]) is not int
        or cursor["version"] != 1
        or not _typed_equal(cursor["identity"], identity)
    ):
        raise ValueError("resume identity differs")
    if (
        type(cursor["epoch"]) is not int
        or not 0 <= cursor["epoch"] <= 5
        or type(cursor["next_component"]) is not int
        or not 0 <= cursor["next_component"] < selected_count
        or (cursor["epoch"] == 5 and cursor["next_component"] != 0)
    ):
        raise ValueError("invalid durable cursor position")
    _validate_progress(cursor["stats"], rows, cursor["epoch"], cursor["next_component"])
    _keys(cursor["stats"], _empty_fit(), "fit counters")
    for key in ("updates", "supervised_presentations"):
        if type(cursor["stats"][key]) is not int or cursor["stats"][key] < 0:
            raise ValueError("invalid cursor counters")
    path = _check_ref(cursor["checkpoint"])
    policy, state = load_checkpoint(path)
    expected = {
        "scaling_boundary": _core(
            identity, cursor["epoch"], cursor["next_component"], cursor["stats"]
        )
    }
    if (
        not _typed_equal(state["metadata"], expected)
        or state["optimizer_state"] is None
        or state["steps"] != cursor["stats"]["updates"]
    ):
        raise ValueError("cursor does not bind saved optimizer/counters")
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    optimizer.load_state_dict(state["optimizer_state"])
    if _adam_steps(optimizer) != cursor["stats"]["updates"]:
        raise ValueError("saved Adam step differs from durable counters")
    if type(cursor["milestones"]) is not dict or any(
        k not in ("1", "3", "5") for k in cursor["milestones"]
    ):
        raise ValueError("invalid cursor milestones")
    expected_milestones = {str(e) for e in (1, 3, 5) if e <= cursor["epoch"]}
    if set(cursor["milestones"]) != expected_milestones:
        raise ValueError("cursor omits or fabricates completed milestones")
    for epoch, milestone in cursor["milestones"].items():
        _validate_milestone(milestone, identity, rows, int(epoch))
    if type(cursor["scores"]) is not dict or any(
        k not in ("0", "1", "3", "5") or int(k) > cursor["epoch"] for k in cursor["scores"]
    ):
        raise ValueError("invalid cursor score epochs")
    _keys(
        cursor["timings"],
        ("load_replay_seconds", "score_seconds", "checkpoint_seconds"),
        "cursor timings",
    )
    if any(
        type(v) is not float or not math.isfinite(v) or v < 0 for v in cursor["timings"].values()
    ):
        raise ValueError("invalid cursor timings")
    return policy, optimizer, cursor


def fit_scaling_arm(
    output_dir,
    *,
    corpus_manifest: str,
    initial_checkpoint: str,
    endpoint: int,
    seed: int,
    protocol: dict,
    resume_cursor: str | None = None,
) -> dict:
    """Fit canonical components using one persistent Adam and immutable cursors."""
    with _execution():
        _protocol(protocol, "fit")
        arm = _arm(endpoint, seed)
        start = protocol["inputs"]["starts"][str(seed)]
        if (
            corpus_manifest != protocol["inputs"]["corpus"]["path"]
            or initial_checkpoint != start["checkpoint"]["path"]
        ):
            raise ValueError("input path expectation differs")
        resume_ref = protocol["inputs"]["resume"]
        if (resume_cursor is None) != (resume_ref is None) or (
            resume_ref is not None and resume_cursor != resume_ref["path"]
        ):
            raise ValueError("resume path/SHA expectation differs")
        deadline = _deadline(protocol, arm)
        _guard(protocol, deadline)
        corpus = _scan_corpus(protocol, deadline)
        rows = corpus["training"][: corpus["endpoints"][str(endpoint)]]
        identity = {
            "protocol": _identity(protocol),
            "corpus": protocol["inputs"]["corpus"],
            "start": start["checkpoint"],
            "endpoint": endpoint,
            "seed": seed,
        }
        if resume_ref:
            policy, optimizer, boundary = _resume(resume_ref, identity, rows)
            cursor_ref = resume_ref
        else:
            policy = _check_start(start, seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
            boundary = {
                "epoch": 0,
                "next_component": 0,
                "stats": _empty_fit(),
                "milestones": {},
                "scores": {},
                "timings": {
                    "load_replay_seconds": 0.0,
                    "score_seconds": 0.0,
                    "checkpoint_seconds": 0.0,
                },
            }
            cursor_ref = None
        output = _output(output_dir, protocol)
        attempted, partial_last, failure = None, None, None
        try:
            if cursor_ref is None:
                cursor_ref, boundary = _boundary(
                    output,
                    policy,
                    optimizer,
                    identity,
                    0,
                    0,
                    boundary["stats"],
                    {},
                    {},
                    boundary["timings"],
                )
            pending_score = str(boundary["epoch"])
            if boundary["epoch"] in (0, 1, 3, 5) and pending_score not in boundary["scores"]:
                begin = time.perf_counter()
                score = _score(policy, corpus["validation"], protocol, deadline)
                scores = {**boundary["scores"], pending_score: score}
                timings = {
                    **boundary["timings"],
                    "score_seconds": boundary["timings"]["score_seconds"]
                    + time.perf_counter()
                    - begin,
                }
                cursor_ref, boundary = _boundary(
                    output,
                    policy,
                    optimizer,
                    identity,
                    boundary["epoch"],
                    boundary["next_component"],
                    boundary["stats"],
                    boundary["milestones"],
                    scores,
                    timings,
                )
            while boundary["epoch"] < 5:
                _guard(protocol, deadline)
                index, epoch = boundary["next_component"], boundary["epoch"]
                row = rows[index]
                begin = time.perf_counter()
                data = _load_component(row["artifact"], row["boards"], protocol, deadline)
                try:
                    episodes = _sequences(data)
                    before_steps = _adam_steps(optimizer)
                    attempted = {
                        "epoch": epoch,
                        "component": index,
                        "updates_observed_adam": 0,
                        "reported": None,
                    }
                    fitted = _fit_component(
                        policy, episodes, optimizer, corpus["phase_weights"], deadline
                    )
                    attempted["reported"] = fitted
                    attempted["updates_observed_adam"] = _adam_steps(optimizer) - before_steps
                    if fitted["status"] != "complete" or fitted["epochs"] != 1:
                        raise TimeoutError("interrupted component; previous cursor retained")
                    stats = json.loads(_json(boundary["stats"]))
                    _add_fit(stats, fitted, corpus["phase_weights"], episodes)
                finally:
                    if "episodes" in locals():
                        del episodes
                    del data
                timings = {
                    **boundary["timings"],
                    "load_replay_seconds": boundary["timings"]["load_replay_seconds"]
                    + time.perf_counter()
                    - begin
                    - fitted["profile"]["wall_seconds"],
                }
                index += 1
                if index == len(rows):
                    epoch, index = epoch + 1, 0
                _guard(protocol, deadline)
                # The checkpoint then cursor is the only commit point. Work on
                # the next component starts after both publications succeed.
                cursor_ref, boundary = _boundary(
                    output,
                    policy,
                    optimizer,
                    identity,
                    epoch,
                    index,
                    stats,
                    boundary["milestones"],
                    boundary["scores"],
                    timings,
                )
                attempted = None
                if index == 0 and epoch in (1, 3, 5):
                    begin = time.perf_counter()
                    score = _score(policy, corpus["validation"], protocol, deadline)
                    scores = {**boundary["scores"], str(epoch): score}
                    timings = {
                        **boundary["timings"],
                        "score_seconds": boundary["timings"]["score_seconds"]
                        + time.perf_counter()
                        - begin,
                    }
                    cursor_ref, boundary = _boundary(
                        output,
                        policy,
                        optimizer,
                        identity,
                        epoch,
                        0,
                        stats,
                        boundary["milestones"],
                        scores,
                        timings,
                    )
            _guard(protocol, deadline)
        except Exception as error:  # noqa: BLE001 -- preserve failed attempts, never retry
            failure = f"{type(error).__name__}: {error}"
            if attempted is not None:
                attempted["updates_observed_adam"] = (
                    _adam_steps(optimizer) - boundary["stats"]["updates"]
                )
            try:
                partial_last = _publish(
                    output / f"partial-last-{uuid.uuid4().hex}.pt",
                    lambda p: save_checkpoint(
                        p,
                        policy,
                        optimizer,
                        steps=_adam_steps(optimizer),
                        metadata={
                            "scaling_partial": {
                                "identity": identity,
                                "last_cursor": cursor_ref,
                                "attempted": attempted,
                                "failure": failure,
                            }
                        },
                    ),
                )
            except Exception as save_error:  # noqa: BLE001 -- original boundary survives publication failure
                failure += f"; partial snapshot failed: {type(save_error).__name__}: {save_error}"
        result = {
            "format": "colonist-rl.scaling-arm",
            "version": 1,
            "identity": identity,
            "arm": arm,
            "status": "complete" if failure is None and boundary["epoch"] == 5 else "partial",
            "completed_epochs": boundary["epoch"],
            "next_component": boundary["next_component"],
            "stats": boundary["stats"],
            "milestones": boundary["milestones"],
            "scores": boundary["scores"],
            "timings": boundary["timings"],
            "cursor": cursor_ref,
            "resume_from": resume_ref,
            "attempted": attempted,
            "partial_last": partial_last,
            "failure": failure,
            "peak_rss_bytes": _rss(),
            "loss": boundary["stats"]["loss_sum"] / boundary["stats"]["supervised_presentations"]
            if boundary["stats"]["supervised_presentations"]
            else None,
        }
        _check_ref(protocol["inputs"]["corpus"])
        for ref in (start["checkpoint"], start["parent"], start["bridge"]):
            _check_ref(ref)
        result["result"] = _publish_json(output / "result.json", result)
        return result


def _validate_progress(stats, rows, epoch, next_component):
    def numeric(value, template):
        if type(template) is dict:
            _keys(value, template, "counter/timing")
            for key in template:
                numeric(value[key], template[key])
        elif type(value) is not type(template) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid counter/timing type or value")

    numeric(stats, _empty_fit())
    expected_updates = epoch * sum(r["summary"]["updates_per_epoch"] for r in rows) + sum(
        r["summary"]["updates_per_epoch"] for r in rows[:next_component]
    )
    expected_presentations = epoch * sum(
        r["summary"]["coverage"]["non_forced_decisions"] for r in rows
    ) + sum(r["summary"]["coverage"]["non_forced_decisions"] for r in rows[:next_component])
    if (
        stats["updates"] != expected_updates
        or stats["supervised_presentations"] != expected_presentations
    ):
        raise ValueError("counters do not prove complete component/epoch progress")


def _validate_milestone(milestone, identity, rows, epoch):
    _keys(milestone, ("complete", "checkpoint", "updates", "supervised_presentations"), "milestone")
    if milestone["complete"] is not True:
        raise ValueError("incomplete milestone")
    path = _check_ref(milestone["checkpoint"])
    with _preserve_rng():
        policy, state = load_checkpoint(path)
    del policy
    _keys(state["metadata"], ("scaling_boundary",), "checkpoint metadata")
    core = state["metadata"].get("scaling_boundary")
    _keys(core, ("identity", "epoch", "next_component", "stats"), "checkpoint boundary")
    if (
        not _typed_equal(core["identity"], identity)
        or type(core["epoch"]) is not int
        or core["epoch"] != epoch
        or type(core["next_component"]) is not int
        or core["next_component"] != 0
    ):
        raise ValueError("milestone checkpoint identity/position differs")
    _validate_progress(core["stats"], rows, epoch, 0)
    if (
        type(milestone["updates"]) is not int
        or milestone["updates"] != core["stats"]["updates"]
        or type(milestone["supervised_presentations"]) is not int
        or milestone["supervised_presentations"] != core["stats"]["supervised_presentations"]
        or state["steps"] != milestone["updates"]
        or state["optimizer_state"] is None
    ):
        raise ValueError("milestone counters differ from saved checkpoint")


def _load_arms(protocol, corpus):
    arms = {}
    for index, arm in enumerate(ARMS):
        row = _read(protocol["inputs"]["results"][arm])
        _keys(
            row,
            (
                "format",
                "version",
                "identity",
                "arm",
                "status",
                "completed_epochs",
                "next_component",
                "stats",
                "milestones",
                "scores",
                "timings",
                "cursor",
                "resume_from",
                "attempted",
                "partial_last",
                "failure",
                "peak_rss_bytes",
                "loss",
            ),
            "arm result",
        )
        endpoint, seed = ENDPOINTS[index // 2], int(arm.split("/")[1])
        identity = {
            "protocol": _identity(protocol),
            "corpus": protocol["inputs"]["corpus"],
            "start": protocol["inputs"]["starts"][str(seed)]["checkpoint"],
            "endpoint": endpoint,
            "seed": seed,
        }
        if (
            row["format"] != "colonist-rl.scaling-arm"
            or type(row["version"]) is not int
            or row["version"] != 1
            or row["arm"] != arm
            or not _typed_equal(row["identity"], identity)
        ):
            raise ValueError("arm result identity differs")
        if row["status"] not in ("complete", "partial"):
            raise ValueError("invalid arm completion status")
        rows = corpus["training"][: corpus["endpoints"][str(endpoint)]]
        policy, optimizer, cursor = _resume(row["cursor"], identity, rows)
        del policy, optimizer
        for key, cursor_key in (
            ("completed_epochs", "epoch"),
            ("next_component", "next_component"),
            ("stats", "stats"),
            ("milestones", "milestones"),
            ("scores", "scores"),
            ("timings", "timings"),
        ):
            if not _typed_equal(row[key], cursor[cursor_key]):
                raise ValueError("arm result differs from authoritative cursor")
        if type(row["milestones"]) is not dict or any(
            key not in ("1", "3", "5") for key in row["milestones"]
        ):
            raise ValueError("invalid milestone fields")
        for epoch, milestone in row["milestones"].items():
            if int(epoch) > row["completed_epochs"]:
                raise ValueError("milestone exceeds completed epochs")
            _validate_milestone(milestone, identity, rows, int(epoch))
        arms[arm] = row
    return arms


def _journal_valid(panel, journal, boards):
    if panel is None or panel.get("status") != "complete" or panel.get("partial_outcomes"):
        raise ValueError("missing/incomplete evaluation panel")
    expected = {(b, s) for b in boards for s in range(4)}
    rows = journal["completed_outcomes"]
    if (
        len(rows) != len(expected)
        or {(r["board_seed"], r["seat"]) for r in rows} != expected
        or not _typed_equal(rows, panel["outcomes"])
    ):
        raise ValueError("missing/mismatched authoritative journal")
    # This existing strict check rejects censored, unresolved or malformed games.
    paired_block_summary(panel, panel)


def evaluate_scaling_study(
    output_dir, *, arm_results: dict[str, str], corpus_manifest: str, protocol: dict
) -> dict:
    """Freeze completion-only selection before deterministic validation gameplay."""
    with _execution():
        _protocol(protocol, "evaluation")
        _keys(arm_results, ARMS, "arm result matrix")
        if (
            any(
                type(path) is not str or path != protocol["inputs"]["results"][arm]["path"]
                for arm, path in arm_results.items()
            )
            or corpus_manifest != protocol["inputs"]["corpus"]["path"]
        ):
            raise ValueError("evaluation input path expectation differs")
        deadline = _deadline(protocol, "evaluation")
        _guard(protocol, deadline)
        corpus = _scan_corpus(protocol, deadline)
        arms = _load_arms(protocol, corpus)
        selection = _select_milestone(arms)
        output = _output(output_dir, protocol)
        _publish_json(
            output / "selection.json",
            {
                **selection,
                "inputs": protocol["inputs"]["results"],
                "criterion": "largest common complete epoch in 1/3/5; no outcome inputs",
            },
        )
        result = {
            "format": "colonist-rl.scaling-evaluation",
            "version": 1,
            "selection": selection,
            "variants": {},
            "journals": {},
            "comparisons": {},
            "comparison_unavailable": {},
            "status": "partial",
        }
        if selection["epoch"] is None:
            result["comparison_unavailable"]["all"] = "no common complete milestone"
        else:
            boards = protocol["partitions"]["validation_games"]
            for name in ("heuristic", *ARMS):
                label = name.replace("/", "-")
                journal = {
                    "variant": name,
                    "status": "running",
                    "completed_outcomes": [],
                    "failure": None,
                }
                panel, policy = None, None
                try:
                    _guard(protocol, deadline)
                    if name != "heuristic":
                        checkpoint = arms[name]["milestones"][str(selection["epoch"])]["checkpoint"]
                        policy, state = load_checkpoint(_check_ref(checkpoint))
                        del state

                    def on_game_end(row, label=label, journal=journal):
                        _publish_json(
                            output / f"{label}-game-{len(journal['completed_outcomes']):04d}.json",
                            row,
                        )
                        journal["completed_outcomes"].append(row)
                        _guard(protocol, deadline)

                    panel = evaluate_blocks(
                        policy=policy,
                        board_seeds=boards,
                        candidate="heuristic" if name == "heuristic" else "graph",
                        baseline="heuristic",
                        max_game_steps=20000,
                        deterministic=True,
                        deadline=deadline,
                        on_game_end=on_game_end,
                    )
                    journal["status"] = panel["status"]
                    _journal_valid(panel, journal, boards)
                except Exception as error:  # noqa: BLE001 -- retain failures and raw journal
                    journal["status"] = "failed"
                    journal["failure"] = f"{type(error).__name__}: {error}"
                finally:
                    del policy
                result["variants"][name] = panel
                result["journals"][name] = _publish_json(output / f"{label}-journal.json", journal)
                _publish_json(output / f"{label}-evaluation.json", panel)
                if journal["failure"]:
                    result["comparison_unavailable"][name] = journal["failure"]
            if not result["comparison_unavailable"]:
                pairs = [("heuristic", arm) for arm in ARMS] + [
                    (ARMS[0], ARMS[2]),
                    (ARMS[1], ARMS[3]),
                ]
                for reference, candidate in pairs:
                    result["comparisons"][reference + "->" + candidate] = paired_block_summary(
                        result["variants"][reference], result["variants"][candidate]
                    )
                result["status"] = "complete"
        _check_ref(protocol["inputs"]["corpus"])
        for ref in protocol["inputs"]["results"].values():
            _check_ref(ref)
        result["result"] = _publish_json(output / "evaluation.json", result)
        return result
