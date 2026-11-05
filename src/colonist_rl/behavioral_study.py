"""Minimal audited teacher-state versus student-state acquisition comparison.

Raw collection components remain immutable. Choice matching creates a separate
full-history view, never a merged or relabeled collection. No resume or PPO path.
"""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from .evaluation import evaluate_blocks, paired_block_summary
from .imitation import (
    _positive,
    _preserve_rng,
    _sequences,
    _source,
    collect_demonstrations,
    load_demonstrations,
    save_demonstrations,
)
from .imitation_study import _state_digest
from .policy import GraphPolicy, policy_architecture
from .student_labels import (
    _atomic,
    _component,
    _digest,
    _fields,
    _json,
    _partitions,
    _primitive,
    _validate_source,
    collect_student_labels,
    corpus_sequences,
    load_corpus,
    save_student_labels,
    write_corpus,
)
from .supervised_fit import (
    EpisodeSequence,
    _batches,
    _validate_sequences,
    fit_sequences,
    score_sequences,
)
from .training import load_checkpoint, save_checkpoint

_ARMS = ("teacher_state", "student_state")
_PROTOCOL_KEYS = {
    "version",
    "stage",
    "seeds",
    "source_revision",
    "partitions",
    "acquisition_boards",
    "input_digests",
    "consumed_evidence",
    "validation_labels_path",
    "query_budget",
    "max_game_steps",
    "baseline",
    "deterministic",
    "epochs",
    "segment_epochs",
    "sequence_length",
    "batch_size",
    "learning_rate",
    "weighting",
    "collection_seconds",
    "fit_seconds",
    "evaluation_seconds",
    "deadline",
}
_LIMITATION = (
    "Queries and newly selected non-forced choices are matched within each seed. "
    "Update counts, weighted loss mass, forced-prefix work and runtime can differ; "
    "this is not an equal-compute or pure state-distribution causal comparison."
)


def select_choice_prefix(
    episodes: tuple[EpisodeSequence, ...], *, choices: int
) -> tuple[tuple[EpisodeSequence, ...], dict]:
    """Retain every observation through the requested non-forced choice.

    Board identities here are visible BoardSpec digests, since EpisodeSequence
    contains no board seed. The driver adds the immutable raw episode's seed.
    Endpoints use zero-based episode indices and exclusive sample stops.
    """
    _positive(choices, "choices")
    coverage = _validate_sequences(episodes)
    if coverage["non_forced_decisions"] < choices:
        raise ValueError("insufficient non-forced choices")
    retained, endpoints, unused, boards = [], [], [], []
    remaining = choices
    for i, episode in enumerate(episodes):
        stop = 0
        if remaining:
            for stop, (view, _) in enumerate(episode, 1):
                remaining -= int(len(view.legal_actions) > 1)
                if not remaining:
                    break
            retained.append(episode if stop == len(episode) else episode[:stop])
            endpoints.append({"episode_index": i, "sample_stop": stop})
            boards.append(
                {
                    "episode_index": i,
                    "board_sha256": _digest(asdict(episode[0][0].board)) if episode else None,
                }
            )
        if stop < len(episode):
            unused.append({"episode_index": i, "sample_start": stop, "sample_stop": len(episode)})
    selected = tuple(retained)
    selected_count = _validate_sequences(selected)
    all_count, non_forced = selected_count["decisions"], selected_count["non_forced_decisions"]
    unused_all = coverage["decisions"] - all_count
    unused_choices = coverage["non_forced_decisions"] - non_forced
    return selected, {
        "rule": "ordered_full_history_through_Qth_non_forced_choice",
        "choices": choices,
        "retained": endpoints,
        "counts": {"all": all_count, "forced": all_count - non_forced, "choices": non_forced},
        "actual_boards": boards,
        "unused": unused,
        "unused_counts": {
            "all": unused_all,
            "forced": unused_all - unused_choices,
            "choices": unused_choices,
        },
    }


def _write(path, value):
    return _atomic(path, _json(value))


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _absolute(path):
    if type(path) is not str or not Path(path).is_absolute():
        raise ValueError("input paths must be absolute strings")
    return str(Path(path).resolve(strict=True))


def _integrity(protocol, students=None):
    source = _source()
    _validate_source(source, known=True)
    if _json(source) != _json({"revision": protocol["source_revision"], "dirty": False}):
        raise ValueError("study requires the expected clean source revision without drift")
    try:
        for path, expected in protocol["input_digests"].items():
            if _file_digest(path) != expected:
                raise ValueError(f"input digest changed: {path}")
    except OSError as error:
        raise ValueError("missing study input") from error
    if students is not None:
        for seed in students.values():
            for arm in _ARMS:
                record = seed.get(arm, {})
                identity = record.get("raw_component")
                if identity is not None:
                    if _file_digest(identity["path"]) != identity["sha256"]:
                        raise ValueError("raw component digest changed")
                    actual, _ = _component(identity["path"])
                    if _json(actual) != _json(identity):
                        raise ValueError("raw component identity/digest changed")
    return {"verified": True, "source": source, "input_digests": dict(protocol["input_digests"])}


def _preflight(initial_checkpoints, training_components, protocol):
    _fields(protocol, _PROTOCOL_KEYS, "protocol")
    _primitive(protocol)
    if (
        type(protocol["version"]) is not int
        or protocol["version"] != 1
        or protocol["stage"] != "student_labels"
        or protocol["baseline"] != "heuristic"
        or protocol["weighting"] != "original_phase_balanced"
        or type(protocol["deterministic"]) is not bool
        or type(protocol["source_revision"]) is not str
        or not protocol["source_revision"]
    ):
        raise ValueError("unsupported protocol settings")
    seeds = protocol["seeds"]
    if (
        type(seeds) is not list
        or not seeds
        or any(type(s) is not int for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds require a nonempty unique integer list")
    for mapping in (initial_checkpoints, training_components):
        if (
            type(mapping) is not dict
            or any(type(s) is not int for s in mapping)
            or set(mapping) != set(seeds)
        ):
            raise ValueError("exact checkpoint/training seed matrix required")
    _fields(protocol["acquisition_boards"], {str(s) for s in seeds}, "acquisition seed matrix")
    _partitions(protocol["partitions"])
    partitions = protocol["partitions"]
    if not all(partitions[k] for k in ("training", "validation_labels", "validation_games")):
        raise ValueError("nonempty training and held-out panels required")
    used = set()
    for entry in protocol["acquisition_boards"].values():
        _fields(entry, set(_ARMS), "acquisition arm matrix")
        for boards in entry.values():
            if (
                type(boards) is not list
                or not boards
                or any(type(b) is not int for b in boards)
                or len(set(boards)) != len(boards)
                or used.intersection(boards)
                or not set(boards).issubset(partitions["training"])
            ):
                raise ValueError("acquisition boards must be disjoint unused training boards")
            used.update(boards)
    for key in (
        "query_budget",
        "max_game_steps",
        "epochs",
        "segment_epochs",
        "sequence_length",
        "batch_size",
    ):
        _positive(protocol[key], key)
    for key in (
        "learning_rate",
        "collection_seconds",
        "fit_seconds",
        "evaluation_seconds",
        "deadline",
    ):
        value = protocol[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (key != "deadline" and value <= 0)
        ):
            raise ValueError(f"{key} must be finite and positive (deadline absolute)")
    evidence = protocol["consumed_evidence"]
    if (
        type(evidence) is not list
        or not evidence
        or any(type(p) is not str for p in evidence)
        or len(set(evidence)) != len(evidence)
    ):
        raise ValueError("ordered nonempty consumed evidence paths required")
    digests = protocol["input_digests"]
    if type(digests) is not dict or any(
        type(p) is not str or type(d) is not str or re.fullmatch("[0-9a-f]{64}", d) is None
        for p, d in digests.items()
    ):
        raise ValueError("input_digests must contain SHA-256 identities")
    required = [*initial_checkpoints.values(), protocol["validation_labels_path"], *evidence]
    for components in training_components.values():
        if (
            type(components) is not tuple
            or not components
            or any(type(p) is not str for p in components)
        ):
            raise ValueError("training components require nonempty ordered path tuples")
        required.extend(components)
    if any(type(p) is not str for p in required):
        raise ValueError("all input paths must be strings")
    try:
        if set(required) != set(digests):
            raise ValueError("exact input digest list required; missing or unlisted inputs")
        resolved = [_absolute(p) for p in digests]
        if len(set(resolved)) != len(resolved) or any(p != r for p, r in zip(digests, resolved)):
            raise ValueError("input paths must be canonical and distinct")
    except OSError as error:
        raise ValueError("missing study input") from error
    _integrity(protocol)
    components, starts, frozen = {}, {}, {}
    for seed in seeds:
        entries = [_component(p) for p in training_components[seed]]
        if len({entry[0]["sha256"] for entry in entries}) != len(entries):
            raise ValueError("duplicate original components")
        for identity, data in entries:
            _validate_source(data["source"], known=True)
            boards = identity["actual_board_seeds"]
            if not set(boards).issubset(partitions["training"]) or used.intersection(boards):
                raise ValueError("original training boards overlap acquisition or held-out boards")
        components[seed] = [entry[0] for entry in entries]
        with _preserve_rng():
            policy, checkpoint = load_checkpoint(initial_checkpoints[seed])
        if type(policy) is not GraphPolicy or any(
            not torch.isfinite(value).all() for value in policy.state_dict().values()
        ):
            raise ValueError("starting checkpoint must be a finite legacy GraphPolicy")
        metadata = checkpoint["metadata"]
        if not isinstance(metadata, dict) or (
            "seed" in metadata and (type(metadata["seed"]) is not int or metadata["seed"] != seed)
        ):
            raise ValueError("starting checkpoint seed mismatch")
        if "fit_calls" in metadata:
            calls = metadata["fit_calls"]
            if (
                type(calls) is not list
                or not calls
                or any(
                    not isinstance(call, dict)
                    or call.get("status") != "complete"
                    or type(call.get("epochs")) is not int
                    or call["epochs"] < 1
                    or type(call.get("requested_epochs")) is not int
                    or call["epochs"] != call["requested_epochs"]
                    or any(
                        type(call.get(k)) is not int or call[k] < 0
                        for k in ("updates", "supervised_presentations")
                    )
                    for call in calls
                )
            ):
                raise ValueError("starting checkpoint requires completed fit-call evidence")
            for field, key in (
                ("completed_epochs", "epochs"),
                ("updates", "updates"),
                ("supervised_presentations", "supervised_presentations"),
            ):
                if type(metadata.get(field)) is not int or metadata[field] != sum(
                    c[key] for c in calls
                ):
                    raise ValueError("starting checkpoint completed-work counters mismatch")
            if type(checkpoint["steps"]) is not int or checkpoint["steps"] != metadata["updates"]:
                raise ValueError("starting checkpoint update count mismatch")
        elif metadata.get("partial") or metadata.get("completed_epochs"):
            raise ValueError("completed starting checkpoint requires full fit-call evidence")
        frozen[seed] = (policy, {"metadata": deepcopy(metadata), "steps": checkpoint["steps"]})
        starts[seed] = {
            "path": initial_checkpoints[seed],
            "sha256": digests[initial_checkpoints[seed]],
            "architecture": policy_architecture(policy),
        }
    validation = load_demonstrations(protocol["validation_labels_path"])
    _validate_source(validation["source"], known=True)
    if set(validation["collection"]["actual_board_seeds"]) != set(partitions["validation_labels"]):
        raise ValueError("validation labels must match the full declared label panel")
    if not validation["coverage"]["non_forced_decisions"]:
        raise ValueError("zero held-out supervision")
    _integrity(protocol)
    return components, starts, frozen, _sequences(validation)


def _work(episodes, protocol, weights):
    """Count the accepted fitter's non-forced-containing chunk schedule."""
    coverage = _validate_sequences(episodes)
    updates = prefixes = chunks = batch_chunks = 0
    masses = []
    for scheduled in _batches(episodes, protocol["sequence_length"], protocol["batch_size"]):
        active = [
            (e, start)
            for e, start in scheduled
            if any(
                len(v.legal_actions) > 1 for v, _ in e[start : start + protocol["sequence_length"]]
            )
        ]
        updates += bool(active)
        if active:
            masses.append(
                sum(
                    weights[v.context.kind]
                    for e, start in active
                    for v, _ in e[start : start + protocol["sequence_length"]]
                    if len(v.legal_actions) > 1
                )
            )
        batch_chunks += len(active)
        prefixes += sum(start for _, start in active)
        chunks += sum(len(e[start : start + protocol["sequence_length"]]) for e, start in active)
    epochs = protocol["epochs"]
    return {
        "epochs": epochs,
        "updates": updates * epochs,
        "supervised_presentations": coverage["non_forced_decisions"] * epochs,
        "prefix_forwards": prefixes * epochs,
        "chunk_forwards": chunks * epochs,
        "batch_chunks": batch_chunks * epochs,
        "weighted_loss_mass": sum(masses) * epochs,
        "decisions_per_epoch": coverage["decisions"],
        "forced_per_epoch": coverage["decisions"] - coverage["non_forced_decisions"],
    }, tuple(masses)


def _save_model(path, policy, optimizer, *, steps, metadata):
    # The accepted saver has a fixed staging filename. Isolate it inside an
    # exclusive same-directory staging directory, then atomically publish.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.") as staging:
        staged = Path(staging) / "checkpoint.pt"
        save_checkpoint(staged, policy, optimizer=optimizer, steps=steps, metadata=metadata)
        staged.replace(path)
    return {
        "path": str(path),
        "sha256": _file_digest(path),
        "architecture": policy_architecture(policy),
    }


def _acquisition_counts(data, arm):
    samples = [s for episode in data["episodes"] for s in episode["samples"]]
    student = arm == "student_state"
    queries = sum(len(e["teacher_calls"]) for e in data["episodes"]) if student else len(samples)
    executed = [
        sample
        for episode in data["episodes"]
        for sample in episode["samples"]
        if (
            sample["behavior_executed"]
            if student
            else sample["observation"].context.version < episode["steps"]
        )
    ]
    return {
        "queries": queries,
        "paired_labels": len(samples),
        "non_forced_labels": data["coverage"]["non_forced_decisions"],
        "executed_decisions": len(executed),
        "executed_choices": sum(len(s["observation"].legal_actions) > 1 for s in executed),
        "unmatched_teacher_calls": queries - len(samples),
        "unexecuted_paired_labels": len(samples) - len(executed),
        "stop_reason": data["collection"]["stop_reason"],
        "actual_board_seeds": data["collection"]["actual_board_seeds"],
        "seconds": data["collection"]["elapsed_seconds"],
    }


def _panel(result, boards, journal):
    expected = {(b, seat) for b in boards for seat in range(4)}
    rows = result["outcomes"]
    keys = [(r["board_seed"], r["seat"]) for r in rows]
    if (
        result["status"] != "complete"
        or result["partial_outcomes"]
        or len(keys) != len(expected)
        or set(keys) != expected
        or _json(rows) != _json(journal)
    ):
        raise ValueError(
            "full declared uncensored evaluation panel and authoritative journal required"
        )
    paired_block_summary(result, result)


def run_behavioral_study(
    output_dir,
    *,
    initial_checkpoints: dict[int, str],
    training_components: dict[int, tuple[str, ...]],
    protocol: dict,
) -> dict:
    """Run only the declared two-arm study, preserving partial/failure evidence.

    Collection and fit ceilings apply per arm/seed; evaluation has one shared
    ceiling across all variants. All ceilings are clipped by the
    global monotonic deadline. Existing nonempty output is never resumed.
    """
    protocol = deepcopy(protocol)
    components, starts, frozen, validation = _preflight(
        initial_checkpoints, training_components, protocol
    )
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; no resume or overwrite")
    output.mkdir(parents=True, exist_ok=True)
    matrix = [f"{arm}/{seed}" for seed in protocol["seeds"] for arm in _ARMS]
    manifest = {
        "format": "colonist-rl.behavioral-study",
        "version": 1,
        "protocol": protocol,
        "source": _source(),
        "original_components": {str(s): c for s, c in components.items()},
        "starting_checkpoints": {str(s): c for s, c in starts.items()},
        "starting_checkpoint_evidence": {str(s): frozen[s][1] for s in protocol["seeds"]},
        "consumed_evidence": [
            {"path": p, "sha256": protocol["input_digests"][p]}
            for p in protocol["consumed_evidence"]
        ],
        "expected_variants": ["heuristic", *matrix],
        "evaluation_schedule": [
            {"board_seed": b, "seat": seat}
            for b in protocol["partitions"]["validation_games"]
            for seat in range(4)
        ],
        "limitation": _LIMITATION,
        "runtime": {"device": "cpu", "torch_threads": torch.get_num_threads(), "workers": 1},
    }
    manifest["configuration_sha256"] = _digest(manifest)
    _write(output / "manifest.json", manifest)
    result = {
        "status": "running",
        "students": {str(s): {} for s in protocol["seeds"]},
        "variants": {},
        "comparisons": {},
        "comparison_unavailable": [],
        "game_progress": {},
        "limitation": _LIMITATION,
        "handoff": {},
        "manifest_sha256": _file_digest(output / "manifest.json"),
    }
    policies, originals, selected, update_masses = {}, {}, {}, {}

    def persist():
        _write(output / "results.json", result)

    def ceiling(seconds):
        return min(protocol["deadline"], time.monotonic() + seconds)

    try:
        # Validate original corpora and freeze weights before any new collection.
        for seed in protocol["seeds"]:
            corpus_path = output / f"original-{seed}-corpus.json"
            write_corpus(
                corpus_path, components=training_components[seed], partitions=protocol["partitions"]
            )
            originals[seed] = corpus_sequences(load_corpus(corpus_path))
            counts = Counter(
                v.context.kind for e in originals[seed] for v, _ in e if len(v.legal_actions) > 1
            )
            if not counts:
                raise ValueError("zero original supervision")
            weights = {p: sum(counts.values()) / (len(counts) * n) for p, n in counts.items()}
            result["students"][str(seed)]["phase_weights"] = {
                "values": weights,
                "original_non_forced_counts": dict(counts),
                "source_components": components[seed],
            }
            for arm in _ARMS:
                name = f"{arm}/{seed}"
                policy = deepcopy(frozen[seed][0])
                policies[name] = policy
                record = {
                    "initial_checkpoint": starts[seed],
                    "initial_state_sha256": _state_digest(policy),
                    "fit_segments": [],
                    "checkpoints": [],
                }
                result["students"][str(seed)][arm] = record
                raw_path = output / f"{arm}-{seed}-labels.gz"
                kwargs = {
                    "board_seeds": protocol["acquisition_boards"][str(seed)][arm],
                    "max_decisions": protocol["query_budget"],
                    "max_game_steps": protocol["max_game_steps"],
                    "baseline": protocol["baseline"],
                    "deadline": ceiling(protocol["collection_seconds"]),
                }
                if arm == "teacher_state":
                    data = collect_demonstrations(**kwargs)
                    save_demonstrations(raw_path, data)
                else:
                    data = collect_student_labels(
                        policy,
                        checkpoint_identity=starts[seed],
                        seed=seed,
                        deterministic=protocol["deterministic"],
                        progress_path=raw_path,
                        **kwargs,
                    )
                    save_student_labels(raw_path, data)
                record["raw_component"] = _component(raw_path)[0]
                record["acquisition"] = _acquisition_counts(data, arm)
                persist()
                actual_boards = data["collection"]["actual_board_seeds"]
                expected_boards = protocol["acquisition_boards"][str(seed)][arm]
                if _json(actual_boards) != _json(expected_boards[: len(actual_boards)]):
                    raise ValueError("acquisition board schedule mismatch")
                for key in ("max_decisions", "max_game_steps", "baseline", "deadline"):
                    if _json(data["collection"][key]) != _json(kwargs[key]):
                        raise ValueError("acquisition collection settings mismatch")
                if _json(data["source"]) != _json(manifest["source"]):
                    raise ValueError("acquisition source identity mismatch")
            q = min(
                result["students"][str(seed)][a]["acquisition"]["non_forced_labels"] for a in _ARMS
            )
            if not q:
                raise ValueError("zero newly acquired non-forced supervision")
            for arm in _ARMS:
                name = f"{arm}/{seed}"
                record = result["students"][str(seed)][arm]
                raw = record["raw_component"]
                corpus_path = output / f"{arm}-{seed}-corpus.json"
                write_corpus(
                    corpus_path, components=(raw["path"],), partitions=protocol["partitions"]
                )
                sequences = corpus_sequences(load_corpus(corpus_path))
                prefix, selection = select_choice_prefix(sequences, choices=q)
                selection.update(component=raw, derivation_source=_source())
                _, data = _component(raw["path"])
                for board in selection["actual_boards"]:
                    board["board_seed"] = data["episodes"][board["episode_index"]]["board_seed"]
                selection["derivation_sha256"] = _digest(selection)
                record["selection"] = selection
                _write(output / f"{arm}-{seed}-selection.json", selection)
                selected[name] = originals[seed] + prefix
                phases = {
                    v.context.kind for e in selected[name] for v, _ in e if len(v.legal_actions) > 1
                }
                if not phases.issubset(weights):
                    raise ValueError("acquired supervision has uncovered phase weights")
                record["expected_work"], update_masses[name] = _work(
                    selected[name], protocol, weights
                )
                persist()
        # Every arm's selection and expected work is fixed before any fitting.
        _integrity(protocol)
        for seed in protocol["seeds"]:
            for arm in _ARMS:
                name = f"{arm}/{seed}"
                policy = policies[name]
                record = result["students"][str(seed)][arm]
                optimizer = torch.optim.Adam(policy.parameters(), lr=protocol["learning_rate"])
                fit_deadline = ceiling(protocol["fit_seconds"])
                actual = dict.fromkeys(
                    (
                        "epochs",
                        "updates",
                        "supervised_presentations",
                        "prefix_forwards",
                        "chunk_forwards",
                        "batch_chunks",
                    ),
                    0,
                )
                actual["weighted_loss_mass"] = 0.0
                record["actual_work"] = actual
                record["fit_status"] = "partial"
                started = time.monotonic()
                while actual["epochs"] < protocol["epochs"]:
                    epochs = min(protocol["segment_epochs"], protocol["epochs"] - actual["epochs"])
                    segment = None
                    record.pop("validation_metrics", None)
                    try:
                        segment = fit_sequences(
                            policy,
                            selected[name],
                            epochs=epochs,
                            sequence_length=protocol["sequence_length"],
                            batch_size=protocol["batch_size"],
                            learning_rate=protocol["learning_rate"],
                            phase_weights=result["students"][str(seed)]["phase_weights"]["values"],
                            optimizer=optimizer,
                            deadline=fit_deadline,
                            profile=True,
                        )
                        record["fit_segments"].append(segment)
                        for key in ("epochs", "updates", "supervised_presentations"):
                            actual[key] += segment[key]
                        for key in ("prefix_forwards", "chunk_forwards", "batch_chunks"):
                            actual[key] += segment["profile"][key]
                        full, remainder = divmod(actual["updates"], len(update_masses[name]))
                        actual["weighted_loss_mass"] = full * sum(update_masses[name]) + sum(
                            update_masses[name][:remainder]
                        )
                    finally:
                        actual["seconds"] = time.monotonic() - started
                        if segment is None:
                            record["work_after_exception"] = (
                                "unknown; checkpoint includes any changes before exception"
                            )
                        checkpoint = _save_model(
                            output / f"{arm}-{seed}-segment-{len(record['checkpoints']) + 1}.pt",
                            policy,
                            optimizer,
                            steps=actual["updates"],
                            metadata={
                                "seed": seed,
                                "arm": arm,
                                "actual_work": actual,
                                "fit_segments": record["fit_segments"],
                                "partial": segment is None or segment["status"] != "complete",
                            },
                        )
                        record["checkpoints"].append(checkpoint)
                        persist()
                    if time.monotonic() < fit_deadline:
                        record["validation_metrics"] = score_sequences(policy, validation)
                        record.setdefault("validation_scores", []).append(
                            {
                                "checkpoint": record["checkpoints"][-1],
                                "metrics": record["validation_metrics"],
                            }
                        )
                    if segment["status"] != "complete":
                        break
                    if actual["epochs"] == protocol["epochs"]:
                        record["fit_status"] = "complete"
                    if time.monotonic() >= fit_deadline:
                        break
                actual["seconds"] = time.monotonic() - started
                result["handoff"][name] = {
                    "checkpoint": record["checkpoints"][-1],
                    "frozen_starting_reference": starts[seed],
                    "fit_status": record["fit_status"],
                    "selection_sha256": record["selection"]["derivation_sha256"],
                }
                persist()
        boards = protocol["partitions"]["validation_games"]
        evaluation_deadline = ceiling(protocol["evaluation_seconds"])
        for name in ["heuristic", *matrix]:
            journal = []
            result["game_progress"][name] = journal

            def on_game_end(row, *, journal=journal, name=name):
                key = (row["board_seed"], row["seat"])
                if key not in {(b, s) for b in boards for s in range(4)} or any(
                    (r["board_seed"], r["seat"]) == key for r in journal
                ):
                    raise ValueError("unexpected or duplicate game callback")
                _write(output / "games" / name.replace("/", "-") / f"{key[0]}-{key[1]}.json", row)
                journal.append(deepcopy(row))
                persist()

            evaluation = evaluate_blocks(
                policies.get(name),
                board_seeds=boards,
                candidate="heuristic" if name == "heuristic" else "graph",
                baseline="heuristic",
                max_game_steps=protocol["max_game_steps"],
                deterministic=protocol["deterministic"],
                deadline=evaluation_deadline,
                on_game_end=on_game_end,
            )
            result["variants"][name] = evaluation
            _write(output / f"evaluation-{name.replace('/', '-')}.json", evaluation)
            try:
                _panel(evaluation, boards, journal)
            except (KeyError, ValueError, TypeError) as error:
                result["comparison_unavailable"].append(f"{name}: {error}")
            persist()
        if set(result["variants"]) != {"heuristic", *matrix}:
            raise ValueError("exact evaluation variant matrix required")
        for seed in protocol["seeds"]:
            for arm in _ARMS:
                record = result["students"][str(seed)][arm]
                if (
                    record["acquisition"]["queries"] != protocol["query_budget"]
                    or record["acquisition"]["stop_reason"] == "deadline"
                    or record["acquisition"]["unmatched_teacher_calls"]
                    or record["acquisition"]["unexecuted_paired_labels"]
                    or record["fit_status"] != "complete"
                    or "validation_metrics" not in record
                ):
                    result["comparison_unavailable"].append(
                        f"{arm}/{seed}: acquisition, fitting or held-out score incomplete"
                    )
        result["final_integrity"] = _integrity(protocol, result["students"])
        if not result["comparison_unavailable"]:
            result["comparisons"] = {
                str(seed): {
                    "student_minus_teacher": paired_block_summary(
                        result["variants"][f"teacher_state/{seed}"],
                        result["variants"][f"student_state/{seed}"],
                    ),
                    **{
                        f"{arm}_minus_heuristic": paired_block_summary(
                            result["variants"]["heuristic"], result["variants"][f"{arm}/{seed}"]
                        )
                        for arm in _ARMS
                    },
                }
                for seed in protocol["seeds"]
            }
        result["status"] = "partial" if result["comparison_unavailable"] else "complete"
        persist()
        return result
    except BaseException as error:
        result.update(
            status="failed",
            comparisons={},
            error={"type": type(error).__name__, "message": str(error)},
        )
        try:
            result["final_integrity"] = _integrity(protocol, result["students"])
        except (OSError, ValueError) as integrity_error:
            result["final_integrity"] = {"verified": False, "error": str(integrity_error)}
        persist()
        raise
