"""Bounded, provenance-preserving observation-only learning experiments."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .evaluation import evaluate_blocks, paired_block_summary, validate_seed_splits
from .imitation import collect_demonstrations, fit_imitation, imitation_metrics, save_demonstrations
from .policy import KINDS, PHASES, GraphPolicy
from .simulator import BACKEND_VERSION
from .training import save_checkpoint


def manifest_digest(value) -> str:
    """Content digest of canonical JSON, excluding filesystem and insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_identity() -> dict:
    """Only clean execution is supported: a commit identifies the exact sources."""
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    revision = git("rev-parse", "HEAD")
    return {
        "revision": revision,
        "dirty": bool(git("status", "--porcelain")),
        "content_identity": "git:" + revision,
    }


def baseline_gate(summary: dict) -> dict:
    """Pure conservative gate; never infer competence from censored or forced data."""
    reasons, comparisons = [], {}
    teacher = summary.get("teacher")
    seeds = summary.get("training_seeds")
    students = summary.get("students")
    # The seed plan is explicit: inferring it from surviving stages could hide
    # a missing seed. Reject malformed metadata before publishing comparisons.
    if (
        not isinstance(seeds, (list, tuple))
        or len(seeds) < 2
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        reasons.append("at_least_two_declared_unique_training_seeds_required")
    if not isinstance(students, list) or any(
        not isinstance(student, dict) or type(student.get("seed")) is not int
        for student in students
    ):
        reasons.append("declared_student_evidence_required")
    elif not reasons and (
        len(students) != len(seeds) or {student["seed"] for student in students} != set(seeds)
    ):
        reasons.append("exactly_one_student_per_declared_seed_required")
    if reasons:
        return {
            "eligible": False,
            "reasons": reasons,
            "teacher_comparisons": {},
            "note": "Descriptive eligibility only; no significance claim.",
        }
    variant_map = summary.get("variants")
    if not isinstance(variant_map, dict):
        variant_map = {}
    required = {
        f"{candidate}/{opponent}"
        for candidate in ("heuristic", "random")
        for opponent in ("heuristic", "random")
    }
    required.update(f"graph/{seed}/{kind}" for seed in seeds for kind in ("untrained", "imitated"))
    missing = required - variant_map.keys()
    reasons.extend(f"{name}:required_variant_missing" for name in sorted(missing))
    if not teacher:
        reasons.append("missing_teacher_evidence")
    elif teacher != variant_map.get("heuristic/heuristic"):
        reasons.append("teacher_variant_evidence_mismatch")
    for student in students:
        for kind in ("untrained", "imitated"):
            name = f"graph/{student['seed']}/{kind}"
            if student.get(kind) != variant_map.get(name):
                reasons.append(f"{name}:student_variant_evidence_mismatch")
    variants = list(variant_map.items())
    all_paired = not reasons
    for name, result in variants:
        try:
            paired = paired_block_summary(teacher, result)
            if paired["paired_boards"] < 8:
                reasons.append(f"{name}:eight_complete_board_blocks_required")
        except (ValueError, TypeError, KeyError, AttributeError):
            all_paired = False
            reasons.append(f"{name}:unresolved_incomplete_or_mismatched_blocks")

    def agreement_passes(score, minimum):
        count, accuracy = score.get("count", 0), score.get("accuracy")
        return (
            type(count) is int
            and count >= minimum
            and type(accuracy) in (float, int)
            and math.isfinite(accuracy)
            and 0.60 <= accuracy <= 1.0
        )

    for student in students:
        seed = str(student["seed"])
        metrics = student.get("validation_metrics") or {}
        score = metrics.get("non_forced", {})
        if not agreement_passes(score, 100):
            reasons.append(f"{seed}:held_out_non_forced_agreement_required")
        phases = metrics.get("by_phase", {})
        if not phases:
            reasons.append(f"{seed}:missing_phase_diagnostics")
        for phase, groups in phases.items():
            score = groups.get("non_forced", {})
            if score.get("count", 0) and not agreement_passes(score, 1):
                reasons.append(f"{seed}:{phase}:non_forced_agreement_below_threshold")
        try:
            paired = paired_block_summary(teacher, student["imitated"])
            if all_paired:
                comparisons[seed] = paired
            if paired["mean_victory_points_difference"] < -1.0:
                reasons.append(f"{seed}:victory_point_tolerance_failed")
            if paired["win_rate_difference"] < -0.10:
                reasons.append(f"{seed}:win_rate_tolerance_failed")
        except (ValueError, TypeError, KeyError, AttributeError):
            reasons.append(f"{seed}:teacher_comparison_unavailable")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "teacher_comparisons": comparisons,
        "note": "Descriptive eligibility only; no significance claim.",
    }


def _write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _artifact(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _metrics(policy, dataset, deadline):
    # The public diagnostic replays an episode. Bound every forward without
    # changing that interface, and always remove the temporary guard.
    def guard(module, args):
        if time.monotonic() >= deadline:
            raise TimeoutError("diagnostic deadline")

    hook = policy.register_forward_pre_hook(guard)
    try:
        return imitation_metrics(policy, dataset)
    finally:
        hook.remove()


def run_experiment(
    output_dir,
    *,
    training_seeds=(11, 29),
    decisions=2500,
    epochs=2,
    hidden_size=32,
    sequence_length=16,
    validation_board_seeds=(90000, 90001),
    validation_demo_seeds=(91000, 91001),
    final_test_board_seeds=(70000017, 70000019),
    max_game_steps=20000,
    wall_seconds=900,
) -> dict:
    """Run one bounded pilot; no automatic restart or PPO on inadequate evidence.

    Seed candidates, not actual consumed boards, are validated before all I/O
    and simulation. Existing manifests are refused to preserve previous runs.
    """
    for value in (decisions, epochs, hidden_size, sequence_length, max_game_steps):
        if type(value) is not int or value <= 0:
            raise ValueError("integer experiment limits must be positive")
    if (
        type(wall_seconds) not in (int, float)
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0
    ):
        raise ValueError("wall_seconds must be finite and positive")
    # Also validate initialization seeds before deriving their board namespaces.
    validate_seed_splits(training_seeds, validation_board_seeds, final_test_board_seeds)
    training = {str(s): [1000 * s + i for i in range(64)] for s in training_seeds}
    train_boards = [b for values in training.values() for b in values]
    validation = list(validation_board_seeds) + list(validation_demo_seeds)
    probe_seeds = [92000]
    validate_seed_splits(train_boards, validation + probe_seeds, final_test_board_seeds)
    source = source_identity()
    if source["dirty"]:
        raise ValueError("experiment requires a clean committed source tree")
    output = Path(output_dir)
    if (output / "manifest.json").exists():
        raise ValueError("output already contains an experiment; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + wall_seconds
    manifest = {
        "format": "colonist-rl.experiment",
        "version": 1,
        "source": source,
        "started_unix_seconds": time.time(),
        "wall_seconds": wall_seconds,
        "deadline_monotonic": deadline,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "catanatron", "gymnasium")
        },
        "python": platform.python_version(),
        "backend_version": BACKEND_VERSION,
        "action_kinds": list(KINDS),
        "model": {
            "hidden_size": hidden_size,
            "sequence_length": sequence_length,
            "epochs": epochs,
            "learning_rate": 3e-4,
            "action_mode": "argmax",
        },
        "runtime": {"device": "cpu", "torch_threads": 1, "workers": 1},
        "seed_splits": {
            "training_initialization": list(training_seeds),
            "training_candidates": training,
            "validation_boards": list(validation_board_seeds),
            "validation_demonstrations": list(validation_demo_seeds),
            "probe_only": probe_seeds,
            "final_test_reserved": list(final_test_board_seeds),
        },
        "rng": {
            "initialization": "Python/NumPy/CPU Torch = training seed; independent per seed",
            "students": "untrained and imitated are copies of the same initial weights",
            "collection": "board; teacher=board+100; seat=board_index%4",
            "evaluation": "board; graph=(4*board+seat)%2**63; baseline=board+31+seat",
        },
        "requested": {
            "demonstrations_per_seed": decisions,
            "validation_blocks": len(validation_board_seeds),
            "comparison_games": len(validation_board_seeds) * 4 * (4 + 2 * len(training_seeds)),
            "max_game_steps": max_game_steps,
        },
    }
    manifest["configuration_sha256"] = manifest_digest(
        {
            k: v
            for k, v in manifest.items()
            if k not in ("started_unix_seconds", "deadline_monotonic")
        }
    )
    _write(output / "manifest.json", manifest)
    result = {
        "status": "running",
        "stages": {},
        "training_seeds": list(training_seeds),
        "students": [],
        "variants": {},
        "artifacts": {},
        "game_progress": {},
        "teacher": None,
        "manifest_sha256": _artifact(output / "manifest.json")["sha256"],
        "ppo_status": "deferred_insufficient_baseline_evidence",
    }
    external = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def persist():
        result["elapsed_seconds"] = time.monotonic() - started
        _write(output / "results.json", result)

    def stage(name, function, *, until=deadline):
        begin = time.monotonic()
        result["stages"][name] = {"status": "running"}
        persist()
        if begin >= until:
            value = None
            result["stages"][name] = {"status": "incomplete", "reason": "deadline"}
        else:
            try:
                value = function(until)
                incomplete = isinstance(value, dict) and (
                    value.get("status") == "incomplete"
                    or value.get("collection", {}).get("stop_reason") == "deadline"
                )
                result["stages"][name] = {"status": "incomplete" if incomplete else "complete"}
            except TimeoutError as error:
                value = None
                result["stages"][name] = {"status": "incomplete", "reason": str(error)}
            except Exception as error:  # noqa: BLE001 -- preserve partial evidence at stage boundary
                value = None
                result["stages"][name] = {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                result["status"] = "failed"
        result["stages"][name]["elapsed_seconds"] = time.monotonic() - begin
        persist()
        return value

    def evaluate(name, policy, boards, baseline, until):
        path = output / "progress" / (name.replace("/", "-") + ".json")
        path.parent.mkdir(exist_ok=True)
        progress = {
            "status": "running",
            "variant": name,
            "manifest_sha256": result["manifest_sha256"],
            "requested_board_seeds": list(boards),
            "requested_games": 4 * len(boards),
            "finished_games": 0,
            "aggregation_scope": "raw_completed_games",
            "completed_outcomes": [],
            "note": "Recovery evidence only; strict statistics require complete matched blocks.",
        }

        def save_progress(row=None):
            if row is not None:
                progress["completed_outcomes"].append(row)
                progress["finished_games"] = len(progress["completed_outcomes"])
            # Replace the journal before referencing it from the main results.
            # A later actor/evaluator exception cannot discard completed rows.
            _write(path, progress)
            result["game_progress"][name] = {
                "path": str(path.resolve()),
                "finished_games": progress["finished_games"],
                "status": progress["status"],
            }
            persist()

        save_progress()
        value = stage(
            name,
            lambda end: evaluate_blocks(
                policy=policy,
                board_seeds=boards,
                candidate="heuristic" if name == "probe" else name.split("/")[0],
                baseline=baseline,
                max_game_steps=max_game_steps,
                deterministic=True,
                deadline=end,
                on_game_end=save_progress,
            ),
            until=until,
        )
        progress["status"] = result["stages"][name]["status"]
        progress["stage"] = result["stages"][name].copy()
        save_progress()
        if value is not None:
            _write(output / (name.replace("/", "-") + ".json"), value)
        return value

    try:
        probe = evaluate("probe", None, probe_seeds, "heuristic", min(deadline, started + 60))
        if probe is None:
            result["status"] = "failed" if result["status"] == "failed" else "partial"
            return result
        probe_seconds = max(0.001, result["stages"]["probe"]["elapsed_seconds"])
        finished = probe.get("finished_games", 0)
        # A partial probe gives a conservative lower bound on per-game time.
        seconds_per_game = probe_seconds / max(1, finished)
        decision_rate = max(1, probe.get("learner_decisions", 0)) / probe_seconds
        remaining = max(0.0, deadline - time.monotonic())
        variant_count = 4 + 2 * len(training_seeds)
        boards = list(validation_board_seeds)
        # Reserve 45% for collection/fitting/diagnostics and a 2x graph-game factor.
        if (
            len(boards) > 1
            and seconds_per_game * 4 * len(boards) * variant_count * 2 > remaining * 0.55
        ):
            boards = boards[:1]
        evaluation_budget = min(
            remaining * 0.55, seconds_per_game * 4 * len(boards) * variant_count * 2
        )
        learning_budget = max(0.0, remaining - evaluation_budget)
        # Prefix replay is expensive: budget 16 observed-teacher time units per demo.
        per_demo_seconds = max(0.05, 16 / decision_rate)
        cap = min(
            decisions, max(1, int(learning_budget * 0.60 / len(training_seeds) / per_demo_seconds))
        )
        result["budget_decision"] = {
            "probe_seconds": probe_seconds,
            "probe_finished_games": finished,
            "seconds_per_game": seconds_per_game,
            "teacher_decisions_per_second": decision_rate,
            "evaluation_graph_safety_factor": 2,
            "demo_cost_safety_factor": 16,
            "demonstrations_per_seed": cap,
            "demonstration_shortfall_per_seed": decisions - cap,
            "validation_board_seeds": boards,
            "comparison_games": 4 * len(boards) * variant_count,
            "reduced_blocks": len(boards) < len(validation_board_seeds),
            "inconclusive": len(boards) < 8,
            "evaluation_seconds_reserved": evaluation_budget,
            "learning_seconds_reserved": learning_budget,
        }
        persist()
        learning_end = min(deadline, time.monotonic() + learning_budget)
        # Shared held-out diagnostics use independent boards and never gradients.
        demo_end = min(learning_end, time.monotonic() + learning_budget * 0.10)
        validation_data = stage(
            "validation_demonstrations",
            lambda end: collect_demonstrations(
                board_seeds=validation_demo_seeds,
                max_decisions=128,
                max_game_steps=max_game_steps,
                deadline=end,
            ),
            until=demo_end,
        )
        if validation_data is not None:
            path = output / "validation-demonstrations.json.gz"
            save_demonstrations(path, validation_data)
            result["artifacts"]["validation_demonstrations"] = _artifact(path)
            result["validation_coverage"] = validation_data["coverage"]
            result["validation_actual_board_seeds"] = validation_data["collection"][
                "actual_board_seeds"
            ]
        policies = {}
        for index, seed in enumerate(training_seeds):
            remaining_seeds = len(training_seeds) - index
            seed_end = min(
                learning_end,
                time.monotonic() + max(0.0, learning_end - time.monotonic()) / remaining_seeds,
            )
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            initial = GraphPolicy(hidden_size)
            student_policy = copy.deepcopy(initial)
            policies[seed] = (initial, student_policy)
            student = {
                "seed": seed,
                "untrained": None,
                "imitated": None,
                "validation_metrics": None,
                "fit": None,
            }
            result["students"].append(student)
            for kind, policy in (("untrained", initial), ("initial_student", student_policy)):
                path = output / f"seed-{seed}-{kind}.pt"
                save_checkpoint(path, policy, metadata={"seed": seed, "source": source})
                result["artifacts"][f"{seed}/{kind}"] = _artifact(path)
            collection_end = min(
                seed_end, time.monotonic() + max(0.0, seed_end - time.monotonic()) * 0.30
            )
            data = stage(
                f"collect/{seed}",
                lambda end, seed=seed: collect_demonstrations(
                    board_seeds=training[str(seed)],
                    max_decisions=cap,
                    max_game_steps=max_game_steps,
                    deadline=end,
                ),
                until=collection_end,
            )
            if data is not None:
                path = output / f"seed-{seed}-demonstrations.json.gz"
                save_demonstrations(path, data)
                result["artifacts"][f"{seed}/dataset"] = _artifact(path)
                student["coverage"] = data["coverage"]
                student["actual_board_seeds"] = data["collection"]["actual_board_seeds"]
                student["dataset_episodes"] = [
                    {k: v for k, v in e.items() if k != "samples"} for e in data["episodes"]
                ]
                student["fit"] = stage(
                    f"fit/{seed}",
                    lambda end, student_policy=student_policy, data=data: fit_imitation(
                        student_policy,
                        data,
                        epochs=epochs,
                        sequence_length=sequence_length,
                        deadline=end,
                    ),
                    until=min(
                        seed_end, time.monotonic() + max(0.0, seed_end - time.monotonic()) * 0.85
                    ),
                )
            path = output / f"seed-{seed}-imitated.pt"
            save_checkpoint(
                path,
                student_policy,
                metadata={"seed": seed, "source": source, "fit": student["fit"]},
            )
            result["artifacts"][f"{seed}/imitated"] = _artifact(path)
            if validation_data is not None:
                student["validation_metrics"] = stage(
                    f"metrics/{seed}",
                    lambda end, student_policy=student_policy: _metrics(
                        student_policy, validation_data, end
                    ),
                    until=seed_end,
                )
                metrics = student["validation_metrics"]
                if metrics:
                    student["uncovered_phases"] = sorted(set(PHASES) - set(metrics["by_phase"]))
                    student["uncovered_action_kinds"] = sorted(
                        set(KINDS) - set(metrics["by_action_kind"])
                    )
            persist()
        schedule = [
            (candidate, baseline, None, None)
            for baseline in ("random", "heuristic")
            for candidate in ("heuristic", "random")
        ]
        schedule += [
            ("graph", "heuristic", seed, kind)
            for seed in training_seeds
            for kind in ("untrained", "imitated")
        ]
        for i, (candidate, baseline, seed, kind) in enumerate(schedule):
            until = min(
                deadline,
                time.monotonic() + max(0.0, deadline - time.monotonic()) / (len(schedule) - i),
            )
            name = f"{candidate}/{baseline}" if seed is None else f"graph/{seed}/{kind}"
            policy = None if seed is None else policies[seed][kind == "imitated"]
            value = evaluate(name, policy, boards, baseline, until)
            result["variants"][name] = value
            if name == "heuristic/heuristic":
                result["teacher"] = value
            if seed is not None:
                next(s for s in result["students"] if s["seed"] == seed)[kind] = value
            persist()
        result["gate"] = baseline_gate(result)
        if result["status"] != "failed":
            result["status"] = (
                "partial"
                if any(s["status"] != "complete" for s in result["stages"].values())
                else "complete"
            )
        return result
    except Exception as error:  # noqa: BLE001 -- preserve partial evidence at stage boundary
        result["status"] = "failed"
        result["failure"] = f"{type(error).__name__}: {error}"
        return result
    finally:
        result.setdefault("gate", baseline_gate(result))
        result["source_at_end"] = source_identity()
        if result["source_at_end"] != source:
            result["status"] = "failed"
            result["source_error"] = "source changed during experiment"
        random.setstate(external[0])
        np.random.set_state(external[1])
        torch.set_rng_state(external[2])
        torch.set_num_threads(threads)
        persist()
