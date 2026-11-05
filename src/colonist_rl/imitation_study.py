"""Controlled uniform versus phase-balanced imitation on immutable saved inputs.

This runner orchestrates existing collection, fitting, metrics and evaluation;
it does not implement a second simulator loop or infer action opportunities.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from .evaluation import evaluate_blocks, paired_block_summary, validate_seed_splits
from .experiments import _artifact, _metrics, _write, manifest_digest, source_identity
from .imitation import (
    _preserve_rng,
    _validate,
    collect_demonstrations,
    fit_imitation,
    load_demonstrations,
    save_demonstrations,
)
from .training import load_checkpoint, save_checkpoint

_ARMS = ("uniform", "phase_balanced")
_FINAL_TEST = (70000017, 70000019)


def phase_weights(dataset) -> dict[str, float]:
    """N/(P*n_phase), using only non-forced choices in this training dataset.

    Phases with no training choices are absent. An all-forced dataset returns
    an empty mapping; fit_imitation still replays forced inputs where needed.
    """
    _validate(dataset)
    counts = Counter(
        sample["observation"].context.kind
        for episode in dataset["episodes"]
        for sample in episode["samples"]
        if len(sample["observation"].legal_actions) > 1
    )
    total = sum(counts.values())
    return {phase: total / (len(counts) * count) for phase, count in counts.items()}


def _state_digest(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def run_imitation_study(
    output_dir,
    *,
    datasets,
    initial_checkpoints,
    epochs=20,
    wall_seconds=900,
    validation_board_seeds=(62100000, 62100001, 62100002, 62100003),
    validation_demo_seeds=(62110000, 62110001, 62110002, 62110003),
) -> dict:
    """Compare saved-input arms with independent Adam and identical per-seed RNG.

    Input mappings are integer training seed -> local artifact path. Each arm
    reloads the same checkpoint and uses the same dataset order, sequence 16,
    learning rate 3e-4 and hidden size 32. Sixty percent of the wall budget is
    split into identical fixed fit allowances; the global deadline also bounds
    collection, metrics and evaluation. No adaptive epoch/board reduction occurs.
    Partial training never supports an equal-budget comparison. Existing output
    directories must be empty; the input artifacts are never overwritten.
    """
    if (
        not isinstance(datasets, Mapping)
        or not isinstance(initial_checkpoints, Mapping)
        or not datasets
        or any(type(seed) is not int for seed in datasets)
        or any(type(seed) is not int for seed in initial_checkpoints)
        or set(datasets) != set(initial_checkpoints)
    ):
        raise ValueError("datasets and initial_checkpoints require matching integer seed mappings")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if (
        type(wall_seconds) not in (int, float)
        or not math.isfinite(wall_seconds)
        or wall_seconds <= 0
    ):
        raise ValueError("wall_seconds must be finite and positive")
    seeds = sorted(datasets)
    # Validate all supplied seed candidates, including unconsumed demo candidates.
    validate_seed_splits(validation_board_seeds, validation_demo_seeds, _FINAL_TEST)
    data = {seed: load_demonstrations(datasets[seed]) for seed in seeds}
    train_boards = [
        board for seed in seeds for board in data[seed]["collection"]["actual_board_seeds"]
    ]
    validate_seed_splits(
        train_boards, list(validation_board_seeds) + list(validation_demo_seeds), _FINAL_TEST
    )
    source = source_identity()
    if (
        source.get("dirty") is not False
        or not isinstance(source.get("revision"), str)
        or not source["revision"]
        or source.get("content_identity") != "git:" + source["revision"]
    ):
        raise ValueError("study requires a known clean committed source tree")
    artifacts = {
        str(seed): {
            "dataset": _artifact(Path(datasets[seed])),
            "initial_checkpoint": _artifact(Path(initial_checkpoints[seed])),
            "dataset_source": data[seed]["source"],
            "coverage": data[seed]["coverage"],
            "phase_weights": phase_weights(data[seed]),
        }
        for seed in seeds
    }
    # Check architecture and initial-checkpoint semantics before creating outputs.
    with _preserve_rng():
        for seed in seeds:
            policy, checkpoint = load_checkpoint(initial_checkpoints[seed])
            if (
                policy.hidden_size != 32
                or checkpoint["steps"] != 0
                or checkpoint["optimizer_state"] is not None
            ):
                raise ValueError(
                    "initial checkpoint requires hidden32, zero steps and no optimizer"
                )
            if any(not torch.isfinite(value).all() for value in policy.state_dict().values()):
                raise ValueError("initial checkpoint contains nonfinite parameters")
            artifacts[str(seed)]["initial_state_sha256"] = _state_digest(policy)
            artifacts[str(seed)]["checkpoint_metadata"] = checkpoint["metadata"]
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; refusing to overwrite study evidence")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + wall_seconds
    fit_seconds = wall_seconds * 0.60 / (len(seeds) * len(_ARMS))
    manifest = {
        "format": "colonist-rl.imitation-study",
        "version": 1,
        "source": source,
        "inputs": artifacts,
        "started_unix_seconds": time.time(),
        "wall_seconds": wall_seconds,
        "model": {
            "hidden_size": 32,
            "epochs": epochs,
            "sequence_length": 16,
            "learning_rate": 3e-4,
            "action_mode": "argmax",
        },
        "arms": list(_ARMS),
        "fit_seconds_per_arm": fit_seconds,
        "seed_splits": {
            "training_initialization": seeds,
            "training_boards": train_boards,
            "validation_boards": list(validation_board_seeds),
            "validation_demonstrations": list(validation_demo_seeds),
            "final_test_reserved": list(_FINAL_TEST),
        },
        "runtime": {"device": "cpu", "torch_threads": 1, "workers": 1},
        "rng": "Reload same initial checkpoint, then Python/NumPy/CPU Torch=seed for each arm",
        "weighting": "Training non-forced N/(P*n_phase); weighted chunk CE sum / choice count",
        "validation": {"max_decisions": 128, "weighted": False},
        "evaluation": {
            "baseline": "heuristic",
            "max_game_steps": 20000,
            "requested_games_per_variant": 4 * len(validation_board_seeds),
        },
    }
    manifest["configuration_sha256"] = manifest_digest(
        {k: v for k, v in manifest.items() if k != "started_unix_seconds"}
    )
    _write(output / "manifest.json", manifest)
    result = {
        "status": "running",
        "training_seeds": seeds,
        "stages": {},
        "students": {str(seed): {} for seed in seeds},
        "variants": {},
        "comparisons": {},
        "comparison_unavailable": {},
        "game_progress": {},
        "equal_training_budget_complete": False,
        "manifest_sha256": _artifact(output / "manifest.json")["sha256"],
        "actual_play_diagnostics": {
            "status": "external_analysis_required",
            "required_counts": [
                "maritime_legal",
                "maritime_selected",
                "passes_with_maritime_legal",
                "feasible_accept",
                "feasible_reject",
                "passes_over_direct_purchases",
            ],
            "note": "Task3 observation-only audit supplies actual opportunity denominators; selected action counts alone are insufficient.",
        },
    }
    threads = torch.get_num_threads()
    policies = {}

    def persist():
        result["elapsed_seconds"] = time.monotonic() - started
        _write(output / "results.json", result)

    def stage(name, function, until=deadline):
        begin = time.monotonic()
        result["stages"][name] = {"status": "running"}
        persist()
        if begin >= until:
            result["stages"][name] = {"status": "incomplete", "reason": "deadline"}
            value = None
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
            except Exception as error:  # noqa: BLE001 -- preserve partial stage evidence
                value = None
                result["stages"][name] = {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
        result["stages"][name]["elapsed_seconds"] = time.monotonic() - begin
        persist()
        return value

    def evaluate(name, policy):
        path = output / "progress" / (name.replace("/", "-") + ".json")
        path.parent.mkdir(exist_ok=True)
        progress = {
            "variant": name,
            "status": "running",
            "finished_games": 0,
            "manifest_sha256": result["manifest_sha256"],
            "aggregation_scope": "raw_completed_games",
            "completed_outcomes": [],
        }

        def journal(row=None):
            if row is not None:
                progress["completed_outcomes"].append(row)
                progress["finished_games"] = len(progress["completed_outcomes"])
            _write(path, progress)
            result["game_progress"][name] = {
                "path": str(path.resolve()),
                "finished_games": progress["finished_games"],
                "status": progress["status"],
            }
            persist()

        journal()
        value = stage(
            "evaluate/" + name,
            lambda end: evaluate_blocks(
                policy=policy,
                board_seeds=validation_board_seeds,
                candidate="graph" if policy is not None else "heuristic",
                baseline="heuristic",
                max_game_steps=20000,
                deterministic=True,
                deadline=end,
                on_game_end=journal,
            ),
        )
        progress["status"] = result["stages"]["evaluate/" + name]["status"]
        journal()
        result["variants"][name] = value
        if value is not None:
            _write(output / (name.replace("/", "-") + "-evaluation.json"), value)
        persist()

    try:
        with _preserve_rng():
            torch.set_num_threads(1)
            validation = stage(
                "validation_demonstrations",
                lambda end: collect_demonstrations(
                    board_seeds=validation_demo_seeds,
                    max_decisions=128,
                    deadline=end,
                ),
            )
            if validation is not None:
                path = output / "validation-demonstrations.json.gz"
                save_demonstrations(path, validation)
                result["validation_dataset"] = _artifact(path)
            for seed in seeds:
                for arm in _ARMS:
                    name = f"{seed}/{arm}"
                    row = {
                        "phase_weights": None
                        if arm == "uniform"
                        else artifacts[str(seed)]["phase_weights"],
                        "initial_state_sha256": artifacts[str(seed)]["initial_state_sha256"],
                        "fit": None,
                        "validation_metrics": None,
                    }
                    result["students"][str(seed)][arm] = row

                    def fit(end, seed=seed, arm=arm, row=row):
                        policy, _ = load_checkpoint(initial_checkpoints[seed])
                        # Independent optimizer (created by fitter), identical RNG start and order.
                        random.seed(seed)
                        np.random.seed(seed % 2**32)
                        torch.manual_seed(seed)
                        policies[seed, arm] = policy
                        return fit_imitation(
                            policy,
                            data[seed],
                            epochs=epochs,
                            sequence_length=16,
                            learning_rate=3e-4,
                            deadline=end,
                            phase_weights=row["phase_weights"],
                        )

                    row["fit"] = stage(
                        "fit/" + name, fit, min(deadline, time.monotonic() + fit_seconds)
                    )
                    if (seed, arm) in policies:
                        path = output / f"seed-{seed}-{arm}.pt"
                        save_checkpoint(
                            path,
                            policies[seed, arm],
                            metadata={
                                "seed": seed,
                                "arm": arm,
                                "source": source,
                                "fit": row["fit"],
                                "inputs": artifacts[str(seed)],
                                "manifest_sha256": result["manifest_sha256"],
                            },
                        )
                        row["checkpoint"] = _artifact(path)
                    persist()
            result["equal_training_budget_complete"] = all(
                (row.get("fit") or {}).get("status") == "complete"
                and row["fit"]["epochs"] == epochs
                and row["fit"]["supervised_presentations"]
                == epochs * data[seed]["coverage"]["non_forced_decisions"]
                for seed in seeds
                for row in result["students"][str(seed)].values()
            )
            for (seed, arm), policy in policies.items():
                row = result["students"][str(seed)][arm]
                row["state_before_validation_sha256"] = _state_digest(policy)
                if validation is not None:
                    row["validation_metrics"] = stage(
                        "metrics/" + f"{seed}/{arm}",
                        lambda end, policy=policy: _metrics(policy, validation, end),
                    )
                row["state_after_validation_sha256"] = _state_digest(policy)
                if row["state_before_validation_sha256"] != row["state_after_validation_sha256"]:
                    raise RuntimeError("validation changed policy parameters")
            evaluate("heuristic", None)
            for (seed, arm), policy in policies.items():
                evaluate(f"{seed}/{arm}", policy)
            if result["equal_training_budget_complete"]:
                pairs = [("heuristic", f"{seed}/{arm}") for seed in seeds for arm in _ARMS]
                pairs += [(f"{seed}/uniform", f"{seed}/phase_balanced") for seed in seeds]
                for reference, candidate in pairs:
                    name = reference + "->" + candidate
                    try:
                        result["comparisons"][name] = paired_block_summary(
                            result["variants"][reference], result["variants"][candidate]
                        )
                    except (ValueError, TypeError, KeyError, AttributeError) as error:
                        result["comparison_unavailable"][name] = str(error)
            statuses = [s["status"] for s in result["stages"].values()]
            result["status"] = (
                "failed"
                if "failed" in statuses
                else ("complete" if all(s == "complete" for s in statuses) else "partial")
            )
    except Exception as error:  # noqa: BLE001 -- retain completed evidence on failure
        result["status"] = "failed"
        result["failure"] = f"{type(error).__name__}: {error}"
    finally:
        torch.set_num_threads(threads)
        try:
            result["source_at_end"] = source_identity()
            if result["source_at_end"] != source:
                result["status"] = "failed"
                result["source_error"] = "source changed during imitation study"
            for seed in seeds:
                for key in ("dataset", "initial_checkpoint"):
                    original = artifacts[str(seed)][key]
                    if _artifact(Path(original["path"])) != original:
                        result["status"] = "failed"
                        result["input_error"] = "immutable input changed during imitation study"
        except Exception as error:  # noqa: BLE001 -- provenance failure cannot erase evidence
            result["status"] = "failed"
            result["provenance_error"] = f"{type(error).__name__}: {error}"
        if result["status"] == "failed":
            result["comparisons"] = {}
            result["equal_training_budget_complete"] = False
        persist()
    return result
