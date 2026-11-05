"""Bounded fresh-PPO curriculum using the existing collector and block evaluator."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .evaluation import (
    _paired_rows,
    _seed_list,
    evaluate_blocks,
    paired_block_summary,
    validate_seed_splits,
)
from .experiments import _artifact, _write, manifest_digest, source_identity
from .imitation import _preserve_rng
from .training import (
    PPOConfig,
    RolloutCollector,
    _expired,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)

LINEUPS = {
    "random": ("random", "random", "random"),
    "one_heuristic": ("heuristic", "random", "random"),
    "heuristic": ("heuristic", "heuristic", "heuristic"),
    "historical": ("heuristic", "heuristic", "historical"),
}
FINAL_TEST = tuple(range(71001000, 71001008))


@dataclass(frozen=True)
class CurriculumConfig:
    compute_seconds: float = 1800
    wall_seconds: float = 3600
    deadline: float | None = None
    max_decisions_per_seed: int = 65536
    rollout_size: int = 128
    evaluation_interval: int = 4096
    max_game_steps: int = 20000
    validation_board_seeds: tuple[int, ...] = (62000000, 62000001, 62000002, 62000003)
    historical_checkpoints: tuple[str, ...] = ()
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(epochs=2))


def _resolved(result, *, baseline=None, lineup=None):
    if not isinstance(result, dict) or (
        result.get("status") != "complete"
        or result.get("aggregation_scope") != "complete_blocks"
        or result.get("candidate") != "graph"
        or result.get("deterministic") is not True
        or result.get("interrupted_game") is not None
    ):
        raise ValueError("complete deterministic graph block evidence required")
    rows = _paired_rows(result)
    # Curriculum evidence is stricter than the legacy paired-comparison API:
    # every selected game must positively establish typed terminal resolution.
    for row in rows.values():
        winner = row.get("winner")
        if (
            row.get("terminated") is not True
            or row.get("truncated") is not False
            or type(winner) is not int
            or winner not in range(4)
            or row["won"] != (winner == row["seat"])
        ):
            raise ValueError("explicit typed and consistent resolved-game evidence required")
    boards = {board for board, _ in rows}
    if len(boards) < 2 or boards & set(FINAL_TEST):
        raise ValueError("two non-final complete board blocks required")
    if (
        set(_seed_list(result.get("requested_board_seeds"))) != boards
        or set(_seed_list(result.get("completed_board_seeds"))) != boards
    ):
        raise ValueError("requested and completed boards must match evidence")
    actual_lineup = result.get("opponent_lineup")
    implicit = {
        "random": LINEUPS["random"],
        "heuristic": LINEUPS["heuristic"],
        "mixed": ("heuristic", "random", "heuristic"),
    }
    actual_lineup = (
        tuple(actual_lineup) if actual_lineup is not None else implicit.get(result.get("baseline"))
    )
    if baseline is not None and (
        result.get("baseline") != baseline or actual_lineup != LINEUPS[baseline]
    ):
        raise ValueError("fixed heuristic opponent panel required")
    if lineup is not None and actual_lineup != lineup:
        raise ValueError("current stage lineup evidence required")
    return {
        "boards": len(boards),
        "win_rate": sum(row["won"] for row in rows.values()) / len(rows),
        "mean_victory_points": sum(row["victory_points"] for row in rows.values()) / len(rows),
    }


def promotion_gate(stage, stage_result, fixed_result, initial_fixed_result) -> dict:
    """Operational gate only: reject missing, censored and mismatched evidence."""
    reasons, metrics = [], {}
    try:
        if stage not in LINEUPS:
            raise ValueError("unknown curriculum stage")
        metrics["stage"] = _resolved(stage_result, lineup=LINEUPS[stage])
        metrics["fixed"] = _resolved(fixed_result, baseline="heuristic")
        _resolved(initial_fixed_result, baseline="heuristic")
        # All gate panels use the same declared validation boards and four seats.
        paired_block_summary(stage_result, fixed_result)
        regression = paired_block_summary(initial_fixed_result, fixed_result)[
            "mean_victory_points_difference"
        ]
        metrics["fixed_vp_difference"] = regression
        if metrics["stage"]["win_rate"] < 0.25:
            reasons.append("current_stage_win_rate_below_0.25")
        if metrics["stage"]["mean_victory_points"] < 4:
            reasons.append("current_stage_vp_below_4")
        if regression < -1:
            reasons.append("fixed_heuristic_vp_regressed_more_than_1")
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        reasons.append("invalid_evidence: " + str(error))
    return {
        "promote": not reasons,
        "stage": stage,
        "reasons": reasons,
        "metrics": metrics,
        "note": "Operational allocation gate, not a competence or significance claim.",
    }


def select_checkpoint(candidates) -> dict:
    """Maximize fixed heuristic VP, then wins, then earliest eligible checkpoint."""
    eligible = []
    rejected = []
    common_initial = None

    def chronological(item):
        index, candidate = item
        order = candidate.get("order", index) if isinstance(candidate, dict) else None
        return (order if type(order) is int and order >= 0 else math.inf, index)

    for index, candidate in sorted(enumerate(candidates), key=chronological):
        try:
            fixed = candidate["fixed_result"]
            metrics = _resolved(fixed, baseline="heuristic")
            initial = candidate["initial_fixed_result"]
            _resolved(initial, baseline="heuristic")
            if paired_block_summary(initial, fixed)["mean_victory_points_difference"] < -1:
                raise ValueError("fixed heuristic VP regressed more than 1")
            order = candidate.get("order", index)
            if type(order) is not int or order < 0:
                raise ValueError("checkpoint order must be a nonnegative integer")
            initial_outcomes = {
                key: (row["winner"], row["won"], row["victory_points"])
                for key, row in _paired_rows(initial).items()
            }
            # Earliest eligible evidence establishes one reference for this
            # selection call, independent of candidate-list and row ordering.
            # Pairing above already requires fixed and initial panels to match.
            if common_initial is None:
                common_initial = initial_outcomes
            elif initial_outcomes != common_initial:
                raise ValueError("checkpoint panel or initial-reference outcomes mismatch")
            eligible.append(
                (
                    metrics["mean_victory_points"],
                    metrics["win_rate"],
                    -order,
                    -index,
                    candidate,
                    metrics,
                )
            )
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            rejected.append(
                {
                    "id": candidate.get("id") if isinstance(candidate, dict) else None,
                    "reason": str(error),
                }
            )
    if not eligible:
        return {"id": None, "status": "unavailable", "rejected": rejected}
    *_, selected, metrics = max(eligible, key=lambda row: row[:4])
    return {**selected, "status": "selected", "metrics": metrics, "rejected": rejected}


def _validate_config(config, initial_checkpoints):
    if not isinstance(config, CurriculumConfig):
        raise TypeError("config must be CurriculumConfig")
    if (
        not isinstance(initial_checkpoints, Mapping)
        or not initial_checkpoints
        or any(type(seed) is not int or seed < 0 for seed in initial_checkpoints)
    ):
        raise ValueError("initial_checkpoints requires nonnegative integer seed mapping")
    for value in (config.compute_seconds, config.wall_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("budgets must be finite and positive")
    if config.deadline is not None and (
        type(config.deadline) not in (int, float) or not math.isfinite(config.deadline)
    ):
        raise ValueError("deadline must be finite absolute monotonic time")
    for value in (
        config.max_decisions_per_seed,
        config.rollout_size,
        config.evaluation_interval,
        config.max_game_steps,
    ):
        if type(value) is not int or value < 1:
            raise ValueError("positive integer decision and game limits required")
    if config.max_decisions_per_seed > 65536:
        raise ValueError("curriculum caps choices per seed at 65536")
    if (
        not isinstance(config.ppo, PPOConfig)
        or config.ppo.epochs < 1
        or config.ppo.sequence_length < 1
    ):
        raise ValueError("valid PPOConfig required")


def run_curriculum(output_dir, *, initial_checkpoints, config=None) -> dict:
    """Run independent seed models; no demonstrations or final-test games.

    Compute includes constructor advancement, collection, profiling (the first
    rollout) and PPO updates. Evaluations share the outer wall/absolute deadline.
    Each of three (four with frozen diversity) allocations gets an equal share of
    each seed's allowance. A failed gate spends the next allocation at that stage.
    """
    config = config or CurriculumConfig()
    _validate_config(config, initial_checkpoints)
    seeds = sorted(initial_checkpoints)
    boards = {seed: [8000000 + 100000 * seed + i for i in range(4096)] for seed in seeds}
    validate_seed_splits(
        [b for values in boards.values() for b in values], config.validation_board_seeds, FINAL_TEST
    )
    source = source_identity()
    if (
        source.get("dirty") is not False
        or not source.get("revision")
        or source.get("content_identity") != "git:" + source["revision"]
    ):
        raise ValueError("curriculum requires known clean committed source")
    inputs = {str(seed): _artifact(Path(initial_checkpoints[seed])) for seed in seeds}
    with _preserve_rng():
        for path in initial_checkpoints.values():
            policy, _ = load_checkpoint(path)
            if any(not torch.isfinite(value).all() for value in policy.state_dict().values()):
                raise ValueError("initial checkpoint contains nonfinite parameters")
        historical = [
            load_checkpoint(path)[0].eval().requires_grad_(False)
            for path in config.historical_checkpoints
        ]
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be empty; refusing to overwrite evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "progress").mkdir()
    started = time.monotonic()
    deadline = min(
        started + config.wall_seconds, config.deadline if config.deadline is not None else math.inf
    )
    allowance = config.compute_seconds / len(seeds)
    stages = ["random", "one_heuristic", "heuristic"] + (["historical"] if historical else [])
    manifest = {
        "format": "colonist-rl.curriculum",
        "version": 1,
        "source": source,
        "config": asdict(config),
        "inputs": inputs,
        "historical_inputs": [_artifact(Path(p)) for p in config.historical_checkpoints],
        "compute_seconds_per_seed": allowance,
        "stages": stages,
        "lineups": {stage: list(LINEUPS[stage]) for stage in stages},
        "seed_splits": {
            "training_initialization": seeds,
            "training_board_candidates": {str(s): boards[s] for s in seeds},
            "fixed_validation": list(config.validation_board_seeds),
            "final_test_reserved": list(FINAL_TEST),
        },
        "runtime": {"device": "cpu", "torch_threads": 1, "workers": 1},
        "first_profile_counts_inside_compute_budget": True,
        "historical_note": "Optional frozen diversity; no stronger-opponent claim.",
        "selection": "fixed heuristic VP, win rate, earliest; uncensored <=1VP regression",
    }
    manifest["configuration_sha256"] = manifest_digest(manifest)
    _write(output / "manifest.json", manifest)
    result = {"status": "running", "manifest": manifest, "seeds": {}, "game_progress": {}}
    prior_threads = torch.get_num_threads()

    def persist():
        result["elapsed_seconds"] = time.monotonic() - started
        _write(output / "results.json", result)

    def evaluate(policy, seed, name, stage):
        key = f"{seed}-{name}"
        path = output / "progress" / (key + ".json")
        progress = {
            "status": "running",
            "completed_outcomes": [],
            "finished_games": 0,
            "aggregation_scope": "raw_completed_games",
        }

        def journal(row=None):
            if row is not None:
                progress["completed_outcomes"].append(row)
                progress["finished_games"] += 1
            _write(path, progress)
            result["game_progress"][key] = {
                "path": str(path.resolve()),
                "finished_games": progress["finished_games"],
                "status": progress["status"],
            }
            persist()

        journal()
        if _expired(deadline):
            progress["status"] = "skipped_deadline"
            journal()
            return None
        try:
            value = evaluate_blocks(
                policy=policy,
                board_seeds=config.validation_board_seeds,
                baseline="heuristic" if stage == "heuristic" else "random",
                opponent_lineup=LINEUPS[stage],
                historical_policies=historical,
                max_game_steps=config.max_game_steps,
                deadline=deadline,
                on_game_end=journal,
            )
            progress["status"] = value["status"]
            _write(output / (key + "-evaluation.json"), value)
            return value
        finally:
            if progress["status"] == "running":
                progress["status"] = "failed"
            journal()

    try:
        with _preserve_rng():
            torch.set_num_threads(1)
            for seed in seeds:
                result["seeds"][str(seed)] = {
                    "stage": "random",
                    "exposed_stages": [],
                    "decisions": 0,
                    "completed_intervals": 0,
                    "compute_seconds": 0.0,
                    "compute_allowance_seconds": allowance,
                    "actual_board_seeds": [],
                    "dropped_partial_intervals": 0,
                    "checkpoints": [],
                    "allocations": [],
                    "completed_updates": 0,
                    "status": "pending",
                }
            persist()
            for seed in seeds:
                row = result["seeds"][str(seed)]
                policy, checkpoint = load_checkpoint(initial_checkpoints[seed])
                # Deliberately new optimizer/RNG: this is fresh PPO from supplied weights.
                optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo.learning_rate)
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.manual_seed(seed)
                row["initial_checkpoint_metadata"] = checkpoint["metadata"]
                initial_fixed = evaluate(policy, seed, "initial-fixed", "heuristic")
                stage_index = 0
                row["checkpoints"].append(
                    {
                        "id": f"{seed}/initial",
                        "order": 0,
                        "path": str(Path(initial_checkpoints[seed]).resolve()),
                        "decisions": 0,
                        "stage": "initial",
                        "fixed_result": initial_fixed,
                        "initial_fixed_result": initial_fixed,
                    }
                )
                candidate_order = 1

                def checkpoint_and_gate(
                    allocation,
                    final=False,
                    *,
                    seed=seed,
                    policy=policy,
                    optimizer=optimizer,
                    row=row,
                    initial_fixed=initial_fixed,
                ):
                    nonlocal candidate_order, stage_index
                    name = f"checkpoint-{candidate_order}"
                    path = output / f"seed-{seed}-{name}.pt"
                    save_checkpoint(
                        path,
                        policy,
                        optimizer,
                        row["decisions"],
                        metadata={
                            "source": source,
                            "seed": seed,
                            "stage": row["stage"],
                            "manifest_sha256": manifest["configuration_sha256"],
                            "initial_checkpoint": inputs[str(seed)],
                        },
                    )
                    fixed = evaluate(policy, seed, name + "-fixed", "heuristic")
                    current = (
                        fixed
                        if row["stage"] == "heuristic"
                        else evaluate(policy, seed, name + "-stage", row["stage"])
                    )
                    gate = promotion_gate(row["stage"], current, fixed, initial_fixed)
                    candidate = {
                        "id": f"{seed}/{name}",
                        "order": candidate_order,
                        "path": str(path.resolve()),
                        "decisions": row["decisions"],
                        "stage": row["stage"],
                        "fixed_result": fixed,
                        "initial_fixed_result": initial_fixed,
                        "gate": gate,
                    }
                    row["checkpoints"].append(candidate)
                    candidate_order += 1
                    allocation["gate"] = gate
                    # Promotion occurs only at allocation end; the next collector
                    # starts a new episode with the new immutable lineup snapshot.
                    if final and gate["promote"] and stage_index + 1 < len(stages):
                        stage_index += 1
                        row["stage"] = stages[stage_index]
                    persist()

                for allocation_index in range(len(stages)):
                    if (
                        _expired(deadline)
                        or row["decisions"] >= config.max_decisions_per_seed
                        or row["compute_seconds"] >= allowance
                    ):
                        break
                    if len(row["actual_board_seeds"]) >= len(boards[seed]):
                        row["stop_reason"] = "board_schedule_exhausted"
                        break
                    allocation = {
                        "index": allocation_index,
                        "stage": row["stage"],
                        "status": "running",
                        "decisions": 0,
                    }
                    row["allocations"].append(allocation)
                    budget = min(allowance / len(stages), allowance - row["compute_seconds"])
                    end = min(deadline, time.monotonic() + budget)
                    compute_start = time.monotonic()
                    collector = RolloutCollector(
                        policy,
                        seed=seed,
                        max_game_steps=config.max_game_steps,
                        historical_policies=historical,
                        opponent_lineup=LINEUPS[row["stage"]],
                        board_seeds=boards[seed][len(row["actual_board_seeds"]) :],
                        deadline=end,
                    )
                    row["actual_board_seeds"].extend(collector.actual_board_seeds)
                    tracked_boards = len(collector.actual_board_seeds)
                    tracked_drops = 0
                    spent = time.monotonic() - compute_start
                    row["compute_seconds"] += spent
                    allocation_spent = spent
                    allocation_cap = math.ceil(config.max_decisions_per_seed / len(stages))
                    next_evaluation = (
                        (row["decisions"] // config.evaluation_interval) + 1
                    ) * config.evaluation_interval
                    while (
                        allocation_spent < budget
                        and not _expired(deadline)
                        and row["decisions"] < config.max_decisions_per_seed
                        and allocation["decisions"] < allocation_cap
                    ):
                        count = min(
                            config.rollout_size,
                            config.max_decisions_per_seed - row["decisions"],
                            allocation_cap - allocation["decisions"],
                            next_evaluation - row["decisions"],
                        )
                        compute_start = time.monotonic()
                        end = min(deadline, compute_start + max(0, budget - allocation_spent))
                        records = collector.collect(count, deadline=end)
                        row["actual_board_seeds"].extend(
                            collector.actual_board_seeds[tracked_boards:]
                        )
                        tracked_boards = len(collector.actual_board_seeds)
                        if records and allocation["stage"] not in row["exposed_stages"]:
                            row["exposed_stages"].append(allocation["stage"])
                        dropped = collector.dropped_partial_intervals - tracked_drops
                        tracked_drops = collector.dropped_partial_intervals
                        row["dropped_partial_intervals"] += dropped
                        row["decisions"] += len(records) + dropped
                        row["completed_intervals"] += len(records)
                        allocation["decisions"] += len(records) + dropped
                        if records and not _expired(end):
                            metrics = ppo_update(
                                policy, optimizer, records, config.ppo, deadline=end
                            )
                            row["completed_updates"] += metrics["completed_updates"]
                            allocation["last_update"] = metrics
                        spent = time.monotonic() - compute_start
                        row["compute_seconds"] += spent
                        allocation_spent += spent
                        if "first_rollout_profile" not in row:
                            row["first_rollout_profile"] = {
                                "seconds": spent,
                                "completed_decisions": len(records),
                                "included_in_compute_seconds": True,
                            }
                        persist()
                        if not records or collector.stop_reason is not None:
                            break
                        if row["decisions"] >= next_evaluation:
                            checkpoint_and_gate(allocation)
                            next_evaluation += config.evaluation_interval
                    allocation.update(
                        status="complete"
                        if allocation["decisions"] >= allocation_cap
                        else "partial",
                        compute_seconds=allocation_spent,
                        stop_reason=collector.stop_reason,
                    )
                    checkpoint_and_gate(allocation, final=True)
                path = output / f"seed-{seed}-last.pt"
                save_checkpoint(
                    path,
                    policy,
                    optimizer,
                    row["decisions"],
                    metadata={
                        "seed": seed,
                        "source": source,
                        "stage": row["stage"],
                        "manifest_sha256": manifest["configuration_sha256"],
                        "actual_board_seeds": row["actual_board_seeds"],
                        "dropped_partial_intervals": row["dropped_partial_intervals"],
                    },
                )
                row["last_checkpoint"] = _artifact(path)
                row["selection"] = select_checkpoint(row["checkpoints"])
                row["status"] = (
                    "complete"
                    if (
                        row["decisions"] >= config.max_decisions_per_seed
                        and row["dropped_partial_intervals"] == 0
                        and row["completed_updates"] > 0
                    )
                    else "partial"
                )
                persist()
        result["status"] = (
            "complete"
            if all(row["status"] == "complete" for row in result["seeds"].values())
            else "partial"
        )
    except Exception as error:  # noqa: BLE001 -- retain durable partial study evidence
        result["status"] = "failed"
        result["failure"] = f"{type(error).__name__}: {error}"
    finally:
        torch.set_num_threads(prior_threads)
        result["source_at_end"] = source_identity()
        if result["source_at_end"] != source:
            result["status"] = "failed"
            result["source_error"] = "source changed during curriculum"
        persist()
    return result
