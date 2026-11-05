"""Reproducible evaluation with complete four-seat board blocks."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from copy import deepcopy

import numpy as np
import torch

from .greedy_no_trade import GreedyNoTrade
from .opponents import HeuristicPolicy, RandomPolicy
from .simulator import FourPlayerGame
from .training import _lineup, _MatchDriver, _wilson


def _seed_list(seeds):
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("seeds must be a nonempty explicit list of unique integers")
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique integers (booleans are not seeds)")
    return list(seeds)


def validate_seed_splits(training, validation, final_test) -> None:
    """Reject empty, duplicate, non-integer, or overlapping seed partitions."""
    splits = [set(_seed_list(seeds)) for seeds in (training, validation, final_test)]
    if any(splits[i] & splits[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("training, validation, and final-test seeds must be disjoint")


def evaluate_blocks(
    policy=None,
    *,
    board_seeds,
    candidate="heuristic",
    baseline="heuristic",
    max_game_steps=20000,
    deterministic=True,
    deadline=None,
    on_game_end=None,
    opponent_lineup=None,
    historical_policies=(),
) -> dict:
    """Evaluate all four seats per board; ``deadline`` is absolute monotonic time.

    Only complete blocks enter ``outcomes`` and summary metrics. Finished games
    from an interrupted block are retained separately in ``partial_outcomes``.
    A step-limited game is a completed protocol game with ``truncated=True``
    and unresolved ``won=None``. Strict win metrics require all games resolved.
    ``on_game_end`` optionally receives a deep copy of each completed game row,
    before the next game starts. It includes step-truncated games, excludes
    interrupted games, and does not change block aggregation. Callback errors
    propagate after external RNG state and every policy module mode are restored.
    """
    seeds = _seed_list(board_seeds)
    return _evaluate_matches(
        policy,
        [(seed, seat) for seed in seeds for seat in range(4)],
        board_seeds=seeds,
        candidate=candidate,
        baseline=baseline,
        max_game_steps=max_game_steps,
        deterministic=deterministic,
        deadline=deadline,
        strict_blocks=True,
        on_game_end=on_game_end,
        opponent_lineup=opponent_lineup,
        historical_policies=historical_policies,
    )


def _evaluate_matches(
    policy,
    schedule,
    *,
    board_seeds,
    candidate,
    baseline,
    max_game_steps,
    deterministic,
    deadline,
    strict_blocks,
    reject_trades=False,
    on_game_end=None,
    opponent_lineup=None,
    historical_policies=(),
):
    """The authoritative evaluation match loop, also used by training.evaluate."""
    if candidate not in ("heuristic", "random", "graph", "greedy_no_trade") or (
        candidate == "graph" and policy is None
    ):
        raise ValueError(
            "candidate must be heuristic, random, greedy_no_trade, or a supplied graph policy"
        )
    if (
        candidate == "greedy_no_trade"
        and policy is not None
        and not isinstance(policy, GreedyNoTrade)
    ):
        raise ValueError("greedy_no_trade candidate requires no policy or a GreedyNoTrade actor")
    if baseline not in ("heuristic", "random", "mixed", "greedy_no_trade"):
        raise ValueError("a known opponent baseline is required")
    if type(max_game_steps) is not int or max_game_steps < 1:
        raise ValueError("max_game_steps must be a positive integer")
    if type(deterministic) is not bool:
        raise ValueError("deterministic must be a boolean")
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise ValueError("deadline must be a finite absolute monotonic time")
    if on_game_end is not None and not callable(on_game_end):
        raise ValueError("on_game_end must be callable or None")
    opponent_lineup = _lineup(opponent_lineup)
    historical = tuple(deepcopy(p).eval().requires_grad_(False) for p in historical_policies)
    if opponent_lineup is not None and "historical" in opponent_lineup and not historical:
        raise ValueError("historical lineup requires a frozen policy pool")
    if baseline == "greedy_no_trade" and historical and opponent_lineup is None:
        raise ValueError(
            "greedy_no_trade with historical policies requires an explicit opponent_lineup"
        )
    greedy_candidate = isinstance(policy, GreedyNoTrade) or candidate == "greedy_no_trade"
    identity = (
        "greedy_no_trade" if greedy_candidate else "graph" if policy is not None else candidate
    )
    action_mode = (
        "deterministic_greedy_no_trade"
        if greedy_candidate
        else ("deterministic" if deterministic else "stochastic")
        if policy is not None
        else ("seeded_" + candidate)
    )
    external_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    modes = [(module, module.training) for module in policy.modules()] if policy is not None else []
    finished = []
    interrupted = None
    expired = False
    try:
        if policy is not None:
            policy.eval()
        for game_index, (board_seed, seat) in enumerate(schedule):
            if deadline is not None and time.monotonic() >= deadline:
                expired = True
                break
            action_seed = (board_seed * 4 + seat) % (2**63)
            torch.manual_seed(action_seed)
            random.seed(action_seed)
            np.random.seed(action_seed % (2**32))
            env = FourPlayerGame(max_steps=max_game_steps)
            env.reset(seed=board_seed)
            driver = _MatchDriver(
                env,
                seat,
                board_seed,
                historical,
                baseline=baseline,
                reject_trades=reject_trades,
                opponent_lineup=opponent_lineup,
            )
            actor = (
                (
                    GreedyNoTrade()
                    if candidate == "greedy_no_trade"
                    else HeuristicPolicy(board_seed + 31 + seat)
                    if candidate == "heuristic"
                    else RandomPolicy(board_seed + 31 + seat)
                )
                if policy is None
                else policy
            )
            memory = None
            phases, kinds = Counter(), Counter()
            latency = []
            decisions = non_forced = 0
            result = None
            while result is None or not (result.terminated or result.truncated):
                if deadline is not None and time.monotonic() >= deadline:
                    expired = True
                    break
                driver.maybe_offer(deadline=deadline)
                if deadline is not None and time.monotonic() >= deadline:
                    expired = True
                    break
                # Offers consume a simulator version without setting its step
                # result's truncation flag. Do not apply one extra action.
                if env.version >= max_game_steps:
                    break
                view = driver.view
                if deadline is not None and time.monotonic() >= deadline:
                    expired = True
                    break
                if view.player_id == seat:
                    start = time.perf_counter()
                    if (
                        reject_trades
                        and not greedy_candidate
                        and view.context.kind == "TRADE_RESPONSE"
                    ):
                        action = next(a for a in view.legal_actions if a.kind == "REJECT_TRADE")
                        if policy is not None:
                            with torch.no_grad():
                                if deadline is not None and time.monotonic() >= deadline:
                                    expired = True
                                    break
                                memory = policy(view, memory).memory.detach()
                    elif policy is not None:
                        if deadline is not None and time.monotonic() >= deadline:
                            expired = True
                            break
                        action, memory = actor.act(view, memory, deterministic=deterministic)
                    else:
                        if deadline is not None and time.monotonic() >= deadline:
                            expired = True
                            break
                        action = actor.select_action(view) if greedy_candidate else actor.act(view)
                    latency.append(time.perf_counter() - start)
                    decisions += 1
                    non_forced += int(len(view.legal_actions) > 1)
                    phases[view.context.kind] += 1
                    kinds[action.kind] += 1
                else:
                    action = driver.opponent_action(deadline=deadline)
                if deadline is not None and time.monotonic() >= deadline:
                    expired = True
                    break
                result = driver.step(action)
            if expired:
                interrupted = {
                    "board_seed": board_seed,
                    "seat": seat,
                    "steps": env.version,
                    "learner_decisions": decisions,
                    "non_forced_decisions": non_forced,
                    "phase_counts": dict(phases),
                    "action_kind_counts": dict(kinds),
                    **_latency_summary(latency),
                }
                break
            # A final offer can reach the cap before another StepResult exists.
            terminated = bool(result is not None and result.terminated)
            truncated = bool(result is not None and result.truncated) or (
                env.version >= max_game_steps and not terminated
            )
            winner = result.winner if result is not None else None
            # Private simulator diagnostics are read only after the game has ended.
            row = {
                "game": game_index,
                "board_seed": board_seed,
                "seat": seat,
                "won": (
                    winner == seat if terminated and not truncated and winner is not None else None
                ),
                "winner": winner,
                "victory_points": env._actual_vp()[seat],
                "terminated": terminated,
                "truncated": truncated,
                "steps": env.version,
                "trade_responses": driver.trade_responses,
                "offers": driver.offers,
                "learner_decisions": decisions,
                "non_forced_decisions": non_forced,
                "phase_counts": dict(phases),
                "action_kind_counts": dict(kinds),
                **_latency_summary(latency),
            }
            finished.append((row, latency))
            if on_game_end is not None:
                # Observers receive only final diagnostic primitives, detached
                # from the rows used for strict block statistics. Keep this in
                # the restoration boundary so persistence errors propagate safely.
                on_game_end(deepcopy(row))
    finally:
        random.setstate(external_rng[0])
        np.random.set_state(external_rng[1])
        torch.set_rng_state(external_rng[2])
        for module, training in modes:
            module.training = training
    seats_by_board = {}
    for row, _ in finished:
        seats_by_board.setdefault(row["board_seed"], set()).add(row["seat"])
    complete_seeds = [seed for seed in board_seeds if seats_by_board.get(seed) == {0, 1, 2, 3}]
    selected, partial = [], []
    selected_latency = []
    for row, latency in finished:
        if strict_blocks and row["board_seed"] not in complete_seeds:
            partial.append(row)
        else:
            selected.append(row)
            selected_latency.extend(latency)
    games = len(selected)
    wins = sum(row["won"] is True for row in selected)
    resolved_games = sum(row["won"] is not None for row in selected)
    unresolved_games = games - resolved_games
    # Strict benchmarks do not estimate wins from a censored subset of games.
    win_metrics_defined = not strict_blocks or (games > 0 and unresolved_games == 0)
    phase_counts, action_counts = Counter(), Counter()
    for row in selected:
        phase_counts.update(row["phase_counts"])
        action_counts.update(row["action_kind_counts"])
    return {
        "status": "incomplete" if expired else "complete",
        "incomplete_reason": "deadline" if expired else None,
        "requested_board_seeds": list(board_seeds),
        "completed_board_seeds": complete_seeds,
        "requested_games": len(schedule),
        "games": games,
        "finished_games": len(finished),
        "aggregation_scope": "complete_blocks" if strict_blocks else "finished_games",
        "wins": wins,
        "resolved_games": resolved_games,
        "unresolved_games": unresolved_games,
        "win_rate_reporting": (
            "strict_all_games_resolved" if strict_blocks else "legacy_truncations_as_losses"
        ),
        "win_rate_denominator": games if win_metrics_defined else None,
        "win_rate": (wins / games if games else 0.0) if win_metrics_defined else None,
        "win_rate_ci95_nominal_wilson": (_wilson(wins, games) if win_metrics_defined else None),
        "confidence_interval_note": "Nominal Wilson interval; games sharing a board are dependent.",
        "mean_victory_points": (
            sum(row["victory_points"] for row in selected) / games if games else 0.0
        ),
        "victory_points_note": (
            "Observed endpoint victory points, including truncations; not final-game competence."
        ),
        "truncations": sum(row["truncated"] for row in selected),
        "terminations": sum(row["terminated"] for row in selected),
        "trade_responses": sum(row["trade_responses"] for row in selected),
        "offers": sum(row["offers"] for row in selected),
        "learner_decisions": sum(row["learner_decisions"] for row in selected),
        "non_forced_decisions": sum(row["non_forced_decisions"] for row in selected),
        "phase_counts": dict(phase_counts),
        "action_kind_counts": dict(action_counts),
        **_latency_summary(selected_latency),
        "candidate": identity,
        "action_mode": action_mode,
        **({"candidate_identity": GreedyNoTrade().identity} if greedy_candidate else {}),
        **(
            {"baseline_identity": GreedyNoTrade().identity}
            if baseline == "greedy_no_trade" and opponent_lineup is None
            else {}
        ),
        **(
            {
                "opponent_lineup_identities": [
                    GreedyNoTrade().identity if name == "greedy_no_trade" else None
                    for name in opponent_lineup
                ]
            }
            if opponent_lineup is not None and "greedy_no_trade" in opponent_lineup
            else {}
        ),
        "deterministic": deterministic,
        "baseline": baseline,
        "opponent_lineup": list(opponent_lineup) if opponent_lineup is not None else None,
        "reject_trades": reject_trades,
        "max_game_steps": max_game_steps,
        "outcomes": selected,
        "partial_outcomes": partial,
        "interrupted_game": interrupted,
    }


def _latency_summary(seconds):
    return {
        "decision_latency_ms_mean": 1000 * float(np.mean(seconds)) if seconds else 0.0,
        "decision_latency_ms_p95": 1000 * float(np.percentile(seconds, 95)) if seconds else 0.0,
    }


def _paired_rows(result):
    if result.get("status", "complete") != "complete" or result.get("partial_outcomes"):
        raise ValueError("paired comparison requires complete evaluation results")
    rows = {}
    for row in result["outcomes"]:
        seed, seat = row["board_seed"], row["seat"]
        if type(seed) is not int or type(seat) is not int or seat not in range(4):
            raise ValueError("pair keys require integer board seeds and seats 0..3")
        key = (seed, seat)
        if key in rows:
            raise ValueError("duplicate board/seat pair")
        if (
            row.get("truncated", False)
            or row.get("terminated") is False
            or ("winner" in row and row["winner"] is None)
            or row["won"] is None
        ):
            raise ValueError("paired comparison requires resolved, non-truncated outcomes")
        if type(row["won"]) is not bool or not isinstance(row["victory_points"], (int, float)):
            raise ValueError("invalid paired outcome")
        if not math.isfinite(row["victory_points"]):
            raise ValueError("paired victory points must be finite")
        rows[key] = row
    seeds = {seed for seed, _ in rows}
    if not rows or set(rows) != {(seed, seat) for seed in seeds for seat in range(4)}:
        raise ValueError("missing pairs: every board requires all four seats")
    return rows


def paired_block_summary(reference: dict, candidate: dict) -> dict:
    """Board-averaged candidate-minus-reference differences; reject censored games."""
    reference_rows, candidate_rows = _paired_rows(reference), _paired_rows(candidate)
    if reference_rows.keys() != candidate_rows.keys():
        raise ValueError("missing matching board/seat pairs between results")
    boards = []
    for seed in sorted({seed for seed, _ in reference_rows}):
        pairs = [(reference_rows[seed, seat], candidate_rows[seed, seat]) for seat in range(4)]
        boards.append(
            {
                "board_seed": seed,
                "win_rate_difference": sum(int(c["won"]) - int(r["won"]) for r, c in pairs) / 4,
                "mean_victory_points_difference": sum(
                    c["victory_points"] - r["victory_points"] for r, c in pairs
                )
                / 4,
            }
        )
    return {
        "paired_boards": len(boards),
        "paired_games": len(reference_rows),
        "boards": boards,
        "win_rate_difference": sum(b["win_rate_difference"] for b in boards) / len(boards),
        "mean_victory_points_difference": sum(b["mean_victory_points_difference"] for b in boards)
        / len(boards),
        "comparison_note": "Descriptive board-level differences; no significance claim.",
    }
