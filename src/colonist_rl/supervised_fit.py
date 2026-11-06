"""Observation-only ordered supervised fitting, shared by dataset adapters.

Independent sequence minibatches are opt-in. Every update replays its own prefixes
under the current parameters; learned embeddings and memories are never cached.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter, deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import record_function

from .contracts import Observation
from .graph_inputs import encoding_seconds
from .policy import KINDS

SupervisedSample = tuple[Observation, int]
EpisodeSequence = tuple[SupervisedSample, ...]


@dataclass(frozen=True)
class ApprovedTarget:
    """Explicit informative target; dataset approval provenance is caller-owned."""

    accepted_indices: tuple[int, ...]
    _: KW_ONLY
    role: str
    train_approved: bool = False
    val_approved: bool = False
    legacy: bool = False


def _validate_accepted_indices(indices, menu_size):
    if (
        type(indices) is not tuple
        or not indices
        or len(indices) >= menu_size
        or any(type(i) is not int or not 0 <= i < menu_size for i in indices)
        or tuple(sorted(set(indices))) != indices
    ):
        raise ValueError("accepted indices must be a canonical nonempty proper legal subset")


def accepted_set_cross_entropy(logits, accepted_indices):
    """Negative log total accepted probability; singleton is ordinary CE exactly."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise ValueError("accepted-set logits must be a one-dimensional tensor")
    _validate_accepted_indices(accepted_indices, len(logits))
    if len(accepted_indices) == 1:
        return F.cross_entropy(
            logits.unsqueeze(0), logits.new_tensor(accepted_indices, dtype=torch.long)
        )
    return -F.log_softmax(logits, dim=0)[list(accepted_indices)].logsumexp(0)


def _validate_approved_targets(episodes, targets, role):
    if type(targets) is not tuple or len(targets) != len(episodes):
        raise ValueError("approved targets must parallel the episode tuple")
    for episode, rows in zip(episodes, targets):
        if type(rows) is not tuple or len(rows) != len(episode):
            raise ValueError("approved targets must parallel every saved row")
        for (view, original_target), target in zip(episode, rows):
            if target is None:
                continue
            if not isinstance(target, ApprovedTarget):
                raise ValueError("approved targets must be ApprovedTarget records or None")
            _validate_accepted_indices(target.accepted_indices, len(view.legal_actions))
            if (
                type(target.role) is not str
                or target.role != role
                or any(
                    type(flag) is not bool
                    for flag in (target.train_approved, target.val_approved, target.legacy)
                )
            ):
                raise ValueError("target role or approval flags do not permit this operation")
            if target.legacy:
                valid = (
                    role == "TRAIN"
                    and not target.train_approved
                    and not target.val_approved
                    and target.accepted_indices == (original_target,)
                )
            else:
                valid = (
                    (role == "TRAIN" and target.train_approved and not target.val_approved)
                    or (role == "VAL" and target.val_approved and not target.train_approved)
                )
            if not valid:
                raise ValueError(
                    "target requires the matching explicit approval or legacy singleton"
                )


def _validate_anchor_schedule(episodes, targets, schedule):
    if type(schedule) is not tuple or not schedule:
        raise ValueError("anchor schedule must be a nonempty ordered tuple")
    for anchor in schedule:
        if (
            type(anchor) is not tuple
            or len(anchor) != 2
            or any(type(i) is not int for i in anchor)
            or not 0 <= anchor[0] < len(episodes)
            or not 0 <= anchor[1] < len(episodes[anchor[0]])
            or targets[anchor[0]][anchor[1]] is None
        ):
            raise ValueError("scheduled anchor must reference an approved nonforced TRAIN row")


def _validate_sequences(episodes):
    if not isinstance(episodes, tuple):
        raise ValueError("episodes must be an ordered tuple of sequences")  # noqa: TRY004
    phases, kinds = Counter(), Counter()
    decisions = non_forced = 0
    for episode in episodes:
        if not isinstance(episode, tuple):
            raise ValueError("episode samples must be an ordered tuple")  # noqa: TRY004
        previous = -1
        for sample in episode:
            if not isinstance(sample, tuple) or len(sample) != 2:
                raise ValueError("samples must contain only observation and legal index")
            view, target = sample
            if (
                not isinstance(view, Observation)
                or view.schema_version != 1
                or view.context.actor != view.player_id
                or type(view.context.version) is not int
                or view.context.version <= previous
                or type(target) is not int
                or not 0 <= target < len(view.legal_actions)
                or any(action.kind not in KINDS for action in view.legal_actions)
            ):
                raise ValueError("invalid observation or legal teacher index")
            previous = view.context.version
            decisions += 1
            non_forced += int(len(view.legal_actions) > 1)
            phases[view.context.kind] += 1
            kinds[view.legal_actions[target].kind] += 1
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


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


class _DeadlineExpired(Exception):
    pass


def _check_deadline(deadline):
    if _expired(deadline):
        raise _DeadlineExpired


@contextmanager
def _scope(name, timing):
    started = time.perf_counter() if timing is not None else None
    try:
        with record_function(f"supervised_fit.{name}"):
            yield
    finally:
        if timing is not None:
            timing[name] += time.perf_counter() - started


def _batches(episodes, sequence_length, batch_size):
    if batch_size == 1:
        # This is the original episode-major update order, including forced chunks.
        for episode in episodes:
            for start in range(0, len(episode), sequence_length):
                yield ((episode, start),)
        return
    pending = deque((episode, 0) for episode in episodes if episode)
    while pending:
        # Snapshot the active count so a short queue never repeats an episode.
        batch = tuple(pending.popleft() for _ in range(min(batch_size, len(pending))))
        for episode, start in batch:
            following = start + sequence_length
            if following < len(episode):
                pending.append((episode, following))
        yield batch


def _memory_many(policy, views, memories):
    many = getattr(policy, "recurrent_many", None)
    step = getattr(policy, "recurrent_step", None)
    if len(views) > 1 and callable(many):
        return many(views, memories)
    if callable(step):
        return tuple(step(v, m) for v, m in zip(views, memories))
    return tuple(policy(v, m).memory for v, m in zip(views, memories))


def _prefixes(policy, batch, deadline, counters):
    memories = [None] * len(batch)
    with torch.no_grad():
        for step in range(max(start for _, start in batch)):
            _check_deadline(deadline)
            active = [i for i, (_, start) in enumerate(batch) if step < start]
            views = tuple(batch[i][0][step][0] for i in active)
            outputs = _memory_many(policy, views, tuple(memories[i] for i in active))
            for i, memory in zip(active, outputs):
                memories[i] = memory.detach()
            counters["prefix_forwards"] += len(active)
    return memories


def _ce(output, target, view, phase_weights):
    return F.cross_entropy(
        output.logits.unsqueeze(0), output.logits.new_tensor([target], dtype=torch.long)
    ) * (1.0 if phase_weights is None else phase_weights[view.context.kind])


def _chunk_losses(policy, batch, memories, sequence_length, deadline, phase_weights, counters):
    chunks = [episode[start : start + sequence_length] for episode, start in batch]
    losses = []
    many = getattr(policy, "forward_many", None)
    if len(batch) == 1 or not callable(many):
        # Custom policies retain ordinary forward semantics and serial CE sums.
        for chunk, memory in zip(chunks, memories):
            for view, target in chunk:
                _check_deadline(deadline)
                if len(view.legal_actions) == 1:
                    memory = _memory_many(policy, (view,), (memory,))[0]
                else:
                    output = policy(view, memory)
                    memory = output.memory
                    losses.append(_ce(output, target, view, phase_weights))
                counters["chunk_forwards"] += 1
        return losses
    for step in range(max(map(len, chunks))):
        _check_deadline(deadline)
        active = [i for i, chunk in enumerate(chunks) if step < len(chunk)]
        forced = [i for i in active if len(chunks[i][step][0].legal_actions) == 1]
        choices = [i for i in active if i not in forced]
        if forced:
            views = tuple(chunks[i][step][0] for i in forced)
            values = _memory_many(policy, views, tuple(memories[i] for i in forced))
            for i, memory in zip(forced, values):
                memories[i] = memory
        if choices:
            views = tuple(chunks[i][step][0] for i in choices)
            outputs = many(views, tuple(memories[i] for i in choices))
            for i, output in zip(choices, outputs):
                view, target = chunks[i][step]
                memories[i] = output.memory
                losses.append(_ce(output, target, view, phase_weights))
        counters["chunk_forwards"] += len(active)
    return losses


def _anchor_loss(policy, episode, start, anchor, memory, target, deadline, counters):
    # Every intervening row contributes differentiable recurrence, never CE.
    for view, _ in episode[start:anchor]:
        _check_deadline(deadline)
        memory = _memory_many(policy, (view,), (memory,))[0]
        counters["chunk_forwards"] += 1
    _check_deadline(deadline)
    output = policy(episode[anchor][0], memory)
    counters["chunk_forwards"] += 1
    return accepted_set_cross_entropy(output.logits, target.accepted_indices)


def fit_sequences(
    policy,
    episodes: tuple[EpisodeSequence, ...],
    *,
    epochs=2,
    sequence_length=16,
    learning_rate=3e-4,
    deadline=None,
    phase_weights=None,
    batch_size=1,
    optimizer=None,
    profile=False,
    approved_targets=None,
    anchor_schedule=None,
) -> dict:
    """Fit ordered chunks with fresh no-grad prefixes before every optimizer step.

    Weighted CE sums use non-forced choice count as the denominator. Batches
    contain at most one chunk from an episode, with one frozen-weight mean loss.
    Optional approved targets consume the caller schedule once, with only its
    anchor contributing CE through a suffix ending at that anchor.
    """
    coverage = _validate_sequences(episodes)
    _positive(epochs, "epochs")
    _positive(sequence_length, "sequence_length")
    _positive(batch_size, "batch_size")
    anchored = approved_targets is not None or anchor_schedule is not None
    if anchored:
        if approved_targets is None or anchor_schedule is None:
            raise ValueError("approved_targets and anchor_schedule must be supplied together")
        if epochs != 1 or batch_size != 1 or sequence_length > 16 or phase_weights is not None:
            raise ValueError(
                "anchored updates require epochs=1, batch_size=1, suffix<=16, no weights"
            )
        _validate_approved_targets(episodes, approved_targets, "TRAIN")
        _validate_anchor_schedule(episodes, approved_targets, anchor_schedule)
    if deadline is not None and (type(deadline) not in (float, int) or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite absolute monotonic time")
    if type(profile) is not bool:
        raise ValueError("profile must be a boolean")
    if phase_weights is not None:
        required = {v.context.kind for e in episodes for v, _ in e if len(v.legal_actions) > 1}
        if (
            not isinstance(phase_weights, Mapping)
            or not required.issubset(phase_weights)
            or any(
                not isinstance(k, str)
                or type(v) not in (int, float)
                or not math.isfinite(v)
                or v <= 0
                for k, v in phase_weights.items()
            )
        ):
            raise ValueError(
                "phase_weights must provide finite positive weights for every choice phase"
            )
    if (
        type(learning_rate) not in (float, int)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    else:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("optimizer must be a Torch optimizer for this policy")
        owned = {id(parameter) for parameter in policy.parameters()}
        supplied = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if not supplied or len(set(supplied)) != len(supplied) or set(supplied) != owned:
            raise ValueError("optimizer must contain every policy parameter exactly once")
    modes = [(module, module.training) for module in policy.modules()]
    counters = {"prefix_forwards": 0, "chunk_forwards": 0, "batches": 0, "batch_chunks": 0}
    timing = (
        dict.fromkeys(("encoding", "prefix", "chunk", "backward", "step"), 0.0) if profile else None
    )
    token = encoding_seconds.set(timing)
    started = time.perf_counter()
    updates = presentations = completed_epochs = 0
    total = 0.0
    expired = False
    try:
        policy.train()
        for _ in range(epochs):
            try:
                schedule = anchor_schedule if anchored else _batches(
                    episodes, sequence_length, batch_size
                )
                for scheduled in schedule:
                    _check_deadline(deadline)
                    if anchored:
                        episode_index, anchor = scheduled
                        episode = episodes[episode_index]
                        start = max(0, anchor - sequence_length + 1)
                        batch = ((episode, start),)
                        with _scope("prefix", timing):
                            memories = _prefixes(policy, batch, deadline, counters)
                        with _scope("chunk", timing):
                            losses = [
                                _anchor_loss(
                                    policy, episode, start, anchor, memories[0],
                                    approved_targets[episode_index][anchor], deadline, counters,
                                )
                            ]
                    else:
                        batch = tuple(
                            (e, start)
                            for e, start in scheduled
                            if any(
                                len(v.legal_actions) > 1
                                for v, _ in e[start : start + sequence_length]
                            )
                        )
                        if not batch:
                            continue
                        losses = []
                        # Without a batching API, retain each sequence's ordinary
                        # prefix-then-chunk call order under the same frozen weights.
                        groups = (
                            (batch,)
                            if callable(getattr(policy, "forward_many", None))
                            else ((entry,) for entry in batch)
                        )
                        for group in groups:
                            with _scope("prefix", timing):
                                memories = _prefixes(policy, group, deadline, counters)
                            with _scope("chunk", timing):
                                losses.extend(
                                    _chunk_losses(
                                        policy,
                                        group,
                                        memories,
                                        sequence_length,
                                        deadline,
                                        phase_weights,
                                        counters,
                                    )
                                )
                    _check_deadline(deadline)
                    loss = torch.stack(losses).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("imitation training loss is not finite")
                    with _scope("backward", timing):
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            policy.parameters(), 0.5, error_if_nonfinite=True
                        )
                    if _expired(deadline):
                        optimizer.zero_grad(set_to_none=True)
                        raise _DeadlineExpired
                    with _scope("step", timing):
                        optimizer.step()
                    total += loss.item() * len(losses)
                    presentations += len(losses)
                    updates += 1
                    counters["batches"] += 1
                    counters["batch_chunks"] += len(batch)
            except _DeadlineExpired:
                expired = True
                break
            completed_epochs += 1
    finally:
        encoding_seconds.reset(token)
        for module, training in modes:
            module.training = training
    result = {
        "loss": total / presentations if presentations else None,
        "decisions": coverage["decisions"],
        "epochs": completed_epochs,
        "requested_epochs": epochs,
        "updates": updates,
        "supervised_decisions": (
            len(set(anchor_schedule)) if anchored else coverage["non_forced_decisions"]
        ),
        "supervised_presentations": presentations,
        "status": "incomplete" if expired else "complete",
        "incomplete_reason": "deadline" if expired else None,
    }
    if profile:
        result["profile"] = {
            "scope_seconds": timing,
            **counters,
            "updates": updates,
            "presentations": presentations,
            "batch_size": batch_size,
            "wall_seconds": time.perf_counter() - started,
        }
    return result


def _empty_score():
    return {"count": 0, "correct": 0, "accuracy": None, "cross_entropy": None}


def score_sequences(
    policy, episodes: tuple[EpisodeSequence, ...], *, approved_targets=None
) -> dict:
    """Replay full histories; optional targets grant VAL-only accepted-set scoring."""
    coverage = _validate_sequences(episodes)
    if approved_targets is not None:
        _validate_approved_targets(episodes, approved_targets, "VAL")
    modes = [(module, module.training) for module in policy.modules()]
    totals = {"all": [], "non_forced": []}
    phases, kinds = {}, {}
    try:
        with _preserve_rng(), torch.no_grad():
            policy.eval()
            for episode_index, episode in enumerate(episodes):
                memory = None
                for row_index, (view, target) in enumerate(episode):
                    output = policy(view, memory)
                    memory = output.memory.detach()
                    if approved_targets is None:
                        ce = F.cross_entropy(
                            output.logits.unsqueeze(0),
                            output.logits.new_tensor([target], dtype=torch.long),
                        ).item()
                        correct = int(output.logits.argmax().item() == target)
                    else:
                        approved = approved_targets[episode_index][row_index]
                        if approved is None:
                            continue
                        ce = accepted_set_cross_entropy(
                            output.logits, approved.accepted_indices
                        ).item()
                        correct = int(output.logits.argmax().item() in approved.accepted_indices)
                    if not math.isfinite(ce):
                        raise FloatingPointError("imitation validation CE is not finite")
                    score = (correct, ce)
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
        "coverage": coverage,
        "note": (
            "Teacher agreement is supervised fit, not playing competence; "
            "forced choices are excluded from non_forced metrics."
            if approved_targets is None
            else "Approved-set agreement is not playing competence; action-kind groups use "
            "the original tuple target, not a unique approved action."
        ),
    }
