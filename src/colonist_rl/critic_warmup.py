"""Bounded value-only fitting on committed complete-terminal episode histories.

The supported built-in graph actors do not consume their independent value head.
Exact non-value tensor preservation therefore certifies actor logits and carried
memory without storing a feature cache or replaying after a deadline.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from .policy import (
    ConsequenceGraphPolicy,
    ContextLinearConsequencePolicy,
    GraphPolicy,
    NonlinearConsequencePolicy,
)
from .training import EpisodeTrace, _expired, _validate_deadline, _validate_trace, resolved_returns

_SUPPORTED = (
    GraphPolicy,
    ConsequenceGraphPolicy,
    ContextLinearConsequencePolicy,
    NonlinearConsequencePolicy,
)


def _validate_policy(policy):
    if type(policy) not in _SUPPORTED or type(policy.value) is not nn.Linear:
        raise ValueError(
            "warmup requires a supported built-in graph policy with a Linear value head"
        )
    value = tuple(policy.value.parameters())
    if (
        len(value) != 2
        or policy.value.in_features != policy.hidden_size
        or policy.value.out_features != 1
        or policy.value.bias is None
    ):
        raise ValueError("warmup requires an independent scalar value head")
    storage = {(p.device, p.untyped_storage().data_ptr()) for p in value}
    actor_tensors = [
        tensor
        for name, tensor in policy.named_parameters(remove_duplicate=False)
        if not name.startswith("value.")
    ] + list(policy.buffers())
    if len(storage) != 2 or any(
        (tensor.device, tensor.untyped_storage().data_ptr()) in storage for tensor in actor_tensors
    ):
        raise ValueError("value head must not share parameters or storage with actor/trunk/buffers")
    if torch.is_inference_mode_enabled():
        raise ValueError("warmup cannot fit inside torch.inference_mode")
    return value


def _prepare(episode_traces):
    if not isinstance(episode_traces, Mapping):
        raise ValueError("episode_traces must be a mapping of committed EpisodeTrace snapshots")  # noqa: TRY004
    items = tuple(episode_traces.items())
    identities, eligible, excluded, counts = set(), [], [], {}
    excluded_samples = excluded_observations = empty = 0
    for key, trace in items:
        if (
            not isinstance(trace, EpisodeTrace)
            or not isinstance(key, str)
            or key != trace.episode_key
            or key in identities
            or type(trace.began_at_reset) is not bool
            or type(trace.history_complete) is not bool
            or (
                trace.incomplete_reason is not None
                and (not isinstance(trace.incomplete_reason, str) or not trace.incomplete_reason)
            )
        ):
            raise ValueError("invalid or duplicate episode mapping identity/provenance")
        identities.add(key)
        # False completeness flags mean exclusion, never permission to skip the
        # existing structural identity/observation/interval binding validation.
        _validate_trace(replace(trace, began_at_reset=True, history_complete=True))
        if any(type(record.episode) is not int for record in trace.transitions):
            raise ValueError("invalid transition episode identity")
        reasons = []
        if not trace.began_at_reset or not trace.history_complete:
            reasons.append("incomplete_history")
        if not trace.finalized:
            reasons.append("pending")
        if trace.truncated:
            reasons.append("truncated")
        if trace.dropped_intervals:
            reasons.append("dropped_intervals")
        if (
            trace.finalized
            and not trace.terminated
            and not trace.truncated
            and not trace.dropped_intervals
        ):
            reasons.append("nonterminal")
        if trace.terminated and trace.winner is None:
            reasons.append("unknown_outcome")
        if reasons:
            excluded_samples += len(trace.transitions)
            excluded_observations += len(trace.observations)
            excluded.append(
                {
                    "episode_key": key,
                    "reasons": reasons,
                    "incomplete_reason": trace.incomplete_reason,
                    "samples": len(trace.transitions),
                    "observations": len(trace.observations),
                }
            )
            for reason in reasons:
                counts[reason] = counts.get(reason, 0) + 1
            continue
        # Do not swallow corrupt supposedly-resolved provenance as an exclusion.
        targets = resolved_returns(trace)
        if any(not math.isfinite(target) for target in targets):
            raise ValueError("nonfinite resolved return")
        eligible.append((trace, targets))
        empty += not targets
    if tuple((key, id(trace)) for key, trace in episode_traces.items()) != tuple(
        (key, id(trace)) for key, trace in items
    ):
        raise ValueError("episode mapping changed during validation")
    targets = [target for _, values in eligible for target in values]
    metrics = {
        "status": "no_data" if not targets else "complete",
        "epochs_started": 0,
        "epochs_completed": 0,
        "states_attempted": 0,
        "states_replayed": 0,
        "samples_optimized": 0,
        "optimizer_steps": 0,
        "squared_error_sum": 0.0,
        "squared_error_count": 0,
        "samples_discarded": 0,
        "batches_discarded": 0,
        "target_min": min(targets) if targets else None,
        "target_max": max(targets) if targets else None,
        "eligible_episodes": len(eligible),
        "eligible_samples": len(targets),
        "empty_episodes": empty,
        "excluded_episodes": len(excluded),
        "excluded_samples": excluded_samples,
        "excluded_observations": excluded_observations,
        "excluded": excluded,
        "exclusion_counts": counts,
    }
    return eligible, metrics


def _actor_tensors(policy):
    return {
        **{
            "parameter:" + name: tensor
            for name, tensor in policy.named_parameters(remove_duplicate=False)
            if not name.startswith("value.")
        },
        **{
            "buffer:" + name: tensor
            for name, tensor in policy.named_buffers(remove_duplicate=False)
        },
    }


def _same_bits(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


@contextmanager
def _preserve_caller(policy):
    parameters, modules = tuple(policy.parameters()), tuple(policy.modules())
    flags, modes = [p.requires_grad for p in parameters], [m.training for m in modules]
    gradients = [(p.grad, None if p.grad is None else p.grad.clone()) for p in parameters]
    actor = {
        name: (tensor, tensor.detach().clone()) for name, tensor in _actor_tensors(policy).items()
    }
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps_rng = torch.mps.get_rng_state() if policy.device.type == "mps" else None
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, flag, (gradient, original) in zip(parameters, flags, gradients):
                parameter.requires_grad_(flag)
                parameter.grad = gradient
                if gradient is not None and not _same_bits(gradient, original):
                    gradient.copy_(original)
            for module, mode in zip(modules, modes):
                module.training = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)
        current = _actor_tensors(policy)
        changed = sorted(
            name
            for name, (tensor, before) in actor.items()
            if name not in current
            or current[name] is not tensor
            or not _same_bits(before, current[name])
        ) + sorted(current.keys() - actor.keys())
        if changed:
            # Retain the changed tensors for diagnosis; never hide corruption or
            # claim a rollback. An in-flight exception remains chained as context.
            raise RuntimeError("warmup actor invariance failed: " + ", ".join(changed))


def warm_value_head(
    policy,
    episode_traces,
    *,
    epochs,
    batch_size=64,
    learning_rate=3e-4,
    deadline=None,
):
    """Return primitive accounting and a fresh value-only Adam; see docs/critic-warmup.md.

    Batches follow mapping/observation order, span episode boundaries and flush
    at each epoch end. Only detached recurrent outputs enter state-weighted MSE.
    Deadline checks are cooperative, matching the existing monotonic convention.
    """
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be an integer from 1 to 64")
    try:
        valid_rate = (
            type(learning_rate) in (int, float)
            and math.isfinite(learning_rate)
            and learning_rate > 0
        )
        _validate_deadline(deadline)
    except OverflowError as error:
        raise ValueError(
            "learning_rate and deadline must be finite representable scalars"
        ) from error
    if not valid_rate:
        raise ValueError("learning_rate must be finite and positive")
    value_parameters = _validate_policy(policy)
    eligible, metrics = _prepare(episode_traces)
    with _preserve_caller(policy):
        optimizer = torch.optim.Adam(value_parameters, lr=learning_rate)
        if not metrics["eligible_samples"]:
            return metrics, optimizer
        if _expired(deadline):
            metrics["status"] = "deadline"
            return metrics, optimizer
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        for parameter in value_parameters:
            parameter.requires_grad_(True)
        for module in policy.modules():
            module.training = False
        features, targets = [], []

        def stop():
            metrics["status"] = "deadline"
            metrics["samples_discarded"] += len(targets)
            metrics["batches_discarded"] += bool(targets)
            features.clear()
            targets.clear()

        def fit_batch():
            if _expired(deadline):
                return False
            with torch.enable_grad():
                inputs = torch.stack(features)
                truth = inputs.new_tensor(targets)
                errors = policy.value(inputs).squeeze(-1) - truth
                squared_error = errors.square().sum()
                if not torch.isfinite(squared_error).item():
                    raise ValueError("nonfinite critic warmup loss")
                optimizer.zero_grad(set_to_none=True)
                if _expired(deadline):
                    return False
                (squared_error / len(targets)).backward()
                if any(
                    p.grad is None or not torch.isfinite(p.grad).all().item()
                    for p in value_parameters
                ):
                    raise ValueError("nonfinite critic warmup gradients")
                if _expired(deadline):
                    return False
                optimizer.step()
            metrics["optimizer_steps"] += 1
            metrics["samples_optimized"] += len(targets)
            metrics["squared_error_sum"] += squared_error.detach().item()
            metrics["squared_error_count"] += len(targets)
            features.clear()
            targets.clear()
            return True

        for _ in range(epochs):
            if _expired(deadline):
                stop()
                break
            metrics["epochs_started"] += 1
            for trace, suffixes in eligible:
                memory = None
                for observation, target in zip(trace.observations, suffixes):
                    if _expired(deadline):
                        stop()
                        return metrics, optimizer
                    metrics["states_attempted"] += 1
                    with torch.no_grad():
                        memory = policy.recurrent_step(observation, memory).detach()
                    metrics["states_replayed"] += 1
                    features.append(memory)
                    targets.append(target)
                    if len(targets) == batch_size and not fit_batch():
                        stop()
                        return metrics, optimizer
            if targets and not fit_batch():
                stop()
                break
            metrics["epochs_completed"] += 1
        return metrics, optimizer
