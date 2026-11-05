"""Shared fitting keeps legacy updates and independent frozen-weight minibatches."""

import importlib
import math
import time
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from colonist_rl.imitation import collect_demonstrations, fit_imitation, imitation_metrics
from colonist_rl.policy import GraphPolicy, PolicyOutput


def shared():
    assert importlib.util.find_spec("colonist_rl.supervised_fit") is not None, (
        "shared observation-sequence fitter is missing"
    )
    return importlib.import_module("colonist_rl.supervised_fit")


@pytest.fixture
def episodes():
    torch.set_num_threads(1)
    result = []
    for seed, count in ((18, 5), (19, 3)):
        data = collect_demonstrations(board_seeds=[seed], max_decisions=count)
        sequence = []
        for i, sample in enumerate(data["episodes"][0]["samples"]):
            view, target = sample["observation"], sample["action_index"]
            if i in (0, 2):
                view, target = replace(view, legal_actions=view.legal_actions[:1]), 0
            sequence.append((view, target))
        result.append(tuple(sequence))
    return tuple(result)


class SensitivePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.calls = []

    def forward(self, view, memory=None):
        self.calls.append((view.context.version, self.weight.item(), torch.is_grad_enabled()))
        memory = self.weight * (view.context.version + 1) + (0 if memory is None else memory)
        logits = torch.stack([memory * (i + 1) for i in range(len(view.legal_actions))])
        return PolicyOutput(logits, memory * 0, memory)


def reference_update(policy, episode_chunks, optimizer, weights=None):
    """Serial SUM at frozen weights, independent of the fitter/batched policy APIs."""
    losses = []
    for samples, start, length in episode_chunks:
        memory = None
        with torch.no_grad():
            for view, _ in samples[:start]:
                memory = policy(view, memory).memory.detach()
        for view, target in samples[start : start + length]:
            output = policy(view, memory)
            memory = output.memory
            if len(view.legal_actions) > 1:
                losses.append(
                    F.cross_entropy(output.logits[None], torch.tensor([target]))
                    * (1 if weights is None else weights[view.context.kind])
                )
    if losses:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
        return loss.item(), len(losses)
    return 0, 0


def test_batch_one_matches_legacy_serial_updates_and_next_prefix_uses_new_weights(episodes):
    fitter = shared()
    policy, expected = SensitivePolicy(), SensitivePolicy()
    optimizer = torch.optim.Adam(expected.parameters(), lr=0.02)
    total = count = 0
    updates = 0
    for _ in range(2):
        for episode in episodes:
            for start in range(0, len(episode), 2):
                if not any(len(v.legal_actions) > 1 for v, _ in episode[start : start + 2]):
                    continue
                loss, n = reference_update(expected, [(episode, start, 2)], optimizer)
                total += loss * n
                count += n
                updates += 1
    actual = fitter.fit_sequences(
        policy, episodes, epochs=2, sequence_length=2, learning_rate=0.02, batch_size=1
    )
    assert actual["updates"] == updates == 6
    assert actual["supervised_presentations"] == count
    assert actual["loss"] == total / count
    assert policy.calls == expected.calls
    torch.testing.assert_close(policy.weight, expected.weight, rtol=0, atol=0)
    assert any(not grad and weight != 0.1 for _, weight, grad in policy.calls)


def test_minibatch_custom_policy_fallback_matches_serial_frozen_weight_sum(episodes):
    fitter = shared()
    policy, expected = SensitivePolicy(), SensitivePolicy()
    optimizer = torch.optim.Adam(expected.parameters(), lr=0.02)
    reference_update(expected, [(e, 0, 16) for e in episodes], optimizer)
    result = fitter.fit_sequences(
        policy, episodes, epochs=1, batch_size=2, learning_rate=0.02, profile=True
    )
    torch.testing.assert_close(policy.weight, expected.weight, rtol=0, atol=0)
    assert result["updates"] == 1
    assert result["supervised_presentations"] == 3
    assert result["profile"]["batches"] == 1
    assert result["profile"]["chunk_forwards"] == 8
    assert result["profile"]["prefix_forwards"] == 0
    assert set(result["profile"]["scope_seconds"]) == {
        "encoding",
        "prefix",
        "chunk",
        "backward",
        "step",
    }
    assert all(math.isfinite(t) and t >= 0 for t in result["profile"]["scope_seconds"].values())


def test_minibatch_never_contains_two_chunks_from_one_episode(episodes):
    fitter = shared()
    actual, expected = SensitivePolicy(), SensitivePolicy()
    optimizer = torch.optim.Adam(expected.parameters(), lr=0.02)
    # First round: chunk 0 from each episode. Next: their chunk 1, with fresh prefixes.
    reference_update(expected, [(e, 0, 2) for e in episodes], optimizer)
    reference_update(expected, [(episodes[0], 2, 2)], optimizer)
    fitted = fitter.fit_sequences(
        actual, episodes, epochs=1, sequence_length=2, batch_size=10, learning_rate=0.02
    )
    assert fitted["updates"] == 2
    assert actual.calls == expected.calls
    torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)


def test_expired_minibatch_never_updates_caller_optimizer_or_parameters(episodes, monkeypatch):
    fitter = shared()
    now = [0.0]
    monkeypatch.setattr(fitter.time, "monotonic", lambda: now[0])

    class ExpiringPolicy(SensitivePolicy):
        def forward(self, view, memory=None):
            output = super().forward(view, memory)
            if len(self.calls) == 6:
                now[0] = 2.0
            return output

    policy = ExpiringPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
    before_weight = policy.weight.detach().clone()
    before_optimizer = deepcopy(optimizer.state_dict())
    result = fitter.fit_sequences(
        policy, episodes, epochs=1, batch_size=2, optimizer=optimizer, deadline=1.0
    )
    assert result["updates"] == 0 and result["status"] == "incomplete"
    assert optimizer.state_dict() == before_optimizer
    assert torch.equal(before_weight, policy.weight)


def test_rejects_foreign_optimizer_and_invalid_sequences_weights_options(episodes):
    fitter = shared()
    policy = SensitivePolicy()
    foreign = torch.optim.Adam(SensitivePolicy().parameters())
    with pytest.raises(ValueError, match="optimizer"):
        fitter.fit_sequences(policy, episodes, optimizer=foreign)
    view, target = episodes[0][0]
    for invalid in ((((view, -1),),), (((view, True),),), (((view, target), (view, target)),)):
        with pytest.raises(ValueError):
            fitter.fit_sequences(policy, invalid)
        with pytest.raises(ValueError):
            fitter.score_sequences(policy, invalid)
    for kwargs in (
        {"batch_size": 0},
        {"batch_size": True},
        {"phase_weights": {}},
        {"phase_weights": {"SETUP_SETTLEMENT": math.nan}},
        {"learning_rate": 0},
        {"deadline": math.inf},
    ):
        with pytest.raises(ValueError):
            fitter.fit_sequences(policy, episodes, **kwargs)
    assert fitter.fit_sequences(policy, (), deadline=time.monotonic() - 1)["updates"] == 0


def test_legacy_wrappers_preserve_results_and_validate_provenance():
    fitter = shared()
    data = collect_demonstrations(board_seeds=[18], max_decisions=4)
    episodes = tuple(
        tuple((s["observation"], s["action_index"]) for s in e["samples"]) for e in data["episodes"]
    )
    policy, expected = SensitivePolicy(), SensitivePolicy()
    assert imitation_metrics(policy, data) == fitter.score_sequences(expected, episodes)
    result = fit_imitation(policy, data, epochs=1, batch_size=1)
    assert result == fitter.fit_sequences(expected, episodes, epochs=1)
    data["source"] = {"revision": None, "dirty": False}
    with pytest.raises(ValueError):
        fit_imitation(policy, data)


def test_real_minibatch_loss_and_all_parameter_gradients_equal_serial_sum(episodes):
    shared()
    torch.manual_seed(42)
    serial, batched = GraphPolicy(8), GraphPolicy(8)
    batched.load_state_dict(serial.state_dict())
    serial_losses, batched_losses = [], []
    serial_memories = [None] * len(episodes)
    batched_memories = [None] * len(episodes)
    for step in range(max(map(len, episodes))):
        active = [i for i, e in enumerate(episodes) if step < len(e)]
        views = tuple(episodes[i][step][0] for i in active)
        outputs = batched.forward_many(views, tuple(batched_memories[i] for i in active))
        for i, output in zip(active, outputs):
            view, target = episodes[i][step]
            expected = serial(view, serial_memories[i])
            for field in ("logits", "memory", "value"):
                torch.testing.assert_close(getattr(output, field), getattr(expected, field))
            serial_memories[i], batched_memories[i] = expected.memory, output.memory
            if len(view.legal_actions) > 1:
                serial_losses.append(F.cross_entropy(expected.logits[None], torch.tensor([target])))
                batched_losses.append(F.cross_entropy(output.logits[None], torch.tensor([target])))
    serial_loss, batched_loss = (
        torch.stack(serial_losses).mean(),
        torch.stack(batched_losses).mean(),
    )
    torch.testing.assert_close(batched_loss, serial_loss)
    serial_loss.backward()
    batched_loss.backward()
    for (name, expected), (_, got) in zip(serial.named_parameters(), batched.named_parameters()):
        assert (got.grad is None) == (expected.grad is None), name
        if expected.grad is not None:
            torch.testing.assert_close(got.grad, expected.grad, atol=2e-6, rtol=2e-5, msg=name)
    assert torch.isfinite(batched.gru.weight_ih.grad).all()


def test_recurrent_routes_match_memory_and_downstream_gradients_after_update(episodes):
    fitter = shared()
    torch.manual_seed(19)
    original = GraphPolicy(8)
    fitter.fit_sequences(original, episodes, epochs=1)
    full, recurrent = deepcopy(original), deepcopy(original)
    views = tuple(e[0][0] for e in episodes)
    expected = tuple(full(v).memory for v in views)
    got = recurrent.recurrent_many(views, (None, None))
    for v, actual, memory in zip(views, got, expected):
        torch.testing.assert_close(actual, memory)
        torch.testing.assert_close(recurrent.recurrent_step(v), memory)
    losses = []
    for policy, memories in ((full, expected), (recurrent, got)):
        loss = sum(policy(e[1][0], m).logits.square().mean() for e, m in zip(episodes, memories))
        losses.append(loss)
        loss.backward()
    torch.testing.assert_close(*losses)
    for (name, a), (_, b) in zip(full.named_parameters(), recurrent.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5, msg=name)


def test_serial_fallback_preserves_episode_prefix_then_chunk_order(episodes):
    fitter = shared()
    actual, expected = SensitivePolicy(), SensitivePolicy()
    optimizer = torch.optim.Adam(expected.parameters(), lr=0.02)
    # At size one, forced chunks are skipped. The choice at position 1 needs prefix 0.
    reference_update(expected, [(e, 1, 1) for e in episodes], optimizer)
    reference_update(expected, [(episodes[0], 3, 1)], optimizer)
    fitter.fit_sequences(
        actual, episodes, epochs=1, batch_size=2, sequence_length=1, learning_rate=0.02
    )
    assert actual.calls == expected.calls
    torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)


def test_real_fitter_minibatch_matches_independent_serial_sum(episodes):
    fitter = shared()
    torch.manual_seed(91)
    actual = GraphPolicy(8)
    expected = deepcopy(actual)
    optimizer = torch.optim.SGD(expected.parameters(), lr=0.01)
    reference_update(
        expected,
        [(e, 0, 16) for e in episodes],
        optimizer,
        weights={"BUILD_INITIAL_SETTLEMENT": 0.25, "BUILD_INITIAL_ROAD": 0.5},
    )
    actual_optimizer = torch.optim.SGD(actual.parameters(), lr=0.01)
    fitted = fitter.fit_sequences(
        actual,
        episodes,
        epochs=1,
        batch_size=2,
        optimizer=actual_optimizer,
        phase_weights={"BUILD_INITIAL_SETTLEMENT": 0.25, "BUILD_INITIAL_ROAD": 0.5},
    )
    assert fitted["updates"] == 1
    for (name, got), (_, want) in zip(actual.named_parameters(), expected.named_parameters()):
        torch.testing.assert_close(got, want, atol=1e-7, rtol=1e-6, msg=name)


def test_caller_optimizer_rejects_missing_policy_parameters(episodes):
    fitter = shared()
    policy = GraphPolicy(8)
    optimizer = torch.optim.Adam([policy.node_in.weight])
    with pytest.raises(ValueError, match="optimizer"):
        fitter.fit_sequences(policy, episodes, optimizer=optimizer)
