"""Root-run numerical tests for the optional sparse approved-target seam.

Synthetic Observation records only: no collector, board generation, engine, model
checkpoint or teacher. This file is authored but must be run only by Root.
"""

from copy import deepcopy
from dataclasses import replace
import math
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from colonist_rl import supervised_fit as fitter
from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, DecisionContext, Observation, PublicPlayer
from colonist_rl.policy import PolicyOutput


def sparse_api():
    assert hasattr(fitter, "ApprovedTarget"), "optional approved-target seam is missing"
    assert hasattr(fitter, "accepted_set_cross_entropy"), "canonical accepted-set CE is missing"
    return fitter.ApprovedTarget


def view(version, choices=3):
    return Observation(
        player_id=0,
        board=BoardSpec((), (), (), (), ()),
        rules=RulesConfig(),
        context=DecisionContext("PLAY_TURN", 0, 0, version, has_rolled=True),
        players=(PublicPlayer(0, 0, 0, 0),),
        hand=(0, 0, 0, 0, 0),
        development=(0, 0, 0, 0, 0),
        buildings=(), roads=(), robber_tile=0,
        legal_actions=tuple(Action("END_TURN", target=i) for i in range(choices)),
    )


def episode(length=8, forced=(0, 2, 5)):
    return tuple((view(i, 1 if i in forced else 3), 0) for i in range(length))


def approved(indices=(0,), *, role="TRAIN", **kwargs):
    Target = sparse_api()
    flags = {"train_approved": role == "TRAIN", "val_approved": role == "VAL"}
    flags.update(kwargs)
    return Target(indices, role=role, **flags)


def targets_for(ep, rows):
    return (tuple(rows.get(i) for i in range(len(ep))),)


class TracePolicy(nn.Module):
    """Small real differentiable recurrence; row-specific heads expose stray CE."""

    def __init__(self, length=32):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.02, dtype=torch.float64))
        self.head = nn.Parameter(torch.tensor([[-0.2, 0.1, 0.3]] * length, dtype=torch.float64))
        self.value_head = nn.Parameter(torch.tensor([0.4], dtype=torch.float64))
        self.calls = []

    def recurrent_step(self, observation, memory=None):
        v = observation.context.version
        self.calls.append((v, self.weight.item(), torch.is_grad_enabled(),
                           None if memory is None else memory.item()))
        return (0 if memory is None else memory) + self.weight * (v + 1)

    def forward(self, observation, memory=None):
        memory = self.recurrent_step(observation, memory)
        n = len(observation.legal_actions)
        coefficients = self.weight.new_tensor([0.0, 1.0, -1.0])[:n]
        logits = self.head[observation.context.version, :n] + memory * coefficients
        return PolicyOutput(logits, self.value_head.sum(), memory)


class RecordingSGD(torch.optim.SGD):
    def __init__(self, policy, lr=0.03):
        super().__init__(policy.parameters(), lr=lr)
        self.zero_calls = 0
        self.step_calls = 0
        self.gradients = []
        self.policy = policy

    def zero_grad(self, *args, **kwargs):
        self.zero_calls += 1
        return super().zero_grad(*args, **kwargs)

    def step(self, closure=None):
        self.step_calls += 1
        self.gradients.append({n: None if p.grad is None else p.grad.detach().clone()
                               for n, p in self.policy.named_parameters()})
        return super().step(closure)


def fit_sparse(policy, ep, target_rows, schedule, **kwargs):
    return fitter.fit_sequences(
        policy, (ep,), approved_targets=targets_for(ep, target_rows),
        anchor_schedule=schedule, epochs=1, batch_size=1, **kwargs,
    )


def independent_anchor_updates(policy, ep, schedule, accepted, length, optimizer):
    """Reference uses full forward everywhere, with ONE independently written loss."""
    losses = []
    for _, anchor in schedule:
        start = max(0, anchor - length + 1)
        memory = None
        with torch.no_grad():
            for observation, _ in ep[:start]:
                memory = policy(observation, memory).memory.detach()
        for observation, _ in ep[start : anchor + 1]:
            output = policy(observation, memory)
            memory = output.memory
        selected = output.logits[list(accepted[anchor])]
        loss = output.logits.logsumexp(0) - selected.logsumexp(0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_accepted_set_loss_and_gradient_sum_probability_instead_of_mean_ce():
    sparse_api()
    logits = torch.tensor([math.log(0.1), math.log(0.2), math.log(0.7)],
                          dtype=torch.float64, requires_grad=True)
    loss = fitter.accepted_set_cross_entropy(logits, (0, 2))
    assert loss.item() == pytest.approx(-math.log(0.8), abs=1e-14)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([-0.025, 0.2, -0.175],
                               dtype=torch.float64), rtol=0, atol=1e-14)


def test_singleton_loss_and_gradient_are_exact_ordinary_cross_entropy():
    sparse_api()
    actual = torch.tensor([-900.0, 700.0, 699.0], dtype=torch.float64, requires_grad=True)
    expected = actual.detach().clone().requires_grad_()
    got = fitter.accepted_set_cross_entropy(actual, (2,))
    want = F.cross_entropy(expected[None], torch.tensor([2]))
    assert torch.equal(got, want)
    got.backward()
    want.backward()
    assert torch.equal(actual.grad, expected.grad)


@pytest.mark.parametrize("indices", [(), (0, 1, 2), (0, 0), (2, 0), (-1,), (3,),
                                          (True,), [0], (0.0,)])
def test_set_loss_rejects_noninformative_or_malformed_target_sets(indices):
    sparse_api()
    with pytest.raises(ValueError):
        fitter.accepted_set_cross_entropy(torch.zeros(3), indices)


def test_only_scheduled_anchor_gets_ce_even_with_other_approved_suffix_rows():
    ep = episode()
    policy = TracePolicy()
    optimizer = RecordingSGD(policy)
    backward_calls = []
    policy.head.register_hook(lambda grad: backward_calls.append(grad.detach().clone()))
    result = fit_sparse(policy, ep, {1: approved((2,)), 3: approved((1, 2))},
                        ((0, 3),), sequence_length=4, optimizer=optimizer, profile=True)
    assert result["updates"] == result["supervised_presentations"] == 1
    assert optimizer.zero_calls == optimizer.step_calls == len(backward_calls) == 1
    grad = optimizer.gradients[0]["head"]
    assert torch.count_nonzero(grad[3]).item() > 0
    assert torch.count_nonzero(torch.cat((grad[:3], grad[4:]))).item() == 0
    assert optimizer.gradients[0]["value_head"] is None
    assert [v for v, *_ in policy.calls] == [0, 1, 2, 3]
    assert result["profile"]["prefix_forwards"] == 0
    assert result["profile"]["chunk_forwards"] == 4


def test_each_slot_replays_all_forced_prefixes_at_current_weights_and_stops_at_anchor():
    ep = episode(24, forced=(0, 2, 5, 11, 17))
    actual, expected = TracePolicy(), TracePolicy()
    schedule = ((0, 18), (0, 19))
    targets = {18: approved((1, 2)), 19: approved((2,))}
    optimizer = RecordingSGD(actual)
    expected_optimizer = RecordingSGD(expected)
    expected_losses = independent_anchor_updates(
        expected, ep, schedule, {18: (1, 2), 19: (2,)}, 16, expected_optimizer)
    result = fit_sparse(actual, ep, targets, schedule, sequence_length=16,
                        optimizer=optimizer, profile=True)
    assert result["updates"] == result["supervised_presentations"] == 2
    assert result["loss"] == pytest.approx(sum(expected_losses) / 2, abs=1e-14)
    assert actual.calls == expected.calls
    assert [v for v, *_ in actual.calls] == list(range(19)) + list(range(20))
    assert [grad for _, _, grad, _ in actual.calls] == ([False] * 3 + [True] * 16
                                                       + [False] * 4 + [True] * 16)
    assert actual.calls[0][3] is actual.calls[19][3] is None
    assert actual.calls[19][1] != actual.calls[0][1]
    for (name, got), (_, want) in zip(actual.named_parameters(), expected.named_parameters()):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-14, msg=name)
    assert result["profile"]["prefix_forwards"] == 7
    assert result["profile"]["chunk_forwards"] == 32


def test_frozen_schedule_order_and_repeats_have_one_update_each_without_dense_fallback():
    ep = episode()
    policy = TracePolicy()
    optimizer = RecordingSGD(policy)
    schedule = ((0, 6), (0, 1), (0, 6))
    result = fit_sparse(policy, ep, {1: approved((1,)), 6: approved((2,))}, schedule,
                        sequence_length=2, optimizer=optimizer)
    assert result["updates"] == result["supervised_presentations"] == optimizer.step_calls == 3
    assert [v for v, *_ in policy.calls] == list(range(7)) + list(range(2)) + list(range(7))
    assert [torch.nonzero(g["head"].abs().sum(1)).flatten().tolist()
            for g in optimizer.gradients] == [[6], [1], [6]]


def test_legacy_singleton_targets_use_identical_sparse_algorithm_without_audit_claim():
    Target = sparse_api()
    ep = episode()
    old, audited = TracePolicy(), TracePolicy()
    legacy = Target((0,), role="TRAIN", legacy=True)
    a = fit_sparse(old, ep, {6: legacy}, ((0, 6),), sequence_length=3,
                   optimizer=RecordingSGD(old))
    b = fit_sparse(audited, ep, {6: approved((0,))}, ((0, 6),), sequence_length=3,
                   optimizer=RecordingSGD(audited))
    assert a["loss"] == b["loss"]
    assert old.calls == audited.calls
    for left, right in zip(old.parameters(), audited.parameters()):
        assert torch.equal(left, right)


@pytest.mark.parametrize("role,flags,indices", [
    ("VAL", {"train_approved": False, "val_approved": True}, (0,)),
    ("EVAL", {"train_approved": True}, (0,)),
    ("TRAIN", {"train_approved": False}, (0,)),
    ("TRAIN", {"train_approved": True, "val_approved": True}, (0,)),
    ("TRAIN", {"train_approved": 1}, (0,)),
    ("TRAIN", {"legacy": True}, (1,)),
    ("TRAIN", {"train_approved": True}, (0, 1, 2)),
])
def test_invalid_role_approval_or_legacy_target_is_rejected_before_any_forward(role, flags, indices):
    Target = sparse_api()
    ep = episode()
    policy = TracePolicy()
    with pytest.raises(ValueError):
        target = Target(indices, role=role, **flags)
        fit_sparse(policy, ep, {3: target}, ((0, 3),))
    assert policy.calls == []


@pytest.mark.parametrize("kwargs", [
    {"approved_targets": ((None,) * 8,)},
    {"anchor_schedule": ((0, 3),)},
    {"approved_targets": (), "anchor_schedule": ((0, 3),)},
])
def test_sparse_options_cannot_silently_fall_back_to_dense_loss(kwargs):
    sparse_api()
    policy = TracePolicy()
    with pytest.raises(ValueError):
        fitter.fit_sequences(policy, (episode(),), epochs=1, **kwargs)
    assert policy.calls == []


@pytest.mark.parametrize("schedule", [((0, 1),), ((0, 2),), ((1, 3),), ((0, 8),),
                                           ((True, 3),), [(0, 3)], ()])
def test_unapproved_forced_or_invalid_scheduled_anchors_are_rejected(schedule):
    ep = episode()
    policy = TracePolicy()
    with pytest.raises(ValueError):
        fit_sparse(policy, ep, {3: approved((0,))}, schedule)
    assert policy.calls == []


@pytest.mark.parametrize("options", [
    {"epochs": 2}, {"batch_size": 2}, {"sequence_length": 17},
    {"phase_weights": {"PLAY_TURN": 1.0}},
])
def test_sparse_path_rejects_options_that_change_one_anchor_update_contract(options):
    sparse_api()
    ep = episode()
    policy = TracePolicy()
    kwargs = {"epochs": 1, "batch_size": 1, "sequence_length": 16}
    kwargs.update(options)
    with pytest.raises(ValueError):
        fitter.fit_sequences(policy, (ep,), approved_targets=targets_for(ep, {3: approved()}),
                             anchor_schedule=((0, 3),), **kwargs)
    assert policy.calls == []


def test_validation_scores_only_val_approved_sets_but_replays_full_history_and_preserves_state():
    ep = episode()
    policy = TracePolicy()
    policy.train()
    before = deepcopy(policy.state_dict())
    random.seed(36)
    np.random.seed(36)
    torch.manual_seed(36)
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    expected = TracePolicy()
    memory = None
    with torch.no_grad():
        for observation, _ in ep[:4]:
            output = expected(observation, memory)
            memory = output.memory
    want = -output.logits.log_softmax(0)[[0, 2]].exp().sum().log().item()
    result = fitter.score_sequences(policy, (ep,), approved_targets=targets_for(
        ep, {3: approved((0, 2), role="VAL")}))
    assert result["all"]["count"] == result["non_forced"]["count"] == 1
    assert result["all"]["correct"] == int(output.logits.argmax().item() in (0, 2))
    assert result["all"]["cross_entropy"] == pytest.approx(want, abs=1e-14)
    assert [v for v, *_ in policy.calls] == list(range(8))
    assert all(not enabled for _, _, enabled, _ in policy.calls)
    assert policy.training
    for name, value in before.items():
        assert torch.equal(policy.state_dict()[name], value)
    assert random.getstate() == rng[0]
    assert np.array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.get_rng_state(), rng[2])


def test_validation_rejects_train_targets_and_reports_unavailable_empty_denominator():
    sparse_api()
    ep = episode()
    policy = TracePolicy()
    with pytest.raises(ValueError):
        fitter.score_sequences(policy, (ep,), approved_targets=targets_for(ep, {3: approved()}))
    assert policy.calls == []
    result = fitter.score_sequences(policy, (ep,), approved_targets=targets_for(ep, {}))
    assert result["all"] == {"count": 0, "correct": 0, "accuracy": None, "cross_entropy": None}
    assert [v for v, *_ in policy.calls] == list(range(8))


def test_default_dense_fit_and_score_keep_legacy_tuple_semantics():
    """Existing API control: no new target API is needed for this test to pass."""
    ep = episode(5, forced=(0, 2))
    actual, expected = TracePolicy(), TracePolicy()
    expected_optimizer = RecordingSGD(expected)
    memory = None
    losses = []
    for observation, target in ep:
        output = expected(observation, memory)
        memory = output.memory
        if len(observation.legal_actions) > 1:
            losses.append(F.cross_entropy(output.logits[None], torch.tensor([target])))
    loss = torch.stack(losses).mean()
    expected_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 0.5, error_if_nonfinite=True)
    expected_optimizer.step()
    result = fitter.fit_sequences(actual, (ep,), epochs=1, sequence_length=16,
                                  optimizer=RecordingSGD(actual))
    assert result["updates"] == 1
    assert result["supervised_presentations"] == 3
    assert result["loss"] == loss.item()
    for left, right in zip(actual.parameters(), expected.parameters()):
        assert torch.equal(left, right)
    score = fitter.score_sequences(actual, (ep,))
    assert score["all"]["count"] == 5
    assert score["non_forced"]["count"] == 3


def test_400_caller_slots_are_consumed_once_with_400_single_target_presentations():
    ep = episode(1, forced=())
    policy = TracePolicy()
    optimizer = RecordingSGD(policy, lr=0.001)
    schedule = ((0, 0),) * 400
    result = fit_sparse(policy, ep, {0: approved((0, 2))}, schedule,
                        optimizer=optimizer, sequence_length=16)
    assert result["updates"] == result["supervised_presentations"] == 400
    assert optimizer.step_calls == optimizer.zero_calls == 400
    assert len(policy.calls) == 400
    assert all(v == 0 and memory is None for v, _, _, memory in policy.calls)


def test_validation_target_anywhere_in_fit_sidecar_is_optimizer_inaccessible():
    ep = episode()
    policy = TracePolicy()
    targets = {1: approved((1,), role="VAL"), 3: approved((0,))}
    with pytest.raises(ValueError):
        fit_sparse(policy, ep, targets, ((0, 3),))
    assert policy.calls == []


@pytest.mark.parametrize("dtype,offset,atol", [
    (torch.float32, 1e8, 1e-7),
    (torch.float64, 1e16, 1e-14),
])
def test_accepted_set_loss_and_gradient_are_invariant_to_large_common_logit_offset(
    dtype, offset, atol
):
    """Equal logits accept2/3 regardless of a common finite logit shift."""
    sparse_api()
    baseline = torch.zeros(3, dtype=dtype, requires_grad=True)
    shifted = torch.full((3,), offset, dtype=dtype, requires_grad=True)
    baseline_loss = fitter.accepted_set_cross_entropy(baseline, (0, 2))
    shifted_loss = fitter.accepted_set_cross_entropy(shifted, (0, 2))
    assert baseline_loss.item() == pytest.approx(math.log(1.5), abs=atol, rel=0)
    assert shifted_loss.item() == pytest.approx(math.log(1.5), abs=atol, rel=0)
    torch.testing.assert_close(shifted_loss, baseline_loss, rtol=0, atol=atol)
    baseline_loss.backward()
    shifted_loss.backward()
    expected_gradient = torch.tensor([-1 / 6, 1 / 3, -1 / 6], dtype=dtype)
    torch.testing.assert_close(baseline.grad, expected_gradient, rtol=0, atol=atol)
    torch.testing.assert_close(shifted.grad, expected_gradient, rtol=0, atol=atol)
    torch.testing.assert_close(shifted.grad, baseline.grad, rtol=0, atol=atol)
