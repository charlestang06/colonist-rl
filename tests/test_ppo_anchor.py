"""Tiny policy fixtures exercise optional frozen-reference PPO without games."""

import importlib.util
import math
import random
import subprocess
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from colonist_rl import training
from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, DecisionContext, Observation, PublicPlayer
from colonist_rl.policy import (
    ConsequenceGraphPolicy,
    ContextLinearConsequencePolicy,
    GraphPolicy,
    NonlinearConsequencePolicy,
)
from colonist_rl.simulator import BACKEND_VERSION
from colonist_rl.training import EpisodeTrace, PPOConfig, Transition, ppo_update

BASE = "08004ea6a7a9afa3422e0a81761cc63d9ea6b371"


def observation(version, forced=False):
    return Observation(
        0,
        BoardSpec.from_hexes([(0, 0)]),
        RulesConfig.standard(),
        DecisionContext("PLAY_TURN", 0, 0, version),
        tuple(PublicPlayer(p, 2, 0, 0) for p in range(4)),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (),
        (),
        0,
        (Action("END_TURN"),) if forced else (Action("END_TURN"), Action("ROLL")),
    )


def batch(*, forced=(False,), policy=None):
    views = tuple(observation(i, f) for i, f in enumerate(forced))
    records = []
    memory = None
    for i, view in enumerate(views):
        logp = 0.0
        if policy is not None:
            with torch.no_grad():
                output = policy(view, memory)
                memory = output.memory
                logp = torch.log_softmax(output.logits, 0)[0].item()
        records.append(
            Transition(
                view,
                0,
                logp,
                0.0,
                0.0,
                0.0,
                False,
                False,
                False,
                torch.full((3,), 999.0),
                0,
                "6:anchor:0",
                i,
                i,
                i + 1,
            )
        )
    trace = EpisodeTrace(
        "6:anchor:0",
        "anchor",
        0,
        11,
        0,
        BACKEND_VERSION,
        "terminal-win+0.2-vp-potential-v1",
        True,
        True,
        views,
        tuple(replace(r, memory=None) for r in records),
        final_version=len(views),
    )
    return records, {trace.episode_key: trace}


def update(policy, reference, records=None, traces=None, **kwargs):
    if records is None:
        records, traces = batch(policy=policy)
    return ppo_update(
        policy,
        kwargs.pop("optimizer", torch.optim.SGD(policy.parameters(), lr=0.01)),
        records,
        kwargs.pop("config", PPOConfig(epochs=1, entropy=0.0, value=0.0, grad=100.0)),
        episode_traces=traces,
        require_current_history=kwargs.pop("require_current_history", True),
        reference_policy=reference,
        anchor_coefficient=kwargs.pop("anchor_coefficient", 1.0),
        **kwargs,
    )


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def constant_logits(policy, first, second):
    # Controlled logits retain differentiable, independent student coordinates.
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()

    def logits_hook(module, args, output):
        output.logits = torch.stack(
            (module.node_in.bias[0] + first, module.node_in.bias[1] + second)
        )[: len(args[0].legal_actions)]
        return output

    policy.register_forward_hook(logits_hook)


def test_kl_reference_to_student_has_hand_computed_loss_and_gradient():
    student, reference = GraphPolicy(4), GraphPolicy(7)
    constant_logits(student, 0.0, 0.0)
    constant_logits(reference, math.log(0.8), math.log(0.2))
    metrics = update(student, reference, anchor_coefficient=0.3)
    expected = 0.8 * math.log(1.6) + 0.2 * math.log(0.4)
    assert metrics["anchor_kl"] == pytest.approx(expected, abs=1e-7)
    assert metrics["loss"] == pytest.approx(0.3 * expected, abs=1e-7)
    assert metrics["anchor_coefficient"] == 0.3
    assert metrics["anchor_states"] == 1
    assert student.node_in.bias[0].item() > 0
    assert student.node_in.bias[1].item() < 0
    reverse = 0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(0.5 / 0.2)
    assert abs(metrics["anchor_kl"] - reverse) > 0.01


def test_forced_states_have_exact_zero_kl():
    student, reference = GraphPolicy(4), GraphPolicy(7)
    records, traces = batch(forced=(True,), policy=student)
    metrics = update(student, reference, records, traces)
    assert metrics["anchor_kl"] == 0.0
    assert metrics["anchor_states"] == 1


@pytest.mark.parametrize("required", [False, True])
def test_zero_coefficient_is_bitwise_base_parity(tmp_path, required):
    path = tmp_path / "base_training.py"
    path.write_bytes(
        subprocess.check_output(["git", "show", f"{BASE}:src/colonist_rl/training.py"])
    )
    spec = importlib.util.spec_from_file_location("colonist_rl._anchor_base", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    student = GraphPolicy(4)
    student.eval()
    other = deepcopy(student)
    records, traces = batch(forced=(True, False, False), policy=student)
    records = [replace(r, memory=None) for r in records]
    base_records = [module.Transition(**r.__dict__) for r in records]
    base_traces = {
        k: module.EpisodeTrace(
            **{
                **t.__dict__,
                "transitions": tuple(module.Transition(**r.__dict__) for r in t.transitions),
            }
        )
        for k, t in traces.items()
    }
    first_opt = torch.optim.Adam(student.parameters(), lr=0.001)
    second_opt = torch.optim.Adam(other.parameters(), lr=0.001)
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    config = PPOConfig(epochs=2, sequence_length=1)
    a = module.ppo_update(
        student,
        first_opt,
        base_records,
        config,
        episode_traces=base_traces,
        require_current_history=required,
    )
    ending = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.setstate(rng[0])
    np.random.set_state(rng[1])
    torch.set_rng_state(rng[2])
    # An unusable reference must remain completely untouched when disabled.
    b = ppo_update(
        other,
        second_opt,
        records,
        config,
        episode_traces=traces,
        require_current_history=required,
        reference_policy=object(),
        anchor_coefficient=0.0,
    )
    assert a == b
    equal_tree(student.state_dict(), other.state_dict())
    equal_tree(first_opt.state_dict(), second_opt.state_dict())
    assert random.getstate() == ending[0]
    equal_tree(np.random.get_state(), ending[1])
    assert torch.equal(torch.get_rng_state(), ending[2])
    assert student.training is other.training is False
    assert [m.training for m in student.modules()] == [m.training for m in other.modules()]
    for first, second in zip(student.parameters(), other.parameters()):
        equal_tree(first.grad, second.grad)


def test_reference_uses_own_forced_prefix_no_future_and_exact_menu():
    student, reference = GraphPolicy(4), GraphPolicy(7)
    records, traces = batch(forced=(True, False, False, False), policy=student)
    expected = []
    hidden = None
    with torch.no_grad():
        for view in traces[records[0].episode_key].observations[:3]:
            out = reference(view, hidden)
            expected.append((None if hidden is None else hidden.clone(), out.logits.clone()))
            hidden = out.memory
    calls = []

    def capture(module, args, output):
        assert not any(p.requires_grad for p in module.parameters())
        assert not output.logits.requires_grad
        assert not output.memory.requires_grad
        calls.append(
            (
                args[0],
                None if len(args) < 2 or args[1] is None else args[1].clone(),
                output.logits.clone(),
                torch.is_grad_enabled(),
                module.training,
            )
        )

    reference.register_forward_hook(capture)
    metrics = update(
        student, reference, records[2:3], traces, config=PPOConfig(epochs=2, sequence_length=1)
    )
    assert metrics["completed_updates"] == 2
    assert len(calls) == 3
    for i, (view, memory, logits, grad, mode) in enumerate(calls):
        assert view is traces[records[0].episode_key].observations[i]
        equal_tree(memory, expected[i][0])
        equal_tree(logits, expected[i][1])
        assert grad is mode is False
    assert reference.hidden_size != student.hidden_size


@pytest.mark.parametrize("stop", ["success", "error", "deadline"])
def test_reference_state_grad_flags_modes_buffers_rng_preserved(monkeypatch, stop):
    student, reference = GraphPolicy(4), GraphPolicy(7)
    reference.train()
    reference.gru.eval()
    reference.register_buffer("sentinel", torch.tensor([8.0]))
    parameters = list(reference.parameters())
    parameters[0].requires_grad_(False)
    parameters[1].grad = torch.ones_like(parameters[1])
    state, modes = deepcopy(reference.state_dict()), [m.training for m in reference.modules()]
    grads = [None if p.grad is None else p.grad.clone() for p in parameters]
    grad_objects = [p.grad for p in parameters]
    flags = [p.requires_grad for p in parameters]
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])

    def disturb(module, args, output):
        module.sentinel.add_(1.0)
        random.random()
        np.random.random()
        torch.rand(1)
        if stop == "error":
            raise RuntimeError("reference fixture failure")
        if stop == "deadline":
            clock[0] = 2.0

    reference.register_forward_hook(disturb)
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    # One chunk's shuffle is length one, so it consumes no Python RNG.
    if stop == "error":
        with pytest.raises(RuntimeError, match="reference fixture failure"):
            update(student, reference, deadline=1.0)
    else:
        metrics = update(student, reference, deadline=1.0)
        assert metrics["completed_updates"] == (stop == "success")
        assert metrics["anchor_states"] == (stop == "success")
    equal_tree(reference.state_dict(), state)
    assert [m.training for m in reference.modules()] == modes
    assert [p.requires_grad for p in parameters] == flags
    for p, before, obj in zip(parameters, grads, grad_objects):
        equal_tree(p.grad, before)
        assert p.grad is obj
    assert random.getstate() == rng[0]
    equal_tree(np.random.get_state(), rng[1])
    assert torch.equal(torch.get_rng_state(), rng[2])


@pytest.mark.parametrize("coefficient", [-1.0, float("nan"), float("inf"), True, "0", None])
def test_invalid_coefficient_rejected_before_mutation(coefficient):
    student, reference = GraphPolicy(4), GraphPolicy(7)
    state = deepcopy(student.state_dict())
    with pytest.raises(ValueError, match="coefficient"):
        update(student, reference, anchor_coefficient=coefficient)
    equal_tree(student.state_dict(), state)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_reference",
        "unsupported_reference",
        "legacy",
        "missing_trace",
        "invalid_trace",
        "same_model",
        "parameter_alias",
        "buffer_alias",
        "optimizer_reference",
        "optimizer_extra",
        "optimizer_missing",
    ],
)
def test_invalid_anchor_contract_rejected_before_optimizer_mutation(problem):
    student, reference = GraphPolicy(4), GraphPolicy(4)
    records, traces = batch(policy=student)
    optimizer = torch.optim.Adam(student.parameters())
    options = {}
    if problem == "missing_reference":
        reference = None
    if problem == "unsupported_reference":
        reference = torch.nn.Linear(1, 1)
    if problem == "legacy":
        options["require_current_history"] = False
    if problem == "missing_trace":
        traces = None
    if problem == "invalid_trace":
        traces = {k: replace(t, history_complete=False) for k, t in traces.items()}
    if problem == "same_model":
        reference = student
    if problem == "parameter_alias":
        reference.node_in.weight = torch.nn.Parameter(student.node_in.weight.detach())
    if problem == "buffer_alias":
        reference.register_buffer("alias", student.node_in.weight.detach())
    if problem == "optimizer_reference":
        optimizer.add_param_group({"params": list(reference.parameters())})
    if problem == "optimizer_extra":
        optimizer.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))]})
    if problem == "optimizer_missing":
        optimizer = torch.optim.Adam(list(student.parameters())[1:])
    state = deepcopy(student.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    modes = [m.training for m in student.modules()]
    with pytest.raises(ValueError):
        update(student, reference, records, traces, optimizer=optimizer, **options)
    equal_tree(student.state_dict(), state)
    equal_tree(optimizer.state_dict(), optimizer_state)
    assert [m.training for m in student.modules()] == modes


@pytest.mark.parametrize("stage", ["prefix", "backward", "step"])
def test_deadline_in_reference_replay_or_before_backward_or_step_never_commits(monkeypatch, stage):
    student, reference = GraphPolicy(4), GraphPolicy(7)
    records, traces = batch(forced=(True, False), policy=student)
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    if stage == "prefix":
        reference.register_forward_hook(lambda *args: clock.__setitem__(0, 2.0))
    elif stage == "backward":
        # zero_grad lies immediately before backward; expiry here must skip backward.
        class ExpiringSGD(torch.optim.SGD):
            def zero_grad(self, **kwargs):
                super().zero_grad(**kwargs)
                clock[0] = 2.0

        optimizer = ExpiringSGD(student.parameters(), lr=0.01)
    else:
        student.node_in.weight.register_hook(
            lambda gradient: (clock.__setitem__(0, 2.0), gradient)[1]
        )
    before = deepcopy(student.state_dict())
    metrics = update(
        student,
        reference,
        records[1:],
        traces,
        deadline=1.0,
        optimizer=optimizer
        if stage == "backward"
        else torch.optim.SGD(student.parameters(), lr=0.01),
    )
    equal_tree(student.state_dict(), before)
    assert metrics["completed_updates"] == metrics["anchor_states"] == 0
    assert metrics["anchor_kl"] == 0.0
    assert metrics["deadline_reached"] == 1
    if stage == "backward":
        assert all(p.grad is None for p in student.parameters())


def test_full_menu_kl_includes_unsampled_third_action():
    student, reference = GraphPolicy(4), GraphPolicy(7)
    view = replace(
        observation(0),
        legal_actions=(Action("END_TURN"), Action("ROLL"), Action("BUY_DEVELOPMENT_CARD")),
    )
    records, traces = batch(policy=student)
    record = replace(records[0], observation=view)
    trace = replace(
        next(iter(traces.values())),
        observations=(view,),
        transitions=(replace(record, memory=None),),
    )

    def three_logits(module, args, output):
        output.logits = module.node_in.bias[:3] * 0.0 + torch.tensor(
            [0.0, 0.0, math.log(8.0)], device=module.device
        )
        return output

    def uniform_logits(module, args, output):
        output.logits = module.node_in.bias[:3] * 0.0
        return output

    reference.register_forward_hook(three_logits)
    student.register_forward_hook(uniform_logits)
    metrics = update(student, reference, [record], {trace.episode_key: trace})
    expected = 0.1 * math.log(0.3) + 0.1 * math.log(0.3) + 0.8 * math.log(2.4)
    assert metrics["anchor_kl"] == pytest.approx(expected, abs=1e-7)


def test_committed_anchor_mean_counts_states_across_unequal_chunks(monkeypatch):
    student, reference = GraphPolicy(4), GraphPolicy(7)
    constant_logits(student, 0.0, 0.0)
    constant_logits(reference, math.log(0.8), math.log(0.2))
    records, traces = batch(forced=(True, False, False), policy=student)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.0)
    # Fixed order makes the second chunk expire during its differentiable forward.
    monkeypatch.setattr(training.random, "shuffle", lambda order: None)
    clock = [0.0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])

    def expire_later(module, args, output):
        if torch.is_grad_enabled() and args[0].context.version == 2:
            clock[0] = 2.0

    student.register_forward_hook(expire_later)
    metrics = update(
        student,
        reference,
        records,
        traces,
        optimizer=optimizer,
        deadline=1.0,
        config=PPOConfig(epochs=2, sequence_length=2, entropy=0.0, value=0.0),
    )
    expected = 0.8 * math.log(1.6) + 0.2 * math.log(0.4)
    assert metrics["completed_updates"] == 1
    assert metrics["requested_updates"] == 4
    assert metrics["anchor_states"] == 2
    assert metrics["anchor_kl"] == pytest.approx(expected / 2.0, abs=1e-7)
    assert metrics["loss"] == pytest.approx(expected / 2.0, abs=1e-7)
    assert metrics["deadline_reached"] == 1


@pytest.mark.parametrize("malformation", ["short", "infinite", "nan", "rank"])
def test_invalid_reference_logits_fail_before_student_mutation(malformation):
    student, reference = GraphPolicy(4), GraphPolicy(7)

    def corrupt(module, args, output):
        output.logits = {
            "short": torch.zeros(1),
            "infinite": torch.tensor([0.0, float("inf")]),
            "nan": torch.tensor([0.0, float("nan")]),
            "rank": torch.zeros(2, 1),
        }[malformation]
        return output

    reference.register_forward_hook(corrupt)
    before = deepcopy(student.state_dict())
    with pytest.raises(ValueError, match="logits"):
        update(student, reference)
    equal_tree(student.state_dict(), before)


def test_huge_integer_coefficient_is_cleanly_rejected():
    student, reference = GraphPolicy(4), GraphPolicy(7)
    with pytest.raises(ValueError, match="coefficient"):
        update(student, reference, anchor_coefficient=10**1000)


def test_optimizer_state_alias_with_reference_is_rejected():
    student, reference = GraphPolicy(4), GraphPolicy(4)
    parameter = student.node_in.weight
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
    optimizer.state[parameter]["momentum_buffer"] = reference.node_in.weight.detach()
    before = deepcopy(reference.state_dict())
    with pytest.raises(ValueError, match="optimizer"):
        update(student, reference, optimizer=optimizer)
    equal_tree(reference.state_dict(), before)


def test_expired_entry_skips_reference_copy_and_all_forwards(monkeypatch):
    student, reference = GraphPolicy(4), GraphPolicy(7)
    records, traces = batch(policy=student)

    def forbidden(*args, **kwargs):
        raise AssertionError("expired update must not copy or forward")

    monkeypatch.setattr(training.time, "monotonic", lambda: 2.0)
    monkeypatch.setattr(training, "deepcopy", forbidden)
    student.register_forward_hook(forbidden)
    reference.register_forward_hook(forbidden)
    metrics = update(student, reference, records, traces, deadline=1.0)
    assert metrics["completed_updates"] == metrics["anchor_states"] == 0
    assert metrics["deadline_reached"] == 1


def test_reference_gradient_alias_with_student_storage_rejected():
    student, reference = GraphPolicy(4), GraphPolicy(4)
    reference.node_in.weight.grad = student.node_in.weight.detach()
    before = reference.node_in.weight.grad.clone()
    with pytest.raises(ValueError, match="storage"):
        update(student, reference)
    equal_tree(reference.node_in.weight.grad, before)


def test_optimizer_state_alias_with_reference_gradient_rejected():
    student, reference = GraphPolicy(4), GraphPolicy(4)
    reference.node_in.weight.grad = torch.ones_like(reference.node_in.weight)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
    optimizer.state[student.node_in.weight]["momentum_buffer"] = reference.node_in.weight.grad
    before = reference.node_in.weight.grad.clone()
    with pytest.raises(ValueError, match="optimizer"):
        update(student, reference, optimizer=optimizer)
    equal_tree(reference.node_in.weight.grad, before)


@pytest.mark.parametrize(
    "kind",
    [
        GraphPolicy,
        ConsequenceGraphPolicy,
        ContextLinearConsequencePolicy,
        NonlinearConsequencePolicy,
    ],
)
def test_supported_reference_architectures_complete_tiny_update(kind):
    student, reference = GraphPolicy(4), kind(7)
    metrics = update(student, reference)
    assert metrics["completed_updates"] == metrics["anchor_states"] == 1
    assert math.isfinite(metrics["anchor_kl"])
