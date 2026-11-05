"""Tiny committed histories exercise warmup without collecting or playing games."""

import copy
import importlib
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, DecisionContext, Observation, PublicPlayer
from colonist_rl.policy import (
    ConsequenceGraphPolicy,
    ContextLinearConsequencePolicy,
    GraphPolicy,
    NonlinearConsequencePolicy,
)
from colonist_rl.simulator import BACKEND_VERSION
from colonist_rl.training import EpisodeTrace, Transition

BOARD = BoardSpec.from_hexes([(0, 0)])
RULES = RulesConfig.standard()


def api():
    return importlib.import_module("colonist_rl.critic_warmup")


def view(version, forced=False):
    return Observation(
        0,
        BOARD,
        RULES,
        DecisionContext("PLAY_TURN", 0, 0, version),
        tuple(PublicPlayer(p, 2, 0, 0) for p in range(4)),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (),
        (),
        0,
        (Action("END_TURN"),) if forced else (Action("END_TURN"), Action("ROLL")),
    )


def trace(run="fixture", rewards=(0.25, -0.75, 0.25), *, initial=9.0, winner=1):
    key = f"{len(run)}:{run}:0"
    observations = tuple(view(2 + i * 3, forced=i % 2 == 0) for i in range(len(rewards)))
    records = tuple(
        Transition(
            observation,
            0,
            0.0,
            0.0,
            reward,
            0.0,
            i == len(rewards) - 1,
            i == len(rewards) - 1,
            False,
            None,
            0,
            key,
            i,
            observation.context.version,
            observation.context.version + 3,
        )
        for i, (observation, reward) in enumerate(zip(observations, rewards))
    )
    return EpisodeTrace(
        key,
        run,
        0,
        101,
        0,
        BACKEND_VERSION,
        "terminal-win+0.2-vp-potential-v1",
        True,
        True,
        observations,
        records,
        initial_reward=initial,
        initial_step_end=2,
        final_version=2 + len(rewards) * 3,
        terminated=True,
        winner=winner,
        finalized=True,
    )


def pending(source):
    records = source.transitions
    if records:
        records = (*records[:-1], replace(records[-1], boundary=False, terminated=False))
    return replace(
        source,
        transitions=records,
        terminated=False,
        winner=None,
        finalized=False,
        incomplete_reason="count_cutoff",
    )


def mapping(*traces):
    return {t.episode_key: t for t in traces}


def actor_history(policy, source):
    result, memory = [], None
    with torch.no_grad():
        for observation in source.observations:
            output = policy(observation, memory)
            memory = output.memory
            result.append((output.logits.clone(), memory.clone(), output.logits.argmax().item()))
    return result


def snapshot(policy):
    return {name: tensor.clone() for name, tensor in policy.state_dict().items()}


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(before):
    python, numpy, torch_rng = before
    assert random.getstate() == python
    current = np.random.get_state()
    assert current[0] == numpy[0]
    assert np.array_equal(current[1], numpy[1])
    assert current[2:] == numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize(
    "policy_type",
    [
        GraphPolicy,
        ConsequenceGraphPolicy,
        ContextLinearConsequencePolicy,
        NonlinearConsequencePolicy,
    ],
)
def test_actual_suffix_mse_changes_only_value_and_preserves_actor_exactly(policy_type):
    policy = policy_type(hidden_size=4)
    source = trace()
    with torch.no_grad():
        policy.value.weight.zero_()
        policy.value.bias.zero_()
    before = snapshot(policy)
    actor_before = actor_history(policy, source)
    sample_rng = torch.get_rng_state().clone()
    action_before, memory_before = policy.act(source.observations[1])
    torch.set_rng_state(sample_rng)
    metrics, optimizer = api().warm_value_head(policy, mapping(source), epochs=1)
    action_after, memory_after = policy.act(source.observations[1])
    assert action_after == action_before
    assert torch.equal(memory_after, memory_before)
    for old, new in zip(actor_before, actor_history(policy, source)):
        assert torch.equal(old[0], new[0])
        assert torch.equal(old[1], new[1])
        assert old[2] == new[2]
    after = policy.state_dict()
    assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith("value."))
    assert all(torch.equal(before[n], after[n]) for n in before if not n.startswith("value."))
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in policy.value.parameters()
    }
    # Rewards cover 2->5->8->11 including opponent actions. Initial +9 is excluded;
    # losing actor 0 still receives the actual signed suffixes [-.25, -.5, +.25].
    assert metrics["target_min"] == -0.5
    assert metrics["target_max"] == 0.25
    assert metrics["squared_error_sum"] == pytest.approx(0.375)
    assert metrics["squared_error_count"] == 3
    assert metrics["samples_optimized"] == 3
    assert metrics["optimizer_steps"] == 1
    assert metrics["epochs_completed"] == 1
    assert metrics["status"] == "complete"
    hidden = torch.stack([a[1] for a in actor_before])
    targets = torch.tensor([-0.25, -0.5, 0.25])
    expected_moment = (-0.2 / 3 * (targets[:, None] * hidden).sum(0)).reshape(1, -1)
    assert torch.allclose(optimizer.state[policy.value.weight]["exp_avg"], expected_moment)
    assert optimizer.state[policy.value.bias]["exp_avg"].item() == pytest.approx(1 / 30)


def test_mapping_order_forced_recurrence_short_batches_and_epoch_loss(monkeypatch):
    policy = GraphPolicy(hidden_size=4)
    first, second = trace("first"), trace("second", (0.5, -0.25))
    histories = [actor_history(policy, source) for source in (first, second)]
    features = torch.stack([row[1] for history in histories for row in history])
    targets = torch.tensor([-0.25, -0.5, 0.25, 0.25, -0.25])
    head = copy.deepcopy(policy.value)
    expected_optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    expected_sse = 0
    for _ in range(2):
        for start in range(0, 5, 2):
            error = head(features[start : start + 2]).squeeze(-1) - targets[start : start + 2]
            expected_sse += error.square().sum().item()
            expected_optimizer.zero_grad()
            error.square().mean().backward()
            expected_optimizer.step()
    calls, batches = [], []
    original = policy.recurrent_step

    def recurrent(observation, memory=None):
        calls.append((observation.context.version, memory is None, torch.is_grad_enabled()))
        return original(observation, memory)

    monkeypatch.setattr(policy, "recurrent_step", recurrent)
    handle = policy.value.register_forward_pre_hook(
        lambda module, inputs: batches.append((len(inputs[0]), inputs[0].requires_grad))
    )
    try:
        metrics, _ = api().warm_value_head(
            policy, mapping(first, second), epochs=2, batch_size=2, learning_rate=0.01
        )
    finally:
        handle.remove()
    assert (
        calls
        == [
            (2, True, False),
            (5, False, False),
            (8, False, False),
            (2, True, False),
            (5, False, False),
        ]
        * 2
    )
    assert batches == [(2, False), (2, False), (1, False)] * 2
    assert torch.equal(policy.value.weight, head.weight)
    assert torch.equal(policy.value.bias, head.bias)
    assert metrics["squared_error_sum"] == pytest.approx(expected_sse)
    assert metrics["squared_error_count"] == metrics["samples_optimized"] == 10
    assert metrics["optimizer_steps"] == 6
    assert metrics["epochs_completed"] == 2
    assert metrics["states_replayed"] == metrics["states_attempted"] == 10


def exclusions():
    p = pending(trace("pending"))
    t = trace("truncated")
    t = replace(
        t,
        transitions=(
            *t.transitions[:-1],
            replace(t.transitions[-1], terminated=False, truncated=True),
        ),
        terminated=False,
        truncated=True,
        winner=None,
        incomplete_reason="truncation",
    )
    d = pending(trace("dropped"))
    d = replace(
        d,
        observations=(*d.observations, view(d.final_version)),
        final_version=d.final_version + 1,
        finalized=True,
        dropped_intervals=1,
        incomplete_reason="deadline_dropped_interval",
    )
    i = replace(trace("incomplete"), history_complete=False)
    u = replace(trace("unknown"), winner=None)
    return p, t, d, i, u


def test_exclusions_empty_terminal_and_no_data_return_fresh_empty_optimizer():
    policy = GraphPolicy(hidden_size=4)
    empty = trace("empty", ())
    before = snapshot(policy)
    metrics, optimizer = api().warm_value_head(policy, mapping(*exclusions(), empty), epochs=2)
    assert metrics["status"] == "no_data"
    assert metrics["eligible_episodes"] == metrics["empty_episodes"] == 1
    assert metrics["eligible_samples"] == 0
    assert metrics["excluded_episodes"] == 5
    assert metrics["excluded_samples"] == 15
    assert {entry["episode_key"] for entry in metrics["excluded"]} == {
        t.episode_key for t in exclusions()
    }
    reasons = {reason for entry in metrics["excluded"] for reason in entry["reasons"]}
    assert {
        "pending",
        "truncated",
        "dropped_intervals",
        "incomplete_history",
        "unknown_outcome",
    } <= reasons
    assert metrics["target_min"] is metrics["target_max"] is None
    for name in (
        "states_replayed",
        "samples_optimized",
        "optimizer_steps",
        "epochs_completed",
        "squared_error_count",
    ):
        assert metrics[name] == 0
    assert optimizer.state == {}
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())
    again, other = api().warm_value_head(policy, {}, epochs=1)
    assert again["status"] == "no_data"
    assert other is not optimizer
    assert other.state == {}


@pytest.mark.parametrize(
    "kind",
    [
        "key",
        "ordinal",
        "gap",
        "pending_identity",
        "incomplete_identity",
        "reward",
        "false_flag",
        "duplicate_namespace",
    ],
)
def test_corrupt_even_excluded_traces_fail_before_model_work(monkeypatch, kind):
    policy = GraphPolicy(hidden_size=4)
    good, bad = trace("good"), trace("bad")
    if kind == "key":
        bad = replace(bad, episode_key="crossed")
    elif kind == "ordinal":
        bad = replace(
            bad,
            transitions=(
                bad.transitions[0],
                replace(bad.transitions[1], observation_index=0),
                bad.transitions[2],
            ),
        )
    elif kind == "gap":
        bad = replace(
            bad,
            transitions=(
                bad.transitions[0],
                replace(bad.transitions[1], step_start=6),
                bad.transitions[2],
            ),
        )
    elif kind == "pending_identity":
        bad = replace(pending(bad), run_id="other")
    elif kind == "incomplete_identity":
        bad = replace(bad, history_complete=False, run_id="other")
    elif kind == "reward":
        bad = replace(
            bad,
            transitions=(*bad.transitions[:-1], replace(bad.transitions[-1], reward=float("nan"))),
        )
    elif kind == "false_flag":
        bad = replace(bad, history_complete=0)
    traces = mapping(good, bad)
    if kind == "duplicate_namespace":
        traces["alias"] = bad
    before, rng = snapshot(policy), rng_state()
    modes = [m.training for m in policy.modules()]
    policy.value.weight.grad = torch.ones_like(policy.value.weight)
    gradient = policy.value.weight.grad
    monkeypatch.setattr(
        policy, "recurrent_step", lambda *args: pytest.fail("validation must precede replay")
    )
    with pytest.raises(ValueError):
        api().warm_value_head(policy, traces, epochs=1)
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())
    assert [m.training for m in policy.modules()] == modes
    assert policy.value.weight.grad is gradient
    assert_rng_equal(rng)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"epochs": True},
        {"epochs": 1.5},
        {"epochs": 1, "batch_size": 0},
        {"epochs": 1, "batch_size": 65},
        {"epochs": 1, "batch_size": True},
        {"epochs": 1, "learning_rate": 0},
        {"epochs": 1, "learning_rate": float("nan")},
        {"epochs": 1, "deadline": float("inf")},
        {"epochs": 1, "deadline": True},
    ],
)
def test_invalid_arguments_do_not_mutate_policy(kwargs):
    policy = GraphPolicy(hidden_size=4)
    before, rng = snapshot(policy), rng_state()
    with pytest.raises(ValueError):
        api().warm_value_head(policy, mapping(trace()), **kwargs)
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())
    assert_rng_equal(rng)


def test_independent_invocations_have_fresh_adam_moments():
    policy = GraphPolicy(hidden_size=4)
    _, first = api().warm_value_head(policy, mapping(trace()), epochs=1)
    first_moment = first.state[policy.value.weight]["exp_avg"].clone()
    _, second = api().warm_value_head(policy, mapping(trace()), epochs=1)
    assert first is not second
    assert first.state[policy.value.weight]["step"].item() == 1
    assert second.state[policy.value.weight]["step"].item() == 1
    assert torch.equal(first.state[policy.value.weight]["exp_avg"], first_moment)
    assert (
        first.state[policy.value.weight]["exp_avg"].data_ptr()
        != second.state[policy.value.weight]["exp_avg"].data_ptr()
    )


@pytest.mark.parametrize(
    "expire_after,expected", [(0, (0, 0, 0, 0)), (1, (1, 0, 0, 1)), (3, (3, 2, 1, 1))]
)
def test_deadline_discards_pending_batch_without_late_steps(monkeypatch, expire_after, expected):
    module, policy = api(), GraphPolicy(hidden_size=4)
    clock = SimpleNamespace(now=float(expire_after == 0))
    monkeypatch.setattr(module, "_expired", lambda deadline: clock.now >= deadline)
    calls = 0
    original = policy.recurrent_step

    def recurrent(observation, memory=None):
        nonlocal calls
        result = original(observation, memory)
        calls += 1
        if calls == expire_after:
            clock.now = 1
        return result

    monkeypatch.setattr(policy, "recurrent_step", recurrent)
    before = snapshot(policy)
    metrics, optimizer = module.warm_value_head(
        policy, mapping(trace()), epochs=2, batch_size=2, deadline=1
    )
    assert metrics["status"] == "deadline"
    assert (
        metrics["states_replayed"],
        metrics["samples_optimized"],
        metrics["optimizer_steps"],
        metrics["samples_discarded"],
    ) == expected
    assert metrics["epochs_completed"] == 0
    if expected[2] == 0:
        assert optimizer.state == {}
        assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())
    else:
        assert all(s["step"].item() == 1 for s in optimizer.state.values())


@pytest.mark.parametrize("phase", ["before_backward", "before_step"])
def test_deadline_gates_backward_and_step(monkeypatch, phase):
    module, policy = api(), GraphPolicy(hidden_size=4)
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(module, "_expired", lambda deadline: clock.now >= deadline)
    backward_calls = []

    def expire(*args):
        clock.now = 1

    observer = policy.value.weight.register_hook(lambda grad: backward_calls.append(True))
    handle = (
        policy.value.register_forward_hook(expire)
        if phase == "before_backward"
        else policy.value.weight.register_hook(expire)
    )
    before = snapshot(policy)
    try:
        metrics, optimizer = module.warm_value_head(
            policy, mapping(trace()), epochs=1, batch_size=2, deadline=1
        )
    finally:
        observer.remove()
        handle.remove()
    assert bool(backward_calls) == (phase == "before_step")
    assert metrics["samples_optimized"] == metrics["optimizer_steps"] == 0
    assert metrics["samples_discarded"] == metrics["states_replayed"] == 2
    assert metrics["epochs_completed"] == 0
    assert metrics["status"] == "deadline"
    assert optimizer.state == {}
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("exit_kind", ["success", "exception", "deadline"])
def test_rng_modes_flags_and_existing_gradient_evidence_restored(monkeypatch, exit_kind):
    module, policy = api(), GraphPolicy(hidden_size=4)
    policy.train()
    policy.gru.eval()
    policy.value.eval()
    policy.node_in.weight.requires_grad_(False)
    policy.value.weight.requires_grad_(False)
    for i, parameter in enumerate(policy.parameters()):
        parameter.grad = torch.full_like(parameter, i + 1.0)
    gradients = [(p, p.grad, p.grad.clone(), p.grad._version) for p in policy.parameters()]
    flags = [p.requires_grad for p in policy.parameters()]
    modes = [m.training for m in policy.modules()]
    before_rng = rng_state()
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(
        module, "_expired", lambda deadline: deadline is not None and clock.now >= deadline
    )
    original = policy.recurrent_step

    def recurrent(observation, memory=None):
        assert not torch.is_grad_enabled()
        assert all(not m.training for m in policy.modules())
        assert all(
            not p.requires_grad for n, p in policy.named_parameters() if not n.startswith("value.")
        )
        random.random()
        np.random.random()
        torch.rand(1)
        result = original(observation, memory)
        if exit_kind == "exception":
            raise RuntimeError("fixture replay failure")
        if exit_kind == "deadline":
            clock.now = 1
        return result

    monkeypatch.setattr(policy, "recurrent_step", recurrent)
    if exit_kind == "exception":
        with pytest.raises(RuntimeError, match="fixture replay failure"):
            module.warm_value_head(policy, mapping(trace()), epochs=1, deadline=1)
    else:
        module.warm_value_head(policy, mapping(trace()), epochs=1, deadline=1)
    assert [p.requires_grad for p in policy.parameters()] == flags
    assert [m.training for m in policy.modules()] == modes
    for parameter, original_grad, value, version in gradients:
        assert parameter.grad is original_grad
        assert torch.equal(parameter.grad, value)
        assert parameter.grad._version == version
    assert_rng_equal(before_rng)


def test_actor_tensor_corruption_fails_visibly_and_restores_caller_state(monkeypatch):
    policy = GraphPolicy(hidden_size=4)
    original = policy.recurrent_step
    original_mode = policy.training
    before_rng = rng_state()

    def corrupt(observation, memory=None):
        result = original(observation, memory)
        policy.node_in.weight.add_(1)
        return result

    monkeypatch.setattr(policy, "recurrent_step", corrupt)
    with pytest.raises(RuntimeError, match="node_in.weight"):
        api().warm_value_head(policy, mapping(trace()), epochs=1)
    assert policy.training == original_mode
    assert_rng_equal(before_rng)


def test_shared_value_actor_parameters_and_custom_policy_are_rejected():
    policy = GraphPolicy(hidden_size=4)
    policy.heads["turn"][2] = policy.value
    with pytest.raises(ValueError):
        api().warm_value_head(policy, mapping(trace()), epochs=1)

    class CustomGraph(GraphPolicy):
        pass

    with pytest.raises(ValueError):
        api().warm_value_head(CustomGraph(hidden_size=4), mapping(trace()), epochs=1)


@pytest.mark.parametrize("field", ["learning_rate", "deadline"])
def test_unrepresentable_numeric_arguments_are_validation_errors_before_mutation(field):
    policy = GraphPolicy(hidden_size=4)
    before, rng = snapshot(policy), rng_state()
    with pytest.raises(ValueError):
        api().warm_value_head(policy, mapping(trace()), epochs=1, **{field: 10**1000})
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())
    assert_rng_equal(rng)


def test_complete_terminal_with_inconsistent_reason_is_corrupt_not_excluded():
    policy = GraphPolicy(hidden_size=4)
    source = replace(trace(), incomplete_reason="deadline")
    before = snapshot(policy)
    with pytest.raises(ValueError, match="complete genuine terminal"):
        api().warm_value_head(policy, mapping(source), epochs=1)
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())


def test_bitwise_buffer_preservation_and_buffer_corruption_detection(monkeypatch):
    policy = GraphPolicy(hidden_size=4)
    policy.register_buffer("audit", torch.tensor([float("nan"), -0.0]))
    before = policy.audit.view(torch.uint8).clone()
    api().warm_value_head(policy, mapping(trace()), epochs=1)
    assert torch.equal(before, policy.audit.view(torch.uint8))
    original = policy.recurrent_step

    def corrupt(observation, memory=None):
        policy.audit[1] = 0.0  # Numerically equal, but sign-bit corruption must be visible.
        return original(observation, memory)

    monkeypatch.setattr(policy, "recurrent_step", corrupt)
    with pytest.raises(RuntimeError, match="buffer:audit"):
        api().warm_value_head(policy, mapping(trace()), epochs=1)


def test_validation_catches_changed_mapping_even_when_entries_are_valid():
    from collections.abc import Mapping

    first, second = trace("first"), trace("second")

    class ChangingMapping(Mapping):
        def __init__(self):
            self.calls = 0

        def __iter__(self):
            return iter((first.episode_key,))

        def __len__(self):
            return 1

        def __getitem__(self, key):
            return first

        def items(self):
            self.calls += 1
            return (
                [(first.episode_key, first)] if self.calls == 1 else [(second.episode_key, second)]
            )

    policy = GraphPolicy(hidden_size=4)
    before = snapshot(policy)
    with pytest.raises(ValueError, match="mapping changed"):
        api().warm_value_head(policy, ChangingMapping(), epochs=1)
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in before.items())


def test_plain_no_grad_caller_can_fit_and_backward_exception_restores_state():
    policy = GraphPolicy(hidden_size=4)
    with torch.no_grad():
        metrics, _ = api().warm_value_head(policy, mapping(trace()), epochs=1)
        assert not torch.is_grad_enabled()
    assert metrics["optimizer_steps"] == 1
    original_grad = torch.ones_like(policy.value.weight)
    policy.value.weight.grad = original_grad
    original_mode, rng = policy.training, rng_state()

    def fail(gradient):
        torch.rand(1)
        raise RuntimeError("fixture backward failure")

    handle = policy.value.weight.register_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="fixture backward failure"):
            api().warm_value_head(policy, mapping(trace()), epochs=1)
    finally:
        handle.remove()
    assert policy.value.weight.grad is original_grad
    assert torch.equal(original_grad, torch.ones_like(original_grad))
    assert policy.training == original_mode
    assert_rng_equal(rng)


def test_parameter_storage_alias_is_rejected_even_with_distinct_parameter_objects():
    policy = GraphPolicy(hidden_size=4)
    policy.value.weight = torch.nn.Parameter(policy.action_in.bias[None, :])
    assert policy.value.weight is not policy.action_in.bias
    with pytest.raises(ValueError, match="share"):
        api().warm_value_head(policy, mapping(trace()), epochs=1)
