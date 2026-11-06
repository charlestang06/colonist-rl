"""Opt-in road wrapper contracts using a counting parent and pure geometry."""

import importlib
import importlib.util
import unittest
from dataclasses import replace

import torch
from torch import nn

from colonist_rl.contracts import Action
from test_road_continuation import view


class CountingParent(nn.Module):
    def __init__(self, selected):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))
        self.selected = selected
        self.output_memory = object()
        self.forward_output = object()
        self.forward_calls = []
        self.act_calls = []

    @property
    def device(self):
        return self.weight.device

    def forward(self, observation, memory=None):
        self.forward_calls.append((observation, memory))
        return self.forward_output

    def act(self, observation, memory=None, *, deterministic=False):
        self.act_calls.append((observation, memory, deterministic))
        return self.selected, self.output_memory


class StochasticParent(CountingParent):
    def act(self, observation, memory=None, *, deterministic=False):
        self.selected = observation.legal_actions[int(torch.rand(()).item() >= 0.5)]
        return super().act(observation, memory, deterministic=deterministic)


class RoadContinuationPolicyTests(unittest.TestCase):
    def policy_class(self):
        self.assertIsNotNone(
            importlib.util.find_spec("colonist_rl.road_continuation_policy"),
            "opt-in road continuation policy wrapper is missing",
        )
        module = importlib.import_module("colonist_rl.road_continuation_policy")
        return module.RoadContinuationPolicy

    def test_act_changes_zero_access_road_once_and_preserves_parent_and_memory(self):
        policy_class = self.policy_class()
        observation = view()
        selected, alternative = observation.legal_actions
        parent = CountingParent(selected)
        parameter = parent.weight
        values = parameter.clone()
        memory = {"hidden": object()}
        rng_before = torch.get_rng_state()
        wrapper = policy_class(parent)
        self.assertIsNone(wrapper.last_decision)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

        action, returned_memory = wrapper.act(observation, memory, deterministic=True)

        self.assertIs(action, alternative)
        self.assertIs(returned_memory, parent.output_memory)
        self.assertEqual(len(parent.act_calls), 1)
        received_view, received_memory, deterministic = parent.act_calls[0]
        self.assertIs(received_view, observation)
        self.assertIs(received_memory, memory)
        self.assertIs(deterministic, True)
        self.assertEqual(parent.forward_calls, [])
        self.assertIs(parent.weight, parameter)
        self.assertTrue(torch.equal(parent.weight, values))
        self.assertTrue(parent.weight.requires_grad)
        self.assertTrue(parent.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertIs(wrapper.last_decision.action, alternative)
        self.assertTrue(wrapper.last_decision.changed)
        self.assertEqual(wrapper.last_decision.reason, "improved_settlement_access")
        self.assertEqual(
            [(score.edge, score.gain) for score in wrapper.last_decision.scores],
            [(2, 0.0), (1, 3.2)],
        )

    def test_fallbacks_preserve_original_choice_and_refresh_diagnostics(self):
        policy_class = self.policy_class()
        base = view()
        nonroad = Action("END_TURN")
        cases = (
            replace(base, legal_actions=(nonroad,) + base.legal_actions),
            replace(base, context=replace(base.context, kind="SETUP_ROAD")),
            replace(base, context=replace(base.context, free_roads=1)),
        )
        parent = CountingParent(base.legal_actions[0])
        wrapper = policy_class(parent)
        previous_decision = object()
        wrapper.last_decision = previous_decision
        memory = object()
        for call_count, observation in enumerate(cases, 1):
            with self.subTest(context=observation.context):
                selected = observation.legal_actions[0]
                parent.selected = selected
                action, returned_memory = wrapper.act(observation, memory)
                self.assertIs(action, selected)
                self.assertIs(returned_memory, parent.output_memory)
                self.assertEqual(len(parent.act_calls), call_count)
                self.assertIs(parent.act_calls[-1][0], observation)
                self.assertIs(parent.act_calls[-1][1], memory)
                self.assertIs(parent.act_calls[-1][2], False)
                self.assertEqual(parent.forward_calls, [])
                self.assertIsNot(wrapper.last_decision, previous_decision)
                self.assertIs(wrapper.last_decision.action, selected)
                self.assertFalse(wrapper.last_decision.changed)
                self.assertTrue(wrapper.last_decision.reason)
                previous_decision = wrapper.last_decision

    def test_forward_returns_exact_parent_output_without_selection_or_diagnostic_change(self):
        policy_class = self.policy_class()
        parent = CountingParent(None)
        wrapper = policy_class(parent)
        diagnostic = object()
        wrapper.last_decision = diagnostic
        # This deliberately is not an Observation: invoking the selector fails.
        observation, memory = object(), object()

        output = wrapper(observation, memory)

        self.assertIs(output, parent.forward_output)
        self.assertEqual(len(parent.forward_calls), 1)
        self.assertIs(parent.forward_calls[0][0], observation)
        self.assertIs(parent.forward_calls[0][1], memory)
        self.assertEqual(parent.act_calls, [])
        self.assertIs(wrapper.last_decision, diagnostic)

    def test_constructor_registers_parent_without_mutating_mode_grad_or_counters(self):
        policy_class = self.policy_class()
        for training, requires_grad in ((False, True), (True, False)):
            with self.subTest(training=training, requires_grad=requires_grad):
                parent = CountingParent(None)
                parent.train(training)
                parent.weight.requires_grad_(requires_grad)
                parameter = parent.weight
                values = parameter.clone()
                rng_before = torch.get_rng_state()

                wrapper = policy_class(parent)

                self.assertIsInstance(wrapper, nn.Module)
                self.assertIs(wrapper.parent, parent)
                self.assertIs(dict(wrapper.named_children())["parent"], parent)
                self.assertIs(dict(wrapper.named_parameters())["parent.weight"], parameter)
                self.assertIs(parent.training, training)
                self.assertIs(parent.weight.requires_grad, requires_grad)
                self.assertTrue(torch.equal(parent.weight, values))
                self.assertEqual(parent.act_calls, [])
                self.assertEqual(parent.forward_calls, [])
                self.assertEqual(wrapper.device, parent.device)
                self.assertIsNone(wrapper.last_decision)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
                wrapper.eval()
                self.assertFalse(wrapper.training)
                self.assertFalse(parent.training)
                wrapper.train()
                self.assertTrue(wrapper.training)
                self.assertTrue(parent.training)
                self.assertIs(parent.weight.requires_grad, requires_grad)

    def test_stochastic_parent_consumes_exactly_one_selection_rng_step(self):
        policy_class = self.policy_class()
        observation = view()
        memory = object()
        original_rng = torch.get_rng_state()
        observed_changes = set()
        try:
            for seed in (0, 1):
                with self.subTest(seed=seed):
                    torch.manual_seed(seed)
                    initial_rng = torch.get_rng_state()
                    reference = StochasticParent(None)
                    selected, _ = reference.act(observation, memory, deterministic=False)
                    expected_rng = torch.get_rng_state()
                    parent = StochasticParent(None)
                    torch.set_rng_state(initial_rng)
                    wrapper = policy_class(parent)

                    action, returned_memory = wrapper.act(observation, memory)

                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertEqual(len(parent.act_calls), 1)
                    self.assertEqual(parent.forward_calls, [])
                    self.assertIs(parent.selected, selected)
                    self.assertIs(parent.act_calls[0][0], observation)
                    self.assertIs(parent.act_calls[0][1], memory)
                    self.assertIs(parent.act_calls[0][2], False)
                    self.assertIs(returned_memory, parent.output_memory)
                    self.assertIs(action, observation.legal_actions[1])
                    self.assertIs(wrapper.last_decision.action, action)
                    expected_change = selected is observation.legal_actions[0]
                    self.assertIs(wrapper.last_decision.changed, expected_change)
                    observed_changes.add(wrapper.last_decision.changed)
            self.assertEqual(observed_changes, {False, True})
        finally:
            torch.set_rng_state(original_rng)


if __name__ == "__main__":
    unittest.main()
