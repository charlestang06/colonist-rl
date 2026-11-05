"""Public transition protocol checks: fake games, no engine or model execution."""

import random
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from colonist_rl import evaluation, training
from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, DecisionContext, Observation, PublicPlayer


class PublicPolicy:
    """A public actor double; deliberately has no domestic-trade proposal hook."""

    def __init__(self, seed=None):
        pass

    def act(self, view):
        return view.legal_actions[0]


class TrapResult:
    """Only flags are public transition inputs; diagnostics stay end-of-game only."""

    def __init__(self, game, *, terminated, truncated):
        self.game = game
        self.terminated = terminated
        self.truncated = truncated

    @property
    def observation(self):
        raise AssertionError("StepResult.observation must not supply the public after view")

    @property
    def rewards(self):
        raise AssertionError("StepResult.rewards must not enter a public transition")

    @property
    def info(self):
        raise AssertionError("StepResult.info must not enter a public transition")

    @property
    def winner(self):
        assert self.game.version == 4, "private outcome read before the game ended"
        self.game.private_reads.append("winner")
        return 0 if self.terminated else None


@pytest.fixture
def fake_games(monkeypatch):
    games = []
    finish = ["terminated"]
    events = []
    board = BoardSpec.from_hexes([(0, 0)])
    rules = RulesConfig.standard()

    class FakeGame:
        def __init__(self, *, max_steps):
            self.version = 0
            self.current_player = 0
            self.observations = []
            self.private_reads = []
            games.append(self)

        def reset(self, *, seed):
            self.seed = seed

        def observe(self, observer=None):
            self.observations.append(observer)
            player = self.current_player if observer is None else observer
            return Observation(
                player_id=player,
                board=board,
                rules=rules,
                context=DecisionContext(
                    "PLAY_TURN", self.current_player, self.current_player, self.version
                ),
                players=tuple(PublicPlayer(p, 2, p + 1, 0) for p in range(4)),
                hand=(player + 1, 0, 0, 0, 0),
                development=(0, 0, 0, 0, 0),
                buildings=(),
                roads=(),
                robber_tile=0,
                legal_actions=(Action("END_TURN"),),
            )

        def step(self, action):
            events.append((self, "step", self.current_player, self.version))
            self.version += 1
            self.current_player = (self.current_player + 1) % 4
            return TrapResult(
                self,
                terminated=self.version == 4 and finish[0] == "terminated",
                truncated=self.version == 4 and finish[0] == "truncated",
            )

        def _actual_vp(self):
            self.private_reads.append("victory_points")
            return [2, 2, 2, 2]

    monkeypatch.setattr(evaluation, "FourPlayerGame", FakeGame)
    monkeypatch.setattr(evaluation, "HeuristicPolicy", PublicPolicy)
    monkeypatch.setattr(training, "HeuristicPolicy", PublicPolicy)
    monkeypatch.setattr(training, "RandomPolicy", PublicPolicy)
    return SimpleNamespace(games=games, finish=finish, events=events)


def test_callback_keeps_original_observer_and_runs_before_the_next_actor(fake_games):
    received = []

    def record(step):
        game = fake_games.games[-1]
        assert game.private_reads == []
        received.append(step)
        fake_games.events.append((game, "callback", step.before.player_id, game.version))

    result = evaluation.evaluate_blocks(board_seeds=[41], on_candidate_step=record)

    assert result["games"] == 4
    assert len(received) == 4
    for seat, (game, step) in enumerate(zip(fake_games.games, received)):
        assert isinstance(step, evaluation.CandidatePublicStep)
        assert step.before.player_id == step.after.player_id == seat
        assert step.after.context.actor == (seat + 1) % 4
        assert step.after.hand == (seat + 1, 0, 0, 0, 0)
        assert step.before.context.version == seat
        assert step.after.context.version == seat + 1
        assert step.action is step.before.legal_actions[0]
        assert game.observations.count(seat) == 1
        assert [
            (kind, actor, version) for g, kind, actor, version in fake_games.events if g is game
        ] == (
            [("step", actor, actor) for actor in range(seat + 1)]
            + [("callback", seat, seat + 1)]
            + [("step", actor, actor) for actor in range(seat + 1, 4)]
        )
        assert [field.name for field in fields(step)] == [
            "before",
            "action",
            "after",
            "terminated",
            "truncated",
        ]
        with pytest.raises(FrozenInstanceError):
            step.after = step.before


@pytest.mark.parametrize(
    "finish, flags", [("terminated", (True, False)), ("truncated", (False, True))]
)
def test_callback_includes_terminal_candidate_action(fake_games, finish, flags):
    fake_games.finish[0] = finish
    received = []
    result = evaluation.evaluate_blocks(board_seeds=[42], on_candidate_step=received.append)

    assert len(received) == 4
    assert [(step.terminated, step.truncated) for step in received] == [
        (False, False),
        (False, False),
        (False, False),
        flags,
    ]
    assert (result["outcomes"][-1]["terminated"], result["outcomes"][-1]["truncated"]) == flags


def test_no_callback_keeps_default_observation_counts_and_legacy_positional_api(fake_games):
    result = evaluation.evaluate_blocks(None, board_seeds=[43])
    assert result["games"] == 4
    assert [game.observations for game in fake_games.games] == [[None] * 11] * 4

    legacy = training.evaluate(None, 1, 43, "heuristic", False, 20000)
    assert legacy["games"] == 1
    assert fake_games.games[-1].observations == [None] * 11


class ModePolicy:
    """Exercise module-mode restoration without constructing a neural network."""

    def __init__(self):
        self.training = True
        self.child = SimpleNamespace(training=False)

    def modules(self):
        return iter((self, self.child))

    def eval(self):
        for module in self.modules():
            module.training = False
        return self

    def act(self, view, memory, *, deterministic):
        assert not self.training and not self.child.training
        return view.legal_actions[0], None


def test_callback_exception_propagates_after_rng_and_module_modes_restore(fake_games):
    policy = ModePolicy()
    modes = [module.training for module in policy.modules()]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    failure = RuntimeError("public pair persistence failed")

    def fail(step):
        assert step.before.player_id == step.after.player_id == 0
        assert fake_games.games[-1].private_reads == []
        random.random()
        np.random.random()
        torch.rand(1)
        policy.training = True
        policy.child.training = True
        raise failure

    with pytest.raises(RuntimeError) as caught:
        evaluation.evaluate_blocks(policy, board_seeds=[44], on_candidate_step=fail)

    assert caught.value is failure
    assert random.getstate() == python_rng
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_rng[0]
    np.testing.assert_array_equal(restored_numpy[1], numpy_rng[1])
    assert restored_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert [module.training for module in policy.modules()] == modes
    assert fake_games.games[-1].version == 1
    assert fake_games.games[-1].private_reads == []


def test_invalid_callback_is_rejected_before_any_game(fake_games):
    with pytest.raises(ValueError, match="on_candidate_step must be callable or None"):
        evaluation.evaluate_blocks(board_seeds=[45], on_candidate_step=object())
    assert fake_games.games == []
