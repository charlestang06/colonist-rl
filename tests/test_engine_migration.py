"""New-engine mechanics preserve the wrapper's existing learner vocabulary."""

import pytest
from catanatron.models.enums import Action as EngineAction
from catanatron.models.enums import ActionPrompt, ActionType
from test_simulator import grant, play_to_main

from colonist_rl.contracts import Action
from colonist_rl.simulator import FourPlayerGame


@pytest.mark.parametrize("phase", [ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES])
def test_native_negotiation_phases_fail_at_the_observation_boundary(phase):
    env = FourPlayerGame()
    env.reset(seed=4)
    env.game.state.current_prompt = phase
    for operation in (env.observe, env._refresh_actions):
        with pytest.raises(NotImplementedError, match="native.*trade|native.*negotiation"):
            operation()


@pytest.mark.parametrize("kind", ["OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE"])
def test_native_trade_payloads_are_never_collapsed_into_wrapper_actions(kind):
    env = FourPlayerGame()
    env.reset(seed=4)
    candidate = EngineAction(env.game.state.current_color(), ActionType[kind], None)
    with pytest.raises(NotImplementedError, match="native.*trade|native.*negotiation"):
        env._convert(candidate)
    env.game.playable_actions = [candidate]
    with pytest.raises(NotImplementedError, match="native.*trade|native.*negotiation"):
        env.observe()


def test_every_exposed_engine_action_kind_fits_unchanged_policy_vocabulary():
    from colonist_rl.policy import KINDS

    native = {"OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE"}
    assert {a.name for a in ActionType} - native <= set(KINDS)
    assert not {"OFFER_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE"} & set(KINDS)
    env = FourPlayerGame(max_steps=500)
    view = env.reset(seed=4)
    for _ in range(500):
        assert {a.kind for a in view.legal_actions} <= set(KINDS)
        result = env.step(view.legal_actions[0])
        view = result.observation
        if result.terminated or result.truncated:
            break


def test_reset_explicitly_uses_official_spiral_and_seed_offset(monkeypatch):
    from colonist_rl import simulator

    actual_game = simulator.Game
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return actual_game(*args, **kwargs)

    monkeypatch.setattr(simulator, "Game", capture)
    env = FourPlayerGame()
    env.reset(seed=0)
    assert calls[0]["seed"] == 1
    assert calls[0]["number_placement"] == "official_spiral"
    assert env.game.playable_actions


def test_roll_reads_action_record_result(monkeypatch):
    env = FourPlayerGame()
    view = play_to_main(env)
    env.step(Action("END_TURN"))
    monkeypatch.setattr("catanatron.apply_action.roll_dice", lambda: (2, 3))
    view = env.step(Action("ROLL")).observation
    assert view.context.has_rolled
    assert any(event.kind == "DICE_ROLL" and event.count == 5 for event in view.events)
    env.assert_resource_conservation()


def test_robber_result_is_visible_only_to_thief_and_victim(monkeypatch):
    env = FourPlayerGame()
    view = play_to_main(env)
    actor = view.player_id
    for p in range(4):
        if p != actor:
            grant(env, p, (1, 0, 0, 0, 0))
    env.game.state.current_prompt = ActionPrompt.MOVE_ROBBER
    env.game.state.is_moving_knight = True
    env._refresh_actions()
    candidate = next(a for a in env.observe().legal_actions if a.partner is not None)
    victim = candidate.partner
    env.step(candidate)
    record = env.game.state.action_records[-1]
    assert len(record.action.value) == 2
    assert record.result is not None
    for observer in range(4):
        event = next(e for e in reversed(env.observe(observer).events) if e.kind == "STEAL")
        assert event.resource == (record.result if observer in (actor, victim) else None)
        assert event.give == event.receive == (0, 0, 0, 0, 0)
    env.assert_resource_conservation()


def test_bought_development_identity_stays_private_and_ages_natively():
    env = FourPlayerGame()
    view = play_to_main(env)
    actor = view.player_id
    grant(env, actor, (0, 0, 1, 1, 1))
    env.game.state.development_listdeck[-1] = "KNIGHT"
    view = env.step(Action("BUY_DEVELOPMENT_CARD")).observation
    assert env.game.state.action_records[-1].result == "KNIGHT"
    assert Action("PLAY_KNIGHT_CARD") not in view.legal_actions
    for observer in range(4):
        event = env.observe(observer).events[-1]
        assert event.kind == "BUY_DEVELOPMENT_CARD" and event.resource is None
        assert env.observe(observer).development[0] == (1 if observer == actor else 0)
    from colonist_rl.opponents import HeuristicPolicy

    policy = HeuristicPolicy(seed=0)
    view = env.step(Action("END_TURN")).observation
    for _ in range(200):
        if view.player_id == actor and view.context.kind == "PLAY_TURN":
            break
        view = env.step(policy.act(view)).observation
    else:
        raise AssertionError("purchaser never reached its next turn")
    assert env.game.state.player_state[f"P{actor}_KNIGHT_OWNED_AT_START"]
    assert Action("PLAY_KNIGHT_CARD") in view.legal_actions
