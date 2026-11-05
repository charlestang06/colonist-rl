import math

import torch

from colonist_rl.policy import GraphPolicy
from colonist_rl.training import (
    PPOConfig,
    RolloutCollector,
    advantages,
    imitation_warmstart,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)


def test_actual_four_player_rollout_ppo_update_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    policy = GraphPolicy(hidden_size=16)
    collector = RolloutCollector(policy, seed=4, max_game_steps=100)
    records = collector.collect(12)
    assert len(records) == 12
    assert all(r.action_index < len(r.observation.legal_actions) for r in records)
    assert all(r.observation.player_id == r.observation.context.actor for r in records)
    previous = {k: v.clone() for k, v in policy.state_dict().items()}
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    metrics = ppo_update(policy, optimizer, records, PPOConfig(epochs=2, sequence_length=4))
    assert all(math.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(previous[k], v) for k, v in policy.state_dict().items())
    path = tmp_path / "policy.pt"
    save_checkpoint(path, policy, optimizer, steps=12)
    loaded, state = load_checkpoint(path)
    assert state["optimizer_state"] is not None
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in policy.state_dict().items())


def test_terminal_vs_truncation_bootstrap_and_zero_reward():
    adv, ret = advantages([0, 0], [0, 0], [0, 0], [False, True])
    assert list(adv) == [0, 0] and list(ret) == [0, 0]
    _, terminal = advantages([1], [0.5], [0], [True])
    _, cutoff = advantages([0], [0.5], [0.8], [True])
    assert terminal[0] == 1 and abs(cutoff[0] - 0.8) < 1e-6


def test_bc_warmstart_uses_real_teacher_decisions():
    torch.set_num_threads(1)
    policy = GraphPolicy(hidden_size=16)
    metrics = imitation_warmstart(policy, decisions=8, epochs=1, seed=3)
    assert metrics["decisions"] == 8
    assert math.isfinite(metrics["loss"])


def test_collector_handles_incoming_learner_trade_without_opponent_lookup():
    from colonist_rl.contracts import RESOURCES, TradeOffer
    from colonist_rl.opponents import HeuristicPolicy

    torch.set_num_threads(1)
    collector = RolloutCollector(GraphPolicy(hidden_size=16), seed=5)
    env = collector.env
    teacher = HeuristicPolicy(seed=5)
    view = env.observe()
    for _ in range(1000):
        if (
            view.context.kind == "PLAY_TURN"
            and view.context.has_rolled
            and view.player_id != collector.learner
        ):
            break
        view = env.step(teacher.act(view)).observation
    proposer = view.player_id
    before = [env._hand(p) for p in range(4)]
    for p, resource in ((proposer, "WOOD"), (collector.learner, "BRICK")):
        env.game.state.player_state[f"P{p}_{resource}_IN_HAND"] += 1
        env.game.state.resource_freqdeck[RESOURCES.index(resource)] -= 1
    env._record_known_changes(before, "PUBLIC_TEST_GRANT", proposer)
    env._refresh_actions()
    env.offer_trade(
        TradeOffer(
            "incoming-learner", proposer, collector.learner, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0)
        )
    )
    before_version = env.version
    assert collector.driver.view == collector.driver.view
    assert env.version == before_version
    record = collector.collect(1)[0]
    assert record.observation.context.kind == "TRADE_RESPONSE"


def test_short_time_limits_can_rotate_past_an_episode_without_learner_decision():
    torch.set_num_threads(1)
    collector = RolloutCollector(GraphPolicy(hidden_size=16), seed=1, max_game_steps=1)
    records = collector.collect(2)
    assert len(records) == 2
    assert all(r.truncated and not r.terminated for r in records)


def test_truncation_bootstrap_uses_actual_passive_learner_state():
    torch.set_num_threads(1)
    policy = GraphPolicy(hidden_size=16)
    collector = RolloutCollector(policy, seed=7, max_game_steps=2)
    records = collector.collect(2)
    assert records[-1].truncated
    view = collector.env.observe(collector.learner)
    assert view.legal_actions == ()
    with torch.no_grad():
        expected = policy(view, collector.memory).value.item()
    assert abs(records[-1].next_value - expected) < 1e-6


def test_evaluation_rotates_seats_and_reuses_each_board_block():
    from colonist_rl.training import evaluate

    result = evaluate(games=4, seed=20, max_game_steps=30)
    assert [g["seat"] for g in result["outcomes"]] == [0, 1, 2, 3]
    assert {g["board_seed"] for g in result["outcomes"]} == {20}
    assert result["truncations"] == 4
    assert result["confidence_interval_note"].startswith("Nominal")
