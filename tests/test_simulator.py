"""Adapter contracts and behavioral regression checks against the pinned engine."""

import importlib.util
import random

import pytest


def simulator():
    assert importlib.util.find_spec("colonist_rl.simulator") is not None, (
        "simulator adapter missing"
    )
    from colonist_rl.simulator import FourPlayerGame

    return FourPlayerGame


def test_reset_exposes_only_immutable_player_view():
    env = simulator()()
    view = env.reset(seed=0)
    from colonist_rl.contracts import Action, Observation

    assert isinstance(view, Observation)
    assert view.player_id == env.current_player
    assert len(view.players) == 4
    assert len(view.board.tiles) == 19
    assert len(view.board.vertices) == 54
    assert len(view.board.edges) == 72
    assert all(isinstance(a, Action) for a in view.legal_actions)
    assert all(a.kind == "BUILD_SETTLEMENT" for a in view.legal_actions)
    assert view.hand == (0, 0, 0, 0, 0)
    assert env.version == 0


def test_interleaved_seeded_games_are_reproducible_and_preserve_global_rng():
    cls = simulator()
    random.seed(140)
    state = random.getstate()
    first, second = cls(), cls()
    a, b = first.reset(seed=0), second.reset(seed=0)
    assert a == b
    for _ in range(100):
        assert a.legal_actions
        action = a.legal_actions[0]
        a = first.step(action).observation
        b = second.step(action).observation
        assert a == b
    assert random.getstate() == state


def test_six_player_rules_fail_explicitly():
    cls = simulator()
    from colonist_rl.board import RulesConfig

    with pytest.raises(NotImplementedError, match="six|6|four|4"):
        cls(rules=RulesConfig.extended())


def test_random_complete_game_and_resource_conservation():
    cls = simulator()
    from colonist_rl.opponents import HeuristicPolicy

    env = cls(max_steps=15000)
    view = env.reset(seed=18)
    policy = HeuristicPolicy(seed=18)
    for _ in range(15000):
        result = env.step(policy.act(view))
        env.assert_resource_conservation()
        view = result.observation
        if result.terminated or result.truncated:
            break
    assert result.terminated, f"game failed to finish: {view.players}"
    assert result.winner in range(4)
    assert len(result.rewards) == 4
    assert result.rewards[result.winner] > 0.75


def play_to_main(env):
    from colonist_rl.opponents import HeuristicPolicy

    policy = HeuristicPolicy(seed=17)
    view = env.reset(seed=17)
    for _ in range(1000):
        if (
            view.context.kind == "PLAY_TURN"
            and view.context.has_rolled
            and not view.context.free_roads
        ):
            return view
        view = env.step(policy.act(view)).observation
    raise AssertionError("never reached main phase")


def grant(env, player, vector):
    from colonist_rl.contracts import RESOURCES

    before = [env._hand(p) for p in range(4)]
    for i, resource in enumerate(RESOURCES):
        env.game.state.player_state[f"P{player}_{resource}_IN_HAND"] += vector[i]
        env.game.state.resource_freqdeck[i] -= vector[i]
    env._record_known_changes(before, "TEST_PUBLIC_GRANT", player)
    env._refresh_actions()


def test_trade_rejection_is_identity_and_acceptance_is_atomic_once():
    cls = simulator()
    from colonist_rl.contracts import Action, TradeOffer

    env = cls()
    view = play_to_main(env)
    p, q = view.player_id, (view.player_id + 1) % 4
    grant(env, p, (2, 0, 0, 0, 0))
    grant(env, q, (0, 2, 0, 0, 0))
    hands = [env._hand(i) for i in range(4)]
    offer = TradeOffer("reject", p, q, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0))
    response = env.offer_trade(offer)
    assert env.current_player == q and response.context.kind == "TRADE_RESPONSE"
    with pytest.raises(ValueError):
        env.step(Action("END_TURN"))
    env.step(Action("REJECT_TRADE", offer_id="reject"))
    assert [env._hand(i) for i in range(4)] == hands
    with pytest.raises(ValueError, match="stale|used"):
        env.offer_trade(offer)
    response = env.offer_trade(TradeOffer("accept", p, q, offer.give, offer.receive))
    accept = Action("ACCEPT_TRADE", offer_id="accept")
    assert accept in response.legal_actions
    env.step(accept)
    assert env._hand(p) == tuple(h - g + r for h, g, r in zip(hands[p], offer.give, offer.receive))
    assert env._hand(q) == tuple(h + g - r for h, g, r in zip(hands[q], offer.give, offer.receive))
    env.assert_resource_conservation()
    with pytest.raises(ValueError):
        env.step(accept)


def test_private_opponent_composition_and_deck_order_do_not_change_view():
    cls = simulator()
    env = cls()
    view = play_to_main(env)
    p = view.player_id
    a, b = [q for q in range(4) if q != p][:2]
    grant(env, a, (1, 0, 0, 0, 0))
    grant(env, b, (0, 1, 0, 0, 0))
    before = env.observe(p)
    env.game.state.player_state[f"P{a}_WOOD_IN_HAND"] -= 1
    env.game.state.player_state[f"P{a}_BRICK_IN_HAND"] += 1
    env.game.state.player_state[f"P{b}_WOOD_IN_HAND"] += 1
    env.game.state.player_state[f"P{b}_BRICK_IN_HAND"] -= 1
    env.game.state.development_listdeck.reverse()
    env._refresh_actions()
    assert env.observe(p) == before
    env.assert_resource_conservation()


def test_road_building_places_two_roads_without_resource_cost():
    cls = simulator()
    from colonist_rl.contracts import RESOURCES, Action

    env = cls()
    view = play_to_main(env)
    p = view.player_id
    before = [env._hand(q) for q in range(4)]
    for i, resource in enumerate(RESOURCES):
        env.game.state.resource_freqdeck[i] += env._hand(p)[i]
        env.game.state.player_state[f"P{p}_{resource}_IN_HAND"] = 0
    env._record_known_changes(before, "TEST_PUBLIC_RETURN", p)
    env.game.state.player_state[f"P{p}_ROAD_BUILDING_IN_HAND"] = 1
    env._refresh_actions()
    view = env.observe()
    play = Action("PLAY_ROAD_BUILDING")
    assert play in view.legal_actions
    view = env.step(play).observation
    assert view.context.free_roads == 2
    assert view.legal_actions and all(a.kind == "BUILD_ROAD" for a in view.legal_actions)
    view = env.step(view.legal_actions[0]).observation
    assert view.context.free_roads == 1
    view = env.step(view.legal_actions[0]).observation
    assert view.context.free_roads == 0 and view.hand == (0, 0, 0, 0, 0)


def test_bought_development_card_cannot_be_played_until_next_turn():
    cls = simulator()
    from colonist_rl.contracts import Action

    env = cls()
    view = play_to_main(env)
    p = view.player_id
    grant(env, p, (0, 0, 1, 1, 1))
    env.game.state.development_listdeck[-1] = "KNIGHT"
    view = env.step(Action("BUY_DEVELOPMENT_CARD")).observation
    assert Action("PLAY_KNIGHT_CARD") not in view.legal_actions


def test_forced_discard_is_player_chosen_and_private_to_discarder(monkeypatch):
    cls = simulator()
    from colonist_rl.contracts import Action

    env = cls()
    view = play_to_main(env)
    q = (view.player_id + 1) % 4
    grant(env, q, (8, 0, 0, 0, 0))
    env.step(Action("END_TURN"))
    monkeypatch.setattr("catanatron.state.roll_dice", lambda: (3, 4))
    view = env.step(Action("ROLL")).observation
    assert view.context.kind == "DISCARD"
    discarder = view.player_id
    assert discarder == q
    observer = (q + 1) % 4
    before_count = sum(view.hand)
    selected = next(a for a in view.legal_actions if a.resource == "WOOD")
    view = env.step(selected).observation
    assert sum(env.observe(q).hand) == before_count - 1
    hidden_event = env.observe(observer).events[-1]
    own_event = env.observe(q).events[-1]
    assert hidden_event.kind == "DISCARD" and hidden_event.resource is None
    assert own_event.resource == "WOOD"
    assert env.observe(observer).bank is None
    while view.context.kind == "DISCARD":
        view = env.step(view.legal_actions[0]).observation
    assert view.context.kind == "MOVE_ROBBER"
    assert view.context.actor == view.context.turn_owner
    env.assert_resource_conservation()


def test_random_policy_always_returns_a_visible_legal_action():
    cls = simulator()
    from colonist_rl.opponents import RandomPolicy

    env = cls(max_steps=600)
    policy = RandomPolicy(seed=10)
    view = env.reset(seed=10)
    for _ in range(600):
        action = policy.act(view)
        assert action in view.legal_actions
        result = env.step(action)
        view = result.observation
        if result.terminated or result.truncated:
            break
    env.assert_resource_conservation()


def play_to_preroll(env):
    from colonist_rl.opponents import HeuristicPolicy

    policy = HeuristicPolicy(seed=17)
    view = env.reset(seed=17)
    for _ in range(100):
        if view.context.kind == "PLAY_TURN" and not view.context.has_rolled:
            return view
        view = env.step(policy.act(view)).observation
    raise AssertionError("never reached the first pre-roll turn")


@pytest.mark.parametrize(
    "card,kind",
    [
        ("ROAD_BUILDING", "PLAY_ROAD_BUILDING"),
        ("YEAR_OF_PLENTY", "PLAY_YEAR_OF_PLENTY"),
        ("MONOPOLY", "PLAY_MONOPOLY"),
        ("KNIGHT", "PLAY_KNIGHT_CARD"),
    ],
)
def test_old_development_card_resolves_before_roll_without_advancing_turn(card, kind, monkeypatch):
    from colonist_rl.contracts import DEVELOPMENT_CARDS, RESOURCES, Action

    env = simulator()()
    view = play_to_preroll(env)
    actor = view.player_id
    other = (actor + 1) % 4
    # Controlled old development hands retain the normal resource bank and ledger.
    for development in ("KNIGHT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"):
        env.game.state.development_listdeck.remove(development)
        env.game.state.player_state[f"P{actor}_{development}_IN_HAND"] += 1
    if card == "MONOPOLY":
        grant(env, other, (2, 0, 0, 0, 0))
    elif card == "ROAD_BUILDING":
        before = [env._hand(p) for p in range(4)]
        for i, resource in enumerate(RESOURCES):
            env.game.state.resource_freqdeck[i] += env._hand(actor)[i]
            env.game.state.player_state[f"P{actor}_{resource}_IN_HAND"] = 0
        env._record_known_changes(before, "TEST_PUBLIC_RETURN", actor)
    env._refresh_actions()
    view = env.observe()
    if card == "YEAR_OF_PLENTY":
        selected = Action(kind, resource="WOOD", second_resource="BRICK")
    elif card == "MONOPOLY":
        selected = Action(kind, resource="WOOD")
    else:
        selected = Action(kind)
    assert selected in view.legal_actions
    before_hands = [env._hand(p) for p in range(4)]
    roads_before = len(view.roads)
    turn_before = view.context.turn_number
    view = env.step(selected).observation
    if card == "ROAD_BUILDING":
        assert view.context.free_roads == 2
        for expected_remaining in (1, 0):
            assert view.legal_actions and all(a.kind == "BUILD_ROAD" for a in view.legal_actions)
            view = env.step(view.legal_actions[0]).observation
            assert view.context.free_roads == expected_remaining
        assert len(view.roads) == roads_before + 2
        assert env._hand(actor) == before_hands[actor]
    elif card == "KNIGHT":
        assert view.context.kind == "MOVE_ROBBER"
        assert not view.context.has_rolled
        view = env.step(view.legal_actions[0]).observation
    elif card == "YEAR_OF_PLENTY":
        assert env._hand(actor) == tuple(n + int(i < 2) for i, n in enumerate(before_hands[actor]))
    elif card == "MONOPOLY":
        assert env._hand(actor)[0] == sum(hand[0] for hand in before_hands)
        assert all(env._hand(p)[0] == 0 for p in range(4) if p != actor)
    assert view.player_id == view.context.actor == view.context.turn_owner == actor
    assert view.context.kind == "PLAY_TURN"
    assert view.context.turn_number == turn_before
    assert not view.context.has_rolled
    assert view.context.has_played_development_card
    assert view.development[DEVELOPMENT_CARDS.index(card)] == 0
    assert Action("ROLL") in view.legal_actions
    assert not any(a.kind.startswith("PLAY_") for a in view.legal_actions)
    env.assert_resource_conservation()
    monkeypatch.setattr("catanatron.state.roll_dice", lambda: (2, 3))
    view = env.step(Action("ROLL")).observation
    assert view.context.has_rolled
    assert view.context.turn_owner == actor
    assert not any(a.kind.startswith("PLAY_") for a in view.legal_actions)
    env.assert_resource_conservation()


def test_new_development_cards_stay_unplayable_before_rolling():
    from colonist_rl.contracts import DEVELOPMENT_CARDS, Action

    env = simulator()()
    view = play_to_preroll(env)
    actor = view.player_id
    for card in ("KNIGHT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"):
        env.game.state.development_listdeck.remove(card)
        env.game.state.player_state[f"P{actor}_{card}_IN_HAND"] += 1
        env._new_development[actor][DEVELOPMENT_CARDS.index(card)] = 1
    env._refresh_actions()
    assert env.observe().legal_actions == (Action("ROLL"),)
