"""Manufactured public projections verify arithmetic and boundaries, never strength."""

import importlib.util
import random
from copy import deepcopy
from dataclasses import replace

import pytest

from colonist_rl.board import BoardSpec, Port, RulesConfig
from colonist_rl.contracts import (
    Action,
    Building,
    DecisionContext,
    Observation,
    PublicPlayer,
    Road,
)


def planner():
    assert importlib.util.find_spec("colonist_rl.public_turn_planner") is not None, (
        "public turn planner is not implemented"
    )
    from colonist_rl import public_turn_planner

    return public_turn_planner


def trade(give_resource=0, receive_resource=3, ratio=4):
    give = tuple(ratio if i == give_resource else 0 for i in range(5))
    receive = tuple(int(i == receive_resource) for i in range(5))
    return Action("MARITIME_TRADE", give=give, receive=receive)


def view(hand=(8, 0, 0, 1, 2), bank=None):
    board = BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,))
    # Large aggregate opponent count makes current menu the only withdrawal proof.
    opponents = 95 - sum(hand) - sum(bank) if bank is not None else 30
    return Observation(
        10,
        board,
        RulesConfig.standard(),
        DecisionContext("PLAY_TURN", 10, 10, 1, has_rolled=True),
        (
            PublicPlayer(10, 1, sum(hand), 0),
            PublicPlayer(20, 2, opponents, 0),
            PublicPlayer(30, 0, 0, 0),
            PublicPlayer(40, 0, 0, 0),
        ),
        hand,
        (0, 0, 0, 0, 0),
        (Building(0, 10),),
        (),
        99,
        (trade(), trade(receive_resource=4), Action("END_TURN")),
        bank=bank,
    )


def test_named_immutable_recipes_and_explicit_geometry_helpers():
    from colonist_rl import consequences, settlement_sites

    assert hasattr(consequences, "PUBLIC_BUILD_COSTS"), "named recipes missing"
    costs = consequences.PUBLIC_BUILD_COSTS
    assert tuple(costs) == ("BUILD_ROAD", "BUILD_SETTLEMENT", "BUILD_CITY", "BUY_DEVELOPMENT_CARD")
    assert costs["BUILD_CITY"] == (0, 0, 0, 2, 3)
    with pytest.raises(TypeError):
        costs["BUILD_CITY"] = (0, 0, 0, 0, 0)
    board = view().board
    assert settlement_sites.connected_settlement_sites(board, (), (Road(1, 10),), 10) == (1, 2)
    # The opponent at 1 blocks continuation, but the road may reach it from 0.
    assert settlement_sites.connectable_road_edges(
        board, (Building(1, 20),), (Road(0, 10),), 10
    ) == (5,)


def test_two_distinct_trades_then_city_returns_original_root_action():
    p = planner()
    observation = view()
    before = deepcopy(observation)
    plan = p.plan_turn(observation)
    assert plan.reason == "planned"
    assert plan.first_action is observation.legal_actions[plan.root_index]
    assert plan.first_action.kind == "MARITIME_TRADE"
    assert plan.root_index == 1  # Canonical tie: ore receive tuple sorts before wheat.
    assert plan.nodes > 6  # Infeasible build/trade candidates are attempted and counted.
    assert [s.kind for s in plan.steps] == ["MARITIME_TRADE", "MARITIME_TRADE", "BUILD_CITY"]
    assert plan.steps[-1].target == 0
    assert plan.score_after_micro > plan.score_before_micro + 250_000
    assert plan.nodes <= 2048 and plan.geometry_visits <= 100000
    assert observation == before


def projection(observation):
    p = planner()
    work = p._Work(p.TurnPlannerConfig(max_geometry_visits=1_000_000))
    geometry = p._Geometry(observation, work)
    return p._initial_projection(observation), geometry


def replay(observation, steps):
    p = planner()
    state, geometry = projection(observation)
    history = [state]
    for step in steps:
        after = p._transition(observation, state, step, geometry.describe(state))
        assert after is not None, step
        # The same spent resources are returned to the bank, never created.
        for h, l, u, h2, l2, u2 in zip(
            state.hand,
            state.bank_lower,
            state.bank_upper,
            after.hand,
            after.bank_lower,
            after.bank_upper,
        ):
            assert h + l == h2 + l2
            assert h + u == h2 + u2
            assert h2 >= 0 and 0 <= l2 <= u2
        state = after
        history.append(state)
    return history


def test_unknown_withdrawal_evidence_is_maximum_and_consumed_once():
    p = planner()
    observation = view(hand=(8, 0, 0, 2, 1))
    state, geometry = projection(observation)
    assert state.bank_lower[4] == 1
    step = p.PlannedStep("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 0, 1))
    after = p._transition(observation, state, step, geometry.describe(state))
    assert after.bank_lower[4] == 0
    assert p._transition(observation, after, step, geometry.describe(after)) is None
    duplicated_menu = replace(observation, legal_actions=observation.legal_actions * 2)
    assert p._initial_projection(duplicated_menu).bank_lower[4] == 1
    plan = p.plan_turn(observation)
    assert not any(s.kind == "BUILD_CITY" for s in plan.steps)


@pytest.mark.parametrize("ore,expected", [(1, False), (2, True)])
def test_exact_public_bank_allows_only_available_repeated_withdrawals(ore, expected):
    p = planner()
    observation = view(hand=(8, 0, 0, 2, 1), bank=(11, 19, 19, 17, ore))
    plan = p.plan_turn(observation)
    assert any(s.kind == "BUILD_CITY" for s in plan.steps) is expected
    if expected:
        assert [s.receive[4] for s in plan.steps] == [1, 1, 0]
        history = replay(observation, plan.steps)
        assert history[-1].hand == (0, 0, 0, 0, 0)
        assert history[-1].cities_left == history[0].cities_left - 1
        assert history[-1].settlements_left == history[0].settlements_left + 1
        assert history[-1].bank_lower == history[-1].bank_upper


@pytest.mark.parametrize(
    "resource,ratio,owned,wood,expected",
    [
        ("WOOD", 2, True, 4, True),
        (None, 3, True, 6, True),
        ("WOOD", 2, False, 4, False),
        ("ORE", 2, True, 4, False),
    ],
)
def test_specific_generic_and_unowned_ports(resource, ratio, owned, wood, expected):
    p = planner()
    observation = view(hand=(wood, 0, 0, 1, 2))
    port = Port((0, 5) if owned else (2, 3), resource, ratio)
    observation = replace(observation, board=replace(observation.board, ports=(port,)))
    actual_ratio = ratio if owned and resource in (None, "WOOD") else 4
    observation = replace(
        observation,
        legal_actions=(
            trade(ratio=actual_ratio),
            trade(receive_resource=4, ratio=actual_ratio),
            Action("END_TURN"),
        ),
    )
    plan = p.plan_turn(observation)
    assert any(s.kind == "BUILD_CITY" for s in plan.steps) is expected
    if expected:
        assert all(sum(s.give) == ratio for s in plan.steps[:2])
        replay(observation, plan.steps)


def chain_view():
    observation = view(hand=(4, 2, 1, 0, 0))
    return replace(
        observation,
        board=replace(observation.board, ports=(Port((0, 5), "WOOD", 2),)),
        roads=(Road(0, 10),),
        legal_actions=(trade(ratio=2), Action("END_TURN")),
    )


def test_trade_road_settlement_chain_and_conserved_stock():
    p = planner()
    observation = chain_view()
    plan = p.plan_turn(observation)
    assert [s.kind for s in plan.steps] == ["MARITIME_TRADE", "BUILD_ROAD", "BUILD_SETTLEMENT"]
    assert [s.target for s in plan.steps] == [None, 1, 2]
    history = replay(observation, plan.steps)
    assert history[-1].hand == (0, 0, 0, 0, 0)
    assert history[-1].roads_left == history[0].roads_left - 1
    assert history[-1].settlements_left == history[0].settlements_left - 1
    assert history[-1].buildings[-1] == Building(2, 10)


@pytest.mark.parametrize(
    "mutation",
    ["no_roads", "enemy_blocks", "distance", "road_stock", "settlement_stock", "occupied_edge"],
)
def test_chain_obeys_connectivity_blocking_distance_and_piece_stock(mutation):
    p = planner()
    observation = chain_view()
    if mutation == "no_roads":
        observation = replace(observation, roads=())
    elif mutation == "enemy_blocks":
        observation = replace(observation, buildings=(*observation.buildings, Building(1, 20)))
    elif mutation == "distance":
        observation = replace(observation, buildings=(*observation.buildings, Building(3, 20)))
    elif mutation == "occupied_edge":
        observation = replace(observation, roads=(*observation.roads, Road(1, 20)))
    else:
        field = "roads_left" if mutation == "road_stock" else "settlements_left"
        observation = replace(
            observation,
            players=(replace(observation.players[0], **{field: 0}), *observation.players[1:]),
        )
    plan = p.plan_turn(observation)
    assert not any(s.kind == "BUILD_SETTLEMENT" for s in plan.steps)
    state, geometry = projection(observation)
    if mutation == "no_roads" or mutation in ("enemy_blocks", "occupied_edge"):
        assert 1 not in geometry.describe(state).edges
    elif mutation == "distance":
        assert 2 not in geometry.describe(state).sites


def test_new_port_settlement_recomputes_ratio_and_build_returns_proven_supply():
    p = planner()
    observation = replace(view(hand=(3, 1, 1, 1, 0)), roads=(Road(0, 10), Road(1, 10)))
    observation = replace(
        observation,
        board=replace(observation.board, ports=(Port((2, 3), "WOOD", 2),)),
        legal_actions=(Action("END_TURN"),),
    )
    state, geometry = projection(observation)
    assert p._ratio(observation, state, 0) == 4
    after = p._transition(
        observation, state, p.PlannedStep("BUILD_SETTLEMENT", 2), geometry.describe(state)
    )
    assert after.hand == (2, 0, 0, 0, 0)
    assert p._ratio(observation, after, 0) == 2
    # Resource spent on the settlement is now certain bank stock.
    assert after.bank_lower[3] == state.bank_lower[3] + 1
    trade_step = p.PlannedStep("MARITIME_TRADE", give=(2, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0))
    after_trade = p._transition(observation, after, trade_step, geometry.describe(after))
    assert after_trade.hand == (0, 0, 0, 1, 0)


def test_reobserve_uses_only_new_actual_menu_without_stale_steps():
    p = planner()
    policy = p.PublicTurnPlannerPolicy()
    observation = view()
    first = policy.act(observation)
    assert first is observation.legal_actions[0] or first is observation.legal_actions[1]
    new_end = Action("END_TURN")
    updated = replace(
        observation,
        hand=tuple(h - g + r for h, g, r in zip(observation.hand, first.give, first.receive)),
        legal_actions=(new_end,),
    )
    updated = replace(
        updated,
        players=(
            replace(updated.players[0], resource_count=sum(updated.hand)),
            *updated.players[1:],
        ),
    )
    assert policy.act(updated) is new_end


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("not_rolled", "unsupported_context"),
        ("other_actor", "unsupported_context"),
        ("free_roads", "unsupported_context"),
        ("unknown_rules", "unsupported_rules"),
        ("unsupported_h", "unsupported_heuristic_action"),
        ("bad_ratio", "public_contract_mismatch"),
        ("bad_exact_bank", "public_contract_mismatch"),
        ("bad_cost_metadata", "public_contract_mismatch"),
    ],
)
def test_visible_fallback_returns_actual_h_view_member(mutation, reason):
    p = planner()
    observation = view()
    if mutation == "not_rolled":
        observation = replace(observation, context=replace(observation.context, has_rolled=False))
    elif mutation == "other_actor":
        observation = replace(observation, context=replace(observation.context, actor=20))
    elif mutation == "free_roads":
        observation = replace(observation, context=replace(observation.context, free_roads=1))
    elif mutation == "unknown_rules":
        observation = replace(observation, rules=replace(observation.rules, version="unknown"))
    elif mutation == "unsupported_h":
        observation = replace(
            observation, legal_actions=(*observation.legal_actions, Action("BUY_DEVELOPMENT_CARD"))
        )
    elif mutation == "bad_ratio":
        observation = replace(observation, legal_actions=(trade(ratio=3), Action("END_TURN")))
    elif mutation == "bad_exact_bank":
        observation = replace(observation, bank=(0, 0, 0, 0, 0))
    else:
        observation = replace(
            observation,
            legal_actions=(
                Action("BUILD_CITY", target=0, give=(0, 0, 0, 2, 3)),
                Action("END_TURN"),
            ),
        )
    plan = p.plan_turn(observation)
    assert plan.reason == reason
    assert plan.first_action is p.HeuristicViewPolicy().act(observation)
    assert plan.steps == ()


def test_h_view_and_planner_are_public_deterministic_and_preserve_global_rng():
    p = planner()
    observation = view()
    original = deepcopy(observation)
    random.seed(73)
    rng_before = random.getstate()
    first = p.plan_turn(observation)
    h_first = p.HeuristicViewPolicy().act(observation)
    proposal = p.PublicTurnPlannerPolicy().propose(observation)
    assert random.getstate() == rng_before
    random.random()
    assert p.plan_turn(deepcopy(observation)) == first
    assert p.HeuristicViewPolicy().act(observation) == h_first
    assert p.PublicTurnPlannerPolicy().propose(observation) == proposal
    assert observation == original
    assert "env" not in vars(p.PublicTurnPlannerPolicy())


def test_deterministic_caps_distinguish_best_prefix_from_geometry_fallback():
    p = planner()
    observation = view()
    geometry_cap = p.plan_turn(observation, p.TurnPlannerConfig(max_geometry_visits=10))
    assert geometry_cap.reason == geometry_cap.search_stop == "geometry_limit"
    assert geometry_cap.geometry_visits == 10
    assert geometry_cap.first_action is p.HeuristicViewPolicy().act(observation)
    shallow = p.plan_turn(observation, p.TurnPlannerConfig(max_depth=2))
    assert shallow.reason == "insufficient_gain"
    assert not any(s.kind == "BUILD_CITY" for s in shallow.steps)
    # The 3-step fixture finds a useful prefix before a subsequent finite cap.
    capped = p.plan_turn(observation, p.TurnPlannerConfig(max_nodes=100))
    assert capped.search_stop == "node_limit" and capped.nodes == 100
    assert capped.reason == "planned"
    assert capped == p.plan_turn(observation, p.TurnPlannerConfig(max_nodes=100))
    empty = replace(observation, legal_actions=())
    with pytest.raises(ValueError, match="no legal action"):
        p.plan_turn(empty)


def test_geometry_cache_hits_charge_identical_logical_work():
    observation = chain_view()
    state, geometry = projection(observation)
    start = geometry.work.geometry_visits
    first = geometry.describe(state)
    cold = geometry.work.geometry_visits - start
    start = geometry.work.geometry_visits
    assert geometry.describe(state) == first
    assert geometry.work.geometry_visits - start == cold


def test_own_known_vp_gate_does_not_infer_road_award_points():
    p = planner()
    observation = view()
    observation = replace(
        observation,
        development=(0, 0, 0, 0, 4),
        players=(
            replace(observation.players[0], victory_points=5, development_count=4),
            *observation.players[1:],
        ),
    )
    plan = p.plan_turn(observation)
    history = replay(observation, plan.steps)
    assert observation.own_victory_points + history[-1].building_point_gain == 10
    assert observation.players[0].victory_points + history[-1].building_point_gain == 6
    road_only = replace(chain_view(), buildings=(), roads=(Road(0, 10),))
    state, geometry = projection(road_only)
    after = p._transition(
        road_only, state, p.PlannedStep("BUILD_ROAD", 1), geometry.describe(state)
    )
    assert after.building_point_gain == 0


def test_option_b_and_opening_configuration_are_not_exposed():
    p = planner()
    with pytest.raises(TypeError):
        p.TurnPlannerConfig(bank_mode="particles")
    with pytest.raises(TypeError):
        p.TurnPlannerConfig(opening_pair=True)


def test_unvalidated_root_menu_under_node_cap_never_overrides_h_view():
    p = planner()
    observation = replace(
        view(hand=(0, 0, 0, 2, 3)),
        development=(0, 0, 0, 0, 4),
        players=(
            PublicPlayer(10, 5, 5, 4),
            PublicPlayer(20, 2, 30, 0),
            PublicPlayer(30, 0, 0, 0),
            PublicPlayer(40, 0, 0, 0),
        ),
        legal_actions=(Action("BUILD_CITY", 0), Action("END_TURN"), trade(ratio=5)),
    )
    capped = p.plan_turn(observation, p.TurnPlannerConfig(max_nodes=1))
    assert capped.reason == "root_validation_limit"
    assert capped.search_stop == "node_limit" and capped.nodes == 1
    assert capped.first_action is p.HeuristicViewPolicy().act(observation)
    assert capped.steps == ()


def test_extension_falls_back_until_its_engine_replay_contract_can_be_verified():
    p = planner()
    observation = replace(view(), rules=RulesConfig.extended(2))
    assert p.plan_turn(observation).reason == "unsupported_rules"


def test_root_menu_all_supported_shapes_are_validated_before_planning():
    p = planner()
    observation = view()
    malformed = replace(observation, legal_actions=(*observation.legal_actions, trade(ratio=5)))
    assert p.plan_turn(malformed).reason == "public_contract_mismatch"


def engine_main_fixture(monkeypatch, hand=(8, 4, 3, 6, 6), port_ratio=None):
    """Replay just setup and one fixed roll, then arrange a public hand fixture."""
    from colonist_rl.contracts import RESOURCES
    from colonist_rl.opponents import HeuristicPolicy
    from colonist_rl.simulator import FourPlayerGame

    monkeypatch.setattr("catanatron.apply_action.roll_dice", lambda: (2, 3))
    env = FourPlayerGame()
    observation = env.reset(seed=17)
    setup = HeuristicPolicy(17)
    for _ in range(18):
        if observation.context.kind == "PLAY_TURN" and observation.context.has_rolled:
            break
        selected = setup.act(observation)
        if port_ratio is not None and not observation.buildings:
            port = next(port for port in observation.board.ports if port.ratio == port_ratio)
            selected = next(a for a in observation.legal_actions if a.target in port.vertices)
        observation = env.step(selected).observation
    else:
        raise AssertionError("setup replay did not reach normal turn")
    actor = observation.player_id
    before = [env._hand(player) for player in range(4)]
    for i, resource in enumerate(RESOURCES):
        difference = hand[i] - before[actor][i]
        env.game.state.resource_freqdeck[i] -= difference
        env.game.state.player_state[f"P{actor}_{resource}_IN_HAND"] += difference
    env._record_known_changes(before, "TEST_PUBLIC_GRANT", actor)
    env._refresh_actions()
    env.assert_resource_conservation()
    return env


def assert_engine_transition(env, action):
    p = planner()
    observation = env.observe()
    assert action in observation.legal_actions
    state, geometry = projection(observation)
    step = p.PlannedStep(action.kind, action.target, action.give, action.receive)
    predicted = p._transition(observation, state, step, geometry.describe(state))
    assert predicted is not None
    updated = env.step(action).observation
    assert predicted.hand == updated.hand
    assert predicted.buildings == updated.buildings
    assert predicted.roads == updated.roads
    own = next(player for player in updated.players if player.player_id == observation.player_id)
    assert (predicted.roads_left, predicted.settlements_left, predicted.cities_left) == (
        own.roads_left,
        own.settlements_left,
        own.cities_left,
    )
    assert (
        updated.own_victory_points == observation.own_victory_points + predicted.building_point_gain
    )
    env.assert_resource_conservation()
    return updated


def test_standard_engine_replays_public_city_road_and_maritime_shapes(monkeypatch):
    env = engine_main_fixture(monkeypatch)
    for kind in ("BUILD_CITY", "MARITIME_TRADE", "BUILD_ROAD"):
        observation = env.observe()
        action = next(a for a in observation.legal_actions if a.kind == kind)
        assert_engine_transition(env, action)


def test_standard_engine_replays_road_to_distance_legal_settlement(monkeypatch):
    p = planner()
    env = engine_main_fixture(monkeypatch)
    for _ in range(3):
        observation = env.observe()
        settlements = [a for a in observation.legal_actions if a.kind == "BUILD_SETTLEMENT"]
        if settlements:
            assert_engine_transition(env, settlements[0])
            break
        state, geometry = projection(observation)
        shape = geometry.describe(state)
        roads = [a for a in observation.legal_actions if a.kind == "BUILD_ROAD"]

        # Pick the real road that exposes the most distance-legal public endpoints.
        def sites_after(
            action, observation=observation, state=state, shape=shape, geometry=geometry
        ):
            projected = p._transition(
                observation, state, p.PlannedStep(action.kind, action.target), shape
            )
            return len(geometry.describe(projected).sites)

        chosen = max(roads, key=sites_after)
        assert_engine_transition(env, chosen)
    else:
        raise AssertionError("short road replay did not expose a settlement")


def test_hidden_engine_compositions_and_deck_do_not_change_plan_or_counters(monkeypatch):
    from test_simulator import grant

    p = planner()
    env = engine_main_fixture(monkeypatch, hand=(8, 0, 0, 1, 2))
    actor = env.current_player
    a, b = [player for player in range(4) if player != actor][:2]
    grant(env, a, (1, 0, 0, 0, 0))
    grant(env, b, (0, 1, 0, 0, 0))
    observation = env.observe()
    first = p.plan_turn(observation)
    env.game.state.player_state[f"P{a}_WOOD_IN_HAND"] -= 1
    env.game.state.player_state[f"P{a}_BRICK_IN_HAND"] += 1
    env.game.state.player_state[f"P{b}_WOOD_IN_HAND"] += 1
    env.game.state.player_state[f"P{b}_BRICK_IN_HAND"] -= 1
    env.game.state.development_listdeck.reverse()
    env._refresh_actions()
    assert env.observe() == observation
    assert p.plan_turn(env.observe()) == first
    # Vary undisclosed bank composition at constant public total and current menu.
    env.game.state.player_state[f"P{a}_WOOD_IN_HAND"] += 1
    env.game.state.player_state[f"P{a}_BRICK_IN_HAND"] -= 1
    env.game.state.resource_freqdeck[0] -= 1
    env.game.state.resource_freqdeck[1] += 1
    env._refresh_actions()
    assert env.observe() == observation
    assert p.plan_turn(env.observe()) == first
    env.assert_resource_conservation()


def test_ephemeral_projection_cannot_enter_observation_export():
    from colonist_rl.adapters.errors import PerceptionError
    from colonist_rl.adapters.observation_io import observation_to_dict

    state, _ = projection(view())
    assert not isinstance(state, Observation)
    with pytest.raises(PerceptionError, match="observation schema"):
        observation_to_dict(state)


def test_fixed_score_and_access_use_bounded_public_graph_distances():
    p = planner()
    observation = view()
    state, geometry = projection(observation)
    shape = geometry.describe(state)
    assert shape.production == pytest.approx(7 / 36)
    assert shape.access == pytest.approx(7 / 72)  # Two eligible sites, two new roads each.
    # 100P + 20Y + 6E + 4B + 12R, with B=.835 and R=1/3.
    assert p._score(observation, state, shape) == 111_812_222
    network = replace(observation, roads=(Road(0, 10),))
    network_state, network_geometry = projection(network)
    assert network_geometry.describe(network_state).access == pytest.approx(7 / 36)
    no_pieces = replace(network_state, roads_left=0)
    assert network_geometry.describe(no_pieces).access == 0
    opponent_road = replace(network, roads=(Road(0, 10), Road(1, 20)))
    blocked_state, blocked_geometry = projection(opponent_road)
    assert blocked_geometry.describe(blocked_state).access == pytest.approx(7 / 144)
    opponent_building = replace(network, buildings=(Building(0, 10), Building(1, 20)))
    blocked_state, blocked_geometry = projection(opponent_building)
    assert blocked_geometry.describe(blocked_state).access == pytest.approx(7 / 144)
    # New geometry is needed when the public robber position changes.
    _, robbed = projection(replace(opponent_building, robber_tile=0))
    assert robbed.describe(blocked_state).production == 0
    assert robbed.describe(blocked_state).access == 0


def test_unsupported_player_count_and_modified_rule_profile_fall_back():
    p = planner()
    for rules in (RulesConfig.standard(2), replace(RulesConfig.standard(), victory_points=12)):
        observation = replace(view(), rules=rules)
        assert p.plan_turn(observation).reason == "unsupported_rules"


def test_geometry_cap_after_useful_prefix_discards_entire_search():
    p = planner()
    observation = view()
    config = p.TurnPlannerConfig(max_geometry_visits=300)
    limited = p.plan_turn(observation, config)
    assert limited.nodes > 3
    assert limited.reason == "geometry_limit"
    assert limited.steps == ()
    assert limited.first_action is p.HeuristicViewPolicy().act(observation)


def test_h_view_act_and_propose_match_production_h_with_separate_digest_domains():
    p = planner()
    from colonist_rl.contracts import ResourceBelief
    from colonist_rl.opponents import HeuristicPolicy

    observation = replace(
        view(),
        beliefs=(
            ResourceBelief(
                20, (0, 1, 1, 1, 1), (0, 0, 0, 0, 0), (19, 19, 19, 19, 19), (0, 0, 0, 0, 0)
            ),
        ),
    )
    act_seed = p._seed(observation, 11, "act")
    propose_seed = p._seed(observation, 11, "propose")
    assert act_seed != propose_seed
    assert p.HeuristicViewPolicy().act(observation) is HeuristicPolicy(act_seed).act(observation)
    assert p.HeuristicViewPolicy().propose(observation) == HeuristicPolicy(propose_seed).propose(
        observation
    )


def test_rejected_root_metadata_consumes_an_attempted_transition():
    p = planner()
    observation = replace(
        view(hand=(0, 0, 0, 2, 3)),
        legal_actions=(Action("BUILD_CITY", 0, resource="WOOD"), Action("END_TURN")),
    )
    result = p.plan_turn(observation)
    assert result.reason == "public_contract_mismatch"
    assert result.nodes == 1


@pytest.mark.parametrize("port_ratio", [2, 3])
def test_standard_engine_replays_owned_specific_and_generic_port_ratios(monkeypatch, port_ratio):
    p = planner()
    env = engine_main_fixture(monkeypatch, port_ratio=port_ratio)
    observation = env.observe()
    owned = {b.vertex for b in observation.buildings if b.owner == observation.player_id}
    port = next(
        port
        for port in observation.board.ports
        if port.ratio == port_ratio and owned.intersection(port.vertices)
    )
    resource = (
        0
        if port.resource is None
        else ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE").index(port.resource)
    )
    state, _ = projection(observation)
    assert p._ratio(observation, state, resource) == port_ratio
    action = next(
        a
        for a in observation.legal_actions
        if a.kind == "MARITIME_TRADE" and a.give[resource] == port_ratio
    )
    assert_engine_transition(env, action)


def test_root_build_trade_end_order_is_stable_under_tiny_operation_budget():
    p = planner()
    # END is canonical after exchanges, so the first attempted trade is scored.
    observation = view()
    plan = p.plan_turn(observation, p.TurnPlannerConfig(max_nodes=1))
    assert plan.reason == "root_validation_limit"
    assert plan.nodes == 1
    assert plan.geometry_visits == 96


def test_standard_engine_own_known_vp_is_public_vp_plus_own_cards(monkeypatch):
    env = engine_main_fixture(monkeypatch)
    actor = env.current_player
    original = env.observe()
    env.game.state.development_listdeck.remove("VICTORY_POINT")
    env.game.state.player_state[f"P{actor}_VICTORY_POINT_IN_HAND"] += 1
    env.game.state.player_state[f"P{actor}_ACTUAL_VICTORY_POINTS"] += 1
    updated = env.observe()
    own_public = next(p for p in updated.players if p.player_id == actor).victory_points
    assert own_public == original.own_victory_points
    assert updated.own_victory_points == own_public + 1 == env._actual_vp()[actor]


def test_poisoned_environment_properties_are_never_queried():
    p = planner()

    class PublicOnlyObservation(Observation):
        @property
        def env(self):
            raise AssertionError("planner queried hidden engine state")

        engine = env
        game = env

    observation = view()
    poisoned = PublicOnlyObservation(**vars(observation))
    assert p.plan_turn(poisoned) == p.plan_turn(observation)


def test_future_maritime_group_uses_canonical_keys_after_prescribed_build_groups():
    p = planner()
    observation = replace(chain_view(), roads=(Road(0, 10), Road(1, 10)))
    state, geometry = projection(observation)
    candidates = list(p._candidates(observation, state, geometry.describe(state)))
    assert [(step.kind, step.target) for step in candidates[:4]] == [
        ("BUILD_CITY", 0),
        ("BUILD_SETTLEMENT", 2),
        ("BUILD_ROAD", 2),
        ("BUILD_ROAD", 5),
    ]
    maritime = candidates[4:]
    assert all(step.kind == "MARITIME_TRADE" for step in maritime)
    pairs = [
        (
            next(i for i, count in enumerate(step.give) if count),
            next(i for i, count in enumerate(step.receive) if count),
        )
        for step in maritime
    ]
    # Canonical tuple order: a later nonzero resource sorts before an earlier one.
    assert pairs == [
        (4, 3),
        (4, 2),
        (4, 1),
        (4, 0),
        (3, 4),
        (3, 2),
        (3, 1),
        (3, 0),
        (2, 4),
        (2, 3),
        (2, 1),
        (2, 0),
        (1, 4),
        (1, 3),
        (1, 2),
        (1, 0),
        (0, 4),
        (0, 3),
        (0, 2),
        (0, 1),
    ]


def test_small_node_cap_scores_canonical_maritime_prefix_instead_of_resource_order():
    p = planner()
    observation = replace(
        view(hand=(0, 4, 0, 2, 7), bank=(19, 15, 19, 17, 12)),
        legal_actions=(Action("BUILD_CITY", 0), Action("END_TURN")),
    )
    # Two root attempts, two unaffordable road attempts, then four exchanges.
    # Canonical ORE gives reach ORE->WOOD; ascending WOOD gives are all rejected.
    plan = p.plan_turn(observation, p.TurnPlannerConfig(max_nodes=8))
    assert plan.nodes == 8 and plan.search_stop == "node_limit"
    assert plan.reason == "planned"
    assert plan.first_action is observation.legal_actions[0]
    assert plan.steps == (
        p.PlannedStep("BUILD_CITY", 0),
        p.PlannedStep("MARITIME_TRADE", give=(0, 0, 0, 0, 4), receive=(1, 0, 0, 0, 0)),
    )
    assert plan.score_after_micro > plan.score_before_micro + 250_000
