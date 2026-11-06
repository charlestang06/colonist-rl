"""Pure original-menu transitions; no engine, policy or search is executed."""

from dataclasses import replace

import pytest

from colonist_rl import public_turn_planner as p
from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import (
    ZERO,
    Action,
    Building,
    DecisionContext,
    Observation,
    PublicPlayer,
    Road,
)


def root_view():
    # Edges 0=(0,1), 1=(1,2), 5=(0,5). Enemy 1 blocks future continuation.
    return Observation(
        player_id=0,
        board=BoardSpec.from_hexes(((0, 0),), ("WHEAT",), (6,)),
        rules=RulesConfig.standard(),
        context=DecisionContext("PLAY_TURN", 0, 0, 1, has_rolled=True),
        players=(
            PublicPlayer(0, 1, 2, 0, roads_left=13, settlements_left=4),
            PublicPlayer(1, 1, 0, 0, settlements_left=4),
            PublicPlayer(2, 0, 0, 0),
            PublicPlayer(3, 0, 0, 0),
        ),
        hand=(1, 1, 0, 0, 0),
        development=ZERO,
        buildings=(Building(1, 1), Building(5, 0)),
        roads=(Road(0, 0), Road(5, 0)),
        robber_tile=0,
        legal_actions=(Action("BUILD_ROAD", 1), Action("END_TURN")),
        bank=(18, 18, 19, 19, 19),
    )


def initial(view):
    state = p._initial_projection(view)
    geometry = p._Geometry(view, p._Work(p.TurnPlannerConfig()))
    return state, geometry.describe(state)


def transition(view, state, action, shape):
    helper = getattr(p, "_root_transition", None)
    assert callable(helper), "dedicated original-root transition is missing"
    return helper(view, state, action, shape)


def test_original_root_menu_certifies_only_first_road_connectivity():
    view = root_view()
    state, shape = initial(view)
    assert shape.edges == (4,)
    assert p._transition(view, state, p.PlannedStep("BUILD_ROAD", 1), shape) is None

    after = transition(view, state, view.legal_actions[0], shape)

    assert after == p.ProjectedTurn(
        hand=ZERO,
        buildings=(Building(1, 1), Building(5, 0)),
        roads=(Road(0, 0), Road(1, 0), Road(5, 0)),
        roads_left=12,
        settlements_left=4,
        cities_left=4,
        building_point_gain=0,
        bank_lower=(19, 19, 19, 19, 19),
        bank_upper=(19, 19, 19, 19, 19),
    )
    assert view.hand == (1, 1, 0, 0, 0)
    assert shape.edges == (4,)
    # Original menu authority cannot be reused on a projected child.
    assert transition(view, after, view.legal_actions[0], shape) is None


def test_equal_or_nonmenu_actions_do_not_certify_a_road():
    view = root_view()
    state, shape = initial(view)
    for action in (replace(view.legal_actions[0]), Action("BUILD_ROAD", 2)):
        assert transition(view, state, action, shape) is None


@pytest.mark.parametrize("target", [None, True, False, 1.0, "1", -1, 6])
def test_root_road_target_requires_an_in_range_integer(target):
    view = replace(root_view(), legal_actions=(Action("BUILD_ROAD", target),))
    state, shape = initial(view)
    assert transition(view, state, view.legal_actions[0], shape) is None


@pytest.mark.parametrize("owner", [0, 1])
def test_original_menu_cannot_certify_an_occupied_edge(owner):
    view = replace(root_view(), roads=(Road(0, 0), Road(1, owner), Road(5, 0)))
    state, shape = initial(view)
    assert transition(view, state, view.legal_actions[0], shape) is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"give": (1, 1, 0, 0, 0)},
        {"receive": (1, 0, 0, 0, 0)},
        {"resource": "WOOD"},
        {"second_resource": "BRICK"},
        {"partner": 1},
        {"offer_id": "extra"},
    ],
)
def test_original_menu_cannot_certify_extra_action_payload(metadata):
    view = root_view()
    view = replace(view, legal_actions=(replace(view.legal_actions[0], **metadata),))
    state, shape = initial(view)
    assert transition(view, state, view.legal_actions[0], shape) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"roads_left": 12},
        {"hand": (2, 1, 0, 0, 0)},
        {"bank_lower": (17, 18, 19, 19, 19)},
        {"building_point_gain": 1},
    ],
)
def test_root_authority_rejects_noninitial_projections(changes):
    view = root_view()
    state, shape = initial(view)
    assert transition(view, replace(state, **changes), view.legal_actions[0], shape) is None


def test_original_menu_does_not_waive_stock_or_resource_cost():
    view = root_view()
    no_stock = replace(view, players=(replace(view.players[0], roads_left=0), *view.players[1:]))
    no_brick = replace(
        view,
        hand=(1, 0, 0, 0, 0),
        bank=(18, 19, 19, 19, 19),
        players=(replace(view.players[0], resource_count=1), *view.players[1:]),
    )
    for bad in (no_stock, no_brick):
        state, shape = initial(bad)
        assert transition(bad, state, bad.legal_actions[0], shape) is None


def test_original_menu_does_not_waive_public_bank_contract():
    view = replace(root_view(), bank=(19, 18, 19, 19, 19))
    # This bank already exceeds supply minus own hand; no initial projection is valid.
    valid_state, shape = initial(root_view())
    assert transition(view, valid_state, view.legal_actions[0], shape) is None


@pytest.mark.parametrize("kind,target", [("BUILD_CITY", 1), ("BUILD_SETTLEMENT", 2)])
def test_other_builds_still_require_strict_geometry(kind, target):
    view = root_view()
    view = replace(
        view,
        hand=(2, 2, 2, 2, 3),
        bank=(17, 17, 17, 17, 16),
        players=(replace(view.players[0], resource_count=11), *view.players[1:]),
        legal_actions=(Action(kind, target),),
    )
    state, shape = initial(view)
    assert transition(view, state, view.legal_actions[0], shape) is None


def test_end_and_maritime_keep_existing_deterministic_semantics():
    view = root_view()
    state, shape = initial(view)
    assert transition(view, state, view.legal_actions[1], shape) == state
    view = replace(
        view,
        hand=(4, 0, 0, 0, 0),
        bank=(15, 19, 19, 19, 19),
        players=(replace(view.players[0], resource_count=4), *view.players[1:]),
        legal_actions=(Action("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 1, 0, 0, 0)),),
    )
    state, shape = initial(view)
    after = transition(view, state, view.legal_actions[0], shape)
    assert after.hand == (0, 1, 0, 0, 0)
    assert after.bank_lower == after.bank_upper == (19, 18, 19, 19, 19)
