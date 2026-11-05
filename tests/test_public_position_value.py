"""Public manufactured fixtures check runtime contracts, never playing strength."""

import importlib.util
import math
from dataclasses import replace

import pytest
from test_public_turn_planner import chain_view, projection, replay, view

from colonist_rl import public_turn_planner as p


def runtime():
    assert importlib.util.find_spec("colonist_rl.public_position_value") is not None, (
        "frozen public position runtime is missing"
    )
    from colonist_rl import public_position_value

    return public_position_value


def test_optional_evaluator_preserves_default_search_and_scores_actual_root():
    observation = view()
    calls = []

    def identity(root, state, handcraft):
        assert root is observation
        assert isinstance(state, p.ProjectedTurn)
        assert type(handcraft) is int
        calls.append((state, handcraft))
        return handcraft

    expected = p.plan_turn(observation)
    actual = p.plan_turn(observation, evaluator=identity)
    assert replace(actual, evaluator_calls=0, evaluator_time_ns=0) == expected
    assert actual.first_action is observation.legal_actions[actual.root_index]
    assert actual.evaluator_calls == len(calls) > 3
    assert actual.evaluator_time_ns >= 0
    assert calls[0][0].hand == observation.hand
    assert expected.evaluator_calls == expected.evaluator_time_ns == 0
    policy = p.PublicTurnPlannerPolicy(evaluator=identity)
    assert policy.act(observation) is actual.first_action


def test_gain_margin_is_strict_in_evaluator_units_and_winning_order_is_preserved():
    observation = view()

    def gain(root, state, handcraft):
        return 500_000 + state.building_point_gain * 10_000

    insufficient = p.plan_turn(observation, evaluator=gain, gain_margin_micro=10_000)
    planned = p.plan_turn(observation, evaluator=gain, gain_margin_micro=9_999)
    assert insufficient.reason == "insufficient_gain"
    assert planned.reason == "planned"
    assert planned.score_before_micro == 500_000
    assert planned.score_after_micro == 510_000
    winning = replace(observation, development=(0, 0, 0, 0, 8))
    assert p.plan_turn(winning, evaluator=lambda *_: 0).reason == "planned"


@pytest.mark.parametrize("bad", [True, 0.1, math.inf, math.nan, "1", None])
def test_evaluator_requires_finite_integer_scores(bad):
    with pytest.raises(ValueError, match="integer"):
        p.plan_turn(view(), evaluator=lambda *_: bad)


@pytest.mark.parametrize("bad", [True, -1, 0.1, math.inf, math.nan])
def test_gain_margin_rejects_invalid_units(bad):
    with pytest.raises(ValueError, match="margin"):
        p.plan_turn(view(), gain_margin_micro=bad)
    with pytest.raises(ValueError, match="margin"):
        p.PublicTurnPlannerPolicy(gain_margin_micro=bad)


def test_unsupported_and_root_validation_fallbacks_preserve_original_identity():
    observation = replace(view(), context=replace(view().context, has_rolled=False))
    plan = p.plan_turn(observation, evaluator=lambda *_: pytest.fail("unsupported scored"))
    assert plan.reason == "unsupported_context"
    assert plan.evaluator_calls == 0
    assert plan.first_action is p.HeuristicViewPolicy().act(observation)
    observation = view()
    plan = p.plan_turn(observation, p.TurnPlannerConfig(max_nodes=1), evaluator=lambda *_: 0)
    assert plan.reason == "root_validation_limit"
    assert plan.first_action is p.HeuristicViewPolicy().act(observation)
    assert plan.evaluator_calls == 2


def test_actual_encoder_hand_checked_public_fields_and_hidden_omissions():
    r = runtime()
    observation = replace(view(), development=(1, 2, 3, 4, 5))
    encoded = r.encode_position(observation)
    expected = {
        "known_own_vp": 6.0,
        "public_own_vp": 1.0,
        "hand_total": 11.0,
        "turn_number": 0.0,
        "roads_left": 15.0,
        "settlements_left": 5.0,
        "cities_left": 4.0,
        "own_settlements": 1.0,
        "own_cities": 0.0,
        "own_roads": 0.0,
        "connected_sites": 0.0,
        "road_frontier": 2.0,
    }
    for resource, count in zip(("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"), (8, 0, 0, 1, 2)):
        expected["hand_" + resource] = float(count)
        expected["production_" + resource] = 5 / 36 if resource == "WHEAT" else 0.0
        expected["unblocked_" + resource] = expected["production_" + resource]
        expected["coverage_" + resource] = float(resource == "WHEAT")
        expected["port_ratio_" + resource] = 4.0
    for name, count in zip(
        ("knight", "plenty", "monopoly", "road_building", "hidden_vp"), range(1, 6)
    ):
        expected["own_dev_" + name] = float(count)
    for offset, vp, cards in ((1, 2, 30), (2, 0, 0), (3, 0, 0)):
        for name, count in {
            "victory_points": vp,
            "resource_count": cards,
            "development_count": 0,
            "roads_left": 15,
            "settlements_left": 5,
            "cities_left": 4,
            "played_knights": 0,
        }.items():
            expected[f"opponent_{offset}_{name}"] = float(count)
    assert encoded == expected
    assert r.encode_position(replace(observation, bank=(1, 2, 3, 4, 5))) == expected
    assert r.encode_position(replace(observation, robber_tile=0))["unblocked_WHEAT"] == 0


def test_projection_encoder_matches_explicit_actual_public_overlay():
    r = runtime()
    observation = chain_view()
    steps = (
        p.PlannedStep("MARITIME_TRADE", give=(2, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0)),
        p.PlannedStep("BUILD_ROAD", 1),
        p.PlannedStep("BUILD_SETTLEMENT", 2),
    )
    state = replay(observation, steps)[-1]
    own = replace(
        observation.players[0],
        resource_count=sum(state.hand),
        victory_points=2,
        roads_left=14,
        settlements_left=4,
    )
    # Only a test oracle: runtime never constructs a synthetic Observation.
    actual_overlay = replace(
        observation,
        hand=state.hand,
        buildings=state.buildings,
        roads=state.roads,
        players=(own, *observation.players[1:]),
    )
    assert r.encode_position(observation, state) == r.encode_position(actual_overlay)
    assert r.encode_position(observation, state)["known_own_vp"] == 2
    road_state = replace(state, building_point_gain=0, buildings=observation.buildings)
    assert r.encode_position(observation, road_state)["known_own_vp"] == 1


def model(kind="public_linear"):
    r = runtime()
    columns = list(r.PUBLIC_COLUMNS) if kind == "public_linear" else ["handcraft_score"]
    return {
        "schema": r.SCHEMA,
        "target": "resolved_eventual_win_probability",
        "continuation": r.CONTINUATION,
        "columns": columns,
        "mean": [0.0] * len(columns),
        "scale": [1.0] * len(columns),
        "coef": [0.0] * len(columns),
        "intercept": 0.0,
        "constant_probability": 0.25,
        "l2": 0.1,
        "steps": 300,
    }


@pytest.mark.parametrize("kind", ["public_linear", "calibrated_handcraft"])
def test_frozen_logistic_probability_and_policy_use_same_fixed_margin(kind, tmp_path):
    import json

    r = runtime()
    data = model(kind)
    data["intercept"] = math.log(3)
    path = tmp_path / "models.json"
    path.write_text(json.dumps({kind: data}))
    evaluator = r.load_evaluator(path, kind=kind)
    state, _ = projection(view())
    assert evaluator(view(), state, 123_000_000) == 750_000
    data["intercept"] = -100  # Parsed runtime does not retain mutable model containers.
    assert evaluator(view(), state, 123_000_000) == 750_000
    policy = r.probability_planner(path, kind=kind)
    assert policy.gain_margin_micro == 10_000
    assert policy.config == p.TurnPlannerConfig()
    assert policy.act(view()).kind in ("MARITIME_TRADE", "END_TURN")


def test_calibration_consumes_raw_handcraft_in_original_units():
    r = runtime()
    data = model("calibrated_handcraft")
    data.update(mean=[100.0], scale=[20.0], coef=[1.0])
    evaluator = r.FrozenPositionEvaluator.from_model(data, kind="calibrated_handcraft")
    observation = view()
    state, _ = projection(observation)
    assert evaluator(observation, state, 100_000_000) == 500_000
    assert evaluator(observation, state, 120_000_000) == 731_059


@pytest.mark.parametrize(
    "field,bad",
    [
        ("schema", "public-position-value/1"),
        ("target", "action_label"),
        ("continuation", "unknown"),
        ("columns", ["opponent_private_cards"]),
        ("columns", ["handcraft_score"]),
        ("columns", []),
        ("mean", [0.0]),
        ("scale", [1.0]),
        ("coef", [0.0]),
        ("intercept", math.nan),
        ("intercept", math.inf),
        ("intercept", True),
        ("constant_probability", math.nan),
        ("l2", math.inf),
        ("steps", True),
    ],
)
def test_model_rejects_wrong_target_columns_shapes_and_nonfinite_numbers(field, bad):
    r = runtime()
    data = model()
    data[field] = bad
    with pytest.raises(ValueError):
        r.FrozenPositionEvaluator.from_model(data, kind="public_linear")


@pytest.mark.parametrize(
    "field,bad",
    [("scale", 0), ("scale", -1), ("scale", math.inf), ("coef", math.nan), ("mean", True)],
)
def test_model_rejects_invalid_array_entries(field, bad):
    r = runtime()
    data = model()
    data[field][0] = bad
    with pytest.raises(ValueError):
        r.FrozenPositionEvaluator.from_model(data, kind="public_linear")


def test_model_rejects_duplicate_columns_and_unknown_kinds():
    r = runtime()
    data = model()
    data["columns"][0] = data["columns"][1]
    with pytest.raises(ValueError):
        r.FrozenPositionEvaluator.from_model(data, kind="public_linear")
    with pytest.raises(ValueError):
        r.FrozenPositionEvaluator.from_model(model(), kind="constant")


def test_public_linear_scores_projected_building_gain_and_preserves_root_identity(tmp_path):
    import json

    r = runtime()
    data = model()
    data["intercept"] = -2.0
    data["coef"][data["columns"].index("known_own_vp")] = 2.0
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"public_linear": data}))
    policy = r.probability_planner(path, kind="public_linear")
    observation = view()
    plan = policy.plan(observation)
    assert plan.reason == "planned"
    assert plan.score_before_micro == 500_000
    assert plan.score_after_micro == 880_797
    assert plan.first_action is observation.legal_actions[plan.root_index]
    assert policy.act(observation) is plan.first_action
    assert plan.evaluator_calls > 3


@pytest.mark.parametrize("intercept,expected", [(-100, 0), (100, 1_000_000)])
def test_frozen_probability_saturation_stays_in_integer_bounds(intercept, expected):
    r = runtime()
    data = model()
    data["intercept"] = intercept
    evaluator = r.FrozenPositionEvaluator.from_model(data, kind="public_linear")
    observation = view()
    state, _ = projection(observation)
    assert evaluator(observation, state, 0) == expected


def test_loading_rejects_absent_selected_models_and_nonobject_bundles(tmp_path):
    r = runtime()
    path = tmp_path / "models.json"
    for payload in ("[]", "{}", '{"constant": {}}'):
        path.write_text(payload)
        with pytest.raises(ValueError):
            r.load_evaluator(path, kind="public_linear")
