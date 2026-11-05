"""Pure toy-anchor inference and synthetic public overlays; no trained-model probes."""

import importlib.util
import json
import math
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
from test_public_turn_planner import chain_view, projection, replay, view

from colonist_rl import public_position_value as old
from colonist_rl import public_turn_planner as planner
from colonist_rl.contracts import ResourceBelief, VisibleEvent


def runtime():
    assert importlib.util.find_spec("colonist_rl.public_residual_value") is not None, (
        "public residual runtime missing"
    )
    from colonist_rl import public_residual_value

    return public_residual_value


@pytest.fixture
def toy(monkeypatch):
    r = runtime()
    anchor = {
        "schema": old.SCHEMA,
        "target": old.TARGET,
        "continuation": old.CONTINUATION,
        "columns": ["handcraft_score"],
        "mean": [100.0],
        "scale": [20.0],
        "coef": [1.0],
        "intercept": 0.0,
    }
    # Only the provenance fingerprint is replaced. No real pilot coefficients are scored.
    fingerprint = sha256(
        json.dumps(anchor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    monkeypatch.setattr(r, "ANCHOR_MODEL_SHA256", fingerprint)

    def model(productive=False):
        columns = list(r.PRODUCTIVE_RESIDUAL_COLUMNS if productive else r.RESIDUAL_COLUMNS)
        return {
            "schema": r.SCHEMA,
            "target": r.TARGET,
            "continuation": r.CONTINUATION,
            "include_productive": productive,
            "columns": columns,
            "mean": [0.0] * len(columns),
            "scale": [1.0] * len(columns),
            "coef": [0.0] * len(columns),
            "intercept": 0.0,
            "residual_bound": 0.5,
            "anchor": {
                "source_sha256": r.ANCHOR_SOURCE_SHA256,
                "entry": "calibrated_handcraft",
                "model": deepcopy(anchor),
            },
        }

    return r, anchor, model


@pytest.mark.parametrize("productive", [False, True])
def test_zero_residual_exactly_matches_old_calibration_in_original_units(toy, productive):
    r, anchor, model = toy
    observation = view()
    state, _ = projection(observation)
    residual = r.PublicResidualEvaluator(model(productive))
    previous = old.FrozenPositionEvaluator.from_model(anchor, kind="calibrated_handcraft")
    for score, expected in ((100_000_000, 500_000), (120_000_000, 731_059)):
        assert (
            residual(observation, state, score) == previous(observation, state, score) == expected
        )


@pytest.mark.parametrize("sign,expected", [(-1, 377_541), (1, 622_459)])
def test_residual_offset_saturates_at_half_a_logit_and_model_is_immutable(toy, sign, expected):
    r, _, model = toy
    data = model()
    data["coef"][data["columns"].index("known_own_vp")] = sign * 1_000
    evaluator = r.PublicResidualEvaluator(data)
    data["intercept"] = -sign * 1_000
    observation = view()
    state, _ = projection(observation)
    assert evaluator(observation, state, 100_000_000) == expected


def test_residual_uses_train_mean_scale_and_intercept_before_tanh(toy):
    r, _, model = toy
    data = model()
    index = data["columns"].index("known_own_vp")
    data["mean"][index] = 3.0
    data["scale"][index] = 4.0
    data["coef"][index] = 2.0
    data["intercept"] = 1.0  # own VP 1 -> standardized -0.5 -> zero residual logit
    observation = view()
    state, _ = projection(observation)
    assert r.PublicResidualEvaluator(data)(observation, state, 100_000_000) == 500_000


@pytest.mark.parametrize(
    "field,bad",
    [
        ("schema", "public-residual-value/0"),
        ("target", "action_label"),
        ("continuation", old.CONTINUATION),
        ("include_productive", 1),
        ("columns", ["private_hand"]),
        ("columns", []),
        ("mean", []),
        ("scale", []),
        ("coef", []),
        ("intercept", math.inf),
        ("intercept", True),
        ("residual_bound", 0.6),
        ("residual_bound", True),
        ("extra", 1),
    ],
)
def test_invalid_residual_schema_columns_parameters_and_bound_reject(toy, field, bad):
    r, _, model = toy
    data = model()
    data[field] = bad
    with pytest.raises(ValueError):
        r.PublicResidualEvaluator(data)


@pytest.mark.parametrize(
    "field,bad",
    [("scale", 0), ("scale", -1), ("scale", math.inf), ("coef", math.nan), ("mean", True)],
)
def test_invalid_standardization_and_weights_reject(toy, field, bad):
    r, _, model = toy
    data = model()
    data[field][0] = bad
    with pytest.raises(ValueError):
        r.PublicResidualEvaluator(data)


def test_residual_rejects_missing_duplicate_reordered_and_wrong_arm_columns(toy):
    r, _, model = toy
    invalid = [None, [], {}]
    missing = model()
    del missing["anchor"]
    invalid.append(missing)
    duplicate = model()
    duplicate["columns"][0] = duplicate["columns"][1]
    invalid.append(duplicate)
    reordered = model()
    reordered["columns"].reverse()
    invalid.append(reordered)
    wrong_arm = model()
    wrong_arm["include_productive"] = True
    invalid.append(wrong_arm)
    for payload in invalid:
        with pytest.raises(ValueError):
            r.PublicResidualEvaluator(payload)


@pytest.mark.parametrize("mutation", ["source", "entry", "model", "continuation", "extra"])
def test_anchor_identity_cannot_relabel_another_model(toy, mutation):
    r, _, model = toy
    data = model()
    if mutation == "source":
        data["anchor"]["source_sha256"] = "0" * 64
    elif mutation == "entry":
        data["anchor"]["entry"] = "public_linear"
    elif mutation == "model":
        data["anchor"]["model"]["coef"] = [9.0]
    elif mutation == "continuation":
        data["anchor"]["model"]["continuation"] = r.CONTINUATION
    else:
        data["anchor"]["extra"] = 1
    with pytest.raises(ValueError):
        r.PublicResidualEvaluator(data)


def test_combined_encoder_preserves_known_own_vp_and_projection_actual_parity(toy):
    r, _, _ = toy
    observation = replace(chain_view(), development=(1, 2, 3, 4, 5))
    steps = (
        planner.PlannedStep("MARITIME_TRADE", give=(2, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0)),
        planner.PlannedStep("BUILD_ROAD", 1),
        planner.PlannedStep("BUILD_SETTLEMENT", 2),
    )
    for state in replay(observation, steps):
        own = replace(
            observation.players[0],
            victory_points=1 + state.building_point_gain,
            resource_count=sum(state.hand),
            roads_left=state.roads_left,
            settlements_left=state.settlements_left,
            cities_left=state.cities_left,
        )
        actual = replace(
            observation,
            hand=state.hand,
            buildings=state.buildings,
            roads=state.roads,
            players=(own, *observation.players[1:]),
        )
        combined = r.encode_residual_features(observation, state, include_productive=True)
        assert combined == r.encode_residual_features(actual, include_productive=True)
        assert combined["known_own_vp"] == 6 + state.building_point_gain
        assert combined["public_own_vp"] == 1 + state.building_point_gain
        assert r.encode_residual_features(observation, state) == old.encode_position(
            observation, state
        )
        assert set(combined) == set(r.PRODUCTIVE_RESIDUAL_COLUMNS)


def test_encoder_ignores_bank_beliefs_events_menu_and_awards(toy):
    r, _, _ = toy
    observation = view()
    expected = r.encode_residual_features(observation, include_productive=True)
    changed = replace(
        observation,
        bank=(1, 2, 3, 4, 5),
        legal_actions=(),
        beliefs=(ResourceBelief(20, (9.0,) * 5, (0,) * 5, (19,) * 5, (8.0,) * 5),),
        events=(VisibleEvent(99, "RESOURCE_GAIN", 20, resource="ORE", count=7),),
        players=tuple(
            replace(p, longest_road=77, has_longest_road=True, has_largest_army=True)
            for p in observation.players
        ),
    )
    assert r.encode_residual_features(changed, include_productive=True) == expected


def test_factory_keeps_ordinary_search_budgets_margin_and_original_root_action(toy, tmp_path):
    r, _, model = toy
    data = model(True)
    data["coef"][data["columns"].index("known_own_vp")] = 1
    path = tmp_path / "residual.json"
    path.write_text(json.dumps(data))
    policy = r.residual_planner(path)
    observation = view()
    assert type(policy) is planner.PublicTurnPlannerPolicy
    assert policy.config == planner.TurnPlannerConfig(6, 2048, 24, 100_000, 11)
    assert policy.gain_margin_micro == 10_000
    plan = policy.plan(observation)
    assert plan.first_action is observation.legal_actions[plan.root_index]
    assert plan.nodes <= 2048 and plan.geometry_visits <= 100_000
    assert type(plan.score_before_micro) is int


@pytest.mark.parametrize("score", [True, 1.0, math.inf])
def test_handcrafted_score_requires_integer_micro_units(toy, score):
    r, _, model = toy
    observation = view()
    state, _ = projection(observation)
    with pytest.raises(ValueError):
        r.PublicResidualEvaluator(model())(observation, state, score)
