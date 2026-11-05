"""Frozen public logistic inference for current-turn projections; no fitting code.

Encoder copied without semantic changes from reviewed value-lane/v3/value_pilot.py,
SHA256 8940a6edae7c67657cff61263c5c89dd02b4c74a1182ad3970c39a5b0db031b5.
The target is eventual win under the declared H continuation, not planner self-play.
Projection features do not resolve longest-road ownership or certify an afterstate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .contracts import RESOURCES, Observation
from .public_turn_planner import (
    ProjectedTurn,
    PublicTurnPlannerPolicy,
    TurnPlannerConfig,
)
from .settlement_sites import connectable_road_edges, connected_settlement_sites

SCHEMA = "public-position-value/2"
CONTINUATION = "evaluation-H-candidate-no-proposals-vs-3H-with-proposals/1"
TARGET = "resolved_eventual_win_probability"
PROBABILITY_GAIN_MARGIN_MICRO = 10_000
_DEFAULT_CONFIG = TurnPlannerConfig()


def encode_position(view, projected=None):
    """One public encoder, with explicit projection overlays; never creates an Observation.

    The projection does not resolve longest-road ownership. Known VP is root known VP
    plus certain building gain; opponent public awards/counts remain root evidence.
    Bank and deck are deliberately omitted, including when an exact bank is supplied.
    """
    own = next(p for p in view.players if p.player_id == view.player_id)
    hand = view.hand if projected is None else projected.hand
    buildings = view.buildings if projected is None else projected.buildings
    roads = view.roads if projected is None else projected.roads
    gain = 0 if projected is None else projected.building_point_gain
    result = {
        "known_own_vp": float(view.own_victory_points + gain),
        "public_own_vp": float(own.victory_points + gain),
        "hand_total": float(sum(hand)),
        "turn_number": float(view.context.turn_number),
    }
    for name in ("roads_left", "settlements_left", "cities_left"):
        result[name] = float(getattr(own if projected is None else projected, name))
    for i, resource in enumerate(RESOURCES):
        result["hand_" + resource] = float(hand[i])
    for i, name in enumerate(("knight", "plenty", "monopoly", "road_building", "hidden_vp")):
        result["own_dev_" + name] = float(view.development[i])
    production = [0.0] * 5
    unblocked = [0.0] * 5
    owned = {b.vertex for b in buildings if b.owner == view.player_id}
    for b in buildings:
        if b.owner != view.player_id:
            continue
        for ti, (tile, vertices) in enumerate(zip(view.board.tiles, view.board.tile_vertices)):
            if b.vertex in vertices and tile.resource in RESOURCES and tile.number:
                i = RESOURCES.index(tile.resource)
                amount = b.level * (6 - abs(7 - tile.number)) / 36
                production[i] += amount
                if ti != view.robber_tile:
                    unblocked[i] += amount
    for i, resource in enumerate(RESOURCES):
        result["production_" + resource] = production[i]
        result["unblocked_" + resource] = unblocked[i]
        result["coverage_" + resource] = float(production[i] > 0)
        result["port_ratio_" + resource] = float(
            min(
                [4]
                + [
                    p.ratio
                    for p in view.board.ports
                    if owned.intersection(p.vertices) and p.resource in (None, resource)
                ]
            )
        )
    result["own_settlements"] = float(
        sum(b.owner == view.player_id and b.level == 1 for b in buildings)
    )
    result["own_cities"] = float(sum(b.owner == view.player_id and b.level == 2 for b in buildings))
    result["own_roads"] = float(sum(r.owner == view.player_id for r in roads))
    result["connected_sites"] = float(
        len(connected_settlement_sites(view.board, buildings, roads, view.player_id))
    )
    result["road_frontier"] = float(
        len(connectable_road_edges(view.board, buildings, roads, view.player_id))
    )
    # Clockwise player order, invariant to replacing seat identifiers consistently.
    players = list(view.players)
    index = next(i for i, p in enumerate(players) if p.player_id == view.player_id)
    for offset in range(1, len(players)):
        p = players[(index + offset) % len(players)]
        for name in (
            "victory_points",
            "resource_count",
            "development_count",
            "roads_left",
            "settlements_left",
            "cities_left",
            "played_knights",
        ):
            result[f"opponent_{offset}_{name}"] = float(getattr(p, name))
    if any(not math.isfinite(x) for x in result.values()):
        raise ValueError("nonfinite public features")
    return result


_OPPONENT_FIELDS = (
    "victory_points",
    "resource_count",
    "development_count",
    "roads_left",
    "settlements_left",
    "cities_left",
    "played_knights",
)
PUBLIC_COLUMNS = tuple(
    sorted(
        (
            "known_own_vp",
            "public_own_vp",
            "hand_total",
            "turn_number",
            "roads_left",
            "settlements_left",
            "cities_left",
            "own_settlements",
            "own_cities",
            "own_roads",
            "connected_sites",
            "road_frontier",
        )
        + tuple(
            "own_dev_" + name
            for name in ("knight", "plenty", "monopoly", "road_building", "hidden_vp")
        )
        + tuple(
            prefix + resource
            for resource in RESOURCES
            for prefix in ("hand_", "production_", "unblocked_", "coverage_", "port_ratio_")
        )
        + tuple(f"opponent_{offset}_{name}" for offset in range(1, 4) for name in _OPPONENT_FIELDS)
    )
)


def _finite_number(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("finite numeric model values required")
    return float(value)


@dataclass(frozen=True)
class FrozenPositionEvaluator:
    """Validated immutable logistic coefficients, scoring in probability millionths."""

    kind: str
    columns: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float

    @classmethod
    def from_model(cls, model: dict, *, kind: str) -> FrozenPositionEvaluator:
        if kind not in ("public_linear", "calibrated_handcraft"):
            raise ValueError("unknown frozen position model kind")
        required = {
            "schema",
            "target",
            "continuation",
            "columns",
            "mean",
            "scale",
            "coef",
            "intercept",
        }
        optional = {"constant_probability", "l2", "steps"}
        if (
            type(model) is not dict
            or not required <= model.keys()
            or model.keys() - required - optional
        ):
            raise ValueError("invalid frozen model fields")
        if (
            model["schema"] != SCHEMA
            or model["target"] != TARGET
            or model["continuation"] != CONTINUATION
        ):
            raise ValueError("model schema, target or continuation mismatch")
        columns = PUBLIC_COLUMNS if kind == "public_linear" else ("handcraft_score",)
        if type(model["columns"]) is not list or model["columns"] != list(columns):
            raise ValueError("model requires exact canonical public columns for its kind")
        arrays = []
        for name in ("mean", "scale", "coef"):
            values = model[name]
            if type(values) is not list or len(values) != len(columns):
                raise ValueError("model array length mismatch")
            arrays.append(tuple(_finite_number(v) for v in values))
        if any(s <= 0 for s in arrays[1]):
            raise ValueError("model scale must be positive")
        intercept = _finite_number(model["intercept"])
        if (
            "constant_probability" in model
            and not 0 <= _finite_number(model["constant_probability"]) <= 1
        ):
            raise ValueError("invalid constant probability")
        if "l2" in model and _finite_number(model["l2"]) < 0:
            raise ValueError("invalid regularization metadata")
        if "steps" in model and (type(model["steps"]) is not int or model["steps"] < 1):
            raise ValueError("invalid fit steps metadata")
        return cls(kind, columns, *arrays, intercept)

    def __call__(self, view: Observation, projected: ProjectedTurn, handcraft_micro: int) -> int:
        if type(handcraft_micro) is not int:
            raise ValueError("integer handcraft score required")
        if self.kind == "calibrated_handcraft":
            features = {"handcraft_score": handcraft_micro / 1_000_000}
        else:
            features = encode_position(view, projected)
            if set(features) != set(self.columns):
                raise ValueError("public feature columns mismatch")
        z = self.intercept + sum(
            c * (features[n] - m) / s
            for n, m, s, c in zip(self.columns, self.mean, self.scale, self.coef, strict=True)
        )
        if not math.isfinite(z):
            raise ValueError("nonfinite model logit")
        probability = 1 / (1 + math.exp(-max(-35, min(35, z))))
        return round(probability * 1_000_000)


def load_evaluator(path: str | Path, *, kind: str) -> FrozenPositionEvaluator:
    """Read one explicitly selected model from the pilot's models.json bundle."""
    bundle = json.loads(Path(path).read_text())
    if type(bundle) is not dict or kind not in bundle:
        raise ValueError("selected frozen model missing")
    return FrozenPositionEvaluator.from_model(bundle[kind], kind=kind)


def probability_planner(
    path: str | Path, *, kind: str, config: TurnPlannerConfig = _DEFAULT_CONFIG
) -> PublicTurnPlannerPolicy:
    """Identical search budgets and one percentage point margin for both controls."""
    return PublicTurnPlannerPolicy(
        config,
        evaluator=load_evaluator(path, kind=kind),
        gain_margin_micro=PROBABILITY_GAIN_MARGIN_MICRO,
    )
