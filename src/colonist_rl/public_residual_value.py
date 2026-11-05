"""Frozen bounded residual inference over public positions; no fitting code.

The embedded anchor is an old H-continuation prior, not a new calibration to the
cap-corrected collecting policy. Provenance identifies both its bundle and entry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .productive_geometry import PRODUCTIVE_COLUMNS, encode_productive_geometry
from .public_position_value import (
    PROBABILITY_GAIN_MARGIN_MICRO,
    PUBLIC_COLUMNS,
    TARGET,
    FrozenPositionEvaluator,
    encode_position,
)
from .public_turn_planner import PublicTurnPlannerPolicy, TurnPlannerConfig

SCHEMA = "public-residual-value/1"
CONTINUATION = "cap-raw-no-proposals-vs-3H-with-proposals/1"
RESIDUAL_BOUND = 0.5
RESIDUAL_COLUMNS = PUBLIC_COLUMNS
PRODUCTIVE_RESIDUAL_COLUMNS = tuple(sorted((*PUBLIC_COLUMNS, *PRODUCTIVE_COLUMNS)))
ANCHOR_SOURCE_SHA256 = "9201d6911bb1c2a7a2d66e88022c603199ece240bfa8a100e11188b9f59d6520"
# Canonical JSON (sorted keys, compact separators) of the exact calibrated_handcraft
# entry in value-lane/pilot-001/fit/models.json, whose file SHA is above.
ANCHOR_MODEL_SHA256 = "036644b1d8f847ce88afe0a734f28432fa0aa080293e1bc83d4cb352727d82e7"


def expected_columns(include_productive=False):
    """Return the canonical model/fit column order for either residual arm."""
    if type(include_productive) is not bool:
        raise ValueError("include_productive must be boolean")
    return PRODUCTIVE_RESIDUAL_COLUMNS if include_productive else RESIDUAL_COLUMNS


def encode_residual_features(view, projected=None, include_productive=False):
    """Reuse the old encoder unchanged, optionally adding the fixed geometry block."""
    expected_columns(include_productive)
    result = encode_position(view, projected)
    if include_productive:
        result.update(encode_productive_geometry(view, projected))
    return result


def _finite(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("finite numeric residual model values required")
    return float(value)


def _anchor(model):
    if type(model) is not dict or model.keys() != {"source_sha256", "entry", "model"}:
        raise ValueError("invalid residual anchor fields")
    if model["source_sha256"] != ANCHOR_SOURCE_SHA256 or model["entry"] != "calibrated_handcraft":
        raise ValueError("residual anchor provenance mismatch")
    anchor = FrozenPositionEvaluator.from_model(model["model"], kind="calibrated_handcraft")
    canonical = json.dumps(model["model"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if sha256(canonical.encode()).hexdigest() != ANCHOR_MODEL_SHA256:
        raise ValueError("residual anchor entry mismatch")
    return anchor


@dataclass(frozen=True, init=False)
class PublicResidualEvaluator:
    """Immutable validated coefficients, returning probability millionths.

    Model JSON is a single object with exact fields documented in grounded-value.md.
    ``handcraft_micro`` retains the planner's original score units. Geometry work is
    evaluator work and appears in evaluator_time_ns, outside the search geometry cap.
    """

    include_productive: bool
    columns: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    anchor: FrozenPositionEvaluator

    def __init__(self, model):
        required = {
            "schema",
            "target",
            "continuation",
            "include_productive",
            "columns",
            "mean",
            "scale",
            "coef",
            "intercept",
            "residual_bound",
            "anchor",
        }
        if type(model) is not dict or model.keys() != required:
            raise ValueError("invalid residual model fields")
        if (
            model["schema"] != SCHEMA
            or model["target"] != TARGET
            or model["continuation"] != CONTINUATION
        ):
            raise ValueError("residual model schema, target or continuation mismatch")
        if type(model["include_productive"]) is not bool:
            raise ValueError("include_productive must be boolean")
        columns = expected_columns(model["include_productive"])
        if type(model["columns"]) is not list or model["columns"] != list(columns):
            raise ValueError("residual model requires exact canonical columns")
        if _finite(model["residual_bound"]) != RESIDUAL_BOUND:
            raise ValueError("residual bound must be exactly 0.5")
        arrays = []
        for name in ("mean", "scale", "coef"):
            values = model[name]
            if type(values) is not list or len(values) != len(columns):
                raise ValueError("residual array length mismatch")
            arrays.append(tuple(_finite(value) for value in values))
        if any(scale <= 0 for scale in arrays[1]):
            raise ValueError("residual scales must be positive")
        for name, value in zip(("mean", "scale", "coef"), arrays, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "include_productive", model["include_productive"])
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "intercept", _finite(model["intercept"]))
        object.__setattr__(self, "anchor", _anchor(model["anchor"]))

    def __call__(self, view, projection, handcraft_micro):
        if type(handcraft_micro) is not int:
            raise ValueError("integer handcraft score required")
        features = encode_residual_features(view, projection, self.include_productive)
        if set(features) != set(self.columns):
            raise ValueError("residual public feature columns mismatch")
        z_anchor = (
            self.anchor.intercept
            + self.anchor.coef[0]
            * (handcraft_micro / 1_000_000 - self.anchor.mean[0])
            / self.anchor.scale[0]
        )
        residual = self.intercept + sum(
            coefficient * (features[name] - mean) / scale
            for name, mean, scale, coefficient in zip(
                self.columns, self.mean, self.scale, self.coef, strict=True
            )
        )
        if not math.isfinite(z_anchor) or not math.isfinite(residual):
            raise ValueError("nonfinite residual model logit")
        z = z_anchor + RESIDUAL_BOUND * math.tanh(residual)
        probability = 1 / (1 + math.exp(-max(-35, min(35, z))))
        return round(probability * 1_000_000)


def load_evaluator(path: str | Path) -> PublicResidualEvaluator:
    """Load one direct residual model object, including its frozen anchor entry."""
    return PublicResidualEvaluator(json.loads(Path(path).read_text()))


def residual_planner(path: str | Path) -> PublicTurnPlannerPolicy:
    """Ordinary public planner with unchanged budgets and probability margin 10000."""
    return PublicTurnPlannerPolicy(
        TurnPlannerConfig(),
        evaluator=load_evaluator(path),
        gain_margin_micro=PROBABILITY_GAIN_MARGIN_MICRO,
    )
