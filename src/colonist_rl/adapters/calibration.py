"""Affine lattice-to-image calibration shared by environment adapters.

An affine transform does not establish perspective correction for arbitrary
physical-board photos; readers remain responsible for verified geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from ..board import BoardSpec
from ..contracts import Action
from .errors import PerceptionError
from .observation_io import _integer, _validate_board

Point = tuple[float, float]
_TARGET_TYPES = {
    "BUILD_SETTLEMENT": "vertex",
    "BUILD_CITY": "vertex",
    "BUILD_ROAD": "edge",
    "MOVE_ROBBER": "tile",
}


@dataclass(frozen=True)
class ScreenCalibration:
    coefficients: tuple[tuple[float, float, float], tuple[float, float, float]]
    confidence: float = 1.0
    max_residual_pixels: float = 0.0

    def __post_init__(self):
        matrix = np.asarray(self.coefficients, dtype=float)
        if (
            matrix.shape != (2, 3)
            or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix[:, :2])) < 1e-10
        ):
            raise PerceptionError("calibration must be a finite nonsingular affine transform")
        if (
            not 0 <= self.confidence <= 1
            or not isfinite(self.max_residual_pixels)
            or self.max_residual_pixels < 0
        ):
            raise PerceptionError("invalid calibration confidence or residual")
        object.__setattr__(
            self, "coefficients", tuple(tuple(float(n) for n in row) for row in matrix)
        )

    @classmethod
    def fit(
        cls,
        anchors: Sequence[tuple[Point, Point]],
        *,
        confidence: float = 1.0,
        max_error_pixels: float = 2.0,
    ) -> ScreenCalibration:
        pairs = np.asarray(anchors, dtype=float)
        if (
            pairs.ndim != 3
            or pairs.shape[1:] != (2, 2)
            or len(pairs) < 3
            or not np.isfinite(pairs).all()
        ):
            raise PerceptionError("at least three finite lattice/pixel anchor pairs are required")
        source = np.column_stack((pairs[:, 0], np.ones(len(pairs))))
        transform, _, rank, _ = np.linalg.lstsq(source, pairs[:, 1], rcond=None)
        if rank < 3:
            raise PerceptionError("calibration anchors are collinear")
        residual = float(np.linalg.norm(source @ transform - pairs[:, 1], axis=1).max())
        if not isfinite(max_error_pixels) or max_error_pixels < 0 or residual > max_error_pixels:
            raise PerceptionError("anchor residual exceeds calibration tolerance")
        return cls(tuple(tuple(row) for row in transform.T), confidence, residual)

    def map_point(self, point: Point, *, min_confidence: float = 0.95) -> Point:
        if self.confidence < min_confidence:
            raise PerceptionError("calibration confidence is too low")
        if len(point) != 2 or not all(isfinite(value) for value in point):
            raise PerceptionError("invalid target point")
        mapped = np.asarray(self.coefficients) @ (point[0], point[1], 1.0)
        return float(mapped[0]), float(mapped[1])

    def action_point(
        self,
        board: BoardSpec,
        action: Action,
        *,
        controls: Mapping[Action, Point] | None = None,
        min_confidence: float = 0.95,
    ) -> Point:
        if self.confidence < min_confidence:
            raise PerceptionError("calibration confidence is too low")
        if controls is not None and action in controls:
            point = controls[action]
            if len(point) != 2 or not all(isfinite(value) for value in point):
                raise PerceptionError("control has invalid screen coordinates")
            return float(point[0]), float(point[1])
        _validate_board(board)
        if action.kind not in _TARGET_TYPES or not _integer(action.target):
            raise PerceptionError("action requires an exact explicitly calibrated control")
        target = action.target
        try:
            if _TARGET_TYPES[action.kind] == "vertex":
                point = board.vertices[target]
            elif _TARGET_TYPES[action.kind] == "edge":
                vertices = [board.vertices[v] for v in board.edges[target]]
                point = tuple(np.mean(vertices, axis=0))
            else:
                vertices = [board.vertices[v] for v in board.tile_vertices[target]]
                point = tuple(np.mean(vertices, axis=0))
        except IndexError as error:
            raise PerceptionError("action target is absent from the board") from error
        return self.map_point(point, min_confidence=min_confidence)
