"""Verified capture records, reader protocol, and offline image annotations.

Annotated captures are verified input, not general screenshot or photo OCR.
This module never opens a browser or imports a simulator/training engine.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from ..contracts import Observation
from .errors import PerceptionError
from .observation_io import _envelope, view_from_dict


@dataclass(frozen=True)
class CaptureObservation:
    observation: Observation
    confidence: float
    stable: bool
    verified: bool
    target_mode: str | None = None
    screenshot_sha256: str | None = None

    def __post_init__(self):
        if (
            not isinstance(self.observation, Observation)
            or not isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
            or type(self.stable) is not bool
            or type(self.verified) is not bool
        ):
            raise PerceptionError("invalid capture metadata")


class AnnotatedCaptureReader:
    """Read only labeled, user-verified screenshots identified by their SHA-256.

    This is an offline annotation/template registry, not terrain/number OCR.
    Unrecognized screenshots fail closed rather than reuse an earlier state.
    """

    def __init__(self, annotations: Sequence[Mapping[str, Any]]):
        self._captures: dict[str, CaptureObservation] = {}
        for annotation in annotations:
            _envelope(annotation, "colonist-rl.capture")
            if annotation.get("verified") is not True:
                raise PerceptionError("capture must have an explicitly verified annotation")
            digest = annotation.get("screenshot_sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise PerceptionError("capture requires a lowercase screenshot SHA-256")
            capture = CaptureObservation(
                view_from_dict(annotation["observation"]),
                annotation["confidence"],
                annotation["stable"],
                True,
                annotation.get("target_mode"),
                digest,
            )
            if digest in self._captures and self._captures[digest] != capture:
                raise PerceptionError("ambiguous annotations for the same screenshot")
            self._captures[digest] = capture

    def read(self, screenshot: bytes) -> CaptureObservation:
        digest = hashlib.sha256(screenshot).hexdigest()
        if digest not in self._captures:
            raise PerceptionError("screenshot has no matching verified annotation")
        try:
            with Image.open(io.BytesIO(screenshot)) as image:
                image.verify()
        except (OSError, ValueError) as error:
            raise PerceptionError("capture is not a valid screenshot image") from error
        return self._captures[digest]


def read_capture(annotation_path: str | Path, screenshot_path: str | Path) -> CaptureObservation:
    """Load a user-annotated local screenshot without any browser access."""
    with Path(annotation_path).open() as source:
        annotation = json.load(source)
    return AnnotatedCaptureReader([annotation]).read(Path(screenshot_path).read_bytes())


class CaptureReader(Protocol):
    def read(self, screenshot: bytes) -> CaptureObservation: ...
