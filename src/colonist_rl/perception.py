"""Compatibility exports for shared capture and Colonist execution adapters.

New code can import the focused modules under :mod:`colonist_rl.adapters`.
Existing imports retain the same classes, functions, protocols, and exception.
"""

from .adapters.calibration import Point, ScreenCalibration
from .adapters.capture import (
    AnnotatedCaptureReader,
    CaptureObservation,
    CaptureReader,
    read_capture,
)
from .adapters.colonist.execution import BrowserAdapter, BrowserTransport, TransitionVerifier
from .adapters.errors import PerceptionError
from .adapters.observation_io import (
    observation_from_dict,
    observation_to_dict,
    view_from_dict,
    visible_events_from_export,
)

__all__ = [
    "AnnotatedCaptureReader",
    "BrowserAdapter",
    "BrowserTransport",
    "CaptureObservation",
    "CaptureReader",
    "PerceptionError",
    "Point",
    "ScreenCalibration",
    "TransitionVerifier",
    "observation_from_dict",
    "observation_to_dict",
    "read_capture",
    "view_from_dict",
    "visible_events_from_export",
]
