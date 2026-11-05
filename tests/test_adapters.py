"""Public adapter compatibility and physical-photo trust boundaries."""

import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import Action, DecisionContext, Observation, PublicPlayer


@pytest.fixture
def photo_observation():
    return Observation(
        player_id=0,
        board=BoardSpec.standard(7),
        rules=RulesConfig.standard(),
        context=DecisionContext("TURN", 0, 0, 12),
        players=tuple(PublicPlayer(i, 2, 3, 0) for i in range(4)),
        hand=(1, 2, 0, 0, 0),
        development=(0, 0, 0, 0, 0),
        buildings=(),
        roads=(),
        robber_tile=0,
        legal_actions=(Action("ROLL"),),
    )


def photo_bytes(color="white"):
    output = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(output, format="PNG")
    return output.getvalue()


def verified_annotation(observation, photo):
    from colonist_rl.perception import observation_to_dict

    return {
        "format": "colonist-rl.capture",
        "version": 1,
        "verified": True,
        "stable": True,
        "confidence": 1.0,
        "screenshot_sha256": hashlib.sha256(photo).hexdigest(),
        "observation": observation_to_dict(observation),
    }


@pytest.mark.parametrize(
    ("module", "names"),
    [
        ("errors", ("PerceptionError",)),
        (
            "observation_io",
            (
                "observation_to_dict",
                "view_from_dict",
                "observation_from_dict",
                "visible_events_from_export",
            ),
        ),
        ("calibration", ("ScreenCalibration",)),
        (
            "capture",
            ("CaptureObservation", "CaptureReader", "AnnotatedCaptureReader", "read_capture"),
        ),
        ("colonist.execution", ("BrowserAdapter", "BrowserTransport", "TransitionVerifier")),
    ],
)
def test_legacy_perception_exports_preserve_object_identity(module, names):
    legacy = importlib.import_module("colonist_rl.perception")
    canonical = importlib.import_module(f"colonist_rl.adapters.{module}")
    for name in names:
        assert getattr(legacy, name) is getattr(canonical, name)


def test_legacy_browser_uses_the_same_launch_patch_boundary(monkeypatch):
    legacy = importlib.import_module("colonist_rl.browser")
    canonical = importlib.import_module("colonist_rl.adapters.colonist.browser")

    class LaunchStopped(RuntimeError):
        pass

    def stop_launch():
        raise LaunchStopped("patched legacy launch boundary")

    monkeypatch.setattr(legacy, "_start_playwright", stop_launch)
    assert legacy.PlaywrightSession is canonical.PlaywrightSession
    assert legacy.BrowserConnectionError is canonical.BrowserConnectionError
    assert legacy.PlaywrightTransport is canonical.PlaywrightTransport
    assert legacy.BrowserCapture is canonical.BrowserCapture
    with pytest.raises(LaunchStopped), canonical.PlaywrightSession():
        pytest.fail("the patched launch must stop session creation")


def test_shared_and_physical_imports_do_not_load_browser_or_training_modules():
    code = """
import importlib
import json
import sys
for name in ('errors', 'observation_io', 'calibration', 'capture', 'physical'):
    importlib.import_module('colonist_rl.adapters.' + name)
forbidden = (
    'playwright', 'torch', 'catanatron', 'colonist_rl.simulator',
    'colonist_rl.browser', 'colonist_rl.adapters.colonist',
)
loaded = [name for name in sys.modules
          if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)]
print(json.dumps(loaded))
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_physical_photo_reads_complete_verified_state_through_injected_reader(
    tmp_path, photo_observation
):
    from colonist_rl.adapters.capture import AnnotatedCaptureReader
    from colonist_rl.adapters.physical import PhysicalBoardAdapter

    photo = photo_bytes()
    path = tmp_path / "board.png"
    path.write_bytes(photo)
    reader = AnnotatedCaptureReader([verified_annotation(photo_observation, photo)])
    capture = PhysicalBoardAdapter(reader=reader).read_photo(path)
    assert capture.observation == photo_observation
    assert capture.observation.hand == (1, 2, 0, 0, 0)
    assert capture.verified is True
    assert capture.screenshot_sha256 == hashlib.sha256(photo).hexdigest()


@pytest.mark.parametrize(
    "photo",
    [photo_bytes("black"), b"invalid photo bytes"],
    ids=("unregistered-photo", "unregistered-invalid-bytes"),
)
def test_physical_photo_rejects_unrecognized_images(tmp_path, photo_observation, photo):
    from colonist_rl.adapters.capture import AnnotatedCaptureReader
    from colonist_rl.adapters.errors import PerceptionError
    from colonist_rl.adapters.physical import PhysicalBoardAdapter

    known = photo_bytes()
    path = tmp_path / "unknown.png"
    path.write_bytes(photo)
    adapter = PhysicalBoardAdapter(
        reader=AnnotatedCaptureReader([verified_annotation(photo_observation, known)])
    )
    with pytest.raises(PerceptionError):
        adapter.read_photo(path)


def test_physical_photo_rejects_hash_matched_malformed_image(tmp_path, photo_observation):
    from colonist_rl.adapters.capture import AnnotatedCaptureReader
    from colonist_rl.adapters.errors import PerceptionError
    from colonist_rl.adapters.physical import PhysicalBoardAdapter

    invalid = b"not an image even with a matching annotation"
    path = tmp_path / "malformed.png"
    path.write_bytes(invalid)
    adapter = PhysicalBoardAdapter(
        reader=AnnotatedCaptureReader([verified_annotation(photo_observation, invalid)])
    )
    with pytest.raises(PerceptionError):
        adapter.read_photo(path)


@pytest.mark.parametrize("missing", ["hand", "context", "legal_actions"])
def test_board_photo_annotation_cannot_fill_missing_player_state(photo_observation, missing):
    from colonist_rl.adapters.capture import AnnotatedCaptureReader
    from colonist_rl.adapters.errors import PerceptionError

    annotation = verified_annotation(photo_observation, photo_bytes())
    del annotation["observation"]["observation"][missing]
    with pytest.raises(PerceptionError):
        AnnotatedCaptureReader([annotation])


def test_physical_adapter_requires_an_explicit_reader():
    from colonist_rl.adapters.physical import PhysicalBoardAdapter

    with pytest.raises(TypeError):
        PhysicalBoardAdapter()
