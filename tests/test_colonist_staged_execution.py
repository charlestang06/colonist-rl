"""Synthetic independent complete snapshots exercise the public staged boundary."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from colonist_rl.adapters.calibration import ScreenCalibration
from colonist_rl.adapters.capture import CaptureObservation
from colonist_rl.adapters.colonist.staged_execution import (
    LocatedUIControl,
    StagedBrowserAdapter,
    StagedCapture,
)
from colonist_rl.adapters.colonist.ui import UIControl, UIState, UITradeResponse, UITradeRow
from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import (
    Action,
    Building,
    DecisionContext,
    Observation,
    PublicPlayer,
    Road,
    TradeOffer,
)

CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "colonist-staged-execution.json").read_text()
)
CALIBRATION = ScreenCalibration(((1, 0, 100), (0, 1, 100)))
GIVE, RECEIVE = (1, 0, 0, 0, 0), (0, 0, 0, 1, 0)


def snapshot(data=None, *, incoming=False, kind="BUILD_SETTLEMENT", phase="SETUP_SETTLEMENT"):
    """Build complete immutable snapshots from literal frame descriptions; no gestures read."""
    data = data or {}
    revision, version = data.get("revision", 10), data.get("version", 12)
    phase = data.get("phase", "TRADE_RESPONSE" if incoming else phase)
    action = (
        Action("ACCEPT_TRADE", offer_id="offer-a")
        if incoming
        else Action(kind, target=data.get("target", 4))
    )
    offer = TradeOffer("offer-a", 1, 0, GIVE, RECEIVE)
    buildings = (Building(4, 0),) if kind == "BUILD_CITY" else ()
    roads = ()
    if data.get("placed"):
        if kind == "BUILD_ROAD":
            roads = (Road(4, 0),)
        else:
            buildings = (Building(4, 0, 2 if kind == "BUILD_CITY" else 1),)
    observation = Observation(
        0,
        BoardSpec.standard(7),
        RulesConfig.standard(),
        DecisionContext(
            phase,
            0,
            1 if incoming else 0,
            version,
            offer=offer if incoming and not data.get("no_offer") else None,
        ),
        tuple(PublicPlayer(i, 2, 6, 0) for i in range(4)),
        tuple(data.get("hand", (1, 1, 1, 1, 2))),
        (0, 0, 0, 0, 0),
        buildings,
        roads,
        0,
        (action,),
    )
    controls, rows = [], []
    confirm = None
    if data.get("confirmation"):
        confirm = UIControl("confirm", revision, action, "click")
        controls.append(LocatedUIControl(confirm, (150, 160)))
    if incoming and not data.get("no_row"):
        control = UIControl("accept-a", revision, action, "click", "row-a", 0)
        responses = (
            (UITradeResponse(data.get("marker_player", 0), data["marker"]),)
            if data.get("marker")
            else ()
        )
        rows.append(
            UITradeRow(
                "row-a",
                "offer-a",
                1,
                tuple(data.get("row_give", GIVE)),
                RECEIVE,
                revision,
                responses,
                (control,),
            )
        )
        controls.append(LocatedUIControl(control, (200, 210)))
    if data.get("other_row"):
        other = Action("ACCEPT_TRADE", offer_id="offer-b")
        control = UIControl("accept-b", revision, other, "click", "row-b", 0)
        rows.insert(
            0,
            UITradeRow(
                "row-b",
                "offer-b",
                2,
                GIVE,
                RECEIVE,
                revision,
                (UITradeResponse(2, "willing"),),
                (control,),
            ),
        )
        controls.insert(0, LocatedUIControl(control, (250, 260)))
    ui = UIState(
        revision,
        version,
        phase,
        active_tool=None if incoming else kind,
        selected_action=action if incoming or confirm else None,
        dialog_stage="confirmation" if confirm else None,
        confirmation=confirm,
        trade_rows=tuple(rows),
    )
    return StagedCapture(
        CaptureObservation(
            observation,
            data.get("confidence", 1.0),
            data.get("stable", True),
            data.get("verified", True),
            None if incoming else kind,
        ),
        ui,
        controls,
    )


class Transport:
    def __init__(self, frames, *, click_error=None, read_error_at=None):
        self.frames = [f"frame-{i}".encode() for i in range(len(frames))]
        self.calls = []
        self.gestures = []
        self.click_error, self.read_error_at = click_error, read_error_at
        self.click_count = 0
        self.reads = 0
        self.adapter = None

    def screenshot(self):
        self.calls.append(("screenshot",))
        self.reads += 1
        if self.read_error_at == self.reads:
            raise RuntimeError("capture disconnected")
        return self.frames.pop(0)

    def click(self, x, y):
        assert self.adapter.needs_review, "lock must be set before every input"
        self.calls.append(("click", x, y))
        self.gestures.append(("click", x, y))
        self.click_count += 1
        if self.click_error and self.click_count == self.click_error[0]:
            raise self.click_error[1]


class Reader:
    def __init__(self, frames):
        self.frames = {f"frame-{i}".encode(): frame for i, frame in enumerate(frames)}
        self.calls = []

    def read(self, screenshot):
        self.calls.append(screenshot)
        return self.frames[screenshot]


def adapter(frames, **kwargs):
    transport = Transport(
        frames,
        click_error=kwargs.pop("click_error", None),
        read_error_at=kwargs.pop("read_error_at", None),
    )
    waits = []
    values = {
        "calibration": CALIBRATION,
        "transport": transport,
        "reader": Reader(frames),
        "enabled": True,
        "authorized": True,
        "wait": waits.append,
    }
    values.update(kwargs)
    result = StagedBrowserAdapter(**values)
    transport.adapter = result
    return result, transport, waits


def run(frames, *, incoming=False, **kwargs):
    instance, transport, waits = adapter(frames, **kwargs)
    action = (
        Action("ACCEPT_TRADE", offer_id="offer-a")
        if incoming
        else frames[0].capture.observation.legal_actions[0]
    )
    result = instance.execute(
        frames[0].capture.observation,
        action,
        expected_revision=10,
        row_id="row-a" if incoming else None,
    )
    return result, instance, transport, waits


@pytest.mark.parametrize("scenario", CATALOG["scenarios"], ids=lambda x: x["name"])
def test_catalog_executes_every_synthetic_stage_with_exact_inputs(scenario):
    assert CATALOG["provenance"] == "synthetic"
    frames = [
        snapshot(
            data,
            incoming=scenario.get("incoming", False),
            kind=scenario.get("kind", "BUILD_SETTLEMENT"),
            phase=scenario.get("phase", "SETUP_SETTLEMENT"),
        )
        for data in scenario["frames"]
    ]
    original = frames[0].capture.observation
    result, instance, transport, waits = run(
        frames,
        incoming=scenario.get("incoming", False),
        max_captures=scenario.get("max_captures", 8),
    )
    assert result.status == scenario["status"]
    assert transport.gestures == [tuple(item) for item in scenario["gestures"]]
    assert transport.reads == scenario["captures"]
    assert waits == [0.05] * scenario.get("waits", 0)
    assert instance.needs_review == (result.status == "ambiguous")
    assert frames[0].capture.observation == original
    if result.status == "committed":
        assert result.observation == frames[-1].capture.observation
        assert result.action.kind in result.evidence[0]
        assert "4" in result.evidence[0] and "0" in result.evidence[0]
    elif result.status == "pending":
        assert result.observation is None
        assert frames[-1].capture.observation == original
        assert "row-a" in result.evidence[0] and "willing" in result.evidence[0]
    else:
        assert result.reason


def test_target_then_current_confirmation_has_exact_ordered_transport_boundary():
    frames = [
        snapshot(),
        snapshot({"revision": 11, "confirmation": True}),
        snapshot({"revision": 12, "version": 13, "placed": True}),
    ]
    result, _, transport, _ = run(frames)
    assert result.status == "committed"
    assert transport.calls == [
        ("screenshot",),
        ("click", 97, 95),
        ("screenshot",),
        ("click", 150, 160),
        ("screenshot",),
    ]


@pytest.mark.parametrize("gesture", [1, 2])
@pytest.mark.parametrize("error", [RuntimeError("click failed"), KeyboardInterrupt()])
def test_exception_or_interruption_preserves_preinput_lock_at_each_click(gesture, error):
    frames = [
        snapshot(),
        snapshot({"revision": 11, "confirmation": True}),
        snapshot({"revision": 12, "version": 13, "placed": True}),
    ]
    instance, transport, _ = adapter(frames, click_error=(gesture, error))
    execute = lambda: instance.execute(
        frames[0].capture.observation, Action("BUILD_SETTLEMENT", target=4), expected_revision=10
    )
    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            execute()
    else:
        assert execute().status == "ambiguous"
    assert instance.needs_review
    assert transport.gestures == [("click", 97, 95), ("click", 150, 160)][:gesture]
    before_calls = list(transport.calls)
    assert execute().status == "blocked"
    assert transport.calls == before_calls


@pytest.mark.parametrize("read_number", [2, 3])
def test_postinput_capture_exception_locks_without_another_gesture(read_number):
    frames = [snapshot(), snapshot({"revision": 11, "confirmation": True}), snapshot()]
    result, instance, transport, _ = run(frames, read_error_at=read_number)
    assert result.status == "ambiguous" and instance.needs_review
    assert transport.gestures == [("click", 97, 95), ("click", 150, 160)][: read_number - 1]


@pytest.mark.parametrize(
    "change",
    [
        {"confidence": 0.5},
        {"verified": False},
        {"stable": False},
    ],
)
def test_initial_untrusted_or_unstable_capture_blocks_without_click(change):
    result, instance, transport, _ = run([snapshot(change)])
    assert result.status == "blocked" and not instance.needs_review
    assert transport.calls == [("screenshot",)]


@pytest.mark.parametrize("change", [{"confidence": 0.5}, {"verified": False}])
def test_postinput_untrusted_capture_is_immediately_ambiguous(change):
    result, instance, transport, _ = run([snapshot(), snapshot(change)])
    assert result.status == "ambiguous" and instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 97, 95), ("screenshot",)]


@pytest.mark.parametrize(
    "mutation",
    [
        "selection",
        "dialog",
        "tool",
        "phase",
        "target_mode",
        "legal",
        "actor",
        "stale",
        "calibration",
    ],
)
def test_initial_preflight_fails_closed(mutation):
    frame = snapshot()
    observation, ui, capture = frame.capture.observation, frame.ui, frame.capture
    supplied = observation
    kwargs = {}
    if mutation == "selection":
        ui = replace(ui, selected_action=Action("BUILD_SETTLEMENT", target=5))
    if mutation == "dialog":
        ui = replace(ui, dialog_stage="unrelated")
    if mutation == "tool":
        ui = replace(ui, active_tool="BUILD_ROAD")
    if mutation == "phase":
        ui = replace(ui, phase="OTHER")
    if mutation == "target_mode":
        capture = replace(capture, target_mode="BUILD_ROAD")
    if mutation == "legal":
        observation = replace(observation, legal_actions=())
    if mutation == "actor":
        observation = replace(observation, context=replace(observation.context, actor=1))
    if mutation == "stale":
        supplied = replace(observation, hand=(9, 0, 0, 0, 0))
    if mutation == "calibration":
        kwargs["calibration"] = replace(CALIBRATION, confidence=0.1)
    capture = replace(capture, observation=observation)
    frame = StagedCapture(capture, ui, frame.controls)
    instance, transport, _ = adapter([frame], **kwargs)
    if mutation != "stale":
        supplied = observation
    result = instance.execute(supplied, Action("BUILD_SETTLEMENT", target=4), expected_revision=10)
    assert result.status == "blocked" and result.reason
    assert transport.gestures == [] and not instance.needs_review


@pytest.mark.parametrize("mutation", ["wrong-point-binding", "missing", "stale-control"])
def test_confirmation_requires_complete_current_located_control(mutation):
    frame = snapshot({"confirmation": True})
    if mutation == "missing":
        controls = ()
    else:
        wrong = replace(
            frame.ui.confirmation,
            **(
                {"action": Action("BUILD_SETTLEMENT", target=5)}
                if mutation == "wrong-point-binding"
                else {"revision": 9}
            ),
        )
        if mutation == "stale-control":
            frame = replace(frame, ui=replace(frame.ui, confirmation=wrong), controls=())
            controls = ()
        else:
            controls = (LocatedUIControl(wrong, (150, 160)),)
    result, instance, transport, _ = run([replace(frame, controls=controls)])
    assert result.status == "blocked" and not instance.needs_review
    assert transport.gestures == []


@pytest.mark.parametrize("changes", [{"enabled": False}, {"authorized": False}])
def test_permission_absence_raises_without_io(changes):
    instance, transport, _ = adapter([snapshot()], **changes)
    with pytest.raises(PermissionError):
        instance.execute(
            snapshot().capture.observation,
            Action("BUILD_SETTLEMENT", target=4),
            expected_revision=10,
        )
    assert transport.calls == []


@pytest.mark.parametrize(
    "action",
    [
        Action("BUILD_SETTLEMENT", target=True),
        Action("BUILD_SETTLEMENT", target=4.0),
        Action("BUILD_SETTLEMENT", target=-1),
        Action("ACCEPT_TRADE", offer_id=""),
        "BUILD_SETTLEMENT",
    ],
)
def test_malformed_action_raises_before_io(action):
    instance, transport, _ = adapter([snapshot()])
    with pytest.raises((TypeError, ValueError)):
        instance.execute(snapshot().capture.observation, action, expected_revision=10)
    assert transport.calls == []


@pytest.mark.parametrize(
    "action",
    [
        Action("ROLL"),
        Action("BUILD_SETTLEMENT"),
        Action("BUILD_SETTLEMENT", target=4, resource="ORE"),
        Action("ACCEPT_TRADE", offer_id="offer-a", partner=0),
    ],
)
def test_well_formed_unsupported_action_shape_blocks_without_gesture(action):
    instance, transport, _ = adapter([snapshot()])
    assert (
        instance.execute(snapshot().capture.observation, action, expected_revision=10).status
        == "blocked"
    )
    assert transport.gestures == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_captures": 0},
        {"max_captures": True},
        {"max_captures": 1.0},
        {"poll_interval": -1},
        {"poll_interval": float("nan")},
        {"poll_interval": 1.01},
        {"poll_interval": True},
        {"min_confidence": True},
        {"min_confidence": float("inf")},
        {"wait": None},
        {"enabled": 1},
        {"authorized": 1},
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        adapter([snapshot()], **kwargs)


@pytest.mark.parametrize(
    "point", [(-1, 0), (True, 1), (1, float("nan")), (1, float("inf")), ("1", 2), (1,), "12"]
)
def test_located_control_rejects_invalid_coordinates(point):
    with pytest.raises((TypeError, ValueError)):
        LocatedUIControl(UIControl("control", 10, Action("ROLL"), "click"), point)


def test_capture_detaches_points_and_controls_and_checks_revision_and_version():
    frame = snapshot({"confirmation": True})
    point = [150, 160]
    located = LocatedUIControl(frame.ui.confirmation, point)
    controls = [located]
    detached = StagedCapture(frame.capture, frame.ui, controls)
    point[0] = 999
    controls.clear()
    assert detached.controls == (located,) and located.point == (150, 160)
    with pytest.raises(FrozenInstanceError):
        located.point = (1, 2)
    with pytest.raises((TypeError, ValueError)):
        replace(detached, controls=(located, located))
    with pytest.raises((TypeError, ValueError)):
        replace(
            detached, controls=(LocatedUIControl(replace(located.control, revision=9), (1, 2)),)
        )
    with pytest.raises((TypeError, ValueError)):
        replace(detached, ui=replace(frame.ui, observation_version=99))


@pytest.mark.parametrize("after", [False, True])
def test_digest_binds_whole_capture_to_exact_transport_bytes(after):
    frame = snapshot()
    bad = replace(
        frame,
        capture=replace(frame.capture, screenshot_sha256=hashlib.sha256(b"different").hexdigest()),
    )
    result, instance, transport, _ = run([frame, bad] if after else [bad])
    assert result.status == ("ambiguous" if after else "blocked")
    assert instance.needs_review == after
    assert transport.gestures == ([("click", 97, 95)] if after else [])


def test_malformed_reader_result_blocks_before_input_and_locks_after_input():
    for frames, expected, gestures in [
        ([None], "blocked", []),
        ([snapshot(), None], "ambiguous", [("click", 97, 95)]),
    ]:
        instance, transport, _ = adapter(frames)
        result = instance.execute(
            snapshot().capture.observation,
            Action("BUILD_SETTLEMENT", target=4),
            expected_revision=10,
        )
        assert result.status == expected
        assert transport.gestures == gestures


@pytest.mark.parametrize(
    "changes",
    [
        {"row_id": "other"},
        {"offer_id": "other"},
        {"proposer": 2},
        {"give": (0, 1, 0, 0, 0)},
        {"revision": 9},
    ],
)
def test_incoming_wrong_or_stale_row_blocks_without_click(changes):
    frame = snapshot(incoming=True)
    row = replace(frame.ui.trade_rows[0], **changes)
    frame = replace(frame, ui=replace(frame.ui, trade_rows=(row,)))
    result, _, transport, _ = run([frame], incoming=True)
    assert result.status == "blocked" and transport.gestures == []


def test_incoming_requires_explicit_row_and_funding():
    frame = snapshot(incoming=True)
    for row_id, current in [
        (None, frame),
        (
            "row-a",
            replace(
                frame,
                capture=replace(
                    frame.capture,
                    observation=replace(frame.capture.observation, hand=(0, 0, 0, 0, 0)),
                ),
            ),
        ),
    ]:
        instance, transport, _ = adapter([current])
        result = instance.execute(
            current.capture.observation,
            Action("ACCEPT_TRADE", offer_id="offer-a"),
            expected_revision=10,
            row_id=row_id,
        )
        assert result.status == "blocked" and transport.gestures == []


def test_import_does_not_start_browser_or_load_training_or_simulator():
    code = """
import importlib, json, sys
importlib.import_module('colonist_rl.adapters.colonist.staged_execution')
forbidden = ('playwright', 'torch', 'catanatron', 'colonist_rl.simulator',
             'colonist_rl.adapters.colonist.browser')
print(json.dumps([name for name in sys.modules
                 if any(name == x or name.startswith(x + '.') for x in forbidden)]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize("which", ["selection", "confirmation"])
def test_even_committed_board_evidence_rejects_a_different_selected_target(which):
    after = snapshot({"revision": 11, "version": 13, "placed": True})
    wrong = Action("BUILD_SETTLEMENT", target=5)
    if which == "selection":
        after = replace(after, ui=replace(after.ui, selected_action=wrong))
    else:
        control = UIControl("wrong-confirmation", 11, wrong, "click")
        after = replace(
            after,
            ui=replace(
                after.ui, selected_action=Action("BUILD_SETTLEMENT", target=4), confirmation=control
            ),
        )
    result, instance, transport, _ = run([snapshot(), after])
    assert result.status == "ambiguous" and instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 97, 95), ("screenshot",)]


def test_stable_commit_can_follow_same_revision_unstable_board_render():
    frames = [
        snapshot(),
        snapshot({"revision": 11, "version": 13, "placed": True, "stable": False}),
        snapshot({"revision": 11, "version": 13, "placed": True}),
    ]
    result, instance, transport, waits = run(frames)
    assert result.status == "committed" and not instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 97, 95), ("screenshot",), ("screenshot",)]
    assert waits == [0.05]


def test_changed_confirmation_coordinates_under_same_revision_are_ambiguous():
    confirm = snapshot({"revision": 11, "confirmation": True, "stable": False})
    moved = replace(
        confirm,
        capture=replace(confirm.capture, stable=True),
        controls=(replace(confirm.controls[0], point=(151, 160)),),
    )
    result, instance, transport, _ = run([snapshot(), confirm, moved])
    assert result.status == "ambiguous" and instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 97, 95), ("screenshot",), ("screenshot",)]


@pytest.mark.parametrize("field", ["player_id", "board", "rules", "version"])
def test_postinput_identity_geometry_rules_and_version_changes_never_commit(field):
    after = snapshot({"revision": 11, "version": 13, "placed": True})
    game = after.capture.observation
    if field == "player_id":
        game = replace(game, player_id=1)
    if field == "board":
        game = replace(game, board=BoardSpec.standard(8))
    if field == "rules":
        game = replace(game, rules=replace(game.rules, victory_points=11))
    if field == "version":
        game = replace(game, context=replace(game.context, version=11))
    after = replace(
        after,
        capture=replace(after.capture, observation=game),
        ui=replace(after.ui, observation_version=game.context.version),
    )
    result, instance, transport, _ = run([snapshot(), after])
    assert result.status == "ambiguous" and instance.needs_review
    assert transport.gestures == [("click", 97, 95)]


@pytest.mark.parametrize("read_number", [2, 3])
def test_interruption_while_reading_after_each_click_keeps_lock(read_number):
    frames = [snapshot(), snapshot({"revision": 11, "confirmation": True}), snapshot()]
    instance, transport, _ = adapter(frames)
    original_read = instance.reader.read

    def interrupted_read(screenshot):
        if screenshot == f"frame-{read_number - 1}".encode():
            raise KeyboardInterrupt()
        return original_read(screenshot)

    instance.reader.read = interrupted_read
    with pytest.raises(KeyboardInterrupt):
        instance.execute(
            frames[0].capture.observation,
            Action("BUILD_SETTLEMENT", target=4),
            expected_revision=10,
        )
    assert instance.needs_review
    assert transport.gestures == [("click", 97, 95), ("click", 150, 160)][: read_number - 1]


def test_incoming_interruption_sets_lock_before_response_input():
    frame = snapshot(incoming=True)
    instance, transport, _ = adapter([frame], click_error=(1, KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        instance.execute(
            frame.capture.observation,
            Action("ACCEPT_TRADE", offer_id="offer-a"),
            expected_revision=10,
            row_id="row-a",
        )
    assert instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 200, 210)]


def test_digest_match_is_accepted_and_absent_digest_trust_remains_supported():
    before, after = snapshot(), snapshot({"revision": 11, "version": 13, "placed": True})
    frames = [
        replace(
            before,
            capture=replace(
                before.capture, screenshot_sha256=hashlib.sha256(b"frame-0").hexdigest()
            ),
        ),
        after,
    ]
    result, _, transport, _ = run(frames)
    assert result.status == "committed"
    assert transport.calls == [("screenshot",), ("click", 97, 95), ("screenshot",)]


@pytest.mark.parametrize(
    "kwargs", [{"expected_revision": True}, {"expected_revision": -1}, {"row_id": 2}]
)
def test_invalid_execute_metadata_raises_before_capture(kwargs):
    instance, transport, _ = adapter([snapshot()])
    values = {"expected_revision": 10}
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        instance.execute(
            snapshot().capture.observation, Action("BUILD_SETTLEMENT", target=4), **values
        )
    assert transport.calls == []


def test_incoming_visible_willing_dialog_can_remove_response_control():
    before = snapshot(incoming=True)
    after = snapshot({"revision": 11, "marker": "willing"}, incoming=True)
    after = replace(
        after,
        ui=replace(
            after.ui,
            dialog_stage="willing",
            trade_rows=(replace(after.ui.trade_rows[0], controls=()),),
        ),
        controls=(),
    )
    result, instance, transport, _ = run([before, after], incoming=True)
    assert result.status == "pending" and result.observation is None
    assert result.evidence and not instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 200, 210), ("screenshot",)]


def test_postinput_phase_change_with_unchanged_game_is_ambiguous():
    after = snapshot({"revision": 11})
    after = replace(after, ui=replace(after.ui, phase="ROBBER"))
    result, instance, transport, _ = run([snapshot(), after])
    assert result.status == "ambiguous" and instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 97, 95), ("screenshot",)]


def test_expected_revision_must_match_initial_capture_before_any_input():
    before = snapshot()
    instance, transport, _ = adapter([before])
    result = instance.execute(
        before.capture.observation, Action("BUILD_SETTLEMENT", target=4), expected_revision=9
    )
    assert result.status == "blocked" and not instance.needs_review
    assert transport.calls == [("screenshot",)]


@pytest.mark.parametrize("mutation", ["malformed-board", "missing-geometry-target"])
def test_invalid_board_or_uncalibratable_legal_target_blocks_before_click(mutation):
    before = snapshot()
    game = before.capture.observation
    action = Action("BUILD_SETTLEMENT", target=4)
    if mutation == "malformed-board":
        game = replace(game, board=replace(game.board, vertices=()))
    else:
        action = Action("BUILD_SETTLEMENT", target=999)
        game = replace(game, legal_actions=(action,))
    before = replace(before, capture=replace(before.capture, observation=game))
    instance, transport, _ = adapter([before])
    result = instance.execute(game, action, expected_revision=10)
    assert result.status == "blocked" and not instance.needs_review
    assert transport.calls == [("screenshot",)]


@pytest.mark.parametrize(
    "changes",
    [
        {"dialog_stage": "unexpected"},
        {"active_tool": "MOVE_ROBBER"},
        {"target_mode": "MOVE_ROBBER"},
        {"active_tool": None},
        {"target_mode": None},
        {"dialog_stage": "confirmation"},
        {"trade_rows": snapshot({"revision": 11}, incoming=True).ui.trade_rows},
    ],
)
def test_exact_placement_effect_cannot_override_incompatible_terminal_ui(changes):
    after = snapshot({"revision": 11, "version": 13, "placed": True})
    values = dict(changes)
    capture_changes = {}
    if "target_mode" in values:
        capture_changes["target_mode"] = values.pop("target_mode")
    after = replace(
        after, capture=replace(after.capture, **capture_changes), ui=replace(after.ui, **values)
    )
    result, instance, transport, waits = run([snapshot(), after])
    assert result.status == "ambiguous" and result.reason and instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 97, 95), ("screenshot",)]
    assert waits == []


@pytest.mark.parametrize("existing_confirmation", [False, True])
def test_exact_placement_effect_allows_completed_ui_with_both_modes_cleared(existing_confirmation):
    before = snapshot({"confirmation": existing_confirmation})
    after = snapshot({"revision": 11, "version": 13, "placed": True})
    after = replace(
        after,
        capture=replace(after.capture, target_mode=None),
        ui=replace(after.ui, active_tool=None),
    )
    result, instance, transport, waits = run([before, after])
    assert result.status == "committed" and result.observation == after.capture.observation
    assert result.evidence and not instance.needs_review
    assert transport.calls == [
        ("screenshot",),
        ("click", 150, 160) if existing_confirmation else ("click", 97, 95),
        ("screenshot",),
    ]
    assert waits == []


def test_unstable_incoming_own_rejection_stops_before_later_willing_snapshot():
    frames = [
        snapshot(incoming=True),
        snapshot({"revision": 11, "marker": "rejected", "stable": False}, incoming=True),
        snapshot({"revision": 12, "marker": "willing"}, incoming=True),
    ]
    result, instance, transport, waits = run(frames, incoming=True)
    assert result.status == "ambiguous" and result.reason and instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 200, 210), ("screenshot",)]
    assert waits == []


@pytest.mark.parametrize("change", [{"no_row": True}, {"phase": "TURN", "no_offer": True}])
def test_unstable_incoming_cancellation_is_immediately_ambiguous(change):
    frames = [
        snapshot(incoming=True),
        snapshot({"revision": 11, "stable": False, **change}, incoming=True),
        snapshot({"revision": 12, "marker": "willing"}, incoming=True),
    ]
    result, instance, transport, waits = run(frames, incoming=True)
    assert result.status == "ambiguous" and result.reason and instance.needs_review
    assert transport.calls == [("screenshot",), ("click", 200, 210), ("screenshot",)]
    assert waits == []


def test_benign_unstable_incoming_render_still_polls_without_repeating_response():
    frames = [
        snapshot(incoming=True),
        snapshot({"revision": 11, "stable": False}, incoming=True),
        snapshot({"revision": 12, "marker": "willing"}, incoming=True),
    ]
    result, instance, transport, waits = run(frames, incoming=True)
    assert result.status == "pending" and result.evidence and not instance.needs_review
    assert transport.calls == [
        ("screenshot",),
        ("click", 200, 210),
        ("screenshot",),
        ("screenshot",),
    ]
    assert waits == [0.05]


@pytest.mark.parametrize("context", [None, "invalid", Action("ROLL"), (), 42])
def test_malformed_nested_observation_context_raises_type_or_value_error_without_io(context):
    before = snapshot()
    instance, transport, _ = adapter([before])
    malformed = replace(before.capture.observation, context=context)
    with pytest.raises((TypeError, ValueError)):
        instance.execute(malformed, Action("BUILD_SETTLEMENT", target=4), expected_revision=10)
    assert transport.calls == []
