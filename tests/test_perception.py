import hashlib
import io
import json
from dataclasses import FrozenInstanceError, replace

import pytest
from PIL import Image

from colonist_rl.board import BoardSpec, RulesConfig
from colonist_rl.contracts import (
    Action,
    Building,
    DecisionContext,
    Observation,
    PublicPlayer,
    ResourceBelief,
    Road,
    TradeOffer,
    VisibleEvent,
)
from colonist_rl.perception import (
    AnnotatedCaptureReader,
    BrowserAdapter,
    CaptureObservation,
    PerceptionError,
    ScreenCalibration,
    observation_to_dict,
    read_capture,
    view_from_dict,
    visible_events_from_export,
)


def observation(extended=False, *, action=None):
    board = BoardSpec.extended(2) if extended else BoardSpec.standard(2)
    rules = RulesConfig.extended() if extended else RulesConfig.standard()
    return Observation(
        player_id=0,
        board=board,
        rules=rules,
        context=DecisionContext("TURN", 0, 0, 7, eligible_partners=(1, 2, 3)),
        players=tuple(PublicPlayer(i, 2, 3, 0) for i in range(rules.player_count)),
        hand=(1, 1, 1, 0, 0),
        development=(0, 0, 0, 0, 0),
        buildings=(Building(0, 1),),
        roads=(Road(0, 1),),
        robber_tile=0,
        legal_actions=(action or Action("ROLL"),),
        beliefs=(
            ResourceBelief(
                1,
                (1.0, 0.0, 0.0, 0.0, 0.0),
                (0, 0, 0, 0, 0),
                (3, 3, 3, 3, 3),
                (1.0, 0.0, 0.0, 0.0, 0.0),
            ),
        ),
        events=(VisibleEvent(6, "DICE_ROLL", 1, count=8),),
        bank=(19, 18, 17, 16, 15),
    )


def calibration(confidence=1):
    return ScreenCalibration.fit(
        [((0, 0), (100, 200)), ((1, 0), (120, 203)), ((0, 1), (104, 230)), ((1, 1), (124, 233))],
        confidence=confidence,
    )


def image_bytes(color="white"):
    output = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize("extended", [False, True])
def test_versioned_observation_roundtrip_preserves_complete_immutable_topology(extended):
    original = observation(extended)
    record = json.loads(json.dumps(observation_to_dict(original)))
    restored = view_from_dict(record)
    assert restored == original
    assert observation_to_dict(restored) == record
    with pytest.raises(FrozenInstanceError):
        restored.context.version = 3
    assert isinstance(restored.board.tile_vertices[0], tuple)
    assert isinstance(restored.rules.supported_phases, tuple)
    assert isinstance(restored.beliefs[0].variance, tuple)


def test_roundtrip_preserves_pending_offer_and_resource_action_vectors():
    old = observation()
    offer = TradeOffer("live", 1, 0, (1, 0, 0, 0, 0), (0, 1, 0, 0, 0))
    context = replace(old.context, kind="TRADE_RESPONSE", turn_owner=1, offer=offer)
    original = replace(
        old, context=context, legal_actions=(Action("ACCEPT_TRADE", offer_id="live"),)
    )
    assert view_from_dict(json.loads(json.dumps(observation_to_dict(original)))) == original


@pytest.mark.parametrize("mutation", ["version", "terrain", "hidden", "hand"])
def test_unknown_schema_malformed_board_and_hidden_fields_are_rejected(mutation):
    record = observation_to_dict(observation())
    if mutation == "version":
        record["version"] = 99
    elif mutation == "terrain":
        record["observation"]["board"]["tiles"][0]["resource"] = "UNKNOWN"
    elif mutation == "hidden":
        record["observation"]["opponent_hands"] = [[1, 2, 3, 4, 5]]
    else:
        record["observation"]["hand"] = [0, -1, 0, 0, 0]
    with pytest.raises(PerceptionError):
        view_from_dict(record)


def test_structured_log_filters_private_events_and_redacts_hidden_resource_identity():
    exported = {
        "format": "colonist-rl.visible-events",
        "version": 1,
        "events": [
            {
                "sequence": 1,
                "kind": "STEAL",
                "actor": 0,
                "recipient": 1,
                "resource": "WOOD",
                "count": 1,
                "visibility": "public",
                "resource_visibility": [0, 1],
            },
            {"sequence": 2, "kind": "PRIVATE_NOTE", "visibility": [0]},
            {
                "sequence": 3,
                "kind": "BUILD_ROAD",
                "actor": 2,
                "visibility": "public",
                "give": [1, 1, 0, 0, 0],
                "resource_visibility": "public",
            },
        ],
    }
    outsider = visible_events_from_export(exported, observer_id=2)
    assert len(outsider) == 2
    assert outsider[0].resource is None and outsider[0].count == 1
    assert outsider[1].give == (1, 1, 0, 0, 0)
    party = visible_events_from_export(exported, observer_id=0)
    assert party[0].resource == "WOOD"


def test_log_ingestion_requires_explicit_visibility_and_monotone_sequences():
    exported = {
        "format": "colonist-rl.visible-events",
        "version": 1,
        "events": [{"sequence": 1, "kind": "STEAL", "resource": "WOOD"}],
    }
    with pytest.raises(PerceptionError):
        visible_events_from_export(exported, observer_id=2)


def test_affine_calibration_maps_lattice_and_all_three_target_types():
    board = BoardSpec.from_hexes([(0, 0)])
    fitted = calibration()
    assert fitted.map_point((2, 3)) == pytest.approx((152, 296))
    assert fitted.action_point(board, Action("BUILD_SETTLEMENT", target=0)) == pytest.approx(
        (92, 140)
    )
    assert fitted.action_point(board, Action("BUILD_ROAD", target=0)) == pytest.approx((104, 156.5))
    assert fitted.action_point(board, Action("MOVE_ROBBER", target=0)) == pytest.approx((100, 200))


@pytest.mark.parametrize("extended", [False, True])
def test_calibration_targets_support_different_board_sizes(extended):
    board = observation(extended).board
    point = calibration().action_point(board, Action("MOVE_ROBBER", target=len(board.tiles) - 1))
    assert all(isinstance(value, float) for value in point)


def test_calibration_rejects_degenerate_anchors_and_unconfident_or_missing_targets():
    with pytest.raises(PerceptionError):
        ScreenCalibration.fit([((0, 0), (0, 0)), ((1, 0), (1, 0)), ((2, 0), (2, 0))])
    board = BoardSpec.from_hexes([(0, 0)])
    with pytest.raises(PerceptionError):
        calibration(0.5).action_point(board, Action("BUILD_ROAD", target=0))
    with pytest.raises(PerceptionError):
        calibration().action_point(board, Action("BUILD_ROAD", target=999))
    with pytest.raises(PerceptionError):
        calibration().action_point(board, Action("ROLL"))


def annotation(view, screenshot):
    return {
        "format": "colonist-rl.capture",
        "version": 1,
        "verified": True,
        "stable": True,
        "confidence": 1.0,
        "screenshot_sha256": hashlib.sha256(screenshot).hexdigest(),
        "observation": observation_to_dict(view),
    }


def test_verified_annotated_screenshot_ingestion_checks_image_identity(tmp_path):
    original = observation()
    screenshot = image_bytes()
    record = annotation(original, screenshot)
    reader = AnnotatedCaptureReader([record])
    captured = reader.read(screenshot)
    assert captured.observation == original
    with pytest.raises(PerceptionError):
        reader.read(image_bytes("red"))
    source = tmp_path / "capture.json"
    image = tmp_path / "capture.png"
    source.write_text(json.dumps(record))
    image.write_bytes(screenshot)
    assert read_capture(source, image).observation == original
    record["verified"] = False
    with pytest.raises(PerceptionError):
        AnnotatedCaptureReader([record])


class FakeTransport:
    def __init__(self):
        self.clicked = []

    def screenshot(self):
        return b"after" if self.clicked else b"before"

    def click(self, x, y):
        self.clicked.append((x, y))


class FakeReader:
    def __init__(self, before, after, *, confidence=1, target_mode=None):
        self.before, self.after = before, after
        self.confidence = confidence
        self.target_mode = target_mode

    def read(self, screenshot):
        return CaptureObservation(
            self.before if screenshot == b"before" else self.after,
            confidence=self.confidence,
            stable=True,
            verified=True,
            target_mode=self.target_mode,
        )


def adapter(before, after, **kwargs):
    transport = FakeTransport()
    browser = BrowserAdapter(
        calibration=calibration(),
        transport=transport,
        reader=FakeReader(before, after),
        controls={Action("ROLL"): (500, 600)},
        **kwargs,
    )
    return browser, transport


@pytest.mark.parametrize("kwargs", [{}, {"enabled": True}, {"authorized": True}])
def test_browser_requires_both_explicit_enablement_and_authorization(kwargs):
    before = observation()
    browser, transport = adapter(before, before, **kwargs)
    with pytest.raises(PermissionError):
        browser.execute(before, Action("ROLL"))
    assert transport.clicked == []


def test_browser_executes_once_and_returns_only_verified_refreshed_state():
    before = observation()
    after = replace(before, context=replace(before.context, version=8, has_rolled=True))
    browser, transport = adapter(before, after, enabled=True, authorized=True)
    assert browser.execute(before, Action("ROLL")) == after
    assert transport.clicked == [(500, 600)]
    assert before.hand == (1, 1, 1, 0, 0)


def test_stale_and_low_confidence_observations_prevent_clicks():
    before = observation()
    browser, transport = adapter(before, before, enabled=True, authorized=True)
    with pytest.raises(PerceptionError):
        browser.execute(replace(before, context=replace(before.context, version=6)), Action("ROLL"))
    browser.reader = FakeReader(before, before, confidence=0.1)
    with pytest.raises(PerceptionError):
        browser.execute(before, Action("ROLL"))
    assert transport.clicked == []


def test_ambiguous_refresh_locks_adapter_after_one_click():
    before = observation()
    browser, transport = adapter(before, before, enabled=True, authorized=True)
    with pytest.raises(PerceptionError):
        browser.execute(before, Action("ROLL"))
    assert browser.needs_review
    with pytest.raises(PerceptionError):
        browser.execute(before, Action("ROLL"))
    assert len(transport.clicked) == 1


def test_board_target_execution_requires_verified_tool_mode():
    action = Action("BUILD_SETTLEMENT", target=2)
    before = observation(action=action)
    after = replace(
        before,
        context=replace(before.context, version=8),
        buildings=before.buildings + (Building(2, 0),),
    )
    transport = FakeTransport()
    browser = BrowserAdapter(
        enabled=True,
        authorized=True,
        calibration=calibration(),
        transport=transport,
        reader=FakeReader(before, after),
    )
    with pytest.raises(PerceptionError):
        browser.execute(before, action)
    assert transport.clicked == []
    browser.reader = FakeReader(before, after, target_mode="BUILD_SETTLEMENT")
    assert browser.execute(before, action) == after
    assert len(transport.clicked) == 1


def test_trade_accept_requires_live_offer_id_and_reported_commit():
    action = Action("ACCEPT_TRADE", offer_id="live")
    original = observation(action=action)
    offer = TradeOffer("live", 1, 0, (0, 0, 0, 1, 0), (1, 0, 0, 0, 0))
    before = replace(
        original,
        context=replace(original.context, kind="TRADE_RESPONSE", turn_owner=1, offer=offer),
    )
    after = replace(
        before,
        context=replace(before.context, kind="TURN", version=8, offer=None),
        hand=(0, 1, 1, 1, 0),
        events=before.events
        + (VisibleEvent(8, "DOMESTIC_TRADE", 0, 0, give=(1, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0)),),
    )
    transport = FakeTransport()
    browser = BrowserAdapter(
        enabled=True,
        authorized=True,
        calibration=calibration(),
        transport=transport,
        reader=FakeReader(before, after),
        controls={action: (3, 4)},
    )
    with pytest.raises(PerceptionError):
        browser.execute(before, replace(action, offer_id="old"))
    assert transport.clicked == []
    assert browser.execute(before, action) == after
    assert before.hand == (1, 1, 1, 0, 0)


def test_trade_commit_requires_actual_reported_event_even_with_a_custom_verifier():
    action = Action("ACCEPT_TRADE", offer_id="live")
    original = observation(action=action)
    trade = TradeOffer("live", 1, 0, (0, 0, 0, 1, 0), (1, 0, 0, 0, 0))
    before = replace(
        original,
        context=replace(original.context, kind="TRADE_RESPONSE", turn_owner=1, offer=trade),
    )
    unreported = replace(
        before,
        context=replace(before.context, kind="TURN", version=8, offer=None),
        hand=(0, 1, 1, 1, 0),
    )
    transport = FakeTransport()
    browser = BrowserAdapter(
        enabled=True,
        authorized=True,
        calibration=calibration(),
        transport=transport,
        reader=FakeReader(before, unreported),
        controls={action: (3, 4)},
        verify_transition=lambda before, action, after: True,
    )
    with pytest.raises(PerceptionError):
        browser.execute(before, action)
    assert browser.needs_review
    assert before.hand == (1, 1, 1, 0, 0)


def test_unstable_unknown_or_missing_geometry_observations_prevent_actions():
    before = observation()
    for board in [
        replace(before.board, tiles=()),
        replace(
            before.board,
            tiles=(replace(before.board.tiles[0], resource="UNKNOWN"),) + before.board.tiles[1:],
        ),
    ]:
        broken = replace(before, board=board)
        browser, transport = adapter(broken, broken, enabled=True, authorized=True)
        with pytest.raises(PerceptionError):
            browser.execute(broken, Action("ROLL"))
        assert transport.clicked == []


def test_trade_reject_uses_exact_control_and_requires_reported_rejection():
    action = Action("REJECT_TRADE", offer_id="live")
    original = observation(action=action)
    trade = TradeOffer("live", 1, 0, (0, 0, 0, 1, 0), (1, 0, 0, 0, 0))
    before = replace(
        original,
        context=replace(original.context, kind="TRADE_RESPONSE", turn_owner=1, offer=trade),
    )
    after = replace(
        before,
        context=replace(before.context, kind="TURN", version=8, offer=None),
        events=before.events + (VisibleEvent(8, "REJECT_TRADE", 0),),
    )
    transport = FakeTransport()
    browser = BrowserAdapter(
        enabled=True,
        authorized=True,
        calibration=calibration(),
        transport=transport,
        reader=FakeReader(before, after),
        controls={action: (3, 4)},
    )
    assert browser.execute(before, action) == after
    assert transport.clicked == [(3.0, 4.0)]


def test_conflicting_annotations_and_high_residual_calibration_are_rejected():
    screenshot = image_bytes()
    first = annotation(observation(), screenshot)
    second = dict(first, confidence=0.3)
    with pytest.raises(PerceptionError):
        AnnotatedCaptureReader([first, second])
    with pytest.raises(PerceptionError):
        ScreenCalibration.fit(
            [((0, 0), (0, 0)), ((1, 0), (10, 0)), ((0, 1), (0, 10)), ((1, 1), (99, 99))],
            max_error_pixels=1,
        )


def test_actual_simulator_observation_is_a_supported_offline_export():
    from colonist_rl.simulator import FourPlayerGame

    game = FourPlayerGame()
    original = game.reset(seed=4)
    exported = observation_to_dict(original)
    assert view_from_dict(json.loads(json.dumps(exported))) == original
    assert "game" not in exported["observation"]
    assert "opponent_hands" not in exported["observation"]


def test_resource_action_choices_remain_immutable_after_json_roundtrip():
    action = Action("MARITIME_TRADE", give=(4, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0))
    original = observation(action=action)
    restored = view_from_dict(json.loads(json.dumps(observation_to_dict(original))))
    assert restored.legal_actions == (action,)
    assert isinstance(restored.legal_actions[0].give, tuple)


def target_transition(kind):
    target = 0 if kind == "BUILD_CITY" else (1 if kind == "MOVE_ROBBER" else 2)
    action = Action(kind, target=target)
    before = observation(action=action)
    after_changes = {}
    if kind == "BUILD_SETTLEMENT":
        after_changes["buildings"] = before.buildings + (Building(target, 0),)
    elif kind == "BUILD_CITY":
        before = replace(before, buildings=(Building(target, 0),))
        after_changes["buildings"] = (Building(target, 0, 2),)
    elif kind == "BUILD_ROAD":
        after_changes["roads"] = before.roads + (Road(target, 0),)
    else:
        after_changes["robber_tile"] = target
    after = replace(before, context=replace(before.context, version=8), **after_changes)
    return action, before, after


@pytest.mark.parametrize("kind", ["BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "MOVE_ROBBER"])
@pytest.mark.parametrize("mode", [None, "WRONG_MODE"])
def test_explicit_board_target_controls_still_require_matching_verified_mode(kind, mode):
    action, before, after = target_transition(kind)
    transport = FakeTransport()
    browser = BrowserAdapter(
        enabled=True,
        authorized=True,
        calibration=calibration(),
        transport=transport,
        reader=FakeReader(before, after, target_mode=mode),
        controls={action: (2, 3)},
    )
    with pytest.raises(PerceptionError):
        browser.execute(before, action)
    assert transport.clicked == []


@pytest.mark.parametrize("kind", ["BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "MOVE_ROBBER"])
def test_explicit_board_target_controls_execute_with_matching_verified_mode(kind):
    action, before, after = target_transition(kind)
    transport = FakeTransport()
    browser = BrowserAdapter(
        enabled=True,
        authorized=True,
        calibration=calibration(),
        transport=transport,
        reader=FakeReader(before, after, target_mode=kind),
        controls={action: (2, 3)},
    )
    assert browser.execute(before, action) == after
    assert transport.clicked == [(2.0, 3.0)]
