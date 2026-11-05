# Offline captures and guarded browser integration

The working offline path reads complete, explicitly annotated observations and
structured visible-event exports. It preserves full four-player or six-player
board topology and rules, and fits board lattice coordinates to screenshot
pixels. The browser boundary is dependency-injected, disabled by default, and
requires a separate explicit authorization flag. This implementation has not
accessed or verified a live Colonist session.

Automatic terrain/number recognition across Colonist themes, zoom levels,
animations, obscured tokens, and board variants remains pending. The current
screenshot reader is an honest annotation registry: an image must have a matching
user-verified annotation and SHA-256 hash. Unknown images raise an error. It does
not label terrain or numbers from arbitrary screenshots or infer private hands.

## Versioned observation JSON

```python
import json
from colonist_rl.perception import observation_to_dict, view_from_dict

payload = observation_to_dict(observation)
with open("observation.json", "w") as output:
    json.dump(payload, output, indent=2)
with open("observation.json") as source:
    restored = view_from_dict(json.load(source))
assert restored == observation
```

`observation_from_dict` is an alias for `view_from_dict`. The envelope is:

```json
{
  "format": "colonist-rl.observation",
  "version": 1,
  "observation": {"schema_version": 1, "...": "all Observation fields"}
}
```

The observation object uses the exact dataclass field names from `contracts.py`.
Its `board` includes tiles, integer lattice vertices, edges, tile-to-vertex and
tile-to-edge incidence, ports, and topology version. Its `rules` includes the
player count, bank, development deck, supported phases, variant, and rule version.
Context, pending offer, legal actions, public players, own resources/development
cards, buildings, roads, robber, beliefs, and visible events are all preserved.
The optional `bank` can be `null` when not visible. JSON arrays become immutable
tuples, including nested topology and belief arrays. Unsupported schema versions,
unknown contract fields, hidden-hand fields, invalid vectors, and incomplete or
unknown board geometry are rejected.

## Structured visible-event exports

The input is a documented local format, not an undocumented Colonist API. Each
row must explicitly declare its visibility. Resource identity and resource
vectors have their own visibility, allowing a public steal/discard count without
revealing its resource to an uninvolved observer.

```python
from colonist_rl.perception import visible_events_from_export

export = {
    "format": "colonist-rl.visible-events",
    "version": 1,
    "events": [
        {"sequence": 12, "kind": "STEAL", "actor": 0, "recipient": 1,
         "resource": "WOOD", "count": 1,
         "visibility": "public", "resource_visibility": [0, 1]},
        {"sequence": 13, "kind": "BUILD_ROAD", "actor": 2,
         "give": [1, 1, 0, 0, 0], "visibility": "public",
         "resource_visibility": "public"},
    ],
}
events = visible_events_from_export(export, observer_id=2)
assert events[0].resource is None
```

`visibility` and `resource_visibility` are either `"public"` or a list of player
IDs. A row invisible to the observer is omitted. Invisible resource information
is normalized to `resource=None` and zero `give`/`receive` vectors; its public
count remains. Resource visibility is required when any resource identity/vector
field is supplied. Event sequences must be nondecreasing; several resource
changes may share one sequence. The result is a tuple of immutable `VisibleEvent`
records. Callers apply only newly reported events to their resource-belief ledger;
a click or an intended action is not evidence of a resource transfer.

## Annotated screenshots

An annotation binds the full observation export to the original image bytes:

```python
import hashlib
from colonist_rl.perception import AnnotatedCaptureReader, observation_to_dict

annotation = {
    "format": "colonist-rl.capture", "version": 1,
    "verified": True,          # Set only after checking the annotation.
    "stable": True,            # No animation, obscured board, or pending UI change.
    "confidence": 1.0,
    "screenshot_sha256": hashlib.sha256(screenshot_bytes).hexdigest(),
    "observation": observation_to_dict(user_verified_observation),
    "target_mode": None,       # e.g. BUILD_ROAD only when that UI tool is selected.
}
reader = AnnotatedCaptureReader([annotation])
capture = reader.read(screenshot_bytes)
```

The reader verifies that the bytes decode as an image and match a registered
SHA-256. Conflicting annotations for one image are rejected. A reader can hold
multiple separately verified captures for an offline sequence. The convenience
function `read_capture(annotation_path, screenshot_path)` loads JSON and image
files. Annotation is the source of terrain, number tokens, ports, player-visible
state, and UI mode; image decoding alone does not establish their correctness.

## Calibration and action coordinates

```python
from colonist_rl.perception import ScreenCalibration

calibration = ScreenCalibration.fit([
    ((0, 0), (100, 200)),  # ((lattice x, lattice y), (pixel x, pixel y))
    ((1, 0), (120, 203)),
    ((0, 1), (104, 230)),
    ((1, 1), (124, 233)),
], confidence=1.0, max_error_pixels=2.0)
point = calibration.action_point(observation.board, action)
```

At least three noncollinear anchors are required. Least-squares fitting supports
translation, scale, rotation, and skew; degenerate transforms or excessive anchor
residuals are rejected. The transform maps settlements/cities to vertices, roads
to edge midpoints, and robber moves to tile centers. It uses the supplied board's
incidence lists, so board size and IDs are data. Missing targets and low-confidence
calibrations raise errors. A different viewport or zoom requires recalibration.
The same viewport must be used for the supplied action controls.

## Authorized transport boundary

A transport implements `screenshot() -> bytes` and `click(x, y) -> None`. A reader
implements `read(screenshot) -> CaptureObservation`. Neither is created by this
module; callers inject implementations for their explicitly authorized test or
browser environment.

```python
from colonist_rl.contracts import Action
from colonist_rl.perception import BrowserAdapter

adapter = BrowserAdapter(
    enabled=True, authorized=True,
    calibration=calibration, transport=authorized_transport, reader=reader,
    controls={Action("ROLL"): (800, 600)},
)
after = adapter.execute(observation, Action("ROLL"))
```

Both flags default to `False`. Before clicking, the adapter reads a current,
verified, stable, confident observation and compares the complete observation to
the caller's snapshot, including its version and pending offer. The action must
be legal for that actor. Controls are keyed by the complete typed `Action`, so a
trade response control includes the live offer ID, and resource actions include
the chosen resources. Every board target action requires a verified matching
`target_mode`, including actions with explicitly mapped controls; this prevents
clicking a board location while the wrong build
tool is selected. Robber actions involving victim selection require an explicit
control. No non-target action gets a guessed coordinate.

After one click, a refreshed observation must have a greater version and an
expected action-specific state change. Built-in checks cover settlement/city/road
placement, robber location, rolling, ending a turn, discarding, and trade response.
Trade acceptance requires the exact own-hand change plus a reported
`DOMESTIC_TRADE` event with matching actor-perspective vectors. Rejection requires
a reported rejection. The adapter never predicts or mutates resource ledgers.
Only its returned, verified observation/events should be ingested.

Other actions require an explicit `verify_transition(before, action, after)`
callback and an exact calibrated control. For built-in actions a custom callback
adds a check; it cannot bypass the built-in transition verification. An ambiguous
post-click capture or transport error leaves `needs_review=True` and prevents
retries through that adapter. Review the actual UI/state before constructing a
new adapter. Browser controls, robust ongoing screenshot recognition, and live
latency/animation behavior still require environment-specific implementation and
authorized end-to-end validation.

## Verification

The test suite uses generated image bytes and fake transports; it performs no
network or browser access. It covers full standard/extended board roundtrips,
immutable nested records, pending offers, unknown/malformed input rejection,
privacy projection, affine fitting and target geometry, verified image identity,
confidence and mode guards, disabled/unauthorized execution, stale observations,
one successful click with a verified refresh, ambiguous-result lockout, and
reported trade acceptance/rejection.
