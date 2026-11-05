# Bounded staged Colonist execution

`colonist_rl.adapters.colonist.staged_execution.StagedBrowserAdapter` is an
opt-in executor for three placement actions and incoming trade willingness.
The caller injects a transport, calibrated board geometry, and a reader of
complete verified screenshot evidence. It does not recognize screenshots or
start a browser on import. The [synthetic fixture catalog](../tests/fixtures/colonist-staged-execution.json)
models the [saved controls](colonist-controls.md); it contains independent
snapshots, not captured transitions or a live replay trace.

## Public boundary and entry conditions

```python
LocatedUIControl(control: UIControl, point: Point)
StagedCapture(capture: CaptureObservation, ui: UIState,
              controls: tuple[LocatedUIControl, ...] = ())
StagedCaptureReader.read(screenshot: bytes) -> StagedCapture
DragBrowserTransport(BrowserTransport, Protocol)
    .drag(start_x, start_y, end_x, end_y, *, steps=10) -> None

StagedBrowserAdapter(*, calibration, transport, reader,
                     enabled=False, authorized=False, min_confidence=0.95,
                     max_captures=8, poll_interval=0.05, wait=time.sleep)
adapter.execute(observation, action, *, expected_revision, row_id=None)
```

The records are frozen. Coordinate and control sequence inputs are detached
into tuples. Coordinates must be finite, nonnegative real numbers, excluding
booleans. Located control identities must be unique, and each control revision
must equal its UI revision. The UI observation version must equal the supplied
game observation version. A UI revision covers the complete semantic payload:
Observation, UIState, target mode, and located controls. A changed coordinate
under the same revision is rejected. Screenshot bytes, digest, confidence, and
stability may change while that semantic payload stays equal.

The reader's verified metadata vouches for **all** game, UI, and coordinate
evidence in the complete screenshot. If it supplies a screenshot SHA-256, the
executor checks the exact transport bytes. An absent digest retains the existing
injected-reader trust contract; no independent coordinate recognition is
claimed. Control IDs alone do not authorize input: a click resolves by equality
of the complete validated UIControl in the current StagedCapture.

Both explicit enablement and authorization are required. Execution begins with
a verified, stable, sufficiently confident capture exactly equal to the caller's
full Observation and expected UI revision. The board and calibration must be
valid, the action must be currently legal, and the observed player must be the
actor. UI phase must match the game decision context. Invalid argument types
or malformed Action payloads raise TypeError/ValueError before transport I/O;
permission absence raises PermissionError. Well-formed unsupported shapes and
safe preflight failures produce a blocked result without a gesture.

## Placement

Canonical `BUILD_SETTLEMENT`, `BUILD_CITY`, and `BUILD_ROAD` actions contain only
their kind and exact target. The ordinary placement path requires both the
capture's target mode and UI's active tool to equal that kind. Settlement/city
can also enter through the focused activation path below. An unrelated
selection, dialog, or confirmation blocks entry. Target selection may initially
be absent in active mode.
The executor-local compatible placement dialog labels are None and
`confirmation`; a labelled confirmation still requires the exact reviewed
action/control binding. Other labels, including `target_confirmation` used by
the unsupported robber path, are outside this executor's vocabulary.

The executor clicks the calibrated target once, then reads again. Direct visible
commitment succeeds with one click. A strictly newer UI revision with the entire
game Observation unchanged can establish an exact selected action and current
confirmation. The reviewed `validate_confirmation` binds that confirmation; its
located control supplies the second click. A pre-existing exact confirmation
can instead be consumed directly, using one click. There are at most two clicks.

Commitment requires a stable verified capture, a newer UI revision, a strictly
later game version, the same player/board/rules, and the existing compatibility
executor's action-specific visible board effect. The result contains the
refreshed Observation and evidence naming kind, target, actor, and versions.
It never predicts a resource deduction. A higher game version, disappearing
dialog, or another selected target cannot establish commitment.
Before accepting the board effect, the executor also checks terminal UI
compatibility: both target mode and active tool must still equal the action
kind, or both must be cleared to None with selection and confirmation cleared.
An absent confirmation requires no dialog, and unrelated dialogs or trade rows
remain incompatible. A matching board effect cannot override contradictory UI
evidence or clear the review lock. Verified unstable completion captures may be
polled only while these same compatibility checks hold.

## Already-focused settlement/city activation

When both target mode and active tool are None, canonical `BUILD_SETTLEMENT`
and `BUILD_CITY` may first use one activation drag. The selected action must
already equal the complete requested action, with no dialog, confirmation, or
trade rows. Exactly one located drag control must bind that action at the
current revision, have no row/partner, and match `ui.focused_control`. Its
verified point supplies the drag start; the calibrated target supplies the end.
An unfocused control or tooltip cannot establish this entry condition. No focus
click is issued. Other active tools or inconsistent mode/tool evidence block.

The transport must provide a callable `drag` method. This additive capability
is described by `DragBrowserTransport`; the original BrowserTransport remains
unchanged, and active-mode placement/incoming paths still accept transports
that only implement click/screenshot.

After the single drag, the executor polls within the same capture budget until
a stable verified capture at a strictly newer UI revision establishes both
mode and tool equal to the action kind, the same exact selected action, no
dialog/confirmation/trade rows, and an exactly unchanged complete Observation.
Only then does it issue the ordinary target click and, when needed, the exact
confirmation click. Changed game state, another selection/tool, an unexpected
confirmation/dialog, or inconsistent revision evidence is immediately ambiguous,
including unstable contradictory captures. Benign unchanged/unstable captures
may be polled. A drag with no established mode times out with the review lock
retained, without repeating the drag or issuing target input. There are at most
one drag and two placement clicks; every gesture reserves a remaining capture.
Road drag activation is unsupported and requires already active road mode.

## Incoming willingness

Canonical `Action('ACCEPT_TRADE', offer_id=...)` requires TRADE_RESPONSE context,
the exact shared offer and recipient, sufficient known own resources through
`validate_trade_offer`, an explicit row_id, and an already selected exact action.
`validate_trade_response` binds the current row/control, proposer, terms,
revision, and acting recipient. Multiple simultaneous rows are allowed; only
the exact requested row is clicked. An existing own willing or rejected marker
blocks without duplicate input.

One response click can produce `pending` only after a strictly newer UI revision,
exactly unchanged complete Observation, the same row/offer/proposer/terms at the
new revision, and that recipient's visible willing marker. The executor calls
`validate_acknowledgement` and attaches marker evidence. Pending carries no
committed Observation and certifies no transfer. A changed hand, disappearing or
replaced row, other partner's selection, or cancellation is ambiguous.
A verified own rejected marker or cancelled/disappearing offer is ambiguous
immediately, even when the frame is unstable; no later willing frame is read to
reverse that outcome. A benign unstable render can still be polled. The allowed
post-response dialog labels are None and `willing`, with exact row/marker
evidence still required.

## Capture bound and review lock

`max_captures` is a positive nonboolean integer counting the initial screenshot
and every subsequent read across all stages. At least one capture must remain
before any input. Unchanged evidence and verified confident unstable renders
may be polled; `wait(poll_interval)` separates those polls. The interval is
finite in [0, 1]. No gesture is retried. The count bound does not promise a hard
wall-clock timeout for synchronous transport, reader, or wait calls.

The adapter sets `needs_review` **before** input. Every ordinary exception or
unexpected post-input outcome returns ambiguous with a reason and retains the
lock. BaseException interruptions, including KeyboardInterrupt, propagate with
the lock preserved. An already locked instance blocks without transport I/O.
Only verified committed placement or pending willingness clears the lock. There
is no reset/retry API; an uncertain action requires external manual review.

## Compatibility and unsupported flows

`BrowserAdapter.execute`, BrowserTransport, legacy imports, schema version 1,
shared actions, policy, simulator atomic trades, CLI, and training are unchanged.
This separate executor does not update a resource/belief ledger or infer history.

Focus acquisition and road drag activation remain unsupported. Keyboard,
development-card/robber/discard sequences, bank and outgoing finalization,
incoming rejection/editing, later acceptance
completion/cancellation monitoring, automatic UI reading, stable live row IDs,
arbitrary photographs, and ledger/history integration remain unsupported.
