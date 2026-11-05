# Offline Colonist UI evidence

`colonist_rl.adapters.colonist.ui` holds immutable descriptions supplied by a
caller and pure binding validators. UI metadata stays outside the shared
`Observation`, whose schema remains version 1. The module does not capture or
recognize screenshots, open a browser, issue gestures, contact services, or
update resource or belief ledgers.

## Public records

All five records are frozen dataclasses. Revisions, game versions, and player
IDs are nonnegative integers; booleans are rejected. Identities are nonempty
strings. Resources use `(WOOD, BRICK, SHEEP, WHEAT, ORE)` order and five
nonnegative integer counts. Valid sequence inputs are copied to immutable
tuples; mutable nested payloads are rejected.

| Record | Constructor fields |
| --- | --- |
| `UIControl` | `control_id: str`, `revision: int`, `action: Action`, `gesture: Literal['click', 'drag']`, `row_id: str \| None = None`, `partner: int \| None = None` |
| `UITradeResponse` | `player_id: int`, `status: Literal['willing', 'rejected']` |
| `UITradeRow` | `row_id: str`, `offer_id: str`, `proposer: int`, `give: ResourceVector`, `receive: ResourceVector`, `revision: int`, `responses: tuple[UITradeResponse, ...] = ()`, `controls: tuple[UIControl, ...] = ()` |
| `UIState` | `revision: int`, `observation_version: int`, `phase: str`, `focused_control: str \| None = None`, `active_tool: str \| None = None`, `selected_action: Action \| None = None`, `dialog_stage: str \| None = None`, `confirmation: UIControl \| None = None`, `trade_rows: tuple[UITradeRow, ...] = ()` |
| `UIExecutionResult` | `status: Literal['committed', 'pending', 'blocked', 'ambiguous']`, `action: Action`, `starting_version: int`, `ui_revision: int`, `observation: Observation \| None = None`, `evidence: tuple[str, ...] = ()`, `reason: str \| None = None` |

`row_id` is a caller-supplied visible row identity independent of `offer_id`.
Do not derive it from screen position, resource terms, or hashes. Multiple rows
may coexist. Duplicate row identities, duplicate player markers within a row,
and ambiguous duplicate control bindings are rejected. Row terms always use the
proposer's perspective. Phase, tool, focus, and dialog names are descriptive
strings supplied by the caller; they do not establish that an action committed.

## Exact bindings

`validate_confirmation(state, action, *, expected_revision)` returns the exact
`UIControl` stored in `state.confirmation`. Both UI and control revision must
match. The selected action and control action must equal the complete supplied
`Action`, including target, resource choices, partner, offer identity and terms.
A confirmation must be a click and have consistent partner metadata. Missing
required descriptor fields or stale/mismatched bindings raise `PerceptionError`.
Construction errors raise `TypeError` or `ValueError`.

Confirmation validation checks exact equality of the caller-supplied descriptor
and its control, plus the module's minimum field requirements. It does not prove
that the descriptor represents a complete legal game action or that a live
interaction sequence is supported. In particular, PLAY_YEAR_OF_PLENTY requires
resource but does not require second_resource, and PLAY_ROAD_BUILDING has no
required road targets in this check. These accepted evidence shapes do not
certify either card's unverified own-card controls. A future executor must
establish the complete legal action and verified card-specific sequence before
issuing gestures.

```python
from colonist_rl.contracts import Action
from colonist_rl.adapters.colonist.ui import UIControl, UIState, validate_confirmation

action = Action('BUILD_SETTLEMENT', target=4)
confirm = UIControl('floating-house', 8, action, 'click')
ui = UIState(8, 12, 'SETUP_SETTLEMENT', selected_action=action, confirmation=confirm)
assert validate_confirmation(ui, action, expected_revision=8) is confirm
# Action('BUILD_SETTLEMENT', target=99) is rejected, even with the same kind.
```

`validate_trade_response(state, action, offer, *, row_id, expected_revision,
player_id)` returns the exact response `UIControl`. It accepts canonical shared
`Action('ACCEPT_TRADE', offer_id=...)` or `REJECT_TRADE` actions. The shared
`TradeOffer` supplies the exact proposer, acting recipient, and proposer-side
terms; all must match the current row. The control's own `row_id`, revision, and
`partner` must match that row and acting recipient. A replaced row or changed
terms under the same offer ID cannot reuse an old response.

`validate_trade_finalization(state, action, *, row_id, expected_revision,
player_id)` returns the exact willing partner's `UIControl`. The UI-only action
convention is:

```python
action = Action('FINALIZE_TRADE', partner=1, offer_id='offer-a',
                give=(0, 0, 0, 0, 1), receive=(0, 0, 1, 0, 0))
```

This convention does not add a policy action or change the shared `Action`
record. It cannot be submitted as a policy decision and is absent from policy
`KINDS`, shared legal actions, simulator, codecs, and schemas. The acting player
must be the row's proposer; terms must equal that current row; both sides must be nonempty. The chosen other player must have one
`willing` marker. The exact control must bind that action, row, revision and
partner. Incoming acceptance and outgoing finalization are distinct bindings.
A willing marker never establishes a resource transfer.

## Acknowledgement and results

`validate_acknowledgement(before_ui, after_ui, before_observation,
after_observation, action)` requires a strictly newer UI revision, exactly equal
full before/after game `Observation` values, both stored observation versions
matching their observations, and the exact action selected afterward. The
previous UI may have no selection: target selection itself can be UI progress.
The function returns `UIExecutionResult('pending', ...)` with the starting game
version and latest UI revision. It neither issues a gesture nor infers completion.
An unchanged UI after a failed drag fails this check. An accepted incoming marker
with an unchanged hand remains pending.

A `committed` result requires an explicitly supplied immutable `Observation`
whose `context.version` exceeds `starting_version`, plus a nonempty tuple of
caller-supplied evidence strings. This is an evidence record; the constructor
cannot independently verify that the supplied evidence proves the action.
`pending` carries no committed observation. `blocked` and `ambiguous` retain a
nonempty reason and also carry no committed observation. No result updates beliefs.

## Synthetic evidence catalog and integration boundary

[The scenario catalog](../tests/fixtures/colonist-ui-stages.json) is explicitly
labelled `provenance: synthetic`. Its independent before/after stage records
model setup preview/confirmation, tooltip/focus, failed drag, Knight confirmation,
robber target/victim stages, Monopoly choices, incoming willingness, and
simultaneous trade rows. `basis: controls_notes` identifies a sequence modeled
from the [controls walkthrough](colonist-controls.md); `basis: observed_image`
identifies stage contents illustrated by its referenced saved image. Those
images are snapshots, not captured transitions. The catalog does not provide
screenshot recognition, captured DOM row identities, or a live replay trace.

A6/A7 capture/execution wiring remains unsupported: these records are not
connected to the capture reader, browser transport, bounded waiting, gesture
sequencing, or event history reconciliation. Control IDs and UI revisions must
be supplied by a future evidence reader. Unsupported Year of Plenty/Road Building
interaction sequences, incoming-offer edits, and unobserved forced decisions are
not certified by these records.

`BrowserAdapter.execute` in
`src/colonist_rl/adapters/colonist/execution.py` remains the unchanged one-click
atomic compatibility path. It immediately requires a newer game observation
with an action-specific committed change. Its current acceptance verifier still
requires an exact immediate hand change and reported `DOMESTIC_TRADE` event.
The simulator's atomic trade behavior and legacy imports are unchanged.
