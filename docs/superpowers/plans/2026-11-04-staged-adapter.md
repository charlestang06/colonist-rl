# Bounded Staged Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Execute verified placement stages and incoming willingness with bounded progress, then add only focused drag activation needed for settlement/city tool entry.

**Architecture:** A new small explicit executor consumes reviewed UI records plus injected verified capture/coordinates. Existing one-click execution remains unchanged. Task 2 adds a public transport method and one optional activation stage; no generic workflow engine.

**Tech Stack:** Python >=3.11, existing dataclasses/contracts, pytest, Ruff, optional public Playwright API.

**Spec:** `docs/superpowers/specs/2026-11-04-staged-adapter-design.md`

## Global Constraints

- Work only in `/private/tmp/colonist-swarm-bxz1k39_/adapter`; original repository is read-only. Baseline is `47a28d93ed1a9955771d4a740b718d152c932ffe`.
- Preserve Python >=3.11, shared Observation/Action/schema version 1, policy action kinds, simulator atomic trades, legacy module identities, and the existing BrowserAdapter.execute and BrowserTransport APIs.
- No live Colonist connection, screenshot/photo recognition, hidden state, private APIs, network installation, ledger mutation, semantic-history guessing, CLI/training changes, or development-card/robber sequence support.
- Use original `.venv/bin/python` and `.venv/bin/ruff` read-only with workspace `PYTHONPATH=src` and `PYTHONDONTWRITEBYTECODE=1`; no uv sync.
- All replay fixtures are explicitly synthetic models grounded in the saved controls; images are snapshots, not recorded transitions. Transport tests use an injected public Page/Mouse double or opt-in loopback HTML only.
- Every gesture is issued at most once for its established stage. Set the review lock before the gesture; preserve it after uncertain outcomes or interruption. Never blindly retry a gesture.
- Runtime/source ownership is limited to Colonist adapter modules plus their new tests and adapter docs; shared contracts, simulator, beliefs, policy, training, CLI, and board remain untouched.

---

### Task 1: Explicit placement and pending-acceptance executor

**Files / ownership:** Create `src/colonist_rl/adapters/colonist/staged_execution.py`, `tests/test_colonist_staged_execution.py`, `tests/fixtures/colonist-staged-execution.json`, `docs/colonist-staged-execution.md`. Modify only the integration-boundary paragraph in `docs/colonist-ui.md` to link the new limited executor. Do not edit existing runtime/test files, contracts, or training. No subagents.

**Interfaces:** Produce `LocatedUIControl`, `StagedCapture`, `StagedCaptureReader`, and `StagedBrowserAdapter` with exact signatures/invariants in the spec's Public boundary. Consume reviewed UI validators, CaptureObservation, ScreenCalibration, validate_trade_offer, and existing `_expected_transition` for three placement actions. Implement the spec's Task 1 paths and common failure/lock/budget rules. Task 2 will add optional focused drag activation to this same explicit placement flow. Keep helpers task-specific; do not add dispatch tables/state-machine frameworks/generic stage plugins.

- [x] Read spec Public boundary, Task 1 paths, acceptance, and global constraints; read ui.py/execution.py/capture.py/calibration.py, saved control notes, and current tests. Build a deterministic fake transport with queued frame bytes and an injected reader returning literal complete StagedCapture snapshots. Keep the fixture explicitly synthetic and name both successful and rejected stages.
- [x] Write failing behavior tests before implementation: exact setup target then current confirmation then board commit; direct placement with one click; existing confirmation with one click; delayed unchanged/unstable render without repeat click; wrong target/stale revision/wrong point binding; unchanged state timeout; exception and BaseException interruption after each gesture; same-version board change; malformed/low-confidence capture; permission/legal/action-shape failures; bounded capture count; already-locked no I/O; incoming exact willingness pending; changed/expired/wrong/simultaneous rows; already-willing no duplicate input. Use exact ordered call lists rather than broad any-call assertions.

```python
# Expected intent for fixture-driven tests, with builder details in the test module:
result = adapter.execute(before.observation, action, expected_revision=10)
assert transport.gestures == [('click', target_x, target_y), ('click', confirm_x, confirm_y)]
assert result.status == 'committed'
assert result.observation == after.observation
assert result.evidence
assert not adapter.needs_review
# A willingness scenario instead has one click, status pending, no result.observation,
# and before.observation exactly equal to the acknowledged capture's Observation.
```

- [x] Run new test file to establish expected RED. Implement the smallest explicit paths with the exact spec API and limits. Call existing reviewed validators at stage boundaries. Geometry effects use `_expected_transition`, not copied settlement/road/city logic. Set needs_review before transport input. All post-input unexpected outcomes return ambiguous with reason; KeyboardInterrupt propagates while lock remains. Initial blocked paths issue zero gestures. Record exact board/marker evidence in results and never predict hand deltas.
- [x] Add constructor/coordinate-detachment checks and import isolation. The fixture catalog must feed test snapshots and expected ordered gesture/result assertions, not merely list scenario names. Validate every declared fixture scenario or explicitly mark catalog-only context.
- [x] Document supported entry preconditions, capture trust/coordinate binding, budget versus wall-clock limitation, lock behavior, pending acceptance semantics, unchanged compatibility executor, and unsupported flows. Update only the prior UI document's wiring paragraph so it no longer claims all execution wiring is absent.
- [x] Run targeted and legacy compatibility suites plus Ruff:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_colonist_staged_execution.py tests/test_colonist_ui.py tests/test_adapters.py tests/test_perception.py
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check src/colonist_rl/adapters/colonist/staged_execution.py tests/test_colonist_staged_execution.py
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff format --check src/colonist_rl/adapters/colonist/staged_execution.py tests/test_colonist_staged_execution.py
```

- [x] Self-review, commit only owned files, and write assigned report with RED/GREEN commands/results, interfaces, exact file list, concerns, commit IDs. Return short status contract. Independent review must pass before Task 2.

### Task 2: Focused drag activation and guarded public transport

**Files / ownership:** Modify `src/colonist_rl/adapters/colonist/staged_execution.py`, `src/colonist_rl/adapters/colonist/browser.py`, `tests/test_colonist_staged_execution.py`, `tests/fixtures/colonist-staged-execution.json`, `docs/colonist-staged-execution.md`, and `docs/browser.md`. Create `tests/test_colonist_drag.py`. Do not modify execution.py, legacy facade modules, shared contracts, existing tests, CLI/training. No subagents.

**Interfaces:** Consume Task 1's exact records/executor. Add `DragBrowserTransport(BrowserTransport, Protocol)` with `drag(start_x, start_y, end_x, end_y, *, steps=10) -> None`. Add compatible method to PlaywrightTransport. Implement only the spec Task 2 already-focused settlement/city activation stage, then reuse Task 1 placement handling and same global capture budget. Existing BrowserTransport does not gain a required method; transports lacking drag remain valid for active-mode paths.

- [x] Read spec Task 2/Common boundaries, Task 1 implementation/report, and existing transport/public Mouse behavior. Write failing tests for exact move/down/move/up sequence; origin/connection check before and after each step; invalid coordinates/steps before input; mid-drag failure release only on original connected allowed origin; original exception preserved; no further move/drop on changed origin; old click/screenshot behavior intact.

```python
transport.drag(10, 20, 80, 90, steps=4)
assert page.mouse.calls == [
    ('move', 10.0, 20.0), ('down',),
    ('move', 80.0, 90.0, 4), ('up',),
]
# Fake Mouse records the public kwargs; expectations may use dicts consistently.
```

- [x] Write staged tests for focused exact drag -> verified active mode -> target -> confirmation/direct commitment; no-effect drag bounded timeout; tooltip/unfocused state blocked before drag; changed action, wrong/ambiguous control, missing transport method; delayed activation, unexpected confirmation/game change, exceptions/interruption, and capture budget shared across activation/placement. Assert one drag maximum and no target input before verified active mode. Add explicit synthetic fixture stages; no live screenshots required.
- [x] Run focused RED then implement transport method and optional activation path. Guard release on the original allowed origin and connection; suppress cleanup failure only to preserve an already-active original exception. Do not alter BrowserTransport or add keyboard/focus acquisition/general card workflow. Keep Board/Observation/Action and previous executor untouched.
- [x] Document the additive method, already-focused entry constraint, origin and timeout limitations, exact supported settlement/city activation, and remaining unsupported sequences. Existing docs' legacy one-click behavior remains accurate; add a link to the new opt-in executor.
- [x] Run focused plus existing browser/compatibility tests and Ruff; use optional loopback Chromium test only if already available, never install dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_colonist_staged_execution.py tests/test_colonist_drag.py tests/test_browser.py tests/test_adapters.py tests/test_perception.py
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check src/colonist_rl/adapters/colonist/staged_execution.py src/colonist_rl/adapters/colonist/browser.py tests/test_colonist_staged_execution.py tests/test_colonist_drag.py
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff format --check src/colonist_rl/adapters/colonist/staged_execution.py src/colonist_rl/adapters/colonist/browser.py tests/test_colonist_staged_execution.py tests/test_colonist_drag.py
```

- [x] Self-review, commit only owned files, and write assigned report with RED/GREEN commands/results, exact API/file list, provenance, concerns, commit IDs. Independent task review and broad final review follow; preserve all SDD artifacts for root.

## Completion status

Both tasks are complete. Task 1's independent review found two Important and one
Minor issue; fix round 1 resolved all three and passed scoped re-review at
`decd6d8`. Task 2 passed independent spec/quality review at `9171194`. The final
Astra whole-wave review approved `47a28d9..9171194` with no findings and confirmed
that the Task 1 fixes remain intact. This status/evidence update changes docs
only. Root integrated all runtime changes at `0f865be`.

Reported final task checks: Task 1, 278 passing tests; Task 2, 345 passing and one
existing opt-in Chromium skip; Ruff/format/whitespace checks clean. Root then
recorded a separate actual localhost Chromium transport smoke with
`completed=1 down=1 up=1 moves=10`; the final reviewer inspected its script, JSON,
and screenshot. This is local transport evidence, not live Colonist or full
staged-recognition validation. Runtime was Python 3.13; actual Python 3.11
execution remains unverified beyond syntax compatibility.

The original integration/work horizon was missed and remains recorded in the
root run's history; this closeout does not claim otherwise. No additional adapter
features or test runs were started after final approval. Root owns authorized
learning catch-up and final integration. Both adapter SDD workspaces, all reports,
briefs, packages, and ledger rulings are preserved for durable recovery.
