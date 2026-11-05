# Colonist Adapter Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide reviewable pure UI safety contracts and conservative visible-log semantic candidates.

**Architecture:** Two independent leaf modules consume existing shared contracts or normalized log evidence. No execution, policy, simulator, or ledger behavior changes. Each task has a fresh implementer and independent task review; the complete branch receives a final review.

**Tech Stack:** Python 3.12+, dataclasses, pytest, Ruff, existing contracts and normalized log rows.

**Spec:** `docs/superpowers/specs/2026-11-04-adapter-evidence-design.md`

## Global Constraints

- Work only in `/private/tmp/colonist-swarm-bxz1k39_/adapter`; original repository is read-only.
- Preserve shared `Observation`, `Action`, simulator atomic trade behavior, legacy aliases, and browser origin protections.
- Read only supplied player-visible evidence; no simulator objects, hidden DOM fields, storage, private APIs, inferred opponent identities, or invented history.
- Issue no UI gestures, start no browser, contact no external service, install no dependencies, and never update a resource or belief ledger.
- New runtime files belong only under `src/colonist_rl/adapters/colonist/`; new tests and adapter documents may accompany them.
- Use original `.venv/bin/python` and `.venv/bin/ruff` with workspace `PYTHONPATH=src` and `PYTHONDONTWRITEBYTECODE=1`; never run uv sync.

---

### Task 1: Immutable UI records and exact pure binding validators

**Ownership:** Create only `src/colonist_rl/adapters/colonist/ui.py`, `tests/test_colonist_ui.py`, `tests/fixtures/colonist-ui-stages.json`, `docs/colonist-ui.md`. Do not edit existing source/test files. Read the spec's UI evidence contracts and Global Constraints plus `docs/colonist-controls.md`, contracts, and current execution for context. No worker subagents.

**Interfaces:** Consume `Action`, `Observation`, `TradeOffer`, `ResourceVector` and `PerceptionError`. Produce frozen `UIControl`, `UITradeResponse`, `UITradeRow`, `UIState`, `UIExecutionResult`; pure `validate_confirmation(state, action, *, expected_revision)`, `validate_trade_response(state, action, offer, *, row_id, expected_revision, player_id)`, `validate_trade_finalization(state, action, *, row_id, expected_revision, player_id)`, `validate_acknowledgement(before_ui, after_ui, before_observation, after_observation, action)`. Return exact validated control/row from the binding validators and a pending UIExecutionResult from acknowledgement. Spell out constructor field names/types in module/docstrings. No runtime dependency on execution.py or browser.py is needed.

- [ ] Write failing tests using independent before/after fixtures for exact target confirmation; stale revision; changed action resource/target; current row with another offer ID; same offer ID with changed terms/proposer; wrong finalization partner; duplicate row identities; duplicate partner markers; accepted marker with unchanged hand; identical/lower UI revision; changed Observation/version rejection; and detached immutable sequence fields. Use complete Action equality, not kind-only comparisons.

```python
from dataclasses import replace
from colonist_rl.contracts import Action

def test_confirmation_rejects_same_kind_other_target(confirmed_state):
    import pytest
    from colonist_rl.adapters.errors import PerceptionError
    from colonist_rl.adapters.colonist.ui import validate_confirmation
    state = confirmed_state
    with pytest.raises(PerceptionError):
        validate_confirmation(state, Action('BUILD_SETTLEMENT', target=99),
                              expected_revision=state.revision)
```

- [ ] Run the new test file to establish failing import/behavior, then implement records with constructor validation and pure functions. `UITradeRow` terms are proposer-perspective vectors with a row identity distinct from offer identity; responses and controls are immutable sequences. Matching responses must use the exact shared offer's proposer/terms and acting recipient. Finalization requires the current player to be proposer and the chosen partner to have a willing marker and exact row-bound control. A row's prior terms must not authorize new terms. A control's action and its own row/partner fields must agree. Unsupported or incomplete bindings fail closed.

```python
# Required invariant pattern inside a confirmation validator:
if state.revision != expected_revision or control.revision != state.revision:
    raise PerceptionError('stale UI revision')
if state.selected_action != action or control.action != action:
    raise PerceptionError('confirmation does not match the complete action')
```

- [ ] Include result construction tests for all four statuses. `committed` requires a nonempty evidence tuple and an Observation whose version is greater than the recorded starting game version; `pending` cannot carry committed Observation; blocked/ambiguous retain reason. Acknowledgement returns pending only on strictly newer UI revision and exactly unchanged full Observation, with before/after observation versions matching it and after selected_action equal to action. This validates evidence; it does not issue a gesture or infer completion.
- [ ] Add a compact JSON scenario catalog with `provenance: synthetic`, named stage records and references to the controls/screenshot evidence. Load it in tests and assert fixture-driven valid/rejected bindings for setup preview/confirmation, tooltip/focus, failed drag, Knight card, robber target/victim, Monopoly choice, incoming willingness, simultaneous rows. Explicitly distinguish scenarios modeled from notes from observed images. No screenshot recognition is claimed.
- [ ] Document public records and examples, invariants, unsupported A6/A7 wiring, and `BrowserAdapter.execute` as the unchanged one-click atomic compatibility path.
- [ ] Verify new tests plus existing compatibility suites and lint/format on owned files. Commands:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest tests/test_colonist_ui.py tests/test_adapters.py tests/test_perception.py -q
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check src/colonist_rl/adapters/colonist/ui.py tests/test_colonist_ui.py
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff format --check src/colonist_rl/adapters/colonist/ui.py tests/test_colonist_ui.py
```

- [ ] Self-review the diff, commit only owned files, and write full report to the assigned task report path including commands, results, commit hashes, concerns. Return status/commits/one-line tests/concerns only.

### Task 2: Conservative visible resource and roll projection

**Ownership:** Create only `src/colonist_rl/adapters/colonist/semantics.py`, `tests/test_colonist_semantics.py`, `tests/fixtures/colonist-semantic-synthetic.json`, `docs/colonist-log-semantics.md`. Do not edit existing files. Read spec semantic projection and Global Constraints, existing logs.py, both real fixtures and controls/log-reader docs. No worker subagents.

**Interfaces:** Consume `LogRow`, `LogSnapshot`, `ResourceVector`. Produce frozen `VisiblePlayerName(name: str, player_id: int)`, `ResourceEvidence(known: ResourceVector, unknown_count: int)`, `SemanticCandidate(kind, actor_name, actor, give, receive, dice)`, `RowProjection(source: LogRow, candidate: SemanticCandidate | None, issues: tuple[str,...])`, and `project_row(row: LogRow, players: Sequence[VisiblePlayerName]) -> RowProjection` plus `project_snapshot(snapshot: LogSnapshot, players: Sequence[VisiblePlayerName]) -> tuple[RowProjection,...]`. Candidate kinds `production`, `bank_trade`, `roll`; actor ID may be None; dice is an immutable pair or empty tuple. Candidate contents are evidence, never ledger-ready events.

- [ ] Write failing tests for all five resource icons and repeated icons; whitespace/split text; missing and duplicate visible names; cardback/unknown resource slots; extra text or icons; incomplete/three dice; duplicate identical rows; real fixture production/roll projection; actual offer row yielding no candidate; synthetic completed bank transfer; player trade/acceptance/build/unsupported evidence retained without transfers. No guessed opponent IDs.

```python
def test_real_offer_is_not_a_transfer():
    import json
    from pathlib import Path
    from colonist_rl.adapters.colonist.logs import normalize_snapshot
    from colonist_rl.adapters.colonist.semantics import project_snapshot
    raw = json.loads((Path(__file__).parent / 'fixtures/colonist-log-001.json').read_text())
    rows = project_snapshot(normalize_snapshot(**raw), ())
    assert rows[-1].candidate is None
    assert rows[-1].source.tokens == normalize_snapshot(**raw).rows[-1].tokens
```

- [ ] Run the new tests to establish failure, then implement exact full-row token grammars. Adjacent text fragments can be joined for grammar matching but source tokens remain untouched. Do not coerce a resource word in body text into an icon. Entire grammar must be consumed; unknown tails remain unparsed. Name parsing must not silently assign missing mappings or arbitrarily select among duplicate name bindings. Resource slots preserve known vectors and unknown card-back counts; unsupported icons produce issues and no false complete amount. Validate frozen record invariants and detach sequence inputs.

```python
# Grammar evidence, where each bracket is one supplied icon token:
# Eilis got [Ore] [Lumber] -> production; receive=(1,0,0,0,1)
# Donni#0773 gave bank [Lumber] [Lumber] [Lumber] [Lumber] and took [Grain]
# -> bank_trade; give=(4,0,0,0,0), receive=(0,0,0,1,0)
# Kati rolled [dice_1] [dice_2] -> roll; dice=(1,2)
# Eilis wants to give [Ore] for [Wool] -> no candidate
```

- [ ] Add explicitly synthetic resource/bank/unknown cases as ordered tokens. Preserve real DOM fixtures byte-for-byte and do not label the bank grammar's synthetic token fixture as a captured DOM sequence. Test both complete dice faces from actual fixtures. Confirm parser imports no browser, simulator, policy or ledger modules in a fresh subprocess.
- [ ] Document the grammar, API examples, unresolved actor/unknown resource issue codes, exact source retention, duplicate rows, and difference between semantic candidate, established event identity, and ledger application. Future historian and live collector integration remain outstanding.
- [ ] Verify owned tests and existing log tests, lint/format:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest tests/test_colonist_semantics.py tests/test_colonist_logs.py -q
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check src/colonist_rl/adapters/colonist/semantics.py tests/test_colonist_semantics.py
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff format --check src/colonist_rl/adapters/colonist/semantics.py tests/test_colonist_semantics.py
```

- [ ] Self-review the diff, commit only owned files, write full assigned report with tests/commands/results/commits/concerns and return the short status contract.
