# Public Candidate Successor Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, public-only candidate successor representation that can be trained on broad student-reached decision data without changing the existing parent.

**Architecture:** A pure encoder returns one fixed-width row for each exact legal action. A strict consequence-policy subclass adds zero-start linear and nonlinear action-conditioned branches. Checkpoint metadata names the new feature schema explicitly.

**Tech Stack:** Python 3.13, PyTorch, Catanatron 3.3.0, pytest, Ruff.

**Spec:** [Public candidate successor encoder](../specs/2026-11-06-candidate-successor-encoder-design.md).

## Global Constraints

- Read only the shared immutable `Observation` and its original legal `Action` objects; no engine state or opponent private hand.
- Do not imply a future legal menu, development-card identity or robber steal. Encode immediate public road and army awards from known state.
- Keep current policy classes, checkpoints and default behavior exact.
- Use `hand_recipe` as the only upgrade parent and a distinct strict architecture identity.
- No fresh games or model-strength claim in this representation task.

---

### Task 1: Versioned public successor rows

**Files:** Create `src/colonist_rl/successor_features.py`; create `tests/test_successor_features.py`.

**Interfaces:** `SUCCESSOR_SCHEMA: str`, `SUCCESSOR_COLUMNS: tuple[str, ...]`, `SUCCESSOR_DIM: int`, `successor_feature_values(view: Observation, action: Action) -> dict[str, float]`, and `successor_features_for_menu(view: Observation, *, device="cpu") -> torch.Tensor`.

- [x] Write failing tests for exact action identity, deterministic hand changes, zero-cost free/setup road, development purchase's unknown draw, END, production/port effects of a new settlement or city, unknown-result masks, full-menu parity and observation immutability.
- [x] Run `PYTHONPATH=src .venv/bin/python -m pytest tests/test_successor_features.py -q` and observe missing API or incorrect behavior.
- [x] Implement the named schema. Scalars are hand-known/uncertainty masks, point and win effects, piece counts, before/after connected sites and city sites, before/after eligible point gaps, trade ratio and own VP. Five-resource blocks are before/after/delta hand, four recipe deficits, nominal/unblocked production gain and after-action port ratios. Normalize counts to bounded scales; unknown after-hand and dependent deficits are zero with their mask off.
- [x] Run focused tests and Ruff, refactor only while green, then commit the encoder/test.

### Task 2: Zero-start opt-in policy and strict checkpoint identity

**Files:** Modify `src/colonist_rl/policy.py`; create `tests/test_successor_policy.py`.

**Interfaces:** `CandidateSuccessorPolicy(ConsequenceGraphPolicy)`, `candidate_successor_from_consequence(parent, *, seed=0)`, strict `policy_architecture` and `policy_from_architecture` cases.

- [x] Write failing tests for exact parent logits/memory/value and RNG state, distinct feature schema and strict metadata rejection, inherited parameter copying/mode/trainability, nonzero loss gradient on the new linear/out weights, and save/load roundtrip.
- [x] Run `PYTHONPATH=src .venv/bin/python -m pytest tests/test_successor_policy.py -q` and observe the expected missing class/identity failures.
- [x] Add one zero-initialized linear feature branch and one hidden-state plus feature MLP whose output weight starts at zero; sum their candidate logits with the inherited policy. Implement strict parent upgrade and checkpoint metadata.
- [x] Run focused existing policy/consequence/checkpoint tests plus new tests and Ruff, then commit.

### Task 3: Decide whether representation is cheap enough to cross with broad learning

**Files:** Create `docs/candidate-successor-encoder-preflight-2026-11-06.md`; add one timing test script only if a repeatable bench is needed.

- [x] Run full-menu warm timing on several genuine public TURN menus, recording parent and variant p50/p95, and check all new feature rows are finite.
- [x] Run `PYTHONPATH=src .venv/bin/python -m pytest tests/test_successor_features.py tests/test_successor_policy.py tests/test_policy.py tests/test_consequences.py tests/test_checkpoint_migration.py -q` plus touched-file Ruff.
- [x] Write the measured preflight and limitations. Report the branch to the broad-distillation coordinator for a separately frozen matched training comparison. Commit the report.

### Task 4: Review repair and semantic versioning

**Files:** Update the encoder, upgrade helper, tests, audit and preflight.

- [x] Reproduce the review findings in red tests: no-victim robber, one-resource Plenty, public Largest Army and Longest Road, mixed parent dtypes and ambient device RNG.
- [x] Encode the deterministic effects and preserve every inherited tensor's exact dtype and value. Bump the representation and checkpoint identity to v2.
- [x] Expand the TRAIN transition audit to deterministic point changes and winning indicators, and add endpoint-blocking road geometry coverage.
- [x] Reproduce a legal ordinary settlement cut, a thresholdless award and hidden tie-order ambiguity in red tests against the pinned engine; encode actor point lower bounds and BUY VP-draw uncertainty in v3.
- [x] Reproduce a second enemy endpoint that makes the cut-player length uncertain and a BUILD_ROAD endpoint that would falsely claim an award; encode conservative bounds in v3.
- [x] Rerun the focused tests, full suite, lint, 16-board TRAIN audit and late-road latency after the final v3 bound repair.
- [x] Obtain fresh read-only review before integration. The v3 scoped review
  independently reproduced the pinned-engine settlement award cases and found
  all four prior findings addressed, with no new Critical or Important blocker.
