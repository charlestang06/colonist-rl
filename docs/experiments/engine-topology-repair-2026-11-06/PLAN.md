# Road topology repair implementation plan

> **For agentic workers:** use test-driven development and review each task
before accepting its result.

**Goal:** Add a source-bound, opt-in corrected Catanatron road topology without
changing the installed upstream package or historical study semantics.

**Architecture:** `CorrectedBoard` owns road connectivity, legality, and longest
trail calculation. `FourPlayerGame` installs it only when requested and
reconciles VP after backend action handlers. A mode-specific identity string
distinguishes new studies from prior backend artifacts.

**Spec:** `docs/experiments/engine-topology-repair-2026-11-06/DESIGN.md`.

## Task 1: Road legality and component cache

**Files:** create `src/colonist_rl/catanatron_topology.py`; create
`tests/test_catanatron_topology.py`.

1. Write the 19-action excluded Board test so it fails on the pinned backend:
   the road from an already blocked vertex must be absent from the legal menu.
2. Run the focused test and confirm that exact failure.
3. Implement `CorrectedBoard.from_board`, `buildable_edges`, `buildable_node_ids`,
   and road/settlement mutation with edge-component rebuilding.
4. Run the test green, then add a separate two-component blocker test and a
   cut-settlement test; watch each fail before extending code.

## Task 2: Longest road and award transfer

**Files:** modify `src/colonist_rl/catanatron_topology.py`; modify
`src/colonist_rl/simulator.py`; extend `tests/test_catanatron_topology.py`.

1. Write failing tests for a five-road path ending at an enemy building, a
   branched/cyclic graph, holder retention on tie, transfer on strict lead,
   and loss to no holder after a cut.
2. Implement exact edge-simple trail scoring and incumbent-aware ownership.
3. Reconcile both VP counters, holder flags, and road lengths after an engine
   road/settlement action. Keep upstream package files untouched.
4. Run all focused tests after each behavior becomes green.

## Task 3: Simulator integration and evidence

**Files:** modify `src/colonist_rl/simulator.py`; extend
`tests/test_simulator.py` and `tests/test_catanatron_topology.py`.

1. Add a failing opt-in reset/deepcopy test and a mode identity assertion.
2. Install corrected board only with `corrected_topology=True` and preserve the
   current default. Verify initial menus and exact legal action identity.
3. Run a complete excluded raw-vs-corrected game in lockstep; require identical
   menus/actions/terminal result when no blocker interaction occurs.
4. Run a separate bounded corrected full-game timing/RSS probe and record its
   source, seed, wall time, peak RSS, and termination in ignored `runs`.
5. Run focused simulator/engine tests, Ruff check/format, and verify a clean
   commit. No fresh study board is opened in this task.
