# Adapter Modularization Implementation Plan

**Goal:** Separate reusable observation and capture handling from Colonist browser
control, and add an honest entry point for verified physical-board photos.

**Architecture:** Keep the immutable contracts, board topology, JSON envelopes,
and browser execution behavior unchanged. Move reusable code into
`colonist_rl.adapters` and browser code into `colonist_rl.adapters.colonist`.
Legacy modules remain compatibility facades with identical exported objects.
The physical adapter only reads through an explicitly supplied capture reader;
the existing verified-annotation reader is one supported implementation.

**Scope:** The approved design in the coordinating chat is the specification.
Use the existing `cytang/initial-implementation` branch; leave changes uncommitted
for review. No CLI, contracts, simulator, policy, dependency, or schema changes.
Do not edit the walkthrough or roadmap documents owned by other agents.

## File map

| File | Responsibility |
| --- | --- |
| `src/colonist_rl/adapters/errors.py` | Shared `PerceptionError` identity |
| `src/colonist_rl/adapters/observation_io.py` | Observation codecs, validation, visible-event projection |
| `src/colonist_rl/adapters/calibration.py` | Existing affine board calibration and target coordinates |
| `src/colonist_rl/adapters/capture.py` | Capture records, reader protocol, hash-bound annotations, local file ingestion |
| `src/colonist_rl/adapters/colonist/browser.py` | Existing optional Playwright transport and session |
| `src/colonist_rl/adapters/colonist/execution.py` | Browser protocol and guarded single-click execution |
| `src/colonist_rl/adapters/physical/__init__.py` | `PhysicalBoardAdapter(reader=...).read_photo(path)` |
| `src/colonist_rl/perception.py` | Explicit compatibility exports |
| `src/colonist_rl/browser.py` | Compatibility module alias preserving the `_start_playwright` patch seam |
| `tests/test_adapters.py` | Compatibility, verified physical input, privacy, and import boundaries |
| `docs/adapters.md` | Package boundaries, compatibility, and physical-photo usage |

## Tasks and status

- [x] Baseline: existing perception/browser/CLI tests: 62 passed, 1 skipped.
- [x] Red: add tests for old/new object identity, legacy browser patching, injected
  photo ingestion, unknown/malformed photos, partial annotations, and subprocess
  import isolation; 15 failed because the new adapter package was missing.
- [x] Green: extract existing implementations without changing schemas or
  behavior, retain compatibility facades, and add the reader-injected photo entry.
- [x] Verify: focused suite 77 passed, 1 skipped; full suite 181 passed, 1 skipped.
  Ruff lint and formatting checks passed. Real Chromium loopback smoke: 1 passed,
  18 deselected. Wheel built; all nine required nested adapter files verified.
  Browser source is byte-identical; all 19 moved perception definitions are
  AST-identical to the original implementation.
- [x] Review handoff: update status and save the implementation report to
  `/tmp/colonist-adapters-implementation-report.md`; do not commit.

## Constraints and checks

- Resource order stays `WOOD, BRICK, SHEEP, WHEAT, ORE`.
- All `colonist-rl.*` format names and version fields stay unchanged.
- Unknown images and incomplete annotations fail; no OCR or missing-state defaults.
- Physical input must not import Colonist browser/execution, Playwright, Torch,
  Catanatron, or the simulator. Shared code may use NumPy and Pillow.
- Existing classes and exceptions are reexported by identity; legacy browser
  tests must still patch the same `_start_playwright` module global.
- Preserve before/after verification, explicit authorization, visibility
  redaction, tool modes, and ambiguous-click lockout.
- Verification commands: `uv run --extra dev pytest -q`,
  `uv run --extra dev ruff check src tests`,
  `uv run --extra dev ruff format --check src tests`, and `uv build --wheel`
  followed by inspecting the archive's nested package entries.
