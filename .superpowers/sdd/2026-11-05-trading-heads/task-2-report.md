# Task 2 implementation report

Status: implemented and locally verified; coordinator-owned independent review is
required before the separate small-fit protocol. No architecture study was run.

## Scope and environment

- Checkout: `/private/tmp/colonist-swarm-bxz1k39_/trading-heads/repo`.
- Branch: `cytang/trading-heads`; clean dispatch BASE
  `16686108385de2d5617c6ae4f5671b16746cfa23`.
- Only copied `/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python`
  was used for package execution, tests and Ruff. All invocations set assigned
  `PYTHONPATH=$PWD/src`, `PYTHONDONTWRITEBYTECODE=1`, and OMP/MKL/OpenBLAS threads 1;
  final focused/full/smoke invocations explicitly set Torch threads 1 as well.
- Bounded smoke verified the imported source path and copied catanatron 3.2.1.
  Original `.venv`, migration checkout, network, installers, and paid services were
  not used. No subagents or worker-dispatched reviewers were created.

## Owned files

- New `src/colonist_rl/trading_context.py`: exact context24 public extractor.
- `src/colonist_rl/policy.py`: new residual subclasses, copy helper, strict factory
  branches and architecture serialization. Existing Graph/34D computation and
  legacy metadata branches remain unchanged.
- New `tests/test_trading_context.py`: literal public features, isolation/equality,
  residual masks/gradients, fixed-weight interaction and bounded shared-fitter tests.
- `tests/test_training.py`: two focused strict architecture/checkpoint tests.
- New `docs/trading-residuals.md`: schema, teacher audit distinctions, mechanisms,
  identities, freeze scope and limitations.
- This report. No fitter, teacher, contracts, simulator, reward/PPO, benchmark,
  roadmap, legacy artifacts or migration-owner files were modified.

## RED / GREEN evidence

1. Wrote the first 23 feature/head cases before implementation. Running only
   `tests/test_trading_context.py -q` produced **23 expected assertion failures**:
   the public context extractor and residual upgrade APIs did not exist.
2. Added strict new architecture and no-Git checkpoint cases before dispatch
   implementation. Running `tests/test_training.py -q -k trading --tb=short`
   produced **2 expected assertion failures** for absent architecture/upgrade APIs.
3. Implemented the compact extractor, zero context-linear offset, nonzero hidden /
   zero final nonlinear head, exact copy helper and strict dispatch. The focused
   context/training suite became **30 passed**.
4. Tightened the literal fixture to nonzero city stock and made opposite public
   settlement/city gate availability explicit. Added a direct parent-connectivity
   input-mask regression test. A controlled local mutation removing only C's
   `features[:, 30:34] = 0` made that test **fail as expected**; restored the source.
5. Final focused suite (`test_trading_context.py`, `test_consequences.py`,
   `test_training.py`, `test_policy.py`) was **58 passed in 1.12s**.

## Behavioral evidence

- All 24 slots have a literal expected vector, including normalized piece stock,
  unclipped public development count, own level-one count, connected legal sites,
  real city availability, phase, teacher gates/fallback, resource-order production
  and coverage. Additional cases cover exhausted settlement/road/city pieces,
  enemy roads/buildings, occupied/distance-blocked geometry, clipped site counts,
  duplicate-resource tile coverage, zero-number tiles, no own production, unknown
  bank, robber invariance, immutable inputs and a no-port legal 4:1 menu.
- Teacher methods are patched to raise while real extraction executes; no teacher
  method is used. Production/connectivity/phase are documented as extra public
  strategic ingredients. Teacher `_utility` omits those terms and ignores city
  stock in its city gate; the context records both that gate and actual stock.
- A deterministic parent with nonzero 34D weights is copied exactly into independent
  storage. Scalar and batched logits/value/memory are bitwise equal initially,
  including both ACCEPT/REJECT and an empty-action observation. Prefix-only
  recurrent paths never invoke residual modules; caller memory is unmodified.
- CPU Torch, Python and NumPy caller RNG remain unchanged. The explicitly seeded
  nonlinear hidden map is reproducible and changes with a different seed. Mixed
  inherited module modes and parent dtype/device are retained. Repeat upgrades,
  connectivity parents, legacy Graph parents and unknown kinds are rejected.
- Deliberately nonzero B/C residual weights affect supported maritime/END and both
  trade-response choices, while BUILD_ROAD/ROLL/BUY_DEVELOPMENT_CARD residuals stay
  exactly zero. The inherited value and memory remain exact.
- C's output receives a nonzero first gradient while its hidden map initially has
  zero gradient. One output update makes both hidden weight and bias gradients
  nonzero. C's existing hand_recipe34 connectivity slots remain masked even when
  all four raw connectivity/readiness fields are nonzero.
- A same-hand fixture with opposite settlement/city goal availability reverses C's
  maritime/END preference using fixed demonstrative tanh weights. B's nonzero state
  offset leaves the parent contrast invariant within floating tolerance. This is
  representational evidence only, with no optimizer search or gameplay claim.
- Both arms complete exactly two hard-label shared-fitter updates with full prefix
  replay, frozen inherited Graph/recurrence/value, trainable parent34D/residual,
  and Adam owning every parameter exactly once. Frozen gradients remain None and
  tensors exact. Independent arms have equal results and independent parameter and
  Adam-state storage. Passing only trainable parameters is correctly rejected by
  the unchanged fitter.
- Strict identities reject extra/missing fields, unknown feature/context schemas,
  other feature sets, bool/float/wrong residual widths and invalid hidden sizes.
  Graph, parent34D, B and C checkpoints roundtrip through existing schema1 loading,
  including optimizer state, from a non-Git temporary cwd with Git unavailable.
  Missing/extra residual keys and wrong tensor shapes are rejected.

## Final verification

- Full legacy suite run **once after focused green**:
  **597 passed, 1 skipped in 29.53s**.
- Copied-runtime Ruff check on all four owned Python files: **All checks passed**.
- Copied-runtime Ruff format check: **4 files already formatted**.
- `git diff --check`: clean.
- Bounded import/checkpoint smoke: exact assigned source and copied 3.2.1 imports;
  both new schema1 identities and all tensors roundtrip. Temporary checkpoints
  were automatically removed. No sustained fit or output artifact was retained.
- Self-review confirmed original Graph and Consequence `_output`, old34 dot
  reduction, recurrence/value paths, fitter and schema1 loader were not edited.

## Concerns and limits

No known implementation blocker. CPU behavior and dtype/device preservation on
copy were checked; accelerator execution was not exercised. B intentionally has
no state-conditioned ranking effect among supported alternatives. C's representational
and gradient checks establish capability and connectivity, not empirical trading
quality. Independent architecture review and the coordinator's separate matched-fit
protocol remain required before actual small fitting. No 512-case fit, held-out
gradient run, gameplay, 100k/PPO or teacher changes were performed.

## Review follow-up: shared settlement-site geometry

The independent review's sole Important finding was duplicated occupied / adjacent
building exclusion / incident-own-road geometry in the two feature extractors.
The coordinator explicitly expanded ownership to permit this narrow extraction
into `src/colonist_rl/settlement_sites.py` and replacement of the old block in
`consequences.py`. This follow-up changes no policy, checkpoint, teacher or fitter
code and runs no fits.

- Added pure `connected_settlement_site_count(view) -> int`, returning the existing
  **unclipped** public count without inspecting phase, stock, hand, bank or engine.
- Both `consequence_features` and `trading_context_features` call that helper at
  their previous geometry step. Each retains its existing `min(sites, 8) / 8`
  normalization. Consequence hand/recipe computation, readiness checks, feature
  assignment order and inherited policy reduction are unchanged.
- Added `tests/test_settlement_sites.py`: eight literal geometry boundary cases
  assert the helper count and **complete** expected 34D and 24D vectors, plus an
  unclipped 54-site standard-board case demonstrating both callers still clip.
  Fixtures cover missing/enemy/own roads, occupied vertices, adjacent buildings,
  shared road endpoints, own city eligibility and input immutability.
- TDD RED: all **9 cases failed** because the shared helper did not exist.
  GREEN: the non-fit focused suite passed **65 tests, 2 deselected**. The two
  `shared_hard_label_fitter` cases were explicitly excluded. No full-suite repeat
  was made because this follow-up prohibits fits and the full suite contains them.
- A bounded read-only comparison against both pre-extraction functions from commit
  `a602a4d79f2477df43fe9df8b80829ae91304845` checked 32 synthetic public views across
  road ownership, occupied geometry and roll state: **32 context24 and 96
  consequence34 vectors were bitwise equal**. It used no teacher or optimizer.
- Copied legacy Python, explicit assigned `PYTHONPATH`, disabled bytecode,
  OMP/MKL/OpenBLAS threads 1 and explicit Torch threads 1 were retained. Ruff check,
  format check and diff checks are clean. The original runtime was not accessed.
- Reviewable diff is the follow-up commit against `a602a4d`; only the shared helper,
  two caller substitutions, new compatibility tests and this report were changed.
  Independent re-review remains pending; no dependent fit is authorized here.
