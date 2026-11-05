# Component-wise Corpus Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. One fresh implementer, independent review and scoped fix loop; no worker subagents.

**Goal:** Support a shared nested10k/100k non-forced corpus and reproducible bounded-memory hand_recipe scaling.

**Architecture:** One narrow collection/fit/evaluation driver over reviewed APIs, immutable four-game components, persistent caller Adam and durable component cursors. Root orchestrates processes and starts actual runs.

**Tech Stack:** Existing upgraded isolated Python/Torch/Catanatron3.3.0 exact commit, pytest/Ruff and standard library; no new dependencies.

**Spec:** `2026-11-05-scaling-design.md` in this planning directory; migrate together to docs/superpowers before implementation and adjust only relative reference.

## Global Constraints

- Separate scaling checkout `/private/tmp/colonist-swarm-bxz1k39_/scaling/repo`, branch cytang/scaling-learning, initial accepted BASE9be7529887ed976e5e48fece302ba3f0513d1750. Capture final HEAD after planning migration for review.
- Use isolated upgraded side .venv only; PYTHONPATH=scaling/repo/src, bytecode off, OMP/MKL/Torch1. Do not import legacy runtime or install dependencies.
- Exact engine identity, immutable own observations/legal menus, unchanged teacher/contracts/rules/rewards; original roadmap and all active/frozen study sources/artifacts untouched.
- No whole-corpus materialization, learned/hidden-state caching, new fitter API, generic framework, scheduler or automatic empirical launch.
- Two-GiB process RSS ceiling; at most two independent fit processes only after measured gate. Root declares all real launch budgets; global deadline wins.
- No sustained collection before legacy DAgger ends and migration passes final integration; bridge is a separate owner, consume root-created new-engine starts.

### Task 1: One-component scaling driver and exact parity gate

**Files:** Create `src/colonist_rl/scaling_study.py`, `tests/test_scaling_study.py`,
`docs/scaling-study.md`; no edits to existing production files or migration files.

**Consumes:** collect_demonstrations/load_demonstrations/save_demonstrations,
fit_sequences/score_sequences, policy_architecture, load_checkpoint/save_checkpoint,
verify_engine_installation, evaluate_blocks/paired_block_summary. Use accepted
safe persistence wrappers or isolate their fixed.tmp stages in unique directories.
Do not call whole-corpus corpus helpers. Initial files are already new-engine,
zero-additive hand_recipe, steps0, optimizerNone, verified parent lineage.

**Produces:**

```python
def collect_scaling_corpus(output_dir, *, protocol: dict) -> dict: ...
def fit_scaling_arm(output_dir, *, corpus_manifest: str, initial_checkpoint: str,
                    endpoint: int, seed: int, protocol: dict,
                    resume_cursor: str | None = None) -> dict: ...
def evaluate_scaling_study(output_dir, *, arm_results: dict[str, str],
                           corpus_manifest: str, protocol: dict) -> dict: ...
```

Arm result keys are exactly `10000/11`, `10000/29`, `100000/11`, `100000/29`;
values reference immutable result files. The public API uses strict documented
versioned dictionaries, not a DSL. Each entry point rejects missing/unknown keys,
wrong JSON types, malformed hashes, changed source/runtime/inputs and invalid caps
before compute. Persist exact launch protocol before work. Protocol must identify
source revision, verified runtime, five disjoint partition lists, prior-reservation
exclusions, input path/SHA expectations, collection thresholds/batch limits,
training settings, explicit stage and global deadlines/RSS ceiling. Scoring/
evaluation settings and seed/size matrix are frozen, never inferred from outcomes.
Implement only fields this three-entry-point driver actually consumes; document
one complete concrete launch example. No permissive **kwargs or silent defaults
for experimental budgets. Root replaces deadlines/caps at launch, not inside code.

- [ ] Write RED component collection tests: one-board fixture exposes seat0-only;
  four-board component rotates0/1/2/3; entire-batch threshold overshoot; nested refs;
  preserved forced history; incomplete/truncated batch stops without replacement.
  Commands use upgraded interpreter, `pytest -q -p no:cacheprovider` on new module.
- [ ] Implement minimal manifest collection. Save every raw result, only complete
  batches count toward eligible endpoints. Verify original source/backend/boards,
  exact teacher replay/call order, legal labels/version order and immutable hashes.
  Persist component/milestone summaries incrementally; stop on deadline/RSS/source
  drift. Validation-label components are separate, never counted as training.
- [ ] Write RED loader/memory tests: bad digest/source/backend, typed metadata
  slippage, illegal label, overlapping/reserved boards and unfinished episodes
  rejected. Monkeypatch whole-corpus helpers to raise; tracked component objects
  must be released before next load. Save only primitive refs/count summaries.
- [ ] Implement sequential validation/count pass and global full-training phase
  weights. First10k gate reports sizes, timings, history quantiles and RSS. No
  primary10k fitting before final shared weights; a smaller stop remains partial.
- [ ] Write RED exact fit parity test using real GraphPolicy-derived hand_recipe,
  two components with forced steps and >16 observations, two global epochs.

```python
# a/b have identical initial parameters and independent Adam states.
full = fit_sequences(a, episodes_a + episodes_b, epochs=2,
                     sequence_length=16, batch_size=1, optimizer=adam_a,
                     phase_weights=weights, profile=True)
parts = []
for epoch in range(2):
    for episodes in (episodes_a, episodes_b):
        parts.append(fit_sequences(b, episodes, epochs=1,
                     sequence_length=16, batch_size=1, optimizer=adam_b,
                     phase_weights=weights, profile=True))
for name, tensor in a.state_dict().items():
    assert torch.equal(tensor, b.state_dict()[name])
# Recursively assert equal Adam tensor/step/group state and integer counters.
assert full['updates'] == sum(p['updates'] for p in parts)
assert full['supervised_presentations'] == sum(p['supervised_presentations'] for p in parts)
assert math.isclose(full['loss'] * full['supervised_presentations'],
    sum(p['loss'] * p['supervised_presentations'] for p in parts),
    rel_tol=1e-12, abs_tol=1e-12)
```

- [ ] Implement canonical component calls/persistent Adam, weighted loss/counter
  aggregation and streamed held-out denominators. Test gradients/parameters differ
  in deliberately reset-optimizer or removed-forced-history controls; parity must
  not pass vacuously. Preserve current-weight prefix logic entirely in fitter.
- [ ] Write RED interruption/cursor tests: expiry inside component, exception
  after some updates, cursor-write failure after checkpoint publication, changed
  resume identity, successful resume from prior boundary, no duplicate committed
  updates or false global epoch. Partial attempted/wasted work stays separately
  visible. Test safe staging/alias preservation and RSS refusal before next load.
- [ ] Implement unique checkpoint publication followed by SHA-bound cursor. Save
  partial-last separately; explicit resume restores optimizer/RNG from prior
  durable boundary and replays whole interrupted component. No automatic retry.
- [ ] Write RED completion-only selection and evaluation tests: exact four-arm
  matrix, largest common complete1/3/5, primary-five failure retained, no outcome
  inputs to selection, missing/censored journal suppresses pairs, no final-board
  play. Implement existing evaluate_blocks callback/paired summaries only.
- [ ] Run GREEN focused suite, then full combined suite once, Ruff/format/diff.
  One development smoke<=60s total: tiny complete-component collection/replay,
  parity/fit and checkpoint/resume with honest counts. Do not attempt100k, real
  game panels, bridge conversion or old-artifact rewriting. Verify installed
  upgraded runtime provenance, not a legacy interpreter import.
- [ ] Document all public fields, memory/cursor guarantees and limits, proposed
  budget envelope, data-plus-training confound, no cross-engine score equivalence
  and user-facing actual launch example. Self-review, commit only owned files plus
  report; send root/coordinator commit, tests and report path. Independent review
  must pass before any sustained use; root supplies final run manifest afterward.
