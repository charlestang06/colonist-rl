# Byte-compatible student persistence optimization

User requested audited student states and throughput. The first4000-query study
missed both student120s caps; its partial result/gates remain unchanged. Read-only
cProfile found1740samples require5220 observation_to_dict calls and10.477s/save;
1240samples3720calls/6.932s. Validation7.421/4.945s; gzip only0.139/0.099s.
Profile overhead/whole collection differ; no3x wall-speed claim.

## Binding design

Serialize each immutable observation ONCE per validation/save call. Reuse the exact
primitive snapshot for strict view_from_dict validation, historySHA and encoding.
Preserve EVERY current field/type/source/backend/label/behavior/boundary/coverage
validation and rejection. Public API and v1 deterministic gzip bytes unchanged.
No cross-call cache, weakened validation, schema/label/collector timing change.
Validation-only callers must not retain an extra whole-component primitive export.

### Task1

Own ONLY src/colonist_rl/student_labels.py, tests/test_student_labels.py,
docs/student-labels.md and report. Original .venv is3.2.1 and read-only; use it with
PYTHONPATH=thisclone/src, bytecodeoff,Torch/BLAS1. Other sources, active/frozen runs,
original roadmap/runtime/artifacts stay unchanged. No subagents/installs/network.

Before production edits, ONE combined<=60s baseline/optimized probe: collect a small
max128-query real student fixture using a verified untrained checkpoint and clean
source, disposable boards119200000..119200003, no primary/final panel. Save original
baseline gzip bytes/hash outside repo student-persistence/probe/. After optimization
save SAME immutable fixture and require byte-identical output/history hashes,
load/label/source/parameter equality. Preserve partials and actuals, no retries.
Do not relabel old/new engine data or inflate fixture by duplicating episodes.

RED tests: one conversion/sample, exact baseline wire bytes, no mutation by
roundtrip parser, forced/unmatched/unexecuted histories, metadata type/label/
observation corruption rejection, atomic alias preservation. Inspect parser for
mutation before reusing its DTO. Private _validate export option/helper allowed;
public signatures and behavior stay unchanged. Do not recompute digest to hide
corruption. Existing focused tests remain passing. No actual4000-query run.

Focused RED/GREEN student/corpus/imitation tests, Ruff/format/diff, combined suite
only if concrete integration warrants it; self-review/commit assigned files.
Report .superpowers/sdd/2026-11-05-student-persistence/task-1-report.md: exact commands,
results, baseline/probe source+byteSHAs, timings/calls/RSS, limitations. Independent
review and root combined port to newengine follow. New empirical study gets a
separate immutable source/budget manifest, never extends the failedrun.

Ruling: Remove redundant conversion with byte-identical output, not weaker checks;
actual profile identifies cost. Any wire/rejection difference blocks acceptance.
