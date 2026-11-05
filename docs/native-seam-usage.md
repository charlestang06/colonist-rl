# Real-engine seam worker usage

The guarded engineering run has since passed all 10 cases with eight authentic
queries. See [current results](training-status-2026-11-05.md) for execution evidence.
The preparation record and launch instructions below describe the earlier stage.

This worker implements the next-stage checks in `../REVIEW.md`. It is an
engineering fixture, not playing-strength evidence. Static AST parsing and Ruff
are the only checks performed on this script during preparation. Runtime behavior
was unverified at preparation. Core gateway/evaluator code was unchanged from
reviewed a67576c.

Root must approve and launch it through the existing v4 phase_entry guard and
registered supervisor, with one CPU worker, one native/Torch thread, 2 GiB RSS and
an external hard deadline no later than 30 seconds from guard start. Do not launch
the argv below directly as an unguarded benchmark. The existing v4 guard sends its
soft signal at hard minus 15 seconds; the worker defaults to 12 internal seconds
so it can persist a result before that signal. No automatic retry or escalation.

The worker argv to place after the v4 guard's `--` is:

```text
<verified-venv>/bin/python
<isolated-clone>/tools/native_seam.py
--authorized-root-go
--namespace engineering-native-seam-20261105-v1
--fixture-seed 905110501
--budget-seconds 12
--output <fresh-root-owned-attempt-directory>/seam-results
```

Environment variables required to equal `1`: OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, VECLIB_MAXIMUM_THREADS. The script additionally
sets Torch intra/inter-op threads to one. Supply PYTHONHASHSEED=0 for reproducible
set-based native action enumeration; report that setting with the guard artifacts.
The script adds its own clone's `src` to sys.path. Its output directory must not
exist. Seed/namespace above are engineering fixture identifiers; they are not a
new evaluation panel. A pre-preparation text search of agent-research JSON/JSONL,
Markdown and Python files found no prior literal use of either identifier; root
still owns the run namespace and external-cap acceptance.

The worker attempts four runner/direct pairs, at most eight authentic decide
entries total: seat 0 VF initial placement; seat 1 AB sparse normal road/end-turn;
seat 2 VF forced roll; seat 3 AB forced end-turn. Both identities get nonforced
coverage, all four seats get exact direct-copy parity, and original native root
menus remain intact. Query counters are written before each entry and after each
return. The script stops at eight; every other check uses actual engine transitions
or explicitly labeled stubs, not native policy queries.

Other cases cover real setup and normal conversion/step matrices, all seat color
bindings, true native state/deck/action-record/version/RNG immutability, exception
isolation on a real copied backend with a synthetic nested record, rolled-seven
opponent discard phases and cache renewal, synthetic reject trade and no-proposal
routing, controlled native82 roll/development Action.value/ActionRecord.result
characterization, and a deadline plumbing stub with per-seat metadata/latency and
no overrun step. All hooks are confined to the engineering script and recorded
separately; installed native source hashes are verified before and after hooks.

`manifest.json` is atomically updated as cases and query entries progress. A passed
worker manifest is necessary but insufficient for acceptance: root must join it
with the v4 finish marker and resource/exit records. Missing finish records,
resource failures, nonzero exit, internal budget exhaustion or external kills are
incomplete/censored. A failed assertion requires inspection and a report, not an
automatic core fix or benchmark. Unreached cases remain explicitly unpassed.
