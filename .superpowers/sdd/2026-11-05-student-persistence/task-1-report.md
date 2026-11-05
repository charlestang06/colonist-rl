# Student persistence optimization report

## Scope and implementation

Base: `575f70ec8ea6cc6896b6f502990763d81bdad3f2`; isolated branch
`cytang/student-persistence`. Assigned production, test, and documentation files
only. Read the task brief and matching plan first; both SHA-256:
`7285ec20ed64f2485230cf22a7a631ef64b615a8dccb0de6d6853cff461036a9`.

`_validate` now creates one detached primitive observation export per sample,
strictly roundtrips that exact export through `view_from_dict`, and hashes the
same ordered primitive samples. Its private `export=True` option returns those
validated exports to `save_student_labels`. Default validation returns `None` and
keeps only the current episode's serialized samples, without a whole-component
export. Public signatures, schema, validation predicates, collection scheduling,
labels, canonical JSON, deterministic gzip options, and atomic writer are unchanged.
No cross-call cache is introduced.

Inspected `adapters/observation_io.py`: `_record` copies mappings with `dict(data)`;
`view_from_dict` constructs new records and tuples rather than modifying its DTO.
Regression coverage confirms both nested parser input and caller component remain
unchanged. Existing history digests are compared with the derived digest; they
are never repaired during validation/save.

Production source SHA-256 before:
`47bc3137a2663a109fb6bd8eed3fa71405e49ea4e14b19dbcd0444c6c5a83076`.
After (also the source measured by the optimized probe):
`b2352e60a77119521a1f8eecd105e7cc683400b130f402af9a0b012b5215eb45`.

## One bounded real fixture and byte parity

Artifacts are outside the repository at
`/private/tmp/colonist-swarm-bxz1k39_/student-persistence/probe/`.
The baseline was collected and saved before production edits, from the clean base
revision, with the preserved legacy runtime only:
`/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python`.
Verified Python 3.13.2 and installed Catanatron **3.2.1**. No installs/network or
original runtime usage.

`persistence_probe.py` SHA-256:
`b805244743e7b4db11128676194d8fcf5f3e032f7e140a67a51575a1c18a2205`.
It seeds Torch with 11, constructs an untrained `GraphPolicy(hidden_size=8)`,
saves/reloads it, and asserts every state tensor equal. Collector identity checking
also verifies checkpoint bytes, architecture, and all tensors before freezing an
evaluation copy. Collection asserts the caller's parameters remain unchanged.
No fitting occurs in either phase. The checkpoint SHA remains recorded unchanged
in the identical loaded component; no policy state is present in the serialized
fixture to mutate during save.

Checkpoint: `untrained-seed11-hidden8.pt`, SHA-256
`c7aba5218a07ef3c179d586018b093daeead68b5295dc7c963aebd8fed50d087`.
Requested boards: `119200000..119200003`; actual:
`[119200000,119200001,119200002]`, seats 0/1/2. Max decisions 128, max game
steps 20000, heuristic opponents, seed 11, stochastic student, absolute 20-second
collection deadline. Actual: **128 decisions**, **51 non-forced**, **36 teacher
vs behavior disagreements**, **128 executed**; stop reason `decision_cap`.
Collection elapsed: **1.939020250 s**. These are actual episodes and one genuine
interrupted final prefix, with no repetition/augmentation or relabeling.

The original immutable fixture was retained as `fixture.pkl`, SHA-256
`6c4f16859448366b9f5cf7c7871e9fd988469a4fb2301feddcfee88da5dc4ec2`.
The optimized invocation reads that same local trusted fixture; it does not collect
again. Both phases assert input-pickle equality, complete loaded-data equality
(including labels, behavior flags, observations, source, checkpoint identity), and
history equality. Optimized output is compared directly to original gzip bytes.

| Measure | Baseline | Optimized |
| --- | ---: | ---: |
| Timed save seconds | 0.312598292 | 0.177286041 |
| Observation conversions during save | 384 | 128 |
| Output bytes | 35713 | 35713 |
| Process peak RSS bytes (macOS `ru_maxrss`) | 253165568 | 247349248 |
| Measured phase seconds | 2.824430125 | 0.466936208 |

`baseline.gz` and `optimized.gz` are byte-identical, SHA-256:
`359bf483d81a135f9f503d1e87b7f984af2f8f0976280f6783212f9bd381c773`.
Original history SHA-256 values remain identical:

- `1b53eeb5e9d457117dbb575b86a41e140bdae0a864329c0944afde32a4408f0d`
- `d1f008fefe1b27a322ea204085680ec866c181894f5385c2134095c385d4f492`
- `72adaff2a5bdf8d6e5c64d93ea9a5bce016229bee08605a8414858cdbc7b764b`

Baseline collection source stays `{revision: BASE, dirty: false}` in both files.
The optimized running source correctly reports the same base with `dirty: true`;
that is recorded separately and does not overwrite the fixture's provenance.

`baseline.json` SHA-256:
`3c82d1ce9af9fc4dc5aef40036665a288955c39b55023f9b74255a1d4be0dbce`.
`optimized.json` SHA-256:
`9abb05abfe8b69afed94c86f41d8b1ae0d24b06e803c52f14d2e8882e5afd0e6`.

Combined measured active phase time: **3.291366333 s**, below the 60-second
combined probe budget. Each invocation has a signal alarm; optimized subtracts
the recorded baseline phase duration from 60 seconds. Phase timing starts after
module imports, and implementation time between the two invocations is excluded.
Both invocations completed on their first attempt. Retained original bytes,
collection progress, actual metadata, and logs. No 4000-query study, primary/final
panel, active/frozen run modification, or unrelated artifact change.

## Commands and verification

Working directory for all repository commands:
`/private/tmp/colonist-swarm-bxz1k39_/student-persistence/repo`.
Every Python probe/test command used this exact environment prefix:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/private/tmp/colonist-swarm-bxz1k39_/student-persistence/repo/src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python
```

Append these arguments to that prefix (listed in execution order):

```sh
/private/tmp/colonist-swarm-bxz1k39_/student-persistence/probe/persistence_probe.py baseline
-m pytest tests/test_student_labels.py -q -k 'persistence or roundtrip_parser or student_atomic'
-m pytest tests/test_student_labels.py -q -k 'persistence or roundtrip_parser or student_atomic'
/private/tmp/colonist-swarm-bxz1k39_/student-persistence/probe/persistence_probe.py optimized
-m pytest tests/test_student_labels.py tests/test_imitation.py tests/test_supervised_fit.py tests/test_imitation_study.py -q
```

RED (before production edits), `red-tests.txt`: **8 failed, 19 passed, 32
deselected**, 4.04 s. All eight failures were expected: validation converted each
sample twice, saving three times, rather than once. Existing-contract cases
passed. GREEN after implementation, `green-tests.txt`: **27 passed, 32
deselected**, 3.89 s. Then strengthened the same conversion test to check two
successive calls each independently serialize once, ruling out cross-call reuse.
Focused integration run, `focused-tests.txt`: **113 passed**, 16.87 s. Includes
student/corpus, imitation, sequence fitting, and imitation study checks.

New regression cases cover:

- One conversion per sample per validation/save, including successive calls.
- Independently generated legacy v1 gzip bytes, returned digest, loaded data,
  and input equality for complete, forced-only, unmatched teacher, and unexecuted
  student tails.
- Strict parser input immutability and snapshot wire integrity.
- Save/load rejection of boolean schema versions, integer source flags, boolean
  collection seeds/labels, integer execution flags, malformed observation tiles,
  and existing history digest corruption. Corruption tests leave history hashes
  untouched, and rejected saves preserve previous destination bytes.
- Atomic destination/staging symlink and hardlink preservation; existing corpus
  alias, strict history, source, and typed metadata tests remain covered.

Lint/format commands (preserved Python, `PYTHONDONTWRITEBYTECODE=1`):

```sh
-m ruff check src/colonist_rl/student_labels.py tests/test_student_labels.py
-m ruff format --check src/colonist_rl/student_labels.py tests/test_student_labels.py
-m ruff format tests/test_student_labels.py
-m ruff check src/colonist_rl/student_labels.py tests/test_student_labels.py
-m ruff format --check src/colonist_rl/student_labels.py tests/test_student_labels.py
```

First lint passed; first format check identified one new long parameter list.
Applied Ruff formatting (whitespace only); final lint **passed**, final format
**2 files already formatted**. `git diff --check` passed. No duplicate long suite
or full repository suite: the narrow change has direct coverage across its actual
student/corpus/imitation/fitting consumers, and the parent will separately review
and verify the combined engine port.

## Self-review and limitations

Reviewed the source diff against the binding plan: all old field, type,
provenance, backend/action, label/behavior, endpoint, call-order, history, coverage,
and finite-metadata checks remain in place. The atomic write path is untouched.
Validation-only memory retains one episode's primitive samples for its digest;
save necessarily retains the full detached export until encoding completes.

This is one small correctness/performance probe, not a statistical benchmark.
Peak RSS covers whole separate processes; baseline includes collection whereas
optimized includes fixture loading, so the RSS figures do not isolate memory
improvement. Save timing includes the conversion counter's identical wrapper in
both phases. No 3x wall-speed or full-study cap/throughput claim is justified.
Independent review remains required. The parent owns porting to the new engine;
this work intentionally measured and tested legacy Catanatron 3.2.1 only.
