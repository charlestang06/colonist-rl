# Task 2 implementation report

Status: DONE. Base: `a13054f50dd4b65e9657508466cc6732f12fc822`.
Branch: `cytang/behavioral-learning`; behavioral clone only.

## Implementation and exact owned files

- `src/colonist_rl/consequences.py`: public immutable-input feature extraction, schema `public-consequence-v1`, 34 slots; maritime/offer-perspective accept/no-change reject/end hand arithmetic; recipe deficits/sufficiency; distance-legal own-road settlement connectivity and explicit normal-play/piece readiness. Absent actions are rejected, unsupported present kinds are zeros, unknown bank stock is never inferred.
- `src/colonist_rl/policy.py`: opt-in `ConsequenceGraphPolicy`, one zeroed bias-free 34→1 projection integrated through inherited scalar/batched `_output`, including trade responses; untouched values/recurrence and memory-only prefixes. Exact-copy legacy upgrade preserves caller Torch RNG and training mode. Explicit architecture helpers and read-only feature-set identity separate recipe-only masking from connectivity.
- `src/colonist_rl/training.py`: only checkpoint architecture save/load dispatch and its imports changed. Schema remains 1, legacy metadata remains exactly `{hidden_size: ...}`, strict parameter loading retained.
- `tests/test_consequences.py`: 20 cases covering literal hand/deficit arithmetic, both offer perspectives, immutability, unknown bank, blocked/occupied/disconnected sites, clipping, phase/piece/city constraints, absent/unsupported actions, exact initial compatibility, independent base parameters, useful first gradient, RNG preservation, scalar/batched additive effects, responses, memory-only routes and feature-set isolation.
- `tests/test_training.py`: 3 focused additions covering both metadata identities and nonzero branch weights through optimizer checkpoint round trips, strict missing-branch rejection, exact architecture-shape/schema/feature-set validation, and zero-step checkpoint through curriculum's shared loader plus collector's independent frozen deep-copy route.
- `docs/consequence-policy.md`: API, slot table, recipe/normal-build interpretation, explicit comparison identity and checkpoint compatibility.
- `.superpowers/sdd/2026-11-05-behavioral-learning/task-2-report.md`: this report.

No curriculum gates, PPO, rewards, simulator, adapters, old run artifacts, production entry defaults, original repository or environment were modified. No subagent or sustained training/profile/game study was run. Full-suite automated tests include their existing bounded training checks.

## TDD evidence

All pytest commands used this exact interpreter/environment prefix from the clone:

```sh
PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python
```

RED, before production implementation:

```sh
PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q tests/test_consequences.py tests/test_training.py
```

Output: **23 failed, 2 passed in 1.85s**. Failures identified the missing feature module and missing policy/architecture APIs; the two pre-existing training cases remained green. Full output: `/private/tmp/task2-red.log`.

GREEN, after implementation:

```sh
PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q tests/test_consequences.py tests/test_training.py
```

Output: **25 passed in 2.02s**.

Focused integration after formatting:

```sh
PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q tests/test_consequences.py tests/test_policy.py tests/test_training.py tests/test_curriculum.py tests/test_supervised_fit.py tests/test_graph_inputs.py
```

Output: **105 passed in 5.60s**.

Full suite, once before commit:

```sh
PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python -m pytest -q
```

Output: **428 passed, 1 skipped in 24.77s**. Full output: `/private/tmp/task2-full.log`.

## Static and wheel portability verification

```sh
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff check src/colonist_rl/consequences.py src/colonist_rl/policy.py src/colonist_rl/training.py tests/test_consequences.py tests/test_training.py
/Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/ruff format --check src/colonist_rl/consequences.py src/colonist_rl/policy.py src/colonist_rl/training.py tests/test_consequences.py tests/test_training.py
git diff --check
```

Output: **All checks passed!**, **5 files already formatted**, clean diff check.

Root supplied the known cached official builder, used successfully without installs/network or original-environment writes:

```sh
/Users/charlestang06/.local/bin/uv build --offline --wheel --cache-dir /private/tmp/colonist-rl-uv-cache --out-dir /private/tmp/colonist-swarm-bxz1k39_/behavioral-wheel-check
```

Output: **Successfully built** `/private/tmp/colonist-swarm-bxz1k39_/behavioral-wheel-check/colonist_rl-0.1.0-py3-none-any.whl`.

The official wheel was extracted to an isolated directory and tested with the mandated interpreter, empty `PATH` (Git unavailable), and only the extracted package on `PYTHONPATH`. Legacy and both opt-in identities loaded through `training.load_checkpoint`; metadata and exact logits/value/memory matched source expectations, including nonzero additive weights. Output status: **passed**, `git_available: false`, `training_steps: 0`. No package-manager installation was performed, consistent with the no-installs instruction. An earlier standard-library rebuilt-wheel smoke passed identically; the official offline-built wheel supersedes that packaging check.

The reusable checkpoint/output smoke script is `/private/tmp/task2-wheel-smoke.py`; official-wheel smoke artifacts are `/private/var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/task2-official-wheel-v57dbv49/` (including `smoke.py`, three checkpoints, expected tensors and the extracted package). The isolated runner imports no Git/source checkout.

## Self-review and concerns

Reviewed owned source/tests/docs against section 2 and the exact brief. Only subclass `_output` adds logits, preserving Task 1 graph isolation and recurrent-prefix behavior and legacy `GraphPolicy` parameter keys/calculations. Strict state loading prevents additive parameters being discarded. The frozen-history regression uses an already expired collector deadline, exercising shared checkpoint dispatch and the actual deep-copy path without creating a game or training.

Root explicitly approved the narrow slot interpretation: slot 30 is the spec table's topology-only count; settlement piece and ordinary own post-roll phase restrict slot 31; slot 32 is own-settlement/city-piece availability; slot 33 adds resources and normal phase. Recipe and connectivity identities remain distinct. No unresolved implementation concerns. Portability was verified through an extracted official wheel without Git, rather than installing into the protected environment.
