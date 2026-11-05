# Final integration fix — I1 / P2 curriculum reservations

Addressed the sole Important finding in `final-review.md` on root integration
base `7e904bdd5335800ca73c47f8056f95a90520c621`. The root working tree was clean
before this wave. No other owner files were changed.

Curriculum now uses `RESERVED_FINAL_TEST`, the union of legacy finals `70000017`
and `70000019` and the current `71001000..71001007` panel, for generated training
candidate/configurable validation split preflight and `_resolved`. Promotion and
selection therefore reject any evidence containing either legacy final, even
when all result panels match and provide complete typed four-seat terminal rows.
Initialization seed `620` is rejected because its generated candidates contain
both legacy finals. Rejection occurs before source/input access, output creation,
or collector/evaluator work.

`FINAL_TEST` and the existing manifest `seed_splits.final_test_reserved` retain
the current panel's meaning. The additive
`seed_splits.all_final_test_reserved` records the complete enforced reservation.
The curriculum protocol describes both fields and the seed-620 boundary. Default
initialization seeds, validation panel, lineups, rewards and gate/selection APIs
are unchanged. No source aliases, old studies, artifacts, original `.venv`, runtime
or roadmap files were changed.

## Test construction and RED evidence

All execution used the upgraded isolated side interpreter below with root `src`
on `PYTHONPATH`, bytecode disabled, and OMP/MKL/Torch threads set to one (the
existing autouse curriculum fixture sets Torch threads). No simulator reset,
gameplay, policy forward, training, or actual final-board execution occurred.
The runner tests stop at split/source/manifest boundaries; the manifest test
captures its write payload and stops before game work. Its only directory
creation is inside pytest's temporary directory.

New regressions cover seed-620 training candidates, each legacy validation seed,
current-panel endpoints, normal default/custom validation panels, manifest
reservations, each promotion/selection evidence role, and fully matching legacy
panels. The initial added subset command was:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_curriculum.py -k 'runner_rejects_reserved_boards or runner_accepts_normal_splits or manifest_records_all_reserved or promotion_rejects_reserved_evidence or selection_rejects_reserved_evidence'
```

Result before production edits: **14 failed, 14 passed, 59 deselected in 2.31s**.
Failures established the missing legacy split/evidence reservation and missing
manifest union; existing current-panel and normal-panel checks passed.

After adding matching-panel regression cases, still before production edits:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_curriculum.py -k 'runner_rejects_reserved_boards or runner_accepts_normal_splits or manifest_records_all_reserved or promotion_rejects_reserved_evidence or selection_rejects_reserved_evidence or complete_matching_reserved_panel'
```

Result: **17 failed, 14 passed, 59 deselected in 2.06s**. To make both consumer
outcomes visible in the same assertion, the matching-panel test was then adjusted
to compute promotion and selection before comparing their result tuple:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_curriculum.py -k complete_matching_reserved_panel
```

Result: **3 failed, 87 deselected in 1.78s**. Each failure showed the old behavior
`(True, 'selected', 'reserved')` instead of `(False, 'unavailable', None)`, for a
panel containing either legacy seed alone and for the legacy pair together.

## GREEN and covering verification

After the minimal production fix, the expanded new-subset command above passed:
**31 passed, 59 deselected in 1.75s**.

The existing pure curriculum gate/selection regressions were then included:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_curriculum.py -k 'runner_rejects_reserved_boards or runner_accepts_normal_splits or manifest_records_all_reserved or promotion_rejects_reserved_evidence or selection_rejects_reserved_evidence or complete_matching_reserved_panel or gate_uses_resolved_rows or selection_rejects_censored or curriculum_requires_explicit_typed_resolution or curriculum_rejects_winner or selection_uses_one_common_panel'
```

Result before formatting: **70 passed, 20 deselected in 1.73s**. Final rerun after
formatting: **70 passed, 20 deselected in 1.75s**. The 20 deselected cases include
simulator-backed tests; the already accepted root broad suite was not rerun.

```sh
PYTHONDONTWRITEBYTECODE=1 /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python -m ruff check --no-cache src/colonist_rl/curriculum.py tests/test_curriculum.py
PYTHONDONTWRITEBYTECODE=1 /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python -m ruff format --check --no-cache src/colonist_rl/curriculum.py tests/test_curriculum.py
```

Lint passed. Initial format check requested three layout-only changes in the new
tests. Applied them with:

```sh
PYTHONDONTWRITEBYTECODE=1 /var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python -m ruff format --no-cache src/colonist_rl/curriculum.py tests/test_curriculum.py
```

Result: **1 file reformatted, 1 file left unchanged**. Repeated lint and format
checks both passed: **All checks passed!**, **2 files already formatted**.
`git diff --check` passed, and the complete owned source/test/protocol diff was
read before committing. This resolves the reservation implementation gap; it
adds no empirical claim about policy strength or curriculum effectiveness.
