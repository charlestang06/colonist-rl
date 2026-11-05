# Task 1 implementation report: plain GreedyNoTrade

Implemented the approved frozen `plain-greedy-no-trade-v1` rules on branch
`cytang/trading-baselines`, starting from clean
`2c6422ffc55c5899080d8160824a148b28850c70` in
`/private/tmp/colonist-swarm-bxz1k39_/trading-baselines/repo`.

## Owned files

- `src/colonist_rl/greedy_no_trade.py`: 91-line plain policy, two pure helpers,
  `nn.Module`, read-only instance identity, original legal action selection,
  `(Action, None)` actor adapter, and no proposals.
- `tests/test_greedy_no_trade.py`: 42 focused cases with real immutable
  `Observation`/`Action` records and a scripted environment boundary for the
  existing evaluator/match driver integration.
- `docs/greedy-no-trade.md`: exact frozen priorities, limitations, evaluator
  invocation, required manifest identity, shared-opponent-panel interpretation,
  censoring/board-level uncertainty, and independent-review-before-benchmark.
- This report.

No existing source, simulator/rule/contract, evaluator, teacher, training,
checkpoint, runtime, or roadmap file was changed.

## Runtime and exact test evidence

All policy, test, Ruff, and smoke execution used the preserved interpreter
`/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python`.
The common command environment was:

```sh
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=/private/tmp/colonist-swarm-bxz1k39_/trading-baselines/repo/src
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

These variables were supplied inline for each command. Runtime verification
reported `catanatron=3.2.1` and `torch_threads=1`. The smoke also explicitly called
`torch.set_num_threads(1)`. No installs, network access, original `.venv`,
subagents, training, checkpoint selection, or actual benchmark were used.

RED, before creating the module:

```sh
/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python -m pytest tests/test_greedy_no_trade.py -q
```

Output: `42 failed in 1.43s`, each failure due to
`ModuleNotFoundError: No module named 'colonist_rl.greedy_no_trade'` in a test
body, confirming the requested public module/interface was absent. This was not
a collection failure. Full captured output:
`/private/tmp/greedy-no-trade-red.log`.

GREEN, after implementing the literal policy:

```sh
/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python -m pytest tests/test_greedy_no_trade.py -q
```

Output: `42 passed in 2.48s`. Full captured output:
`/private/tmp/greedy-no-trade-green.log`.

Full legacy suite, run once after formatting:

```sh
/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python -m pytest -q
```

Exit 0: `571 passed, 1 skipped in 41.01s`. Full captured output:
`/private/tmp/greedy-no-trade-full-suite.log`.

Ruff first reported one test import-order issue. `python -m ruff check --fix`
fixed that issue; `python -m ruff format` formatted the two new Python files.
Final exact checks, using the common environment:

```sh
/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python -m ruff check src/colonist_rl/greedy_no_trade.py tests/test_greedy_no_trade.py
/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python -m ruff format --check src/colonist_rl/greedy_no_trade.py tests/test_greedy_no_trade.py
git diff --check
```

Results: `All checks passed!`; `2 files already formatted`; clean whitespace.

## Strict behavior and integration checks

- Rejects a free incoming offer before city purchase; never accepts, makes player
  proposals, or selects a maritime trade. The 4:1 bank fixture has no owned port
  and treats the bank exchange as legal; refusing it is the baseline restriction.
- Empty and unsupported/forbidden-only menus raise `ValueError`; unknown kinds
  cannot trigger arbitrary fallback; `END_TURN` remains a supported fallback.
- Purchases follow city > settlement > lowest-ID road > development purchase,
  with roll before purchases/development; development kinds follow the frozen
  road-building > knight > plenty > monopoly order.
- Setup/ordinary settlements and cities use additive raw pips with smallest
  vertex ties. Fixtures cover unnumbered tiles, resource/port/robber invariance,
  existing production, and roads toward a poorer destination.
- Largest-own-count discards use shared resource-order ties and only legal
  resources. Robber and development choices use numeric target/victim ties,
  missing victims last, resource lexicographic order, then original menu position.
  Legacy `DISCARD` uses the same canonical choice, without payload inference.
- Repeated calls across Python/NumPy/Torch seeds, deterministic flags, and supplied
  memory select the same original action, return no memory, preserve the input
  and RNG state, and have no parameters/buffers/state-dict entries.
- Real `evaluate_blocks` and `_MatchDriver` run the module through `act` in the
  scripted fixture: eight rejects across four game resets, incoming memory
  always `None`, policy in evaluation mode during decisions, original mode and
  external RNG restored afterward. Candidate counts contain no accept/maritime.
- Self-review found only the pure pip/canonical helpers and literal priority
  loops: no teacher delegation, utility, frontier search, resource weighting,
  jitter, lookahead, learned value, hidden engine access, or retained game state.

## Single bounded real-engine smoke

Exactly one standalone real-engine smoke was run, separate from the existing
suite. It invoked:

```python
start = time.monotonic()
result = evaluate_blocks(
    GreedyNoTrade(), board_seeds=[41009], candidate="graph", baseline="heuristic",
    deterministic=True, max_game_steps=120, deadline=start + 20,
)
```

Evaluation elapsed time: `0.4845861669164151` seconds, below the 20-second
configured deadline and the 30-second task bound. Catanatron was 3.2.1.
Four protocol games completed at the step cap: zero terminations, four
truncations, no interrupted game, and `win_rate=None`. These are integration
checks, not resolved competence or benchmark evidence.

Candidate counts (114 total decisions):

```json
{"BUILD_ROAD": 10, "BUILD_SETTLEMENT": 8, "DISCARD_RESOURCE": 26,
 "END_TURN": 30, "MOVE_ROBBER": 4, "REJECT_TRADE": 5, "ROLL": 31}
```

Explicit zero-count assertions passed for `ACCEPT_TRADE` and `MARITIME_TRADE`.
Smoke summary is `/private/tmp/greedy-no-trade-smoke.json`. No full fixed-panel
run or performance-based rule changes occurred.

## Freeze and remaining concerns

SHA-256 after final formatting, before any benchmark:

```text
cc2482ecd73c0dea1a5f5d9a5b7b35aa3329a059c61f08ae701446e204f7ef45  src/colonist_rl/greedy_no_trade.py
ee70226d81bc0224f1d4f39fb0413a9c357da9a652cbdadf313072ef949038ec  tests/test_greedy_no_trade.py
6ec6c8ad1835135a2fdde9aea94f9f168c0c5d8c84aa6a6131497f36b696ac6e  docs/greedy-no-trade.md
```

No blocking interface concerns found. Canonical ties assume normalized legal
kinds carry their contract-defined fields (for example resource discards have a
valid resource); the policy does not repair malformed observations. Strategic
limitations are deliberate frozen rules, not implementation issues. The smoke
was wholly censored and makes no win/strength claim. Independent review remains
required before the coordinator starts the actual 192-game shared-opponent
panel. The existing evaluator's generic `graph` label must be qualified by
`plain-greedy-no-trade-v1` in its run manifest.
