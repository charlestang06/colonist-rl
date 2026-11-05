# Core state implementation report

## Delivered interfaces

`src/colonist_rl/board.py` contains frozen `Tile`, `Port`, `BoardSpec`, and
`RulesConfig` records. Nested topology sequences are normalized to tuples even
when the direct constructor receives lists.

- `BoardSpec.from_hexes(coordinates, resources=None, numbers=None)` preserves
  input tile order. Tile, vertex, and edge ids are dense tuple indices.
- Integer pointy-hex centers are `(2*q+r, 3*r)`. Clockwise corner offsets start
  north: `(0,-2), (1,-1), (1,1), (0,2), (-1,1), (-1,-1)`.
- `tile_edges[t][i]` joins `tile_vertices[t][i]` to the next clockwise vertex.
- `BoardSpec.standard(seed=0)` has 19 tiles, 54 vertices, 72 edges, and 9 ports.
- `BoardSpec.extended(seed=0)` has 30 tiles, 80 vertices, 109 edges, and 11 ports.
  Its explicit row lengths are 3, 4, 5, 6, 5, 4, 3. It is a deterministic usable
  extension topology, not a claim about a backend's tile numbering.
- Generated six/eight number tokens are separated by at least one hex edge.
- `RulesConfig.standard(player_count=4)` has bank counts `(19,)*5` and 25 dev
  cards. `RulesConfig.extended(player_count=6)` has `(24,)*5`, 34 dev cards,
  versioned special-building support, and a `special_build` phase.

`src/colonist_rl/beliefs.py` contains `BeliefLedger` and
`BeliefInconsistency(ValueError)`:

```python
BeliefLedger(known_hands=None, totals=None, *, player_count=None,
             seed=0, max_particles=256)
ledger.apply_delta(player, signed_delta)
ledger.transfer(source, target, count, resource=None)
ledger.discard(player, count, known=None)
ledger.reconcile(totals, own_player=None, own_hand=None)
ledger.summaries()  # tuple[ResourceBelief, ...], ascending player id
```

Known hands are mapping entries of actually visible information. Totals may be
a mapping or sequence. `player_count` alone starts a game with all empty hands.
Player ids may be noncontiguous. Diagnostic properties `particles`, `weights`,
`exhaustive`, and the `rejuvenations` counter are available.

Hidden card transfers branch joint states with the victim's per-card resource
probabilities. For `(3,1,0,0,0)`, a one-card hidden theft has probabilities 3/4
wood and 1/4 brick. Conditioning the thief's observed hand also conditions the
victim's hand. Public costs, known transfers, and visible discards filter all
joint states. Contradictory evidence raises without changing ledger summaries.

Finite support is enumerated exactly when it fits the particle budget. Larger
support is sampled deterministically from the supplied seed. Public resource
bounds and known global inventories are propagated separately from samples.
`exact=True` requires logically equal bounds; agreement among sampled hands is
never sufficient. Missing-history initialization keeps conservative bounds.
No simulator objects or hidden simulator hands are accepted by this subsystem.

## Test-first evidence

The first board test run failed because the module was absent:

```text
.venv/bin/python -m pytest tests/test_board.py -q
ModuleNotFoundError: No module named 'colonist_rl.board'
```

The first belief test run likewise failed on the missing module:

```text
.venv/bin/python -m pytest tests/test_beliefs.py -q
ModuleNotFoundError: No module named 'colonist_rl.beliefs'
```

Additional tests exposed mutable nested topology inputs and adjacent red
number tokens before those behaviors were implemented (2 failed, 16 passed).
A separate port-mutation test also failed before endpoint normalization.

Final scoped verification:

```text
.venv/bin/python -m pytest tests/test_contracts.py tests/test_board.py tests/test_beliefs.py -q
.....................                                                    [100%]
21 passed in 0.02s

.venv/bin/python -m ruff check src/colonist_rl/board.py src/colonist_rl/beliefs.py tests/test_board.py tests/test_beliefs.py
All checks passed!
```

Tests cover shared hex corners/edges, both board sizes, tile-order preservation,
metadata validation, mutable-input isolation, port endpoints, variable player
counts, visible accounting, 3:1 hidden-theft likelihood, joint conditioning,
nonnegativity, conservation, bounded deterministic sampling, false-certainty
prevention, public spending evidence, and own-hand reconciliation after sample
depletion.

## Limits

Capped-particle means and variances are estimates. Hidden discards use a
card-proportional prior; actual chosen discard strategy is not inferred.
Missing-history hand compositions use an uninformative composition prior rather
than knowledge of how resources were acquired. If conditioning eliminates a
nonexhaustive sample, joint rejuvenation retains current public bounds and known
global resource conservation, but may broaden higher-order historical
correlations and changes the approximate posterior; `rejuvenations` makes this
observable. It never turns that approximation into an exactness claim. Reconcile
requires event-updated public totals; it raises on a missed count-changing event
instead of fabricating a card flow. Bounds are conservative and may remain wider
than all feasible histories after sampling. Stock layouts are independent of the
engine's own random-board generator and use their own versioned numbering.
