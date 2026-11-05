# Native opponents at the trusted evaluation boundary

`evaluation.evaluate_blocks` accepts `baseline="native_vf_82aae93"` or
`baseline="native_ab2_82aae93"`. An explicit three-entry `opponent_lineup` can mix
these identities with existing policies; entries are relative to candidate seat.
Native identities are rejected as candidate names and as training lineups. A
`NativeOpponentRunner` supplied as a candidate policy is also rejected. The
existing learner/teacher Observation interface is unchanged.

The new runner is constructed after simulator reset with
`env.game.state.colors[opponent_seat]`, an independent integer planning seed, and
the selected exact native class. `select_action()` takes no learner input and
returns the actual matching member of the current immutable wrapper legal tuple.
It does not execute the engine: the driver executes only through `env.step`.
Conversion uses `env._convert`. Native order is preserved for both the passed
root list and the copied game's own list (the native AB implementation reads the
latter). A deep copy protects real state, actions, ordered deck, and nested action
records. Native planning saves/restores module-global random under the simulator's
shared lock, using its own stream; it never reads or enters `env._rng_state` or
`env._randomness`. Driver planning seeds are `board_seed * 97 + opponent_seat`.

This adapter is an **asymmetric stress test: public-only candidate versus native
privileged-state opponents**. Raw native objects and native decisions must not be
used as candidate observations, learner search inputs, training targets, or labels.
Native implementations can inspect real hidden hands, development deck and action
records inside their isolated copies. This is not an equal-information comparison.

VF delegates the exact `ValueFunctionPlayer` with base defaults and epsilon=None.
AB delegates the exact `AlphaBetaPlayer`, depth=2, prunning=False, base default
weights and epsilon=None. Backend provenance must be 3.3.0 at full Git commit
82aae93ab1f7c267218be0566df573ce477ec3d8; 12 independently audited source hashes
must also match. Module hashes and configuration are emitted per opponent in each
game row. The dependencies already installed are sufficient; upstream license is
GPL-3.0-or-later, and this repository declares GPL-3.0-only. No source is vendored.

The adapter identifies itself as
`<native_identity>+atomic_discard_reject_trade_v1`; full_rule_equivalence is false:

- Domestic proposals are disabled for all seats whenever a lineup contains a
  native opponent. Native classes have no invented proposal head. Synthetic
  trade responses use an explicitly reported reject shim.
- Native DISCARD is forced random bulk discard. The adapter samples with
  `random.sample` from that actor's own visible hand using the planning stream,
  caches only those resource names and public version/count identifiers, then
  selects wrapper DISCARD_RESOURCE members one card at a time. Native search
  still models bulk discards. Missing remaining counts or discontinuous sequences
  fail closed; native decisions are not fabricated at this phase.
- Native engine negotiation remains unsupported by the simulator. Illegal native
  returns or conversions absent from current wrapper legality fail closed. No H
  or Greedy substitution occurs.
- The authentic native82 roll/development Action.value / ActionRecord.result
  mismatch is preserved, not repaired or described as exact chance enumeration.

Native evaluations require an explicit absolute monotonic `deadline`. Existing
between-decision checks classify overruns as incomplete and retain native metrics
in the interrupted row. AB's internal 20-second deadline is soft, checked at
recursion entry, and spectrum expansion can exceed it. An external worker hard
wall/memory cap is still required; this adapter does not provide process isolation.
An externally killed run is censored, never evidence of completion. Search counts
and internal timeout counts are unavailable (`null`), not zero. Per-seat native
metrics report native/nonforced decision counts, shim counts, per-phase count and
planning seconds, and total planning seconds, separately from candidate latency.

CLI `simulate` and `evaluate` expose the two baseline identities plus
`--candidate {heuristic,random,greedy_no_trade}`, three-entry `--opponent-lineup`,
and `--wall-seconds`. A supplied evaluation checkpoint retains the graph candidate
path. Legacy defaults and positional APIs remain intact.

Preparation verification uses fixture native players only. It checks exact
constructor delegation, all seat colors, preserved root order, authoritative
conversion routing for build/robber/maritime/dev/forced actions, membership
rejection, backend and both RNG streams, discard/reject shims, default driver
routing, source identity, and candidate/training exclusion. It does not establish
real engine conversion correctness or baseline playing strength. Before any game
benchmark, separately schedule real-engine seam checks and native direct-copy
parity under identical isolated RNG; cover the real discard and synthetic trade
phases, all coordinate/resource conversions, immutability, and forced chance
characterization. No actual VF/AB decision or real game was run during preparation.
