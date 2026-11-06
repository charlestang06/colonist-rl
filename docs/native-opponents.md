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
characterization. No actual VF/AB decision was run during preparation. The broader 38-test
verification command also included a GraphPolicy checkpoint/dataset identity
smoke test with real collect_demonstrations simulator activity; exact operation
counts were not instrumented. Those side effects are not native seam evidence.

## Separate focal native reference evaluation

`evaluate_native_reference_blocks(reference=..., board_seeds=...,
opponent_lineup=..., deadline=..., max_game_steps=20000, no_offers=True,
on_game_end=None)` is an evaluation-only entry for a focal VF/AB seat. It rotates
all four seats per board using the same complete-block, deadline, censoring and
final-game callback semantics as evaluate_blocks. Reference must be an exact
native ID; deadline and a three-entry background lineup are explicit. The first
protocol should declare a fixed non-native background such as three
GreedyNoTrade actors, rather than confuse this with a three-native stress panel.

This entry accepts no policy/model, historical policy pool or candidate-step
callback. A runner is owned locally after reset and selects the focal seat action;
only the existing match driver steps the environment. Native IDs/runners remain
rejected by ordinary candidate and training routes. No Observation-policy wrapper,
teacher route, training-label callback or native root-action trajectory is added.

Public evaluate_blocks gains an opt-in strict boolean no_offers keyword. Set it
true in the matched public-candidate arm, and use the identical background lineup,
boards, all focal seats, source, step bound and external resource limits. Reference
evaluation requires true. This suppression works when only the focal seat is
native, because it does not depend on the driver's native-opponent dictionary.
The returned no_offers field describes effective suppression, including the
pre-existing automatic suppression when native background opponents are present.
Legacy default routing and public-candidate synthetic-response decisions remain
unchanged; reject_trades is not repurposed as a no-offer control.

Focal native planning seed is board_seed*97+focal_seat. Native algorithms/defaults,
copy/RNG isolation, original root order, legal-member conversion, discard/reject
shims, native82 chance mismatch and full_rule_equivalence=false are unchanged.
Matching environment seeds does not make trajectories identical after differing
actions. The reference retains privileged native state, while the public candidate
retains Observation-only inputs; this is not an equal-information comparison.

Reference results declare focal_role=native_reference, focal_identity,
evaluation_only=true and training_labels_allowed=false. Completed and interrupted
rows carry separate native_focal metadata/metrics; native_opponents still describes
only background seats. Evaluation-local metadata copies identify factual focal
and background roles/information regime without changing the runner algorithm or
its default opponent metadata. Unavailable search/timeout work remains null.
Only detached final diagnostic primitives are exposed through on_game_end, with
no step-label callback. Reference outcomes and diagnostics remain EVAL-only.

Native decisions can overrun the absolute between-decision deadline; no overrun
action is stepped. Existing Root hard wall/RSS guards, one native/Torch thread and
global concurrency constraints remain external. Compare candidate-minus-VF and
candidate-minus-AB separately after binding matching protocols and complete
board/seat outcomes; the existing paired outcome helper alone does not certify
protocol equivalence. Preserve sealed final boards until the confirmation freeze.

Focused tests/test_native_focal_evaluation.py uses fake game/driver/native/public
actors and blocks authentic engine/native/NN/process side effects. Collection
imports existing runtime dependencies, so Root owns its guarded execution. This
source extension itself launches no workers or tests.
