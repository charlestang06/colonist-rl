# Optional trade models

`colonist_rl.trades` has no simulator or neural-network dependency. It consumes
immutable `Observation` and `DecisionContext` records. The observation contains
only the actor's exact hand, public information, and player-visible beliefs and
events. A trade model must never receive an engine object or opponents' hands.
The trade critic is independent of the PPO state-value critic.

The three structural protocols are:

```python
class TradeGenerator(Protocol):
    def propose(self, player_view: Observation, context: DecisionContext,
                max_candidates: int) -> Iterable[TradeOffer]: ...

class TradeCritic(Protocol):
    def score(self, player_view: Observation, context: DecisionContext,
              offers: Sequence[TradeOffer]) -> list[TradeAssessment]: ...

class TradeController(Protocol):
    def choose(self, player_view: Observation,
               context: DecisionContext) -> TradeDecision: ...
```

A critic returns exactly one immutable `TradeAssessment` for each supplied offer,
with the original offer record. `benefit_relative_to_no_trade` is the actor's
expected benefit after accounting for acceptance, in the model's documented
units. It is a finite float. `uncertainty` is a finite nonnegative float.
`acceptance_probability` may be omitted, or lie in `[0, 1]`.
`benefit_if_accepted` is optional and finite. These last three fields are reported
as diagnostics; the default controller selects the largest expected benefit.

The controller always includes a zero-benefit no-trade candidate, represented
by an assessment whose `offer` is `None`. A negative or exactly zero trade score
selects no trade. Offers supplied to the critic retain the proposer's direction;
use `offer.for_player(player_view.player_id)` to obtain the actor's give/receive
vectors. Incoming affordability uses this same normalization.

`TradeDecision` is immutable and has `kind` (`ACCEPT`, `REJECT`, `PROPOSE`, or
`NO_OFFER`), `offer`, `context_version`, and an optional tuple of `assessments`.
`include_assessments=True` retains diagnostics, including the no-trade baseline.
The default leaves this tuple empty.

## Enabling models

```python
from colonist_rl.trades import TradeOrchestrator, validate_trade_decision

# generator and critic are any objects implementing their respective protocols.
controller = TradeOrchestrator(
    generator=generator,
    critic=critic,
    enabled=True,
    max_candidates=32,
    timeout_seconds=0.1,
    binary_callback=lambda observation, context: policy_accepts(observation, context),
    include_assessments=True,
)
decision = controller.choose(observation, observation.context)

# Obtain a fresh observation immediately before the engine's atomic commit.
if validate_trade_decision(decision, latest_observation, latest_observation.context):
    engine_handle_trade_decision(decision)
```

The callback must return an actual `bool`: `True` accepts, `False` rejects.
Adapt a learned policy's action to a boolean before returning. No callback means
reject. A callback exception, malformed result, or timeout also rejects.

Models are disabled by default, even if supplied. They can be enabled
independently: `TradeOrchestrator(critic=critic, critic_enabled=True)` assesses
incoming offers without any generator. Explicit `generator_enabled` and
`critic_enabled` values override the common `enabled` setting. Outgoing proposals
require both an enabled generator and an enabled critic; a missing model returns
`NO_OFFER`. An incoming critic failure uses the binary callback. An outgoing
model failure returns `NO_OFFER`.

A critic can also be used directly: `critic.score(view, context, offers)` requires
only these public records and can be backed by an independent network, a
heuristic, or an external service. Direct callers are responsible for validating
the inputs and outputs; the orchestrator provides that boundary.

## Validation and limits

Both resource vectors must have five nonnegative integer entries, each side
must contain at least one resource, and no resource can be both given and
requested. The actor must have the resources it gives. Both parties must exist
and differ. An outgoing proposer must be the current rolled turn owner, the
context must be `TURN` or the simulator's `PLAY_TURN`, and the recipient must be an eligible partner. An
incoming offer must exactly match the live `TRADE_RESPONSE` offer, including
its ID, parties, and terms. Only the actor's exact inventory is checked here;
the engine must check both inventories when it atomically executes a trade.

Generator output is validated and deduplicated by ID and by terms. The default
candidate limit is 32; at most `4 * max_candidates` source items are inspected,
even when an unbounded generator repeatedly emits invalid or duplicate offers.
`max_input_items` can override this inspection limit. Collection stops as soon
as the candidate limit is met and never exhausts an unbounded generator.

A decision budget defaults to 0.1 seconds for all stages combined. Incoming
scoring reserves 25% for the binary callback when present. Each plugin call runs
on a daemon worker; a stuck call cannot make shutdown wait. A per-controller
in-flight lock prevents repeated timed-out model calls from accumulating workers;
a separate lock protects the binary callback. A timed-out result is abandoned
and cannot alter the returned decision. Python threads cannot terminate a stuck
model or enforce a hard CPU/memory sandbox. Process isolation is appropriate for
untrusted plugins; the thread budget is a latency/fallback boundary for local
trusted models, with ordinary scheduling overhead.

Neither the controller nor a model transfers resources. Revalidation must use
the latest observation and happen at the engine's atomic commit boundary.
A changed version or a changed incoming offer invalidates the old decision,
even when the replacement offer reuses its ID. This rejects late responses.
