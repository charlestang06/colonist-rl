# Trade plugin implementation report

Implemented `src/colonist_rl/trades.py`, its usage guide in `docs/trades.md`, and
44 trade-focused tests in `tests/test_trades.py`. No simulator, engine, learning,
or contract files were modified by this worker.

Public API:

- `TradeGenerator.propose(player_view, context, max_candidates)` returns offer records.
- `TradeCritic.score(player_view, context, offers)` returns one immutable assessment per offer.
- `TradeController.choose(player_view, context)` returns an immutable decision.
- `TradeOrchestrator` implements the controller, with disabled-by-default,
  independently enabled models and an optional boolean policy callback.
- `validate_trade_offer(offer, player_view, context)` validates an offer from
  the acting player's public view.
- `validate_trade_decision(decision, latest_view, latest_context)` provides the
  required execution-time version/identity/terms/affordability recheck.

Example integration:

```python
controller = TradeOrchestrator(
    generator=my_generator,
    critic=my_critic,
    enabled=True,
    binary_callback=lambda view, ctx: learned_policy_accepts(view, ctx),
    include_assessments=True,
)
decision = controller.choose(observation, observation.context)
# The simulator obtains latest_view inside its atomic commit boundary.
if validate_trade_decision(decision, latest_view, latest_view.context):
    simulator_handle_decision(decision)
```

An incoming-only optional critic uses
`TradeOrchestrator(critic=my_critic, critic_enabled=True)`. A critic may also run
standalone against the immutable public records; it does not require a PPO
critic, simulator, or learning-network inputs. Assessment benefit is expected
benefit relative to the no-trade baseline; optional acceptance probability and
benefit conditional on acceptance remain separately available diagnostics.

Validation covers five integer, nonnegative resource entries, positive amounts
on both sides, disjoint resource kinds, known distinct parties, own inventory,
turn/roll/eligibility for proposals (including simulator `PLAY_TURN`), and exact live offer matching for responses.
Incoming own inventory uses the offer's recipient-normalized direction.
No resources are changed by models or the controller. Opponent exact inventory
must only be inspected inside the simulator's atomic commit boundary.

The default candidate cap is 32, and only 4 times that many source items may be
inspected. Duplicate IDs or terms are filtered without exhausting unbounded
iterables. The no-trade candidate is always present with benefit zero and wins
ties. The shared model budget defaults to 0.1 seconds; incoming scoring reserves
a quarter of it for the callback. Daemon workers and one in-flight lock per
model/callback stage bound controller latency without interpreter shutdown waits
or accumulating stuck workers. Timed-out results cannot update a returned
frozen decision. A Python thread does not terminate a stuck native extension or
sandbox untrusted code; process isolation remains appropriate for that use case.

Test-first evidence:

```text
.venv/bin/python -m pytest -q tests/test_trades.py
# Initial red: missing colonist_rl.trades module (collection failure).
# Follow-up regression red: no-trade diagnostics absent when plugins disabled.
# After implementation:
............................................                             [100%]
44 passed in 0.07s
```

Final focused verification:

```text
.venv/bin/python -m pytest -q tests/test_contracts.py tests/test_trades.py
..............................................                           [100%]
46 passed in 0.07s

.venv/bin/python -m ruff check src/colonist_rl/trades.py tests/test_trades.py
All checks passed!

.venv/bin/python -m ruff format --check src/colonist_rl/trades.py tests/test_trades.py
2 files already formatted
```

Tests include varied list/yielding generators and independent critics;
standalone incoming scoring; disabled models; positive/negative/tied expected
benefits; immutable assessments; invalid numeric diagnostics; candidate caps;
unbounded duplicate generation; malformed vectors/parties/contexts; stale
versions and changed IDs or terms; no hidden-hand input; own-hand direction and
no mutation; model exceptions and malformed scores; and blocked generator,
critic, and callback behavior.

Integration follow-up: independent review exposed a backend phase-name mismatch.
The real simulator emits `PLAY_TURN`, while the abstract fixtures used `TURN`.
The controller now accepts both canonical names. A regression test first failed
with `NO_OFFER` and zero generator calls on an actual funded post-roll simulator
view, then passed after the fix. It verifies a generator call, a validated
`PROPOSE` decision, and acceptance into the simulator's pending-offer boundary.

```text
.venv/bin/python -m pytest -q tests/test_trades.py -k real_post_roll
1 passed, 44 deselected in 0.08s
```
