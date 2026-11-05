# Public position value runtime

`public_position_value.py` contains inference only. It scores the actual root
Observation plus a `ProjectedTurn`; it never constructs an Observation, queries
an environment, invents a future menu, fits parameters, or imports an optimizer.

The encoder preserves the reviewed value-lane/v3 `value_pilot.py` encoder semantics
(source SHA256 `8940a6edae7c67657cff61263c5c89dd02b4c74a1182ad3970c39a5b0db031b5`).
Frozen pilot journals still pin that external source. Runtime tests check actual
features and explicit projection overlays, including known versus public VP,
piece stock, development cards, production, ports, connectivity and opponents.
Bank, deck and unknown opponent hands are omitted. Projection features retain
root public opponent evidence and do not resolve longest-road ownership. They do
not certify a simulator afterstate. No playing-strength result follows from the
unit tests or the declared prediction target.

The only accepted model kinds are `public_linear` and `calibrated_handcraft` from
the pilot `models.json` bundle. Models must declare schema
`public-position-value/2`, target `resolved_eventual_win_probability`, and
continuation `evaluation-H-candidate-no-proposals-vs-3H-with-proposals/1`.
This is eventual learner win under the declared H continuation, not a value
trained under this planner's own continuation. Linear columns must exactly match
the canonical public feature list. The calibrated control accepts only
`handcraft_score`, in original score units (raw planner integer / 1,000,000).
Array dimensions, finite numbers, positive scales and supplied fit metadata are
validated before coefficients are copied into immutable tuples. Inference uses
the same standardized logistic equation and [-35, 35] logit clipping as v3,
returning rounded integer probability millionths in [0, 1,000,000].

```python
from colonist_rl.public_position_value import probability_planner
from colonist_rl.public_turn_planner import PublicTurnPlannerPolicy

learned = probability_planner("/path/to/models.json", kind="public_linear")
calibrated = probability_planner("/path/to/models.json", kind="calibrated_handcraft")
raw = PublicTurnPlannerPolicy()

plan = learned.plan(actual_view)
assert plan.first_action is actual_view.legal_actions[plan.root_index]
action = learned.act(actual_view)  # ordinary policy boundary; reobserve next call
```

All three use the same existing depth 6, nodes 2048, beam 24 and geometry budget.
Both probability factories set a fixed gain margin of 10,000 (one percentage
point). Raw handcraft retains the original 250,000 raw-score margin. The planner
still requires strict gain above the margin, retains its known-win priority,
authoritative root validation, fallbacks, ordering, and resource evidence rules.
The factory also accepts an explicit `TurnPlannerConfig` for equal-budget studies.

`plan_turn(..., evaluator=fn, gain_margin_micro=n)` and
`PublicTurnPlannerPolicy(..., evaluator=fn, gain_margin_micro=n)` provide the small
public hook. `fn(actual_root, projected, raw_handcraft_micro)` must be pure and
return an integer score; invalid scores raise `ValueError`. The caller must use
a margin in the evaluator's score units. `TurnPlan.evaluator_calls` counts actual
calls; `evaluator_time_ns` sums their wall time. Both are zero for the default raw
path. Timing is diagnostic and never participates in search ordering or budgets.
`policy.plan(view)` returns this report while `act(view)` returns only the original
root-menu Action. No global score patch is needed.
