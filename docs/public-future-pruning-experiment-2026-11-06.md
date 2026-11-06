# Public future pruning: frozen mechanism and screen

`TurnPlannerConfig(prune_invalid_future=True)` enables this experiment. The
default remains `False`. For generated **future** build and maritime intentions,
the planner skips a candidate only when the projected hand lacks a known build
cost or trade give, or conservative `bank_lower` lacks the trade receive. The
candidate stream and its order stay unchanged. Root menu validation and the
full `_transition` checks remain authoritative. No projected intention becomes
an executable action; the selected action is still an original legal root.

`pruned_invalid_candidates` counts skipped future candidates separately from
charged attempts. `nodes = root_attempts + future_attempts =
feasible_transitions + invalid_transitions`, and `feasible_transitions =
unique_successors + duplicate_successors` still hold. Pruning may improve
coverage under a node cap, so capped scores and choices may differ. An uncapped
search must retain the same feasible search tree, plan, and scores.

For a paired screen, freeze board seeds, seats, opponents, planner settings,
and node/geometry caps for off/on arms. Report complete and resolved coverage,
action and plan differences, attempts, feasible and skipped counts, plus
`searched_selection_latency_ms` (count, mean, p95) per board and arm. This
latency subset includes only planner selections whose `search_stop` is neither
`not_searched` nor absent; `searched_plan_reason_counts` exposes its mix. Keep
`selection_latency_ms` for all selections as a separate descriptive measure.
Promote the flag only after a completed paired screen shows useful time or
feasible-coverage gain without a material decision regression. This branch
provides pure and synthetic checks only, not gameplay evidence.
