# Grounded public residual value runtime

This runtime tests a bounded correction to an old H-continuation prior. It does
not establish calibration or playing strength for the collecting policy. The old
position encoder, old model format and planner are unchanged.

`encode_productive_geometry(view, projected=None)` uses public board geometry and
own piece stock, with explicit `ProjectedTurn` overlays. It never constructs an
Observation, inspects `legal_actions`, certifies a future menu, or infers road
awards. Geometry is independent of phase and current hand affordability.

The following eight added features are fixed before collecting any data. Pips are
nominal number-token weights `6 - abs(7 - number)` summed over productive adjacent
tiles. Nominal own production is building level times those pips divided by 36,
per resource. Both include the temporarily robber-blocked tile.

| Column | Meaning and units |
| --- | --- |
| `pg_connected_sites` | Count of vacant own-road endpoints outside every building's distance exclusion; zero without settlement stock. |
| `pg_connected_best_pips` | Best nominal pip sum among those sites; zero when absent. |
| `pg_one_road_new_sites` | Union count of NEW distance-legal endpoints reached by one unoccupied publicly connectable road, excluding already connected sites; zero without road or settlement stock. |
| `pg_one_road_best_pips` | Best nominal pip sum among those new sites; zero when absent. |
| `pg_expansion_edges_without_site` | Count of unoccupied publicly connectable edges with neither endpoint an immediate distance-legal settlement destination. All such edges count when settlement stock is exhausted; zero without road stock. These edges can support later expansion. |
| `pg_owned_port_utility` | Sum over resources of nominal own production times `(1 / owned_ratio - 1 / 4)`. Units: expected maritime receive-card equivalents per roll relative to bank 4:1. |
| `pg_connected_port_gain` | Best incremental utility at one connected site, replacing each owned ratio by the better of that ratio and the site's port ratio. Same units; current own production only. |
| `pg_one_road_port_gain` | Best incremental utility at one NEW one-road site. Same units and production support. |

Spacing excludes occupied vertices and all neighbors for every owner. Existing
opponent buildings block road continuation; occupied roads cannot be added.
One-road sites need no claim about another player's next action. Port potential
does not assume additional production from the prospective settlement and does
not add gains across mutually exclusive site choices.

`encode_residual_features(view, projected=None, include_productive=False)` reuses
the unchanged `encode_position` result and optionally adds this block.
`expected_columns(include_productive=False)` returns the exact sorted column
tuple for fitting and validation. There are 63 old columns or 71 combined columns
on four-player Observations. The old encoder still retains its original raw
`connected_sites` and `road_frontier` features alongside the added stock-aware
columns. Bank contents, belief vectors, events and longest-road details are not
inputs. Known own VP includes own hidden VP plus certain projected building gain;
public own VP excludes hidden VP. Opponent aggregate public counts remain inputs.

`PublicResidualEvaluator(model)` accepts one direct JSON object, with exactly:

```text
schema: "public-residual-value/1"
target: "resolved_eventual_win_probability"
continuation: "cap-raw-no-proposals-vs-3H-with-proposals/1"
include_productive: boolean
columns: list(expected_columns(include_productive))
mean, scale, coef: numeric lists of exactly the column length
intercept: finite number
residual_bound: 0.5
anchor:
  source_sha256: "9201d6911bb1c2a7a2d66e88022c603199ece240bfa8a100e11188b9f59d6520"
  entry: "calibrated_handcraft"
  model: exact unchanged calibrated_handcraft entry from pilot-001/fit/models.json
```

The anchor bundle is `agent-research/value-lane/pilot-001/fit/models.json` in the
external study directory. The canonical entry JSON (sorted keys, compact
separators, finite numbers) SHA256 is
`036644b1d8f847ce88afe0a734f28432fa0aa080293e1bc83d4cb352727d82e7`.
The embedded anchor retains old schema `public-position-value/2` and continuation
`evaluation-H-candidate-no-proposals-vs-3H-with-proposals/1`. Validation checks
both provenance identities and the exact entry fingerprint; a new continuation
cannot relabel a different prior. No fit metadata enters this strict runtime
object; keep optimizer/data manifests beside it.

All numeric parameters must be finite non-boolean numbers; every scale is
positive. Columns, field sets and array lengths are exact. Parsed tuples and the
anchor are immutable. Inference computes
`sigmoid(z_anchor + 0.5 * tanh(intercept + coef @ standardized_features))`, with
the old calibration's original handcraft-score units and final probability in
integer millionths. It uses the old numerical sigmoid clamp of ±35 logits and
micro rounding. Zero residual matches the old calibrated evaluator.

`residual_planner(path)` loads this direct model and returns the ordinary
`PublicTurnPlannerPolicy`, with depth 6, nodes 2048, beam 24, geometry visits
100000, seed 11 and strict probability gain margin 10000. The ordinary policy's
proposal behavior remains intact; evaluation must separately disable candidate
proposals for the declared continuation.

Added geometry uses linear public scans and shared exclusion/connectivity
helpers, without per-edge projected-state construction. It reruns at evaluator
calls, alongside the unchanged older encoder. This work is measured by existing
`evaluator_time_ns`, outside the planner's geometry counter and cap; unchanged
search budgets do not imply unchanged wall time. No runtime benchmark, trained
model probe or strength claim is provided by synthetic unit tests. Runtime code
adds no optimizer, NumPy, Torch, environment or training imports; the existing
planner dependency retains its prior transitive imports.
