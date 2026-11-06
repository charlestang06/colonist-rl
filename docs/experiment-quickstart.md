# Fast paired experiment loop

Use one JSON manifest to compare public policies on the same board seeds and
four focal seats. The [paired CLI](../src/colonist_rl/experiment_cli.py) calls
the existing evaluator; the [analysis command](../src/colonist_rl/experiment_analysis.py)
reads saved artifacts without replaying games. The
[first search-budget screen](search-budget-results-2026-11-06.md) is a complete
worked example. The [purchase-option](public-option-purchase-results-2026-11-06.md)
and [future-pruning](public-future-pruning-results-2026-11-06.md) screens show
changed-decision and timing readouts.

## 1. Freeze one question

Start from the archived [budget manifest](../runs/swarm-2026-11-04/agent-research/search-budget-001/screen-manifest.json)
and give the new study **new, disjoint board seeds**. Set `role` to `smoke`,
`development`, `validation`, or `final_test`; this declaration does not check
other archives for reused seeds. Require at least two arms and name the control
with `reference_arm`. A public planner arm records every `TurnPlannerConfig`
field and `gain_margin_micro`. The `city_before_purchase` and
`public_option_purchase` arms are opt-in. A new public planner defaults to
`prune_invalid_future: true`; set it explicitly to `false` to reproduce the
older charged-invalid search. A
`graph_checkpoint` arm requires an exact checkpoint SHA-256. The same evaluator
supports three named random, heuristic, greedy-no-trade, or pinned native
opponents in the lineup. This CLI currently requires `no_offers: true` and has
no historical-policy pool.

The control can itself be a `graph_checkpoint`: matched parent and trained
checkpoints can run as a neural-only panel. The declared `reference_arm` runs
first, and every checkpoint keeps its own exact SHA-256 binding. A
[one-board engine smoke](graph-only-paired-smoke-2026-11-06.md) verified this
path with two identical parent checkpoint arms and matching public traces.

Keep the hypothesis, changed configuration field, control, and stop rule in a
small protocol before running. Begin with one excluded smoke board, then a
modest fresh development panel. Reserve independent validation and final
boards for a frozen promising candidate. Changed actions, early/mid
predictions, and teacher agreement are diagnostics, not playing-strength wins.

## 2. Run and inspect

From the repository root in an environment with the pinned dependencies:

```sh
PYTHONPATH=src python -m colonist_rl.experiment_cli \
  --manifest /absolute/path/to/new-manifest.json \
  --output-root /absolute/path/to/experiment-runs

PYTHONPATH=src python -m colonist_rl.experiment_analysis \
  --run-dir /absolute/path/to/experiment-runs/paired-... \
  > /absolute/path/to/analysis.json

# Add first-divergence turn paths when candidate steps were captured.
PYTHONPATH=src python -m colonist_rl.experiment_analysis \
  --run-dir /absolute/path/to/experiment-runs/paired-... \
  --followthrough --detail-limit 32 \
  > /absolute/path/to/followthrough-analysis.json
```

The CLI prints the new run directory. It writes `manifest.json`, an append-only
`games.jsonl`, and `summary.json`. Set `capture_candidate_steps: true` when
first action divergences and public after-action confirmation matter; then it
also writes `candidate_steps.jsonl`. Each completed game row is synced after
its captured steps. The `wall_seconds` deadline starts after preflight, is
checked between evaluator decisions, and is a soft gameplay limit. An
incomplete or invalid panel returns nonzero but preserves finished journals.

The run records the input and expanded manifest, project source-tree SHA,
pinned engine installation identity, optional checkpoint hashes, and checks
source/manifest/checkpoint identity again after play. The analyzer verifies
saved artifact digests and the declared reference. It compares only complete
boards shared by an arm and its control. A win-rate difference is present only
if every game in those matched boards resolved; search-work totals remain
descriptive over attempted selections. Trace differences identify where play
first diverged, not which choice was optimal. The `search_work` section keeps
searched-decision latency and reason counts separate from latency over all
policy selections. A lower attempted-node count alone is not a runtime gain;
compare timing and coverage on the same boards.

The optional `--followthrough` report uses the same verified saved artifacts.
For each matched seat whose **first** divergence is a different action in the
same captured state on the focal player's own turn, it records both action
sequences through that turn, their first city or settlement, whether the next
action is a point build, and their known victory points at a recorded turn end.
The aggregate covers every eligible pair even when `--detail-limit` omits case
details; it reports the eligible, omitted, complete and incomplete counts.
Known turn-end points are `null` if capture ends without an `END_TURN`, a
recorded later turn, or termination. Setup and opponent-turn divergences are
unassessed. Final saved wins appear beside each case only as later outcomes;
neither those wins nor the turn-end point comparison establish which first
action was better. `--followthrough` requires `capture_candidate_steps: true`.

For at least eight matched complete boards, each comparison also reports a
deterministic 20,000-resample percentile interval for paired win-rate and mean
victory-point differences. It resamples whole four-seat boards and omits a win
interval if any matched game is unresolved. With fewer boards it explicitly
reports `insufficient_boards`. These are descriptive uncertainty readouts, not
automatic promotion tests.

## 3. Decide quickly

Read the matched result, intervention frequency, first changed public states,
follow-through, search coverage, latency, truncations, and adverse games.
Retire a mechanism that never changes a decision, fails its mechanical premise,
or has a clear adverse fresh screen. A promising exploratory result earns a
separately frozen confirmation under the [canonical roadmap](agent-research-roadmap.md).
Do not enlarge a weak screen solely to rescue its point estimate.
