# Fast paired experiment loop

Use one JSON manifest to compare public policies on the same board seeds and
four focal seats. The [paired CLI](../src/colonist_rl/experiment_cli.py) calls
the existing evaluator; the [analysis command](../src/colonist_rl/experiment_analysis.py)
reads saved artifacts without replaying games. The
[first search-budget screen](search-budget-results-2026-11-06.md) is a complete
worked example.

## 1. Freeze one question

Start from the archived [budget manifest](../runs/swarm-2026-11-04/agent-research/search-budget-001/screen-manifest.json)
and give the new study **new, disjoint board seeds**. Set `role` to `smoke`,
`development`, `validation`, or `final_test`; this declaration does not check
other archives for reused seeds. Require at least two arms and name the control
with `reference_arm`. A public planner arm records every `TurnPlannerConfig`
field and `gain_margin_micro`. A `city_before_purchase` arm is opt-in, and a
`graph_checkpoint` arm requires an exact checkpoint SHA-256. The same evaluator
supports three named random, heuristic, greedy-no-trade, or pinned native
opponents in the lineup. This CLI currently requires `no_offers: true` and has
no historical-policy pool.

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
first diverged, not which choice was optimal.

## 3. Decide quickly

Read the matched result, intervention frequency, first changed public states,
follow-through, search coverage, latency, truncations, and adverse games.
Retire a mechanism that never changes a decision, fails its mechanical premise,
or has a clear adverse fresh screen. A promising exploratory result earns a
separately frozen confirmation under the [canonical roadmap](agent-research-roadmap.md).
Do not enlarge a weak screen solely to rescue its point estimate.
