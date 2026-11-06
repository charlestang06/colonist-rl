# Catan agent research roadmap

## Objective and success criterion

Develop a player-observable Catan agent whose complete-game results reproducibly
exceed authentic, version-pinned VF and AB baselines in declared harness protocols.
Limited imitation diagnostics, transparent public search, outcome-trained value
and continuous case validation are independent research tools. BC success is not
a gate; PPO remains deferred.

“Outperforms” means a positive predeclared candidate-minus-reference win comparison
on untouched boards, separately for VF and AB. Rotate candidate and reference
through the same seats, boards and opponent lineups; freeze those lineups, reference
configuration, model, decision rule and resource limits before final evaluation.
Predeclare the superiority threshold and board-level uncertainty rule; resample
whole boards for uncertainty. A nominal 25% win rate alone is insufficient;
VP, legality, completion and compute support the primary win comparison. Small
pilots can complete successfully while remaining strategically inconclusive.

Our candidate, teachers and search use public information. Authentic native VF/AB
are separately isolated privileged-state stress opponents; success against them
would not establish equal-information superiority. Custom `HeuristicPolicy` H,
H_view and GreedyNoTrade are separate controls. A public reconstruction of VF/AB
would be a separately named optional experiment.

## Accepted evidence

All recent experiments below use pinned Catanatron
`3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`; legacy 3.2.1 results
remain a separate cohort in the [training history](training-history-through-checkpoint08.md).
The [current status](training-status-2026-11-05.md) links the underlying artifacts.

- **BC/control:** the separate v4 recovery completed FULL25 and 288 resolved games.
  Exact agreement on 48 selected training cases improved 28→35 and 28→38; qualified
  target-or-approved-alternative agreement improved 32→42 and 35→42. Seed 11 wins
  regressed from 3/32→0/32 against H and 11/16→6/16 against Greedy. Seed 29 stayed
  1/32 against H and changed 5/16→6/16 against Greedy, with lower mean VP in both
  panels. Retain stable parents and diagnose retention. This establishes neither
  fundamental cloning failure nor a memory defect.
- **Public planning:** 96 new development games resolved, with verified reused
  controls counted once. Planner versus H_view won 10/32 versus 8/32 against H,
  and 15/16 versus 16/16 against Greedy. The respective paired win differences
  were +6.25 points [0, 15.625] and −6.25 points [−18.75, 0] in descriptive
  board-bootstrap intervals. No reliable general advantage is established. Only
  137/4,317 decisions were marked `planned`; 4,180 fell back. Root-action identity
  failures and truncations were zero; planner p95 was about 8.4–8.9 ms versus
  0.68–0.93 ms for H_view. Inspect coverage and changed decisions before more depth.
- **Native references:** reviewed gateway code is integrated (core `ea36614`, seam
  worker `d1e11bd`; isolated sources `a67576c`/`1fb9d32`). The real-engine seam
  passed all 10 cases with 8 authentic query returns. This is engineering evidence;
  the first exposed-board H profiles resolved all eight games: 0/4 versus VF and
  3/4 versus AB2. No superiority result is established.
- **Outcome value:** v3 passed 14 pure integrity tests, then the guarded smoke
  resolved 12 games across three disjoint boards and fitted the three fixed models
  in 10.895 seconds. It produced 495 positions (133 train, 186 validation, 176
  diagnostic test). This validates the pipeline, not value strength. The separate
  192-game pilot on 32/8/8 train/validation/diagnostic-test boards completed,
  producing 8,674 positions with all outcomes resolved. Public-linear/calibrated
  handcrafted validation Brier were essentially equal (0.16086/0.16094); diagnostic
  Brier favored public-linear (0.15442/0.19835). This mixed offline result requires
  gameplay testing; it is not playing-strength evidence.
- **First fixed evaluator comparison completed:** all 96 new games resolved under
  six accepted guards, exit 0, at most two heavy workers and 170.861 seconds root
  wall time. Raw/calibrated/linear wins were 3/2/3 per 16 against H and 3/2/3
  against native VF. The panels share four exposed board clusters. Linear gained
  one win over calibrated in each panel, but descriptive board-bootstrap intervals
  touch zero and VP differences have opposite signs; it gained no wins over raw.
  No evaluator promotion or authentic VF/AB superiority follows. The
  [empirical review](../runs/swarm-2026-11-04/agent-research/value-search-experiment/run-001/EMPIRICAL-REVIEW.md)
  and [compact analysis](../runs/swarm-2026-11-04/agent-research/value-search-experiment/run-001/compactanalysis.json)
  preserve coverage, compute, comparisons and the evaluation-only conceptual audit.
- **Cap correctness and paired repeat:** a separate old TRAIN case showed a
  completed known win discarded at the geometry cap. The narrow fix passed
  [independent review](../runs/swarm-2026-11-04/agent-research/planner-cap-fix/REVIEW.md)
  at isolated `796e120` and is integrated as `b778830`; root's affected-path check
  passed 98 tests in 1.30 seconds. It retains only completed known wins after full
  root validation; ordinary fallback, budgets, ordering and models are unchanged.
  The separately frozen cap follow-up completed all 96 new games, reusing the
  old 96 as paired controls on the same four boards and adding no independent
  controls. All six arm/panel win counts and mean VP match the old cohort; this
  comparison shows no gameplay improvement on these boards. The fixture repair
  remains valid, without establishing broad strength or trajectory identity.

## Milestone ladder

| Milestone | Work and completion evidence |
|---|---|
| 1. Reliable comparisons | The seam and eight exposed-board native reference games completed. Expand fixed candidate/reference completion/legality/timeout comparisons against authentic VF/AB. Pin engine, opponent, adapter and independent RNG identities; account for every scheduled outcome. This is an integration milestone, requiring no strength assertion. |
| 2. Genuine public outcome-value data | The pinned 192-game board-separated pilot and data audit completed. Ground the next proposed targets in actual public after-action states. Audit true terminal joins, both outcome classes and phase support; retain null/censored outcomes. Compare calibration with constant and exact handcrafted-score controls, counting boards/games rather than independent positions. |
| 3. Isolate search and evaluation | The first fixed 96-game evaluator comparison and its cap-corrected paired repeat completed without an established advantage. Test the proposed grounded after-action value controls next. Keep H, H_view and fixed handcrafted/learned controls explicit. Hold depth/node/beam, fallback and action support fixed; report evaluator calls and inference time. Inspect changed decisions for projection/evaluator exploitation. |
| 4. Repeatable development improvement | Expand predeclared boards, seats and lineups for a promising candidate, retaining fixed parents and native references. Require repeated complete-game benefit, legality and acceptable resources. Failed candidates redirect a lane; they do not end the research objective. This stage selects candidates. |
| 5. Sealed confirmation | Freeze the candidate/protocol and use untouched final boards once. Preserve `123100000..123100063` and legacy finals until then. Report VF and AB comparisons separately, board-level uncertainty, information asymmetry and compute. An inconclusive final remains inconclusive; exposed boards cannot become final again. |

Planning and outcome-value milestones proceed in parallel. Modest BC retention/
tie-aware controls and 50–100-case background batches support every stage; neither gates search/value.
Chance-aware information-set MCTS is considered only when simpler-planner evidence
identifies a useful limitation. Review public-consistent hidden sampling and
strategy fusion before adopting it. More model capacity, features, recurrence,
search and compute each need controlled ablations. PPO remains deferred.

## Immediate experiments

**Broader authentic-opponent comparisons.** The gateway seam, eight reference
games and first H/VF evaluator comparisons are complete. A larger pinned
candidate/reference development schedule, including native AB, remains necessary.
Keep H_view, fixed handcrafted and frozen-parent controls where useful; freeze
candidate choices before each comparison. Pin VF base defaults and AB depth=2, `prunning=False`, source hashes,
root-mask/seat mapping and independent planning RNG. Declare
`+atomic_discard_reject_trade_v1`, no domestic proposals and explicit response
rejection: these are phase shims, with `full_rule_equivalence=false`. Preserve AB's
native chance-model defects and coordinated-minimizer behavior as disclosed
limitations; correcting them would define another baseline. AB's soft 20-second
decision deadline requires a conservative external whole-attempt cap and complete
incomplete-game inventory.

**Completed cap-corrected repeat.** The reviewed evaluator hook was integrated as
`2aa2c61` (isolated frozen source `3dc8bcf`). Its original trial closed with 96 new
complete games, three frozen arms, four boards/all seats, H and native VF panels,
and no reliable advantage. The separately frozen
[cap follow-up protocol](../runs/swarm-2026-11-04/agent-research/cap-followup/protocol.json)
(`024c5251548a1ae480f73e34c1b16d3bd0b366cff61d2772046df29e408a880b`)
completed 96 new cap-fixed games against the reused original 96 on the same
schedule: six guard exits 0, at most two heavy workers and 82.385 seconds root wall
time. All six arm/panel wins and mean VP matched the original cohort. The
[completed run](../runs/swarm-2026-11-04/agent-research/cap-followup/run-001/attempt-status.json)
supplies no gameplay improvement on these four boards. Paired metadata comparison
remains separate from any claim that complete trajectories are identical. The
shorter wall time is not a controlled speedup estimate. Models, score units/margins,
depth 6, 2,048 nodes, beam 24 and 100,000 geometry visits stayed fixed; equal limits
do not imply equal compute. Reused controls do not double effective sample size.
AB expansion remains necessary for the overall VF/AB objective.

**Current grounded value study.** The public transition callback, eight productive
geometry features and bounded residual runtime are integrated and independently
reviewed. The fixed study learns from genuine public before/after observations and
resolved outcomes under cap-corrected raw-planner continuation versus H. Compare
zero residual, existing-feature residual and the same residual with productive
features; fix game weights and 600 optimizer updates before collection. Preserve
normal-turn passes and card purchases as eligible before states. No native action
labels or evaluation cases enter training. The [study specification](../runs/swarm-2026-11-04/agent-research/grounded-value/spec.md)
records interfaces, exact feature/anchor identities and controlled work.

The four-game smoke passed. The first main attempt verified all 96 training games,
then stopped on a validation road accepted by the engine's original legal menu but
rejected by inherited public connectivity. No fit was run. Keep the failed attempt
immutable; correct only original-root projection, preserve strict future geometry,
and rerun the complete 128-game development schedule under frozen reviewed source.
Every unexplained projection mismatch blocks fitting. If this passes, separately
freeze 128 gameplay games: the three residual controls plus unchanged raw, H/VF,
four new boards and all seats. Raw is a practical benchmark with different score
units/margin; the three probability controls form the matched ablation. Offline
loss improvement earns a gameplay test, never promotion.

**Independent setup control.** A completed 32-case public TRAIN audit and critique
left one fallible clearly bad H choice and 14 ambiguous ones. The actual 32-root
H_view probe passed; 13/16 second placements disagreed with the public handcrafted
proposal. The subsequent 64-game second-settlement-only comparison resolved all
games across eight boards. Its independently audited win difference was +9.375 percentage
points [0, 21.875], with mean VP difference −0.09375 [−0.9375, 0.625]. This small
mixed result establishes no promotion or native-strength advantage. The scorer is
a handcrafted control; privileged native VF choices never become teacher labels.
The [setup study](../runs/swarm-2026-11-04/agent-research/grounded-value/setup-control/REPORT.md)
preserves the intervention, fixed continuation and limitations.

**Priorities after these controls.** First validate value grounding and complete-game
transfer; then audit trade/build/pass choices on actual student states and use
small, independently reviewed DAgger/retention experiments where labels are clear.
Keep strict GreedyNoTrade, H/H_view and authentic native reference panels explicit.
Do not increase model size or collect bulk cloning data merely to improve label
agreement. Use terminal wins for outcome targets. VP, hand efficiency and setup
scores are diagnostics; any reward shaping needs a separate controlled experiment
showing it preserves the winning objective. Chance-aware search or PPO requires a
specific observed limitation and a frozen comparison against the simpler policy.

## Information, search and value contracts

Execute an actual legal root action, reobserve and replan. H_view changes historical
H's RNG and requires its own control. Public projections are not actual observations;
unknown bank/deck quantities stay unknown. Our search never reads actual hidden
hands, development cards or future game RNG. Native privileged state stays inside
the isolated opponent runner and cannot enter learner inputs, teachers or labels.

The reviewed value pilot predicts outcomes from actual post-roll public positions
under a declared H continuation; it is not yet a validated projected-afterstate
evaluator. Source/data-contract review precedes smoke, collection, fitting and
same-budget gameplay. Distribution shift and longest-road projection remain
limitations. Use a legitimate public afterstate encoder when that integration is
reviewed; do not fabricate a legal `Observation` for a projected state.

BC scalar heads were not outcome-trained and retain their initial tensors. The
old 64 histories lack winner labels. True resolved outcomes supply labels;
missing/truncated outcomes remain null and suggested H actions never become
outcomes. Keep train/validation/diagnostic boards separate. Freeze checkpoints
and selection rules before comparative results, and keep final boards sealed.
The 126-series panels are exposed development; reused controls add no independent
data. Report wins, VP, completion, uncertainty, action support, p50/p95/max latency
and actual work. Equal nodes do not imply equal compute.

## Background data and validation

Continue 50–100-case batches across full gameplay alongside the main experiments.
Persist each immutable review immediately; incorporate approved data only at
deliberate training checkpoints. Cover minority actions, varied boards/seats and
learner-reached errors: multi-step trades/builds, sensible passes, waste, ports,
scarcity, development cards, robber decisions, road/army races and immediate wins.
Accept justified alternatives and preserve ambiguity. H labels are suggestions;
zero errors on a weak selected-case rubric is not general competence. Keep
evaluation cases separate from training labels.

Batch 001 reuses 64 old training states. All 64 mechanical consequence cases are
certified, and [final independent strategy review](../runs/swarm-2026-11-04/roadmap-resume/background-cases/final-audit-001/README.md)
is complete: 53 reasonable, 11 ambiguous. Forty targets are approved only for a
future limited BC checkpoint, including ten differing from stored H; 13 reasonable
card/development targets remain withheld for unsupported effects, as do the 11
ambiguous targets. Use the [final envelope](../runs/swarm-2026-11-04/roadmap-resume/background-cases/final-audit-001/final-envelope.json)
and its effective amendments, not a raw cases-directory glob. No training has
incorporated these labels.

[Batch 002](../runs/swarm-2026-11-04/roadmap-resume/background-cases/batch-002/README.md)
is a metadata-only draft of 64 unused old training states across eight full-gameplay
strata and 16 boards, balanced by parent and seat. It is not a fresh rollout,
approved label set or certified consequence batch. Independent judgments and
phase-appropriate validation remain future work.

The completed evaluator study's conceptual audit contains 22 of 24 planned slots
(two declared shortfalls): 9 reasonable, 9 ambiguous and 4 clearly bad, drawn from
18 linear and 4 raw changes with no calibrated cases. Its small biased change-only
sample supplies neither population error rates nor evaluator-arm quality rates.
Blocked-road choices versus legal productive settlements motivate hypotheses;
projected continuations remain uncertified. These EVAL packets and judgments must
never become training data.
Counterfactual validators may reconstruct privately solely to test fixed public
proposals; hidden state cannot enter policy/teacher inputs. The learner's domestic-
trade interface currently supports responses only; bank/port trades remain ordinary
supervised actions. Initiating or countering domestic offers requires a separate
interface.

Run at most two heavy workers, one native/Torch thread and 2 GiB each, with root
scheduling and existing guards. Pin each lane's source/runtime cohort, preserve
failed/incomplete attempts and resume useful independent work within bounded runs.
No paid compute is assumed. Historical evidence and superseded plans remain in the
[archive](training-history-through-checkpoint08.md); user edits in
[docs/roadmap.md](roadmap.md) remain preserved.
