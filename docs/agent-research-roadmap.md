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
  the same-budget gameplay test; it is not playing-strength evidence.

## Milestone ladder

| Milestone | Work and completion evidence |
|---|---|
| 1. Reliable comparisons | Finish bounded authentic VF/AB completion/legality/timeout games after the passed seam. Pin engine, opponent, adapter and independent RNG identities; account for every scheduled outcome. This is an integration milestone, requiring no strength assertion. |
| 2. Genuine public outcome-value data | Complete the pinned board-separated pilot after the passed smoke. Audit true terminal joins, both outcome classes and phase support; retain null/censored outcomes. Compare calibration with constant and exact handcrafted-score controls, counting boards/games rather than independent positions. |
| 3. Isolate search and evaluation | Pair H, H_view, fixed handcrafted planner and fixed learned-value planner in complete development games. Hold depth/node/beam, fallback and action support fixed; report evaluator calls and inference time. Inspect changed decisions for projection/evaluator exploitation. |
| 4. Repeatable development improvement | Expand predeclared boards, seats and lineups for a promising candidate, retaining fixed parents and native references. Require repeated complete-game benefit, legality and acceptable resources. Failed candidates redirect a lane; they do not end the research objective. This stage selects candidates. |
| 5. Sealed confirmation | Freeze the candidate/protocol and use untouched final boards once. Preserve `123100000..123100063` and legacy finals until then. Report VF and AB comparisons separately, board-level uncertainty, information asymmetry and compute. An inconclusive final remains inconclusive; exposed boards cannot become final again. |

Milestones 1 and 2 proceed in parallel. Modest BC retention/tie-aware controls and
50–100-case background batches support every stage; neither gates search/value.
Chance-aware information-set MCTS is considered only when simpler-planner evidence
identifies a useful limitation. Review public-consistent hidden sampling and
strategy fusion before adopting it. More model capacity, features, recurrence,
search and compute each need controlled ablations. PPO remains deferred.

## Immediate experiments

**Authentic-opponent reference games.** With the gateway integrated and seam passed,
run a small fixed paired board×seat completion/legality/timeout pilot under the v4
guard. Then, only after that evidence passes, run a pinned development schedule
comparing H_view and the unchanged handcrafted planner with native VF/AB, keeping
H/Greedy/frozen-parent controls where useful. No search tuning belongs in this first
comparison. Pin VF base defaults and AB depth=2, `prunning=False`, source hashes,
root-mask/seat mapping and independent planning RNG. Declare
`+atomic_discard_reject_trade_v1`, no domestic proposals and explicit response
rejection: these are phase shims, with `full_rule_equivalence=false`. Preserve AB's
native chance-model defects and coordinated-minimizer behavior as disclosed
limitations; correcting them would define another baseline. AB's soft 20-second
decision deadline requires a conservative external whole-attempt cap and complete
incomplete-game inventory.

**Outcome-value pilot, then a fixed planner comparison.** Audit the completed
`value-lane/pilot-001` collection under its frozen source/runtime and prior-use-audited
board namespace. Fit the predefined constant, calibrated exact handcrafted score
and regularized linear public-feature predictors, with declared H continuation.
A promising offline result earns a separately pinned handcrafted-versus-learned
planner comparison at depth 6, 2,048 nodes and beam 24, with consistent score units,
selection margins, identical fallback/support and measured evaluator cost.
Offline Brier improvement alone never promotes a model. This lane continues
independently of BC and native-opponent strength.

The reviewed evaluator hook and frozen public-value runtime are integrated as
`2aa2c61` (isolated reviewed source `3dc8bcf`). The next fixed experiment is 96
games: three evaluator arms, four boards/all seats, H and native VF panels. AB
expansion follows this cheaper probe; the overall VF/AB objective remains unchanged.

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

Batch 001's 64 selections reuse existing training states. All 64 deterministic
consequence checks completed under the guarded validator; independent final
strategy review is in progress. They are neither fresh rollouts nor approved labels.
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
