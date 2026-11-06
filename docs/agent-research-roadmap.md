# Catan agent research roadmap

## Goal and success criterion

Develop a player-observable Catan agent that reproducibly outperforms authentic,
version-pinned VF and AB in complete-game reference comparisons under declared
harness protocols. Playing strength is the objective. Audited decisions, supervised
learning, public search and outcome-trained value are tools for reaching it.
BC supports diagnosis and warm starts; it does not gate other learning. PPO
remains deferred.

The candidate may use its own hand/cards, the visible board, public events and
beliefs derived from those events. Opponent hands, deck order and future RNG
cannot enter its policy, teacher or search inputs. Authentic native VF/AB are
isolated privileged-state evaluation references, so a result against them has
that explicit information asymmetry.

Success requires separately positive, predeclared candidate-minus-reference win
comparisons for VF and AB, with fixed focal seats, boards, background lineups,
model, decision rule and resource limits. Freeze a practical effect threshold and
board-level uncertainty rule before confirmation. Rotate all seats and resample
whole boards. VP, completion, legality, decision quality and compute support the
win result. A nominal 25% win rate, label agreement, or a small successful pilot
cannot establish the final strength claim.

Preserve final boards `123100000..123100063` and legacy finals until a candidate
and confirmation protocol are frozen. No authentic VF/AB superiority is established.

## What the evidence changes

The [current results](training-status-2026-11-05.md) preserve source pins, exact
counts and independent reviews. Recent work uses Catanatron
`3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`; legacy 3.2.1 is separate.

Targeted BC improved selected-label agreement while regressing or mixing actual
game results. Fixed public planning and the first value/search comparisons did
not establish a reliable advantage. Keep stable parents and diagnose retention;
do not make more cloning the default response to a failed game comparison.

The new grounded study completed 128 genuine outcome games, two fixed 600-update
residual fits, and 128 separate gameplay games. Data, arithmetic and gameplay
integrity passed independent audits. Productive features added almost no
validation benefit over existing features. H wins for raw/zero/existing/productive
were 3/3/2/4 per 16; native-VF wins were 7/7/7/7 per 16. Four shared development
boards and coarse intervals do not support selecting or promoting a model.

A 24-case public decision audit found 18 reasonable and 6 ambiguous choices, with no
unmistakable error. The selected sample is biased toward seat 0, was selected after
aggregate results were visible, and contains no root settlement option. It cannot
prove general competence, arm quality, or elimination of older failure cases.
Its packets and judgments remain evaluation-only.

The separate eight-board second-settlement pilot motivated a fresh 256-game
transfer study on corrected source `3f0a298`, with raw margin 10000 in both arms.
Across 16 new shared boards and all seats, baseline/control wins were 20/19 per 64
against H and 26/32 per 64 against native VF. Control-minus-baseline win differences
were −1.5625 percentage points [−12.5,10.9375] for H and +9.375 [−4.6875,21.875]
for VF, using the predeclared whole-board bootstrap. All 256 outcomes and 128
pre-second public-root/prefix pairs passed independent integrity checks.

The primary H advancement rule failed. Deprioritize this unchanged placement
intervention; higher VP and the positive VF point estimate cannot rescue it.
Do not pool the old source cohort or retune on these exposed boards. No candidate
is promoted. The [empirical review](../runs/swarm-2026-11-04/agent-research/setup-transfer/run-001/EMPIRICAL-REVIEW.md)
records provenance, measured work and the separate preparation-test incident.

## What to pursue next

**1. Diagnose and isolate trading decisions.** First
audit trading versus passing on student-reached TRAIN states. Inspect passes for
useful root-legal trades that could unlock settlements, cities or expansion.
Check public resource consequences, build deficits and connected-site access;
a complete recipe alone does not certify a legal future build. Accept justified
alternatives and ambiguity rather than assuming H's choice is ground truth.
Check label consistency, action encoding and gradient flow before increasing
training scale. Bank trading and ending share a phase, so phase balancing alone
cannot establish that the student distinguishes their consequences.

A frozen 32-case TRAIN sample is prepared from existing historical neural-student
trajectories: 14 apparent single-trade recipe/site opportunities, nine other passes
and nine executed bank trades. It spans 11 boards/15 episodes and is clustered,
with older checkpoint occupancy distinct from the current raw planner. Public
judgment and any fresh scoped model checks remain diagnostic; selection screens
are neither certified future menus nor approved training labels. Reuse applicable
completed checks rather than rerunning the old panels.

The 32-case public audit and six-case historical replay are now complete. Saved
H labels are legal and accepted by the diagnostic judgments; the six reproduced
passes have matching action encodings and finite nonzero relevant gradients.
This subset exposes no disconnected-gradient or action-index defect. The parents
already use 30 hand/recipe consequence channels. The remaining four channels add
site/context eligibility; some are action-constant or duplicate existing recipe
flags at eligible roots, so the city stalls do not establish their benefit.
Check for informative TRAIN contrasts before fitting a bounded feature ablation.

The probe retains a resource HOLD: two samples recorded four OS threads despite
configured Torch intra/inter-op counts of one. This is not evidence of four active
compute threads. Diagnose the runtime separately before fitting; preserve the
failed identity-check attempt and the resource discrepancy. No optimizer update,
training-label approval or strength promotion resulted from these probes.

Next, isolate explicit public action-consequence features against the existing
feature set. Hold TRAIN data, model family, optimizer work, continuation and
evaluation controls fixed so the comparison tests that feature intervention.
Only after this comparison, aggregate independently audited student-gameplay
examples and test the added data as a separate intervention, with frozen labels,
accepted alternatives and board-separated splits. Change one intervention at a
time; do not combine feature changes and new examples in one claim. BC/DAgger is
an optional use of the aggregated data; genuine terminal outcomes can train value
independently of cloning success. The aggregation and separate data test remain
required follow-up steps.

Judge each intervention in separate fixed GreedyNoTrade, trading H and authentic
VF/AB panels, with paired complete games, whole-board uncertainty and a public
decision audit. Native stress panels remain distinct from focal-reference
comparisons, which still require the evaluation extension in section 2. Retain
frozen parents and check retention. Reuse applicable completed checks and TRAIN
data instead of repeating closed work. Never turn privileged native choices or
EVAL artifacts into training labels; the current 24 EVAL cases remain excluded.

Keep terminal wins as outcome targets; VP, resource efficiency and handcraft
scores remain diagnostics. Keep PPO and reward shaping deferred until the
failure mechanism is understood and simpler methods yield convincing complete-game
gains. These are ordered follow-ups, not newly launched studies or a strength
promotion.

**2. Establish authentic reference comparisons in parallel.** Current native games are
candidate-versus-three-native stress panels. The gateway currently permits native
opponents only; a focal native reference requires a narrowly reviewed evaluation
extension. It must remain outside policy/training/teacher routes. Then compare
candidate and focal VF/AB separately on the same declared background lineups,
boards and seats. Keep strict GreedyNoTrade, H/H_view and frozen parents as
explicit, separately named controls.

Pin the native Git `82aae93` sources and defaults, VF weights and AB depth 2 with
`prunning=False`.
Preserve upstream behavior and declare `+atomic_discard_reject_trade_v1`, synthetic
trade rejection, native-panel no-offer behavior, the declared native chance-semantics
mismatch and `full_rule_equivalence=false`.
Changing a native chance/search behavior defines another baseline. AB needs
conservative whole-attempt deadlines and truthful incomplete-game accounting.

## Milestones

| Stage | Required evidence |
|---|---|
| Reliable execution and data | Completed for the current wave: reviewed public callbacks/features, deterministic parity, real outcome joins, fixed-fit arithmetic and complete guarded games. Each new change still needs affected-path checks. |
| Repeatable development improvement | An isolated frozen intervention improves complete-game wins on fresh boards under its predeclared rule; inspect decisions, completion and measured resources. Small mixed pilots do not qualify. |
| Both native references | Positive candidate-minus-focal-VF and candidate-minus-focal-AB development comparisons under explicit information/rule regimes. Stress panels alone do not satisfy this stage. |
| Sealed confirmation | Freeze candidate, protocol, threshold and uncertainty rule, then use untouched final boards once. An inconclusive final stays inconclusive. |

## Continuous validation and adapter scope

Build small independent TRAIN case batches across setup, trades/builds/passes,
robber/discard, development cards, road/army races and held VP. Accept justified
alternatives and preserve ambiguity. Batch 001 has 40 targets approved only for a
future limited BC checkpoint; none has been used. Batch 002 is a 64-case metadata
draft needing phase-appropriate validation and independent judgments. Neither
gates value/search. Counterfactual validators may inspect a private environment
solely to certify fixed public proposals; hidden state cannot select proposals or
enter policy/teacher inputs.

Keep adapters behind the shared Observation/Action contract. Colonist UI controls,
log parsing and future physical-board recognition translate into that contract;
they do not define the learning algorithm. Maintain legality/staleness/effect
checks while learning remains the primary research priority. Root menus are the
current-action authority; projected future menus remain uncertified. Learner
maritime trades and domestic responses are supported; initiating/countering a
domestic offer requires a separate interface.

Run at most two heavy workers, one native/Torch thread and 2 GiB each, under existing
Root guards. Freeze source/runtime/model/schedules, retain failed attempts and
null outcomes, and avoid adaptive retries or tuning on exposed evaluation boards.
No paid compute is assumed. Preserve histories and the user's edits in
[docs/roadmap.md](roadmap.md). Earlier versions and detailed studies remain in the
[research archive](../runs/swarm-2026-11-04/research-roadmap-history/).
