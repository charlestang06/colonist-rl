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

The separate second-settlement control won 7/32 versus 4/32 across eight boards,
with slightly lower mean VP. That is a concrete, inexpensive hypothesis for a
fresh controlled follow-up, not an established improvement. The independent
[post-study research decision](../runs/swarm-2026-11-04/agent-research/grounded-value/post-study-priorities.md)
explains the choice and its alternatives.

## What to pursue next

**1. Test one isolated source of better decisions.** The next selected experiment
is a second-settlement-only transfer replication: two arms × 16 fresh development
boards × four seats × H/VF = 256 new games. Keep the original public scorer and tie
rules frozen. Both arms use the corrected current source and the same raw
continuation at margin 10000, depth 6 / nodes 2048 / beam 24 / geometry 100000.
First placement, setup roads, continuation algorithm, proposal conventions and
opponent settings are held fixed; realized later states naturally differ.

This is a new source cohort: the earlier setup test used `56cb6bc`; the corrected
core is `3f0a298`. Main grounded raw used margin 250000. Do not pool these results or
reuse their controls as new evidence. Before launch, freeze source/runtime/scorer,
audit the 16 boards, pin H/VF phase shims, and review the minimal integration.
Use existing guards and complete scheduled-outcome inventory.

Predeclare H as the primary replication and VF as transfer stress. Advance this
unchanged intervention only with a positive H win estimate whose 95% whole-board
bootstrap lower endpoint exceeds zero, a nonnegative VF point estimate, complete
integrity and acceptable measured work. Report VF uncertainty separately. Mixed
or inconclusive results remain so; a nonpositive H replication or negative VF
estimate deprioritizes this intervention. Higher setup scores or VP cannot rescue
a failed win hypothesis. This experiment is planned, not launched or promoted.

**2. Improve learning only against an observed mechanism.** Continue genuine
outcome-value research independently of BC. If separate TRAIN evidence identifies
errors on learned-policy states missing from raw-policy data, test one controlled
continuation-data/policy-iteration step. If coverage is adequate but action ranking
is wrong, compare a bounded pair of public-selected legal actions with genuine
continuation outcomes. Freeze proposals, continuation, budgets and splits first;
never turn privileged native choices or EVAL cases into teacher labels.

For a limited BC/DAgger experiment, audit labels and justified alternatives on
actual student states, check encoding/gradients and retention, then judge complete
games. Bank trading and ending share a phase, so phase balancing alone is
insufficient. Public build deficits, connected-site affordability and trade
consequences are candidate features only when a controlled error analysis calls
for them. More rows, capacity, recurrence or search depth require an ablation.

Keep terminal wins as outcome targets. VP, resource efficiency and handcraft
scores are diagnostics. Any reward shaping needs a separate controlled test
judged by wins. Profile actual bottlenecks before increasing training work; cached
or batched recurrent states must remain consistent with the current weights.
PPO or chance-aware search needs a specific remaining limitation and comparison
against the strongest frozen simpler policy.

**3. Establish authentic reference comparisons.** Current native games are
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
