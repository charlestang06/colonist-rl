# Strength roadmap: public planning and resolved-outcome learning

Trusted human direction overrides earlier useful/near-perfect-BC prerequisites for
STARTING controlled PPO.100k remains HOLD; target-objective implementation is lower
priority. PlainGreedy rules and old evidence stay frozen.

## Latest execution status

The reviewed implementation now includes opt-in current-weight PPO history,
resolved-return critic warmup, an optional frozen-BC KL anchor, and a conservative
public current-turn planner. Combined verification:1400 passed,3 optional skips.
These capabilities have not yet demonstrated improved game strength.

The targeted3.3 attempt collected64 episodes and64 training candidates plus16
decision cases, then stopped before fitting because the independent judgment
envelope missed its release deadline. Public packet verification succeeded;
zero labels were approved. Preserve that incomplete attempt without after-deadline
salvage or a repeated run under its authorization. Its operational bottleneck was
auditing and assembling80 cases, not model training. A future separately specified
experiment should first measure audit throughput on smaller batches and persist
each judgment promptly. Keep the scientific success criteria fixed and reserve
fresh decision cases; the confirmation boards remain untouched. No broad PPO,
reward change, or100k scaling has run.

## Current diagnostic result, without moving its gate

The closed diagnostic confirms all six small-fit arms completed common100 epochs,5,700 updates and
51,200 presentations each within715.79/1200seconds. C improved training bank/END CE
by15.51%/13.38% and held-out full-menu regret, but directional accuracy124/182 lost
to linear135/182 and132/182, and held-out hard CE/accuracy worsened. Predeclared
quantitative gate FAIL both seeds. No full10k/100k C scaling or posthoc gate change.
The independent fixed11-case audit found identical choices within each initialization across architectures:9defensible,2ambiguous,0clearlybad per arm. Those selected training cases do not establish general strategic quality. Future PPO uses unchanged FULL15
hand_recipe A parents, never tinyfit100/C-selected weights.

## Parallel direction: bounded public current-turn planning

First option is a shallow, budgeted trade->build improvement operator over H;
BC+planner is an explicitly separate deployment arm. Freeze depth/node/time budget,
ordering/ties and public ranking before games. Only own hand, visible pieces,
public topology/buildings/production and current legal menu enter search. Immediate
point/production gain is a provisional criterion, not calibrated win value.

Only execute an action present in ACTUAL current legal menu, then reobserve and
replan. A projected future build's affordability is not guaranteed future legality.
Unknown bank/development deck remain unknown; no invented supply/draw values, hidden
engine clone, private hand/deck rollout or oracle. Avoid plans requiring unknown
supply, retain legal fallback. Current menu proves current exchange feasibility only.

Opening-pair planning is a separate toggle/study: public joint production/coverage
can rank nonadjacent pairs, but opponents can take the second site. Recompute at
actual second placement. Do not conflate opening and trade gains.

A planner must prove SAME-engine complete-game improvement over H before labels
become a stronger teacher for DAgger/Expert Iteration. This teacher-strength gate
does NOT block outcome-based PPO exploration. Teacher priorities/simple utility
are not win-calibrated; local attractive exchanges are not proof of game strength.

## Immediate direction: controlled3.3 pipelines and integrity first

Fresh exact3.3 cohort, disjoint training/validation/reserved boards. Existing APIs
support GreedyNoTrade/H/frozen historical lineups; reuse collector/PPO/evaluator.
Legacy FULL15 weights use reviewed explicit weight-only bridge, immutable parent
hash, fresh optimizer/RNG/steps and new identity; old scores never become3.3 metrics.

Core arms, fixed starts and seeds11/29: frozenBC; critic-warmedBC->PPO; frozenBC+
public planner. Warm only value head with actor/trunk frozen, verify unchanged
actor logits, then fresh on-policy PPO. This compares pipelines; add matched
BC->PPO without warmup as a separate control to attribute warmup's benefit.

Freeze training mix/schedule of Greedy/H/frozen snapshots and record actual counts/
RNG/checkpoint IDs. Historical opponents provide diversity, not assumed monotonic
strength. Keep validation lineups fixed outside evolving pool and check forgetting.
Small KL-to-frozen-BC or imitation anchor is a SEPARATE optional ablation, not an
assumed existing API. Keep unanchored control, predeclare coefficient and reduction/
withdrawal, monitor KL/clip/entropy/value loss and wins. No permanent teacher ceiling.

Production source correction: PPOConfig has no gamma field; advantages uses
reward+next_value−value and summed off-turn rewards, effectivegamma1, lambda0.95.
Reward terminalwin plus0.2potential difference, terminal potential0. Warm targets
must be actual complete resolved reward-suffix sums with terminal bootstrap0.
Shaped critic is not automatically win probability. Separate outcome head or gamma
change requires explicit target/calibration review, not relabeling old value output.

Full-history audit is required before sustained PPO: collector self.memory can
survive collect() calls across changed actor weights; update chunks use stored
record.memory across updated weights. Distinguish truncated gradients from stale
initial recurrent context. Controlled current-weight mode must replay complete own
observation prefixes, including forced steps, without future information. Audit
must also prove true episode/winner/board/seat linkage and all reward intervals;
current aggregate episode/win counters are not a finished-episode dataset API.
Truncated/deadline outcomes are censored, not losses or MC targets. Preserve partial
records. Existing nonterminal bootstrap semantics are not resolved outcome labels.

PPO needs genuine fresh sampled actor actions/log-probabilities. Teacher/planner
choices cannot masquerade as on-policy behavior; primary PPO has no hidden planner
override. Limited terminal-win coverage is reported rather than fabricated, and
is not another near-perfect-imitation prerequisite.

## Concrete bounded implementation next

Expected first stage, finalized by the read-only audit: a minimal opt-in audited
current-history path in existing training.py plus focused tests. Preserve legacy
constructors; controlled PPO explicitly enables correct history. Add sufficient
owned-history/episode-result records and durable terminal linkage for later MC
warmup. Recompute no-grad current-weight prefixes before collector decisions after
updates and before PPO chunks; keep true boundaries, deadline/partial rules and
on-policy logging. No new simulator, framework, reward, anchor or teacher changes.
Acceptance: changed-weight collector/update outputs match literal full-prefix
reference; forced and cross-call history retained; no future/other-episode context;
true-terminal reward suffixes include off-turn rewards; truncated/missing intervals
never become outcomes; partial persistence/cursor identity survives interruption.
Independent review gates this before a separately bounded critic-warm/PPO pilot.

Planner work can be a separate disjoint module/benchmark. After integrity review,
implement critic-only warmup as a small opt-in consumer of verified complete
records, then run the existing PPO pipeline with fixed starts/pool/panel. Root
sets actual throughput-based episode/step/compute caps and integration reserve.
No near-perfect BC gate, but no sustained PPO atop a known history/target defect.

## Stage 3: win-primary milestone and later branches

Keep declared existing win-plus-potential reward fixed initially; NO trade/build/
action bonuses. Terminal-only versus shaping is a separate reward study.
Fresh fixed all-seat validation plus frozen larger confirmation panel, both neural
initializations, common opponents/seeds, authoritative per-game journals. Report
wins/VP/censoring/work and paired BOARD uncertainty (four seats are correlated).
Desired milestone: H-relative wins positive in both seeds and positive lower bound
of predeclared board-clustered interval on frozen confirmation comparison. VP is
secondary; insufficient power stays inconclusive, no sampling until significance
or choosing on final boards. Deterministic references are not new independent data
when repeated for each neural seed.

Verified stronger planner can seed audited DAgger/Expert Iteration; verified PPO
improvement can continue measured curriculum. Failure calls for one outcome/behavior
diagnosis, not automatic100k copies of old weak teacher. First DAgger remains PARTIAL
with no benefit established despite4fits/320games; persistence-fixed repeat gets
separate engine/budgets. Original roadmap/frozen inputs untouched by this note.


The search/distillation direction is motivated by [Expert Iteration](https://arxiv.org/abs/1705.08439),
which separates search improvement from neural generalization. Its results in
Hex do not establish a Catan benefit. For any later search across hidden
information, [Information Set Monte Carlo Tree Search](https://pure.york.ac.uk/portal/en/publications/information-set-monte-carlo-tree-search/)
is a methodological reference: sampled hidden states must be consistent with
public evidence, never copied from the simulator's actual hidden state.
The first bounded current-turn planner should use public deterministic projections
and explicit unknowns; broader information-set search is future work.
