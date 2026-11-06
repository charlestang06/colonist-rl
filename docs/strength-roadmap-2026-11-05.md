# Strength objective and immediate direction

The objective is a player-observable Catan agent that reproducibly beats authentic,
pinned VF and AB in complete-game reference comparisons. The
[canonical research roadmap](agent-research-roadmap.md) defines the success rule,
next experiment, public-information contract and sealed confirmation stages.
BC is optional; PPO remains deferred.

The latest wave completed 128 grounded outcome games, both fixed 600-update fits,
and 128 new four-arm H/VF gameplay trials. All passed independent integrity audits.
H raw/zero/existing/productive wins were 3/3/2/4 per 16; VF wins were 7/7/7/7 per 16.
Productive features added almost no incremental validation benefit. No arm is
promoted and no native-VF/AB superiority is established. The 24-case diagnostic audit found 18
reasonable / 6 ambiguous / 0 clearly bad, with explicit post-run and seat-selection
bias; it is never training data or population competence evidence.

The subsequent 256-game second-settlement transfer study completed on corrected
source `3f0a298`, with common raw margin 10000 and 16 fresh shared boards. H wins
were baseline 20/64 versus control 19/64; VF wins were 26/64 versus 32/64. The
control-minus-baseline H difference was −1.5625 percentage points
[−12.5,10.9375], while VF was +9.375 [−4.6875,21.875]. Independent provenance and
arithmetic checks passed. The predeclared primary H rule failed, so deprioritize
this placement intervention. Higher VP does not change that decision.

Next, diagnose student-reached TRAIN trades versus passes, check newly audited
misses for label/encoding/gradient problems, then isolate consequence features.
Only afterward test independently audited added data as a separate intervention.
A frozen 32-case public TRAIN sample is prepared; its historical neural parents
remain distinct from the current raw planner. No new fit or reward shaping has
been launched from these results.

Current native panels use three native opponents. The opponent-only gateway needs
a narrow isolated evaluation extension for focal VF/AB references before the
reference milestone can be satisfied. Keep final boards `123100000..123100063` and
legacy finals sealed.

[Current results](training-status-2026-11-05.md) link exact reviews and artifacts.
[Historical results](training-history-through-checkpoint08.md) and
[previous roadmaps](../runs/swarm-2026-11-04/research-roadmap-history/) remain preserved.
