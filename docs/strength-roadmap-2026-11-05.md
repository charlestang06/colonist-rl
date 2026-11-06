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

The next controlled hypothesis is the existing second-settlement scorer on fresh
boards and corrected source `3f0a298`, with the same raw continuation at margin
10000 in both arms. Freeze two arms × 16 boards × four seats × H/VF = 256 games
before launch, measure wins separately against H and VF, and use the
predeclared whole-board advancement rule. This is a planned transfer replication,
not an exact replay or a claimed improvement. No new fit or reward shaping follows
from the current mixed results.

Current native panels use three native opponents. The opponent-only gateway needs
a narrow isolated evaluation extension for focal VF/AB references before the
reference milestone can be satisfied. Keep final boards `123100000..123100063` and
legacy finals sealed.

[Current results](training-status-2026-11-05.md) link exact reviews and artifacts.
[Historical results](training-history-through-checkpoint08.md) and
[previous roadmaps](../runs/swarm-2026-11-04/research-roadmap-history/) remain preserved.
