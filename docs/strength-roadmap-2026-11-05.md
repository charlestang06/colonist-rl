# Strength objective and next work

Direction revised 2026-11-06. Build a strong player-observable Catan decision maker:
first demonstrate sensible heuristic-level play, then reproducible complete-game
superiority over authentic pinned VF and AB. The [canonical roadmap](agent-research-roadmap.md)
defines the goal, execution order, evidence gates and adapter boundaries.

The main loop is fresh student games, public decision diagnosis, trustworthy
teaching targets, one isolated repair and matched gameplay. Check the teacher
rather than treating imitation accuracy as quality. Bank trades stay legal;
GreedyNoTrade deliberately declines them. Preserve useful passes, acceptable
alternatives and ambiguous judgments. Cover placement, trades/build/pass,
domestic responses, robber/discard, development-card buying and play, road/army
races and VP/endgame. Missing coverage prevents a broad competence claim.

Next: repair the accepted-set loss's numerical instability, finish the 43-case and
legacy regression review, then collect/audit current-student data and run the
frozen added-data-only comparison. In a separate bounded lane, complete the
role-aware terminal-history collector and its reviewed 320-game schedule; test
value prediction before policy deployment. A small tactical repair from fresh
TRAIN failures can proceed alongside these experiments; it need not wait for BC.
Validate/profile public cross-turn transitions before larger search. BC is optional.
Compare the strongest competent candidate directly with VF/AB; PPO is not a
prerequisite. PPO and new shaping follow demonstrated competence and a specific
hypothesis. Maintain adapters behind the common Observation/Action
contract while learning gets priority.

No candidate is promoted. The 384-game eligibility comparison failed its primary
rule: every candidate won 2/16 against trading H. The explanatory EVAL audit's
17 reasonable/two ambiguous unique cases covered setup/bank/robber only. The
native focal pilot scored raw/VF/AB2 3/0/4 per 16, which establishes no superiority
over both references. These exposed data remain outside training.

The separate complete-history cache repair passed engineering review on four
saved games/361 inputs, preserving the first failed attempt. Main value collection
and fitting remain unstarted. The sparse loss passed 41 initial tests; two later
offset regressions reproduced an uncovered issue, so it remains pending repair.
Neither result establishes stronger play. [Current status](training-status-2026-11-05.md)
records evidence and limitations.

Retire unchanged failed interventions. Compare on fresh boards, rotate every seat,
retain strict GNT/trading-H controls and separate focal-native benchmarks from
three-native stress panels. Scale after a primary measured benefit. Keep final
boards `123100000..123100063` and legacy finals untouched until confirmation is
frozen. [Historical results](training-history-through-checkpoint08.md) and
[previous roadmaps](../runs/swarm-2026-11-04/research-roadmap-history/) remain available.
