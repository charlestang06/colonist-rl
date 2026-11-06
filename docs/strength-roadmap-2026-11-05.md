# Strength objective and next work

Direction revised 2026-11-06. Build a player-observable Catan agent with sensible
heuristic-level decisions, then demonstrate reproducible complete-game superiority
over authentic pinned VF and AB. The [canonical roadmap](agent-research-roadmap.md)
defines priorities, proposed competence gates, experiment boundaries and confirmation.

Pursue three connected avenues: audit current-student decisions and collect complete
histories; learn terminal-outcome value from fresh board-separated games; and test
bounded public planning across turns. BC is an optional tool. PPO and reward shaping
remain deferred until evidence motivates them. Adapter maintenance continues behind
the shared Observation/Action contract while learning gets priority.

Close the already trained feature intervention with its prepared 384-game comparison,
then complete the independently audited current-student added-data-only experiment.
Keep these controlled comparisons separate from new value/search interventions.
No feature arm is selected: four fits passed integrity audit but extra eligibility
channels gave no exact-label validation gain. The older outcome cohort cannot supply
full recurrent histories; all 128 episodes omit some own observation views.

The latest native pilot scored raw 3/16, VF 0/16 and AB2 4/16. Raw-minus-AB was
−6.25pp [−25,12.5] across four boards; there is no superiority claim or finalist.
Earlier residual and placement studies also failed to establish reliable gains.
Use these results to retire unchanged interventions, not to equate audits or more
training with strength. Keep all evaluation artifacts out of training and final
boards `123100000..123100063` plus legacy finals sealed.

[Current results](training-status-2026-11-05.md) preserve exact counts and reviews.
[Historical results](training-history-through-checkpoint08.md) and
[previous roadmaps](../runs/swarm-2026-11-04/research-roadmap-history/) remain available.
