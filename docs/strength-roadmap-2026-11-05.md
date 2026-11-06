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

The 384-game feature comparison is complete and independently accepted. Pooled
parent/control/eligible wins were GNT 17/25/26 per 32, trading H 4/4/4, native-VF
stress 3/7/7 and native-AB stress 3/7/7. The primary eligibility-minus-control H
contrast was zero on every board; the feature treatment does not advance. Secondary
refit gains are not focal-reference superiority or heuristic competence.

All 24 explanatory slots were judged, representing 19 unique public cases:
17 reasonable, two ambiguous, none clearly bad. Coverage was setup/bank/robber,
with no development-card or discard packet. These dependent EVAL cases cannot
become training targets. Required new current-student audited-data work remains.

The new engineering collector retained four complete games/361 chronological own
inputs. Its first cache failed on a fingerprint-format mismatch before observation
forwards; the loaded parent was not trained. Preserve the failed attempt and repair
only the cache on those saved histories. Main value collection/fitting remains
unstarted; no success claim follows from data persistence alone.

The latest native pilot scored raw 3/16, VF 0/16 and AB2 4/16. Raw-minus-AB was
−6.25pp [−25,12.5] across four boards; there is no superiority claim or finalist.
Earlier residual and placement studies also failed to establish reliable gains.
Use these results to retire unchanged interventions, not to equate audits or more
training with strength. Keep all evaluation artifacts out of training and final
boards `123100000..123100063` plus legacy finals sealed.

[Current results](training-status-2026-11-05.md) preserve exact counts and reviews.
[Historical results](training-history-through-checkpoint08.md) and
[previous roadmaps](../runs/swarm-2026-11-04/research-roadmap-history/) remain available.
