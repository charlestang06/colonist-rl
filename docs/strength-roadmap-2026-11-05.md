# Strength objective and next work

Direction revised 2026-11-06. The [canonical roadmap](agent-research-roadmap.md)
now defines the full goal, four milestones, evidence gates and execution queue.
Build an intelligent Catan decision maker from its player's observations, then
reproducibly outperform authentic pinned VF and AB in complete games. Preserve
all frozen experiment commitments and the user's separate [roadmap](roadmap.md).

1. Establish reliably good decisions and heuristic-level play. Diagnose costly
   mistakes in student-reached states, audit the teacher and legal alternatives,
   and test one targeted remedy. Cover placement; bank/build/pass; domestic
   responses; robber/discard; development-card purchase and play; road/army races;
   and VP/endgame. Missing coverage stays unknown.
2. Improve longer consequences with bounded public planning. Value learning may
   run alongside it when a concrete planning use justifies the study. Prediction
   gains require a separate gameplay deployment test. Neither mechanism depends
   on completing PPO or more BC.
3. Advance through harder opponents with a fixed evaluation panel. Compare the
   best credible candidate directly with focal VF/AB, rotate all seats, account
   for board-level uncertainty and replicate on fresh boards. PPO is optional
   after competence and an identified strategic limitation.
4. Transfer the decision maker through the shared Observation/Action contract.
   Keep adapter extraction, beliefs, action mapping and visible execution separate.
   Live Colonist capability and simulator strength require their own evidence.

The fixed two-panel audit completed 176 first passes and 49 adjudications for
88 selected cases. Its informative-data gate failed: a TRAIN upper bound of six
candidates across four strata, below the required 32 across six. No data-only fit
ran; no claim about BC or data benefit follows. Public fact checking confirms two student roads end
in enemy-blocked expansion leaves; award/defensive value remains uncertain, and
the teacher selected reasonable alternatives on both. A flagged trade has an
omitted city-funding route and is not certainly bad. Next primary work diagnoses
the road-choice mechanism and tests one narrow reachability/continuation remedy,
including adverse award, defensive and free-road cases, followed by fresh gameplay.
The separate value lane retains 208 independently audited TRAIN games. Collect
the missing 112 only with a concrete planning use and a corrected successor;
continuation 001 is on source-review HOLD and has not run. The original 320-game
attempt remains rejected; no automatic retry or cache/head fit follows.
Use further TRAIN evidence to choose the following tactical/planning remedy. No
new broad training run, model expansion or shaping is the default next step.

No candidate is promoted. The 384-game eligibility treatment failed its primary
rule; each candidate won 2/16 against trading H. The focal raw/VF/AB2 pilot scored
3/0/4 per 16 and establishes no superiority over both references. The limited
17-reasonable/two-ambiguous audit cannot establish broad competence. All 48 student
histories/4,729 own inputs survive a failed controller finalization. Separate
recovery passed 9,458 strict pairs and independent metadata review; approved labels
and data benefit remain unestablished. Main value collection is incomplete and
fitting remains unstarted. Engineering checks establish mechanisms, not strength.
See [current results](training-status-2026-11-05.md) for measured work and limits.

Bank trades remain legal whenever the rules permit them. Strict GreedyNoTrade
intentionally declines them; trading H is a separate control. Audit useful trades
and sensible passes. Resource scarcity and trade valuations are state-dependent
hypotheses. New reward shaping requires an isolated test of a demonstrated problem;
trade frequency or VP cannot rescue a failed winning comparison.

Keep final boards `123100000..123100063` and legacy finals untouched until a frozen
confirmation. Preserve failures, nulls, scoped resource HOLDs and exposed EVAL
roles. Reuse the existing harness and applicable reviews. Report actual collected
games, audited labels, fitted updates and strength results separately.
