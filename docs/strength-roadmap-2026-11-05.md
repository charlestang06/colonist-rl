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
2. Improve longer consequences with bounded public planning and the independent
   terminal-outcome value experiment. Prediction gains require a separate gameplay
   deployment test. Neither mechanism depends on completing PPO or more BC.
3. Advance through harder opponents with a fixed evaluation panel. Compare the
   best credible candidate directly with focal VF/AB, rotate all seats, account
   for board-level uncertainty and replicate on fresh boards. PPO is optional
   after competence and an identified strategic limitation.
4. Transfer the decision maker through the shared Observation/Action contract.
   Keep adapter extraction, beliefs, action mapping and visible execution separate.
   Live Colonist capability and simulator strength require their own evidence.

Next primary work is strict zero-query recovery of the 48 recorded student games,
then the prepared independent case audit and frozen added-data-only comparison.
The independent bounded lane completes the 320-game terminal-history study.
Rank fresh TRAIN failures to choose the following tactical/planning remedy. No
new broad training run, model expansion or shaping is the default next step.

No candidate is promoted. The 384-game eligibility treatment failed its primary
rule; each candidate won 2/16 against trading H. The focal raw/VF/AB2 pilot scored
3/0/4 per 16 and establishes no superiority over both references. The limited
17-reasonable/two-ambiguous audit cannot establish broad competence. All 48 student
histories/4,729 own inputs survive a failed controller finalization; recovery,
approved labels and data benefit remain unestablished. Main value collection and
fitting remain unstarted. Engineering checks establish mechanisms, not strength.
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
