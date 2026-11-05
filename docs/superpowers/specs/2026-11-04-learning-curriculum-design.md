# Decision quality and sustained learning design

User authorized autonomous work until approximately06:54UTC on2026-11-05; reserve the last25minutes for integration. This is a separate experiment, preserving the original900second pilot unchanged. Pilot: clean1350de2 source,5000demonstrations,64complete validation games,92.30seconds. Both imitated students won0/8 and achieved32/83 non-forced agreement. All-choice60.16% masks weak choice quality. These observations do not diagnose insufficient capacity.

## Binding constraints

- Work only in /private/tmp/colonist-swarm-bxz1k39_/learning; original repository/environment read-only.
- Existing /Users/charlestang06/Documents/cs/colonist-rl/.venv/bin/python and ruff; PYTHONPATH points at learning/src, PYTHONDONTWRITEBYTECODE=1; one CPU worker and Torch thread, no installs/network/paid services.
- Observation-only actor, teacher and audit inputs; no hidden hands, deck order, private VP or future outcomes justify decisions.
- Preserve current imports/CLI/checkpoints/schema. No simulator/adapters/contracts/board/beliefs/opponent-strategy edits without root agreement.
- Clean committed unchanged source for real runs; exact source/config/seed/model/optimizer/RNG/artifact identities. Preserve old pilot bytes and all weak/failed outcomes.
- Censored outcomes remain unknown, strict block pairing and per-game persistence remain binding. Forced-only accuracy, loss and training authorization do not establish strength or checkpoint promotion.
- Fresh implementer, independent review, reviewed fixes and broad final review; coordinator does not implement production code.

## Approach

First compare longer uniform imitation with a controlled phase-balanced loss, using the same saved pilot datasets and initial_student weights. Then perform explicitly exploratory fresh PPO with gated opponent progression. Root/user permit exploratory PPO despite the old pilot's failed gate; preserve the old deferred verdict and make the new authorization/budget explicit. Do not expand architecture/reward/features before diagnosing optimization and choices.

Independent strategic audit already exports actual choices to learning-audits. Reuse its script and deterministic24–48choice stratification before/after learning. Every judgment is clearly_bad/reasonable/ambiguous, with confidence, public-state rationale and legal alternative for bad judgments. Teacher disagreement or higher production alone is not proof. Report opportunity denominators and absent phases. Frozen historical policies are diversity opponents, not inherently harder than heuristics.

## Supervised comparison

Seeds11/29, saved2500-example datasets and original initial_student checkpoints. Two independent configurations: uniform and phase-balanced, each20epochs, hidden32, sequence16, lr3e-4, gradclip0.5. Phase weights are training-only N/(P*n_phase), counting non-forced choices; divide weighted chunk CE sum by its non-forced choice count (global mean training weight is one). Forced observations advance recurrence without loss; unweighted validation metrics remain unchanged. Independent optimizer/RNG, identical starting weights/order per seed/config. Retain all runs and partial epochs.

Supervised validation boards62100000..62100003; diagnostic demonstrations62110000..62110003 with1024decision cap. Select initialization only on fixed heuristic meanVP, then winrate, then earliest deterministic tie; retain original imitation comparator. If unequal budgets complete, label unmatched comparisons inconclusive.

## PPO curriculum

Reuse collector/updater/checkpoints with optional explicit episode seeds, baseline lineup and cooperative deadline. Stages: all random; one heuristic/two random; all heuristic. Optional frozen diversity: two heuristics plus one immutable earlier checkpoint. Existing default mixed behavior must remain unchanged. Opponent changes apply only at new games. Independent models/optimizers/RNG/logs per seed; fresh current-policy trajectories only.

Check deadline before/after actors/proposals and before subsequent simulator mutations or optimizer updates. An interrupted learner interval is dropped/counted rather than fabricated; complete records keep correct bootstraps. No new calls after expiry, though an in-flight call can finish. Explicit seed exhaustion stops cleanly. Frozen parameters are eval/no-grad and have separate recurrent memory.

At4096decisions or stage end evaluate fixed heuristic panel62000000..62000003 and current fixed lineup on same boards. Advance only after>=2complete uncensored blocks, current-lineup win>=0.25,VP>=4,and <=1VP regression versus initial fixed-heuristic result. Failed gates retain stage and later budget there; report unreached stages. These are operational selection gates, not significant strength claims. Select checkpoint by fixed heuristic VP then win then earliest, reject truncation/>1VPregression; retain last too.

## Budget and held-out evaluation

Separate max4500second study: supervised<=900,PPO<=2700,evaluation/audit reserve900. Respect earlier absolute06:29UTC cutoff; shorten equal-seed budgets explicitly if time runs short. Profile<=120seconds and512fresh choices plus oneupdate within PPO budget, retain progress. Cap65536choices/seed, rollout128,PPOepochs2,seq16,lr3e-4,entropy0.01,value0.5,grad0.5. Measured throughput sets equal achievable caps.

Curriculum seed s board candidates=[8000000+100000*s+i for i in range(4096)], seeds11/29. Post-audit63000000/63000001,probe64000000/64000001,reservedfinal71001000..71001007. Validate disjointness against all prior actual/candidate splits; old70000017/70000019 final namespace stays untouched. Freeze selection before final32games per variant/seed(initial/selected) against heuristics. No post-test tuning; partial blocks remain unpaired/unknown. Report actual decisions/stages/walltime, wins/VP/truncations, paired board differences/dependence caveat, entropy/KL/clip/value/gradient/throughput, strategic audit patterns and failures. Large artifacts stay ignored runs; compact evidence/docs only after source-end verification.

## Final independent audit evidence

Read baseline-strategic-audit.md in learning-audits: 36 deterministic stratified actual choices, 24 reasonable, 11 ambiguous and one clear missed bank-trade-to-settlement sequence. Neither student selected a maritime trade across 143 legal opportunities. All 113 non-forced passes faced maritime alternatives; none passed a directly legal purchase. All 29 affordable incoming offers were accepted, which establishes missing refusal evidence rather than proving those trades bad. Near-uniform setup logits justify optimization diagnostics, not a capacity conclusion.

Phase balancing is a controlled hypothesis. Bank trade versus END_TURN shares PLAY_TURN, so phase weights alone may not fix conversion avoidance. For every compared checkpoint retain opportunity-aware diagnostics from actual play: maritime legal/selected counts, END_TURN when maritime legal, feasible accept/reject counts, direct-purchase legal count and passes over direct purchases. Use denominator counts, not raw chosen-action frequency. Reuse the audit replay script; no action-family weighting or strategic feature changes in this comparison. Any later change requires evidence and a separate ablation.

## Portability and prior diagnostic evidence

Production APIs must remain portable. The independent audit script lives in an external temporary directory and must not be imported by imitation_study or other library code through an absolute path. Task3 may reuse that analysis script as an experiment tool, or inject a narrow callback explicitly; the authoritative evaluator remains the only game loop. Legacy fit/warmstart continue to work without Git; real experiment runners still require exact clean source identity.

The separately recorded uniform diagnostic completed20epochs per seed in120.45/132.93seconds, improved non-forced agreement to43/83 and40/83, but stayed0/8wins with VP3.875/3.125. It is not the controlled uniform-versus-balanced study. Retain its provenance and do not replace equal-budget controls with those diagnostic results. Four similarly sized fits appear feasible within the new900second supervised cap, subject to actual deadlines.
