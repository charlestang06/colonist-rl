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

The 32-case TRAIN audit and six historical replays found consistent action
encoding and relevant gradients in the selected cases. The parents already use
30 recipe features. A separate eligibility contrast screen supports a bounded
four-channel comparison; two discarded profiles completed exact work and passed
their prospective resource audit. Four fixed-data fits then completed and passed independent metadata/resource
audit. Eligibility adds no validation exact-label gain over control; keep all four
weights and unchanged parents for complete-game comparisons. Those games and the
required separate audited-data experiment remain next; no model is promoted.

The evaluation-only focal VF/AB API is now implemented and passed 12 real
engineering games plus independent selector/outcome/resource audit. Raw/VF/AB
scores of 4/3/4 per four against GreedyNoTrade on one exposed board are not
strength evidence. The later fresh 48-game H-no-offers pilot completed and passed
independent audit: raw 3/16, VF 0/16, AB2 4/16. Paired raw-minus-VF was +18.75pp
[0,37.5]; raw-minus-AB was −6.25pp [−25,12.5]. Four clusters do not establish
superiority and the negative AB point cannot be rescued by VF. Keep this cohort
EVAL-only and do not retune on it. The extra macOS executable was pinned before
this new pilot; the older engineering provenance limitation remains historical. Three-native stress panels remain distinct. Keep final
boards `123100000..123100063` and legacy finals sealed.

[Current results](training-status-2026-11-05.md) link exact reviews and artifacts.
[Historical results](training-history-through-checkpoint08.md) and
[previous roadmaps](../runs/swarm-2026-11-04/research-roadmap-history/) remain preserved.
