# Accepted current execution rulings

Learning lead accepts the PM protocol below with root-approved overrides:

- Expanded control missed40 under its unchanged840-second caps; largest common
  completed five-epoch checkpoint is30. Partial-last benchmarks stay separate.
- Root's live hand_recipe run is frozen b4c012e, primary30 with completion-only
  fallback30/25/20/15/10/5, NEW1,200-second ceiling per seed. This supersedes the
  historical840-second recommendation below; report unequal compute ceilings.
- Task4 implements only acquisition comparison and prefix selection plus strict
  consumed-evidence checks; no feature/baseline reruns or general runner modes.
- Next acquisition uses4,000 queries per arm/seed from completed legacy30 starts,
  unchanged heuristic opponents and identical original components. Root declares
  actual fit/time ceilings before running. Query/choice counts match; update,
  weighting and forced-prefix costs can differ and must be disclosed.
- Original authorized unstaged roadmap L6–L10/data-scaling edits stay untouched.

These overrides and the exact Task4 brief supersede conflicting recommendations.
The following protocol does not itself start a run or extend a historical budget.

---

# Immediate BC and DAgger execution protocol

November 5, 2026. **Recommended rulings for the learning lead to accept**, not a
change to the current source/spec. Task 1 fitter and Task 2 consequence policy are
reviewed; Task 3 collection is active. Prioritize actual comparisons over a general
experiment framework. Do not modify the user's original, unstaged L6–L10 roadmap
additions; root preserves and integrates them.

## 1. Consume the expanded control; do not retrain it

Use `/private/tmp/colonist-swarm-bxz1k39_/expanded-bc/artifacts/manifest.json`,
its training records, saved checkpoints and completed evaluation evidence. Frozen
control source is `e845e99138b3c8830e25c332c08d78758e411280`. Each seed 11/29 uses
its original pilot initial checkpoint, exactly 10,000 expanded training
observations, ordered forced-inclusive histories, training-only phase weights,
sequence length 16, batch size 1, learning rate 0.0003 and one persistent Adam
object across five-epoch calls. The expanded 20,000 teacher labels replay exactly.
The shared held-out file has 1,024 observations / 593 non-forced choices. Preserve
every input byte and its original provenance.

**Endpoint ruling before feature training:** target 40 completed epochs. If both
control seeds do not complete 40 within their 840-second fit/curve/checkpoint caps,
choose the largest five-epoch milestone fully completed by both, without consulting
CE, wins or VP. Record this common endpoint `E` and its exact checkpoint digests
before feature fitting. “Five-epoch checkpoint” means a saved full workload
milestone, not a new epoch-5 initialization or optimizer reset.

Eligibility uses complete fit-call status and exact counters, not only checkpoint
metadata `partial`: current intermediate saves can carry `partial=true` while a
five-epoch call completed. A partial final call can also contain updates beyond its
reported completed epochs. Exclude such a state from a full-epoch endpoint unless
an earlier exact full checkpoint is available. Never relabel a 38-epoch/part-epoch
last state as a 40-epoch control. Keep its original operational evaluation separate.

## 2. Run only the hand-recipe feature comparison first

For each seed, load the exact original initial checkpoint used by the control;
upgrade with reviewed `consequence_from_legacy(..., feature_set="hand_recipe")`.
Assert identical base weights and initial logits/value/memory, and an all-zero
additive branch. Record the new architecture identity. Use the exact same training
files, episode/menu order, frozen phase-weight mapping, held-out file, default
batch size 1 and persistent Adam setup as the control. Train to `E` with saved
five-epoch milestones; do not warm-start from a trained control while calling the
comparison equal-start. Keep connectivity features disabled.

Give each feature seed the same declared 840-second ceiling, including curves and
checkpoints. Predeclare the endpoint fallback ladder `E, E-5, ..., 5` before work:
if an arm cannot reach `E`, the primary target is incomplete; a secondary comparison
may use the largest checkpoint fully completed across both control and both
feature seeds. Select solely by completion counters, never performance. Report the
missed primary target and actual presentations/updates/time. If there is no common
full checkpoint, do not publish a feature-effect comparison.

Reuse control game results only when checkpoint digest, deterministic policy mode,
opponents and full schedule match the selected endpoint. Otherwise evaluate the
already-saved control checkpoint; this repeats evaluation, not training. The
primary panel is **72010000..72010015 × all four seats = 64 games per arm/seed**,
against three fixed heuristics. Reuse the matching heuristic reference. Strict
paired results require the full matching uncensored panel; retain partial rows
without subset salvage. Preserve both initialization results and the unweighted
593-choice held-out metrics with phase/action denominators.

## 3. Next prioritize student-state labels versus equal-query teacher-state labels

After Task 3 review, freeze one starting architecture/checkpoint per seed before
acquisition; use the same starting checkpoint in both acquisition arms. Prefer
the completed legacy BC baseline unless the separately measured feature result
justifies a recorded choice. Do not jointly change architecture, teacher, opponent
mixture and data acquisition. Use a fixed declared integer query cap `N` per
arm/seed and equal collection wall ceilings; all learner decisions, including
forced ones, count as teacher queries. The original teacher remains unchanged.

Collect DAgger with actual student behavior and audited legal teacher labels on
the same player-visible observations. Collect the teacher-state control under
teacher behavior with the same opponent setting and matched disjoint training
board schedules. Record actual queried and executed decisions separately. Preserve
every ordered student observation, forced history, teacher RNG call order,
behavior/label distinction, component source/checkpoint identities and interruption
flags. No hidden-state labels or fake merged legacy v1 dataset are permitted.

**Equal-work ruling:** equal all-decision queries do not imply equal non-forced
supervision. After collection, before any fitting or outcome-based selection, set
`Q_s = min(non_forced_teacher_s, non_forced_student_s)` for each initialization.
If either collection misses `N`, retain it as partial and do not label the planned
equal-query comparison complete. If `Q_s=0`, no supervised comparison is available.

For each new component, deterministically retain the ordered episode prefix through
the `Q_s`-th non-forced observation. Retain **every earlier observation**, including
forced steps and entire earlier episodes. Cut only the final retained episode's
suffix; never cherry-pick rare labels, splice unrelated contexts or reset recurrent
memory in the middle. Original shared training components remain identical in both
arms. Raw component files, collection counts, episode completion flags and sources
are unchanged; only a separate derived selection manifest defines the training
view. A clipped training view is not falsely marked as a completed raw episode.

The selection manifest records component digest; ordered retained episode/sample
endpoints; `Q_s`; forced/all observations retained; unused suffix counts/locations;
actual board identities; derivation-source identity and selection rule. Project
this view directly to reviewed `EpisodeSequence` inputs, not to an invented v1
collection. Preserve complete prefixes for current-weight memory replay.

Reuse one frozen training-only phase-weight dictionary across both arms, recording
its source. Preflight that every retained non-forced phase has an explicit positive
weight; never drop an uncovered phase. Use the same epochs, batch size, sequence
length, learning rate and equal fresh/persistent optimizer treatment in both arms.
Do not compare a resumed optimizer on one side with a reset optimizer on the other.

This matches **acquisition queries** and **new non-forced presentations per epoch**
within each seed, not every unit of work. Different forced-history/chunk structure
can produce different optimizer-update counts, weighted loss mass, prefix forwards
and runtime. Precompute expected updates from the reviewed 16-observation chunk
schedule, then record actual updates, presentations, weights, prefix work and time.
Do not claim “identical optimization budget” or attribute every gain solely to
state distribution. Keep that limitation explicit; exact update-matching is a
separate diagnostic only if the result warrants it, not a new framework now.

## 4. Keep Task 4 a small reproducible driver

Use existing load/save/checkpoint, `fit_sequences`, `score_sequences`, corpus,
`evaluate_blocks`, paired summaries and atomic JSON helpers. One explicit driver
with feature and acquisition-comparison modes is enough; no plugin registry,
generic study DSL, replacement simulator loop or alternate PPO implementation.
The prefix selector is a small pure function plus manifest, not a dataset rewrite.

Required focused tests before real use:

- Exact control digest/settings checks; complete-milestone selection ignores all
  metrics and rejects partial-call states with extra updates.
- Feature zero-start/base-parameter equality, persistent Adam across calls, and
  strict checkpoint architecture loading; no automatic connectivity upgrade.
- Prefix selection retains forced observations before the cutoff, handles a cutoff
  inside an episode, selects exactly `Q_s` choices, records all unused suffixes,
  leaves raw bytes/source metadata unchanged, and does not cross episode boundaries.
- Disjoint training/held-out/game/final partitions, legal labels, full component
  provenance, missing-arm/seed rejection and actual-work accounting.
- Deadline/exception recovery leaves last checkpoint and completed-game journals;
  per-game persistence remains authoritative, source/input drift fails closed,
  censored/missing games cannot become wins/losses or a complete comparison.

Before compute, persist the reviewed source identity, exact inputs, selection
manifests, expected arm/seed matrix, caps, endpoint rules and evaluation schedule.
Persist checkpoints/progress after each completed fit segment and each game;
retain partial/failure artifacts and do not restart silently. Finish with source
and input hash verification. The learning lead accepts these rulings and declares
actual query/time caps; this protocol adds no automatic horizon extension.

## 5. Keep the remaining order clear

Consume control → hand-recipe comparison → student-state/equal-query control →
complete-game and public-decision audit → existing PPO curriculum last. Keep its
fixed 16-board panel aligned with BC/DAgger and reward unchanged. Reserved final
boards remain unused until a frozen selection; no post-test tuning.

The shaped PPO critic is **not a win probability**; BC does not train it through
the classification loss. Report its target correctly. A public trade→build
lookahead teacher variant remains conditional after these experiments: first
show the teacher itself misses useful visible continuations, then benchmark the
variant against the unchanged heuristic. Student disagreement alone is not that
evidence. L6–L10 and longer-term features must not postpone these concrete runs.
