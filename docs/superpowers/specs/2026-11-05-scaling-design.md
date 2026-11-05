# Shared non-forced corpus scaling design

User-authorized November 5 stage; planning only. Root-created checkout:
`/private/tmp/colonist-swarm-bxz1k39_/scaling/repo`, branch
`cytang/scaling-learning`, accepted BASE `9be7529` (root0c612b4 plus reviewed
engine Task1). Root schedules one fresh implementer; no worker is spawned here.

## Objective and boundaries

Compare shared nested10k/100k **non-forced observations** across initializations
11/29 with hand_recipe only. Immutable four-game components bound memory; one
component at a time feeds the existing batch-one fitter and persistent Adam.
A monolithic corpus risks multi-GB memory; single-board collector calls repeat
learner seat0; cached learned/hidden state would change recurrent optimization.

One narrow driver `src/colonist_rl/scaling_study.py`, focused tests and usage doc.
No generic framework, CLI, new fitter API, internal concurrency scheduler, teacher,
contract, rule, reward, connectivity, priors or mixed-teacher changes. Do not touch
original roadmap/runtime, active legacy DAgger, frozen studies or migration owners.

Use the isolated upgraded interpreter and sibling Ruff only:
`/var/folders/0l/p7x7d7755vq4wc8qty73kcy40000gn/T/colonist-master-upgrade-trvuh2kq/repo/.venv/bin/python`
with PYTHONPATH=scaling/repo/src, bytecode disabled, OMP/MKL/Torch1. No installs.
Require verify_engine_installation(), exact backend
`3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`, official_spiral, seed offset1,
and clean imported-source Git identity. No cross-engine performance parity claim.
Root supplies accepted weight-only bridged **untrained** starts, then zero-additive
hand_recipe conversion: hidden32, steps0, optimizerNone, preserved parent/bridge
identity. Within each seed both sizes start from the identical file/digest. Do not
import the absent active checkpoint_migration module or implement its behavior.

## Immutable component corpus

Collect unchanged heuristic against three heuristics with collect_demonstrations
on four distinct ordered boards per call: max_decisions80004, max_game_steps20000,
per-batch deadline min(collection/global deadline, now+120s). Each component must
have four natural terminal episodes, learner seats0/1/2/3, all forced/earlier own
observations and legal label/menu order. This balances seats across four different
boards, not four seats on one board. Count distinct non-forced observations using
board/seat/version identities; repeated presentations/opponent moves do not count.

Finish whole batches before threshold checks. Save raw incomplete/truncated
components with reasons and stop primary collection; never replace difficult
boards. Eligible corpus is preceding complete prefix. Milestones10k/25k/50k/100k
are first complete-component prefixes crossing those counts, with honest overshoot.
Only10k and100k are fitted; both seeds use exactly the same ordered component refs.

Freeze unique/disjoint training, validation_labels, validation_games, audit and
final_test lists and check prior manifests before launch. Proposed ranges:
training114000000..114016383; games115000000..115000015;
labels116000000..116000031; audit117000000..117000003;
final118000000..118000015. These are candidates, not asserted fresh. Preserve all
older final reservations. Validation labels use separate complete components.

Study-specific manifest format `colonist-rl.scaling-corpus`, version1: ordered
path/SHA refs, original source/backend, board/seat/phase/kind/count coverage,
completeness, nested endpoint ref counts, partitions/config/runtime/derivation
identity. Never fabricate a merged demonstrations-v1 source or rewrite raw bytes.
Before each load verify digest, then exact typed metadata, legal labels, ordered
versions, full histories and source/runtime/partition agreement. Replay original
teacher board+100 RNG and call order per component, including forced decisions.
Use unique exclusive staging/atomic publication; isolate accepted fixed.tmp save
helpers rather than exposing inputs to predictable-stage alias corruption.

## Bounded-memory fitting

Never use whole-corpus write_corpus/load_corpus/corpus_sequences. Sequentially
validate/replay/count one component, release it, retain only primitive summaries.
Freeze N/(P*n_phase) from the finalized full **training** corpus, shared by both
sizes/seeds; missing supervised phases fail. Report that10k uses phase-distribution
weights derived from100k training data, never validation. Primary fitting waits
for these shared weights. Stopping at the first10k gate means100k remains unmet;
a smaller-only diagnostic requires a separate declaration.

Fixed hidden32 hand_recipe, sequence16, batch1, LR0.0003, five global epochs.
Order: epoch -> component -> episode -> existing chunks. Call fit_sequences with
epochs1, shared phase_weights, profile=True and the SAME caller Adam per arm.
Release the component before next load. Episodes never cross component boundaries;
current-weight prefixes replay before updates; no learned/hidden cache. Avoid
imitation wrappers' extra full scoring. Aggregate loss numerator as
sum(call.loss * call.supervised_presentations), not mean of component means.
Record weighted presentation mass, updates/presentations/forward counts, and fit,
load/replay, scoring/checkpoint seconds separately.

Stream held-out score_sequences at initialization and complete epochs1/3/5;
aggregate count/correct/CE sums and phase/kind denominators, not averages of means.
Check peak RSS at component load/release, fit/score/checkpoint boundaries. Root's
hard monitor complements cooperative checks during one component. Cap2GiB/process;
only measured host-safe execution permits two processes, each Torch1. Unit tests
verify one live component, not flaky RSS deltas. First10k prefix is a measured gate:
bytes, load/replay time, episode lengths, prefix work, RSS and remaining-cost estimate.

## Durable boundaries and exact optimization gate

Publish a unique boundary checkpoint containing optimizer/RNG/counters/cursor,
then atomically publish cursor referencing its SHA. Cursor identifies source,
config/corpus/start hashes, seed/size, completed global epochs and next component.
Crash between files leaves an orphan, never a cursor claiming an unsaved state.
Partial/exception snapshot is distinct; do not advance completed cursor. Explicit
resume verifies all identities, loads last boundary optimizer/RNG, replays the whole
interrupted component and records rollback/attempted work separately. Never resume
a recurrent suffix, double-count committed progress, silently restart or extend
caps. Epoch completes only after every selected component completes.

Parity test: two complete components, forced observations and multiple chunks,
two epochs; compare one concatenated fit with canonical component calls and one
persistent Adam. Require bitwise policy tensors, Adam steps/moments, counters and
unchanged input bytes. Aggregated returned loss agrees at float64 relative/absolute
1e-12 tolerance (root-approved): existing API rounds per-call means, so scalar
regrouping is not bitwise-loss evidence. Actual update ordering is identical.
Tests must detect optimizer reset, order change or removed forced history.

Public entry points preserve caller Python/NumPy/CPU-Torch RNG and policy modes;
checkpoint RNG remains the explicit saved continuation state. Test restoration on
normal and exceptional exits.

## Results and budgets

Four arms:10k/100k ×11/29. Preserve immutable complete epoch1/3/5 checkpoints.
Before gameplay choose largest complete common checkpoint in {1,3,5}, solely from
completion/counters. No common milestone means no comparison. Show primary-five
failure even if fallback exists. Equal epochs intentionally mean unequal updates/
presentations: data-plus-training scaling, not a pure data-volume causal effect.

Use existing evaluate_blocks and authoritative per-game journal for four selected
candidates plus unchanged teacher on same fresh16boards×4seats, deterministic vs
three heuristics. Strict paired_block_summary rejects missing/censored/mismatched
panels. Keep all failures and board-level uncertainty. No final test by default,
no PPO or automatic promotion; useful imitation and decision audit precede PPO.

Root finalizes launch caps; proposed collection/validation/replay<=5400s,
each10k arm<=900s, each100k arm<=7200s, evaluation<=1200s, whole<=14400s.
Earlier global/stage deadline always wins. Even two-worker maxima sum14700s before
overhead, so caps do not promise four-hour completion; retain honest milestones.
Development smoke<=60s TOTAL, meaningful tests and independent review first.
Actual100k collection waits for legacy DAgger completion and migration final
integration. This plan does not launch work or reset prior horizons/budgets.
