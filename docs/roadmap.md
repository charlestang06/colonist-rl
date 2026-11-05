# Product roadmap

Planning snapshot: November 4, 2026, updated after the completed guest game and
accepted evaluation, imitation, UI-contract and semantic-parser foundations.
Status distinguishes delivered foundations
from remaining work. Learning is the primary workstream: establish a competent
teacher, a competent imitated policy, and measured PPO improvement. Further
browser work is secondary and does not block simulator training or evaluation.
Adapters target the shared player-visible Observation/Action contracts already
used by the Catanatron wrapper, not the engine's privileged full state.

## Starting point

- Four-player simulation, graph/GRU PPO, beliefs, incoming trade decisions,
  checkpoint/resume, and seat-rotated evaluation already exist. Historical combined
  verification after adapter modularization and the offline log reader: **205 tests passed in
  5.06 seconds, no skips**, with the real loopback Chromium smoke test enabled.
  Ruff lint/format checks passed across 39 files and the wheel built
  successfully. Independent adapter and log-reader reviews found no actionable
  issues. These checks establish integration and packaging, not policy strength.
- Strict evaluation with explicit board/seed splits, censored-outcome accounting
  and resolved four-seat paired comparisons is implemented and accepted after
  re-review. Persisted observation-only demonstrations, current-weight recurrent
  imitation and deadline guards are also implemented and accepted. See the
  [learning protocol](learning-protocol.md). The clean-source pilot completed
  at `1350de2`; its first-wave Task 3 passed scoped re-review and was integrated.
  Broad learning review remains pending with an installed-wheel legacy warm-start
  Git-dependency regression awaiting correction and re-review.
- An **intermediate** combined integration check passed **437 tests, 1 optional
  browser skip in 8.96 seconds**, with Ruff lint/format clean across 49 source/test files.
  The final broad adapter review was clean and both documentation minors were
  resolved. This is not final verification of the pending review fixes or subsequent
  cleanup/status changes, and does not claim a new wheel or live-browser run.
- Shared observation JSON, visible-event exports, calibration, and capture
  readers now live in `adapters/`; Colonist browser transport/execution live in
  `adapters/colonist/`; physical input lives in `adapters/physical/`. Legacy
  perception/browser imports remain compatible. See [adapters](adapters.md).
- The screenshot reader recognizes verified annotations by image hash. It does
  not recognize an arbitrary live screenshot or physical board photograph.
- `BrowserAdapter.execute` performs one click, then requires an immediate
  committed game change. `BrowserTransport` has no drag operation. The live
  [controls evidence](colonist-controls.md) demonstrates intermediate
  confirmations, focus/drag requirements, several simultaneous offer rows, and
  incoming acceptance that expresses willingness before any transfer.
  The new [UI contracts](colonist-ui.md) provide immutable records, pure exact
  bindings and pending/committed result descriptions. They issue no gestures and
  do not change the shared observation schema or wire staged execution.
- The offline [log reader](colonist-log-reader.md) now preserves ordered text/
  icon tokens and reports conservative snapshot overlap, ambiguity, and gaps.
  Two real ordered DOM captures are fixtures. Their row IDs are all absent;
  unique content overlap is a hypothesis, not established event identity.
  [Semantic candidate parsing](colonist-log-semantics.md) now supports production,
  completed bank exchanges and two-die rolls, preserving unknown evidence and
  unresolved actors. It establishes no event IDs and performs no ledger updates.
  Broader grammars, continuous DOM collection, history recovery and ledger
  integration remain pending. `BeliefLedger.reconcile` intentionally
  rejects unexplained count changes rather than inventing a transfer.
- The manually controlled guest game against three Easy bots ended in a visible
  **10-point victory, 67 turns, 27:27**, saved in
  [16-victory.jpg](assets/colonist-walkthrough-2026-11-04/16-victory.jpg). Three
  Knights earned Largest Army; Longest Road at five was retained through a tie;
  two Monopolies collected seven Ore and five Wool; two private VP cards counted
  automatically and the second triggered victory. Own Road Building and Year
  of Plenty were not drawn; Year of Plenty was observed on bots. The
  [controls guide](colonist-controls.md) records verified and missing flows.
  This establishes manual control and termination, not autonomous policy play,
  complete resource-ledger reconstruction, or learned-policy strength.
- The [completed learning pilot](learning-pilot-2026-11-04.md) took 92.301 seconds:
  5,000 demonstrations, 64 natural validation terminations and four probe games,
  with no evaluation truncations. The teacher won 8/8 against random opponents
  and 3/8 against heuristics; both students won 0/8 against heuristics and matched
  only 32/83 non-forced held-out choices. The competence gate failed and no PPO
  ran in that pilot. A separate 20-epoch uniform-imitation diagnostic improved
  agreement to 43/83 and 40/83, but both still won 0/8; mean VP was 3.875 and 3.125
  versus 3.375 and 3.250 at two epochs. These tiny reused validation boards show
  optimization progress without established playing strength or capacity limits.
- The public-observation strategic audit found 24 reasonable, 11 ambiguous and
  one clearly bad choice among 36 stratified samples. Across the actual policy
  decisions it observed zero maritime trades in 143 opportunities and all 29
  feasible incoming trades accepted, but no end turn skipped a directly legal
  build or development purchase. Investigate intermediate conversions and offer
  selectivity rather than treating all passes or teacher disagreements as errors.
  See [the result-driven plan and evidence](swarm-roadmap.md).

## Order and parallel work

| Priority | Item | Outcome | Status / dependencies |
| --- | --- | --- | --- |
| P0 | L1 | Benchmark the teacher and establish competent imitation | Pilot completed; students weak; test planned balanced imitation and coverage/optimization diagnostics |
| P0 | A9 | Reproducible training and evaluation comparison | Pilot scoped review complete; broad installed-wheel compatibility fix pending; larger playing evidence needed |
| P1 | L2 | Critic warm start and measured PPO improvement | After the imitation baseline; fresh learner experience |
| Done | A1 | Shared, Colonist, and physical adapter boundaries | Implemented, independently reviewed, combined verification passed |
| P2 | A2 | Small, replayable evidence fixture corpus | Ordered log fixtures and synthetic UI/semantic examples delivered; observed transition traces remain |
| P2 | A3 | Rendered log normalization and semantic projection | Production/bank/dice candidate parser accepted; broader grammars and event integration pending |
| Done | A4 | Explicit UI state and execution-result contract | Pure records/validators accepted; no gestures or executor wiring |
| P2 | A5 | Continuous log history and honest gap handling | Offline overlap/gap reporting delivered; recovery and ledger integration pending; A3 |
| P2 | A6 | Verified placement and development-card sequences | Staged Task 1 fixing post-input terminal evidence/lock findings; unaccepted; drag gated |
| P2 | A7 | Pending trade responses and exact finalization | Bounded staged support under review; full finalization pending; A4; A2 |
| P2 | A8 | Verified observation assembly and bounded end-to-end replay | Pending; A5; A6; A7 |

Adapter modularization, offline log normalization, bounded semantic candidate
parsing and pure UI contracts are delivered and reviewed. L1/A9 evaluation and
imitation infrastructure is accepted. The next priority remains L1 together
with A9: resolve the installed-package regression, test coverage and optimization
through the planned balanced-imitation comparison, and run the separately
authorized bounded fresh-PPO curriculum against measured harder opponents.
The original pilot's gate remains failed; exploratory PPO does not change that
record or imply promotion. L3–L5 remain conditional on measured limitations rather
than prerequisites, with no architecture expansion inferred from the tiny studies.

**Current autonomous horizon:** The user authorized three hours without
questions, **03:54–06:54 UTC on November 5**, to judge actual policy decisions
conceptually and train extensively against harder opponents. Reserve the final
approximately 25 minutes, starting **06:29 UTC**, for integration and verification.
Use completed pilot/audit/diagnostic results to prioritize bounded longer training with
competent fixed/frozen opponents while preserving fixed validation benchmarks.
The baseline audit is complete; the follow-on plan covers balanced imitation and
configurable fresh-PPO curriculum;
this does not add mandatory adapter or architecture scope. Strength gates govern
promotion, while correctly instrumented exploratory training can use the
remaining declared budget. Audit only player-visible observations, distinguish
dominated choices from reasonable/ambiguous ones, and never equate teacher
disagreement with error. See the [horizon and audit criteria](swarm-roadmap.md).

When adapter work resumes, A4 provides reviewed immutable UI records and pure
validators separating acknowledgement from a committed game action. A2 observed
transition evidence, broader A3 grammars and A5 history/event identity remain;
A6/A7 staged executor work is correcting post-input evidence and lock findings;
acceptance and further gesture work remain gated on re-review. No live gameplay
is claimed, and supporting evidence remains necessary.
Preserve the shared observation/action boundary and audit
rule differences before deployment, especially multi-stage trades. Physical-board
photos still need supplementary hand/history/context information. Neither UI
gestures nor photo recognition belong in the learning loop.

### A1 — Separate adapters while preserving public behavior

**Status:** Implemented in the chosen `adapters/` root layout, with compatibility
facades, physical reader entry point, focused isolation/identity tests, wheel
coverage, and clean independent review. Final combined tests, lint/format, and
wheel checks passed; this item does not change one-click executor semantics.

**Goal:** Allow simulator training, Colonist capture/execution, and physical
capture to evolve independently.

**Scope:** Move codecs/validation, calibration, capture metadata, readers, and
errors into the shared adapter layer; move transport and UI execution into the
Colonist package; add a small physical input entry point accepting an injected
reader or complete verified annotation. Keep `contracts.py` and `board.py`
backend independent. Retain legacy `colonist_rl.perception` and
`colonist_rl.browser` imports as compatibility facades. Shared modules live at
the `adapters/` root.

**Dependencies:** None. Complete the move before changing UI semantics.

**Acceptance checks:** Existing tests and CLI export/capture behavior remain
unchanged; legacy and new imports have identical class/exception identity;
version-1 JSON roundtrips remain unchanged; the existing browser startup
monkeypatch seam still works; importing shared/physical modules imports no
Playwright or browser session; wheel contents include the new modules. A photo
with incomplete evidence cannot silently become a complete `Observation`.

### A2 — Turn walkthrough evidence into offline replay fixtures

**Status:** Partially delivered. Two ordered real DOM log snapshots are saved in
`tests/fixtures/colonist-log-001.json` and `colonist-log-002.json`; durable control
screenshots include the final victory. Full annotated UI transition replay
fixtures remain to be created. The accepted UI scenario catalog and semantic
bank-exchange examples are explicitly synthetic; they are not observed
transition traces or ordered bank DOM captures.

**Goal:** Make later adapter work independently testable without another live
game or network access.

**Scope:** Promote a small set of observed before/intermediate/after sequences:
placement confirmation, focus/drag and Knight confirmation, robber confirmation
with and without a victim chooser, incoming willingness with no transfer,
completed/cancelled player trade, simultaneous outgoing offers, and bank trade.
Save screenshots and ordered rendered log-row text/icon tokens with source and
capture order. Label observed versus synthetic cases explicitly. Use synthetic
fixtures for missing controls until they are actually observed.

The older `runs/colonist-walkthrough-2026-11-04/trace-*.json` files contain body
text and flat image `alt`/context lists. They are examples, not proof of unique
message identity or complete ordering. Do not reconstruct ambiguous repeated
rows from those flat lists. New `log-rows-001.json` and `log-rows-002.json` retain
ordered row tokens, but their sources supplied no stable row IDs. A screenshot
alone is not a transition fixture.

**Dependencies:** None; coordinate file ownership with A1 and the live recorder.

**Acceptance checks:** Fixtures load locally, record expected UI stages and
committed game changes separately, and include ambiguous/timeout outcomes.
Identical text from two distinct events is represented. Fixture playback never
opens an external page. A short coverage table names remaining unobserved flows
such as own discard, port trade, incoming edit, own Road Building/Year of Plenty,
and post-victory controls. Termination, Monopoly, Largest Army and a Longest Road
tie now have observed evidence; they still need explicit replay assertions.

### A3 — Project rendered log evidence into explicit semantic records

**Status:** Immutable token/row/snapshot normalization and conservative history
reporting are delivered. A pure semantic parser for production, completed bank
exchanges and complete two-die rolls is now implemented and accepted; final broad
adapter review found no outstanding issues. It preserves source rows, unknown
resources, unresolved actors and unsupported wording. Outputs are candidates,
not established events: no event IDs, `VisibleEvent` creation, history-completeness
claim or ledger integration is added. Bank ordering is covered by explicitly
synthetic evidence; live ordered bank DOM capture and automated collection remain
pending. See [semantic candidates](colonist-log-semantics.md).

**Goal:** Recover resource identities that body `innerText` omits without
inventing hidden information.

**Scope:** Build on the existing token normalizer with a pure semantic parser;
start with production and completed bank-trade messages. Resolve players through
an explicit visible-name mapping and retain source rows, unknown identities, and
unparsed evidence. Return candidate operations separately from history identity
and ledger application. Add build, completed player trade, steal, and discard
grammars incrementally. A later optional rendered DOM collector can supply the
same snapshot format; selector changes must produce a diagnostic rather than an
empty successful log. Do not mutate the ledger in the semantic parser.

**Dependencies:** A2 examples; integrate under the A1 Colonist package.

**Acceptance checks:** Tests cover mixed text/icons, repeated resource icons,
unknown/card-back icons, duplicate player names or missing player mapping, and
both complete visible dice faces. Unknown identities stay unknown. An offer or
acceptance marker produces no transfer event. Private identity is included only
when the rendered observer view disclosed it. No internal game objects, hidden
DOM fields, storage, or private API responses enter the parser.

### A4 — Model intermediate UI states explicitly

**Status:** Delivered and accepted: immutable UI/control/trade/result records,
pure exact-action/revision/row/partner validators and a synthetic scenario
catalog. Final broad review found no outstanding issues. The module issues no
gestures, starts no browser and changes no shared `Observation` schema. It does
not independently verify caller-supplied commitment evidence. The existing
one-click executor remains unchanged; staged executor wiring is A6/A7. See
[UI contracts and limits](colonist-ui.md).

**Goal:** Describe a successful gesture that has not yet committed a game action.

**Scope:** Add Colonist-specific immutable UI state and execution results outside
the policy's `Observation`. Track UI revision, game observation version, phase,
focused/active tool, exact selected action/target, dialog stage, confirmation
control, and visible trade rows/response markers. Specify results for committed,
pending, blocked, and ambiguous outcomes. Bind each confirmation to the complete
typed action; bind trade rows to proposer, terms, identity and current UI
revision, with a partner when finalizing. Keep simulator atomic trades intact.

**Dependencies:** A1 and representative A2 stages. Review before A6/A7.

**Acceptance checks:** A UI acknowledgement can advance UI revision without
changing resources or requiring a higher game observation version. Tests reject
stale revisions, same-kind confirmations for another target, changed offer
terms, wrong rows, and wrong partners. No click or pending response updates the
belief ledger. Existing `execute` behavior remains available through an
explicit compatibility path; it must not silently report pending as committed.

### A5 — Reconcile virtualized history and handle gaps

**Status:** The offline historian now reports idempotent capture replay,
candidate overlap, ambiguity, gaps, and incomplete initial history; its
independent review was clean and combined verification passed. It deliberately
never certifies complete history
and creates no `VisibleEvent`, resource delta, or ledger update. Continuous
collection, scroll recovery, event identity, and safe ledger epochs remain work.

**Goal:** Apply each observed resource event once and retain uncertainty when
history is incomplete.

**Scope:** Extend the offline candidate-overlap foundation with continuous
collection and explicit coverage/recovery handling. Use a stable rendered
message ID when available, without fabricating one when absent. Retain
source-local event identity separately from shared decision versions. Preserve
legitimate identical messages. Recover accessible older rows by ordinary log
scrolling when possible. Convert only established events into ledger operations;
`new_rows` relative to an unresolved window cannot establish complete history.

If recovery fails, preserve the old ledger and mark the interval incomplete.
When current own hand and public totals are verified, allow an explicit new
ledger epoch initialized from that evidence, with conservative opponent bounds
and a provenance link to the gap. Do not replay guessed deltas or force the old
ledger to accept contradictory totals. Pause resource-dependent execution while
the current observation cannot be established.

**Dependencies:** A3. Reuse current `BeliefLedger` operations and exactness rules.

**Acceptance checks:** Ordered overlap, repeated identical rows, re-rendered
rows, reconnects, no-overlap snapshots, scroll recovery, and late startup all
have deterministic replay tests. Replaying a snapshot changes no hand twice.
Unresolved gaps never certify exact opponent hands; sample agreement remains
insufficient. Recovered complete history matches a ground-truth visible-event
fixture. Failure/recovery does not mutate a previously valid ledger halfway.

### A6 — Execute verified placement and development-card stages

**Status:** Follow-on staged Task 1 is implemented but not accepted. Independent
review found two Important post-input terminal-evidence/review-lock gaps, now
being corrected, plus a minor nested-argument exception boundary. Drag work is
gated. The first-wave pure contracts remain accepted; no live staged execution
or autonomous gameplay is claimed.

**Goal:** Turn a legal action into the observed sequence of gestures and
confirmations, with bounded progress and a verifiable outcome.

**Scope:** Add a drag capability to the Colonist transport; establish focus/tool
entry, select the exact target, inspect its confirmation, confirm once, and
verify commitment. Support the observed shorter direct-placement path too.
Implement Knight as separate card-confirmation, robber-target-confirmation, and
optional victim-selection stages. Add the observed Monopoly confirmation and
resource-choice sequence after those stages work. Poll for stable UI changes
within a fixed deadline/step budget. Own Road Building, Year of Plenty and
discard controls still require evidence before live support. Keyboard shortcuts
require verified focus and phase.

**Dependencies:** A4, A2 fixtures, and existing calibration.

**Acceptance checks:** Offline replay covers a drag with no effect, tooltip
instead of tool entry, delayed rendering, changed target confirmation, automatic
single-victim steal, multiple-victim selection, and interruption after each
gesture. Each gesture is issued at most once per established stage. Unexpected
or timed-out post-gesture states preserve the review lock; they never trigger a
blind retry. A completion result identifies committed evidence, not merely a
drag return or vanished dialog. Existing transport origin checks remain intact.

### A7 — Separate trade willingness from resource commitment

**Status:** Bounded pending-response support belongs to the unaccepted staged
Task 1 correction/re-review above. Full outgoing finalization and the broader
flows remain pending. Manual evidence confirms willingness, commitment, row
selection and composer behavior; it is not an autonomous execution result.

**Goal:** Execute existing binary trade decisions correctly on Colonist's
multi-stage UI.

**Scope:** Accept/reject the exact live incoming row and return a pending result
after willingness is verified. Observe later completion, cancellation, or
selection of a different partner. For explicitly controlled outgoing trades,
bind finalization to the correct row and willing partner. Inspect retained
composer quantities before a new offer; clear or adjust them explicitly.
Verify completed bank trades separately. This adds UI support, not a new learned
proactive trade policy.

**Dependencies:** A4 and A2; A3/A5 supply transfer evidence for live accounting.

**Acceptance checks:** Incoming acceptance with unchanged hand is pending;
cancellation and selection of another player leave the hand unchanged; a visible
completed exchange updates once. Same terms in different rows, changed terms,
expired offers, multiple willing partners, stale controls, retained composer
values, and bank-cleared selections are tested. A disappearing offer alone is
insufficient evidence of commitment. The simulator's existing atomic acceptance
and trade-plugin checks still pass unchanged.

### A8 — Assemble observations and prove a bounded vertical slice

**Status:** Pending. Manual victory establishes a separate controls milestone;
it does not satisfy the integrated policy/adapter replay below.

**Goal:** Connect verified capture evidence to policy input and execution without
claiming general screenshot recognition.

**Scope:** Assemble complete observations from verified board annotation/
calibration, current visible HUD/own hand, the continuous log, and UI state.
Keep provenance and uncertainty per source outside the model as needed. Reject
incomplete legal-action/context/geometry evidence. Replay one complete offline
decision sequence through reader, policy or scripted action, executor, event
history and beliefs; then run a bounded authorized live slice using supported
controls. Publish support by action kind before expanding it.

**Dependencies:** A5, A6 and A7 for the relevant action kinds. Physical capture
continues to use an injected verified reader; it does not require the web path.

**Acceptance checks:** Hidden-state noninterference and stale-state tests pass;
the offline sequence reconciles own resources and public totals; repeated
captures do not duplicate events; unsupported controls fail explicitly. Live
results include before/after evidence, final status, and unresolved gaps. Full
game completion is reported only after visible termination; autonomous play is
reported only when the integrated policy/adapter actually performed it.

### A9 — Measure learning with a reproducible comparison

**Status:** Evaluation infrastructure is implemented and accepted after the
censoring correction and independent re-review. Explicit disjoint seed lists,
four-seat schedules, raw/partial/interrupted outcomes and RNG restoration are
available. Strict win rate is undefined when a selected outcome is unresolved;
paired comparison rejects incomplete, truncated or unresolved inputs. A finished
schedule is not evidence of four resolved games. See the
[learning protocol](learning-protocol.md). The clean-source pilot at `1350de2`
completed in 92.301 seconds under its 900-second cap, and Task 3 passed scoped
re-review. The competence gate failed; no PPO ran. Broad review's installed-wheel
warm-start Git regression still requires correction/re-review. Fresh final-test board seeds 70000017 and
70000019 remain reserved; the historical 100000 range may have been used in smoke
evaluation. The [swarm roadmap](swarm-roadmap.md) records current decision gates.

**Goal:** Decide whether the existing policy improves before adding model scope.

**Scope:** Save a fixed experiment manifest containing source revision, engine/
action schema, model/configuration, training seeds and decision budget,
checkpoint digest, and disjoint evaluation board seeds. Compare untrained PPO,
warm-start-only, PPO after the same declared training budget, and the heuristic
baseline. Use complete four-seat board blocks and retain all per-game outcomes.
Use the existing incoming-trade/reject-all comparison where there are enough
response opportunities. Report wins, VP, truncations, response count, and latency.

**Dependencies:** A1 compatibility validation; independent of live integration.

**Acceptance checks:** A short local pilot establishes runtime, then a bounded
run uses multiple training seeds and the same held-out board blocks across
variants. Raw results and manifest are saved. Summaries aggregate paired
differences by board block; any interval accounts for shared-board dependence or
retains the existing explicit nominal label. Tiny or heavily truncated samples
yield an inconclusive result. No strength claim is inferred from passing tests,
finite PPO losses, a manually won browser game, or the existing zero-of-four
smoke evaluation. Use results to choose the next learning change.

## Milestones and scope control

Primary execution order is **L1 + A9 -> L2**, with explicit local compute budgets
and held-out evidence at each gate. The adapter milestones below describe the
secondary deployment track; they are not prerequisites for learning progress.

1. **Modular baseline:** A1 is implemented and independently reviewed; A2 has
   ordered log fixtures and a completed manual-game evidence set. Final combined
   verification passed. Shared observations remain the only
   policy/trade-plugin input.
2. **Honest offline adapter:** A3/A5 have token normalization and conservative
   history reporting; A3 adds production/bank/dice candidates and A4 adds reviewed
   pure UI contracts. Established event identity, observed transition extraction,
   staged gesture execution and ledger integration remain before A3–A7 satisfy
   the full milestone.
3. **Verified integration:** A8 demonstrates the supported vertical slice;
   unsupported flows and history gaps remain explicit. A9 provides a separate
   assessment of policy strength.

Keep complete six-player simulation, arbitrary image recognition, learned
proactive/counter trade generation, broader variants, and unmeasured model
architecture expansion deferred until a milestone exposes a measured need.
Physical board support begins with the shared contracts and explicit reader
boundary, rather than requiring a second perception system in this iteration.

## Learning track — reach competent play with limited compute

Added November 4, 2026. This track can run independently of the browser adapter
after simulator/import compatibility is established. These are proposed
experiments, not completed training or demonstrated playing strength.

**Working hypothesis:** Standard Catan has a manageable structured observation
and legal-candidate action space. Long-horizon credit assignment, dice variance,
partial observability, opponent diversity, and simulator fidelity are larger
risks than raw input dimensionality. The current graph has roughly 149 entities
on a standard four-player board. Its small graph/GRU model is a baseline, not
evidence that competent learned play is infeasible. Competitive human trading
remains a separate, harder objective.

### L1 — Establish a competent teacher and imitation baseline (first priority)

**Status:** Persisted observation-only demonstration collection/validation and
current-weight recurrent imitation are implemented and accepted after independent
re-review, including the shared deadline guard fix. Training supervises
non-forced choices while forced observations still advance recurrent state;
held-out metrics report both groups by phase and action kind. The completed pilot
and separate 20-epoch diagnostic found better teacher agreement with more
optimization, but both students still won 0/8 on the two known boards. Competent
imitation is not established. The next planned study tests balanced exposure and
bounded fresh-PPO curriculum while retaining fixed evaluation and the original
failed gate. Teacher candidates and the remaining experiment goals below are
unchanged; these results do not justify a model-capacity claim.

**Goal:** Obtain playable behavior before asking PPO to discover basic strategy.

**Teacher candidates:** The pinned Catanatron 3.2.1 installation includes
`RandomPlayer`, `WeightedRandomPlayer`, and greedy `VictoryPointPlayer`; these
are baseline opponents, not established strong teachers. Upstream documentation
also describes `ValueFunctionPlayer` and `AlphaBetaPlayer`, which are absent from
this installed package. Evaluate integration/version compatibility before using
them. Benchmark a value-function teacher against this project's existing
production-aware `HeuristicPolicy` in four-player games; published small 1v1
comparisons do not establish four-player strength. Audit any engine-native
teacher's full-game access before admitting labels to player-observation training.

**Teacher access and integration tasks (planned):**

- Obtain upstream source in a separate teacher environment, pinning the source
  revision and dependencies; preserve the working `catanatron==3.2.1` learner
  environment. Initial inspected upstream revision:
  `ecf931181b9a65bb4116a2153fb78c16f1438e00`.
- Inspect and integrate
  [ValueFunctionPlayer](https://github.com/bcollazo/catanatron/blob/ecf931181b9a65bb4116a2153fb78c16f1438e00/catanatron/catanatron/players/value.py)
  and
  [depth-two AlphaBetaPlayer](https://github.com/bcollazo/catanatron/blob/ecf931181b9a65bb4116a2153fb78c16f1438e00/catanatron/catanatron/players/minimax.py)
  through an explicit action/observation compatibility boundary. These are
  source-available heuristic/search teachers, not pretrained neural checkpoints.
- Audit multiplayer assumptions: the inspected value function explicitly
  considers one enemy, while AlphaBeta treats other players as minimizing our
  value. Benchmark these assumptions in four-player games before selecting a
  teacher; neither candidate is established as the best available four-player bot.
- Compare both candidates and our `HeuristicPolicy` on common held-out board/seat
  blocks, recording wins, truncations, decision latency and decisions per second.
  Verify hidden-information use and rule/action compatibility before collecting
  labels. Separate teacher-environment game exports require validated conversion;
  they must not be assumed compatible with the learner's engine.
- Prefer the strongest efficient teacher for bulk demonstrations. Use more
  expensive search selectively if its label quality justifies its measured cost.
  Save teacher revision/configuration and observation provenance with the dataset.

**Scope:** Benchmark the information-limited heuristic against random and fixed
opponents on held-out boards and rotated seats. Generate simulator demonstrations
covering full games, setup, forced decisions, bank trades, development cards and
incoming trade responses. Start with a bounded pilot, for example 10,000 learner
decisions, and expand only after measuring runtime and coverage. Keep training
and validation boards disjoint. Train on ordered sequences and recompute memory
under the current model rather than treating old hidden states as timeless
features. Cache static board topology to reduce collection/encoding overhead.

**Acceptance checks:** Report teacher performance, demonstration coverage,
held-out action agreement by decision phase, and complete-game student results.
The student should approach teacher performance before expensive RL runs. If
imitation fails, investigate representation, missing inputs and optimization
before simply adding PPO experience. A tiny demonstration set or high agreement
on mostly forced actions does not establish competent play.

### L2 — Warm-start the critic, then improve with fresh PPO experience

**Goal:** Improve the imitated policy without losing its basic competence.

**Scope:** Pretrain value estimates from completed teacher games, explicitly
labeling these as returns under teacher continuation. Fine-tune with fresh
on-policy learner rollouts against a mixture of competent fixed opponents and
frozen historical checkpoints. Optionally retain a decaying imitation or KL
anchor to the teacher to limit early collapse; compare against no anchor.
Historical demonstrations remain supervised data, not fresh PPO trajectories.

**Bounded parallel experiment tasks (planned):**

- Build an experiment runner for two configurations times three training seeds:
  ordinary imitation-initialized PPO, and the same initialization with a gradually
  weakened imitation/KL anchor. Keep other hyperparameters, opponent mixtures and
  training budgets fixed for this first comparison.
- Give every run independent actor/critic parameters, optimizer state, RNG state,
  metrics and checkpoints. Pair the same three rollout seeds across configurations
  and record the shared frozen initial actor/critic checkpoint digest.
- Profile local throughput before choosing worker concurrency; queue the six runs
  with a bounded worker count rather than assuming six simultaneous jobs are
  efficient. Record resource limits, wall-clock time and learner decisions.
- Compare runs against identical fixed opponents and complete four-seat board
  blocks. Keep training boards, model-selection validation boards and the final
  test boards disjoint. Retain seed-level outcomes, including failed or collapsed
  runs, rather than reporting only the best checkpoint.
- Use short pilots to catch correctness failures and estimate cost. Allocate
  additional compute based on adequately sampled validation results, accounting
  for dice and seed variance; do not eliminate a configuration after a few unlucky
  games. Reserve the final test set for the selected approach.

The current critic predicts shaped return, not literal win probability. Compare
this baseline with an outcome-value head that separates the known potential
term: for the existing undiscounted shaping, V_shaped(s) = V_win(s) - Phi(s),
where Phi is the actual configured own-VP potential and terminal Phi is zero.
Handle truncations with bootstrapping; never label them automatic losses.

**Acceptance checks:** Compare teacher, imitation-only, and imitation-plus-PPO
under A9's fixed evaluation protocol and declared wall-clock/decision budgets.
Track win rate, paired board-block differences, truncations, critic error,
entropy, KL, clip fraction, and throughput. Evaluate outcome calibration against
frozen-policy completed games. Promote checkpoints only after credible held-out
improvement; a lower training loss is insufficient. Self-play against only the
latest copy must not replace fixed external benchmarks.

### L3 — Improve information quality before increasing model size

**Goal:** Avoid asking a larger network to recover information discarded upstream.

**Scope:** Compare compact strategic features for resource production/balance,
building affordability, expansion access and road/army races. Replace event-type
counts with ordered actor/resource-aware event encoding where it adds information
beyond the existing belief ledger. Preserve unknown information and history gaps.
Evaluate whether opponent-hand means and variances omit useful correlations;
introduce richer belief summaries or sampled hands only for a measured failure.

**Acceptance checks:** Use controlled ablations with equal compute and held-out
boards. Actor inputs and legal masks remain free of simulator-private information.
Test long-road/expansion positions where two local graph layers cannot directly
propagate distant topology. A recurrent model cannot recover event details that
were never supplied to it.

### L4 — Run small actor–critic architecture and efficiency ablations

**Goal:** Spend additional compute only where it improves measured play.

**Scope:** After L1/L2 establish a baseline, separately compare modestly wider
networks, deeper graph processing or strategic skip features, and longer recurrent
sequences with a history warm-up. Profile graph construction, simulator time,
network inference and updates first; batch/vectorize independent simulations
where profiling justifies it. Keep candidate-action scoring and legal masks;
use bounded trade generators before considering unrestricted negotiation.

**Acceptance checks:** Change one major factor at a time, retain multiple training
seeds, and compare improvement per wall-clock budget as well as per learner
decision. Record memory/throughput costs. Defer large transformers and arbitrary
trade enumeration until smaller baselines expose a specific limitation.

### L5 — Escalate to search-assisted teachers only if the baseline plateaus

**Goal:** Improve teacher quality without demanding discovery through random play.

**Scope:** Try shallow lookahead or a bounded candidate tournament in the simulator
to generate better action rankings for imitation. Sample hidden hands from valid
beliefs and use plausible opponent continuations; do not grant the deployed actor
oracle access. Keep search budget explicit and separate privileged diagnostics
from deployable policy evaluation.

**Acceptance checks:** Search-assisted labels improve held-out teacher/student
performance enough to justify their generation cost. Report continuation-policy
and hidden-state assumptions. This is a later experiment, not a prerequisite for
the first playable learned policy.

**Execution order:** L1 -> L2; use L3 when diagnostics identify missing information,
then L4 for targeted capacity/efficiency changes. L5 follows a measured plateau.
The first feasibility gate is teacher -> competent imitation -> demonstrable PPO
improvement, independently of successful manual browser gameplay.
