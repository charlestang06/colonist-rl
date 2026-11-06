# Raw planner versus native references: comparison results

The separately frozen comparison completed **384 games on 32 fresh boards**, all
four seats, with three persistent H backgrounds and no domestic offers. Maritime
trades remained legal. All 96 guarded jobs exited zero in about 587 seconds.

| Focal policy | Wins / games | Win rate |
|---|---:|---:|
| Unchanged public raw planner |32/128|25.00%|
| Authentic native VF |19/128|14.84%|
| Authentic native AB2 |26/128|20.31%|

Raw led on this fixed panel, but **the qualification gate failed**. No superiority,
parity or model promotion is established. There is no automatic extension or
confirmation. This comparison is separate from the earlier eight-board H/raw/NN
screen; outcomes are not pooled to rescue either gate.

## Paired evidence and limits

Each contrast averages four seat-win differences within each board, then averages
32 board differences. The frozen primary rule requires all 384 resolved games and
both simultaneous lower bounds strictly above a five-point practical margin.

| Contrast | Observed difference | Simultaneous bounded-KL 95% interval |
|---|---:|---:|
| Raw minus VF |+10.16 percentage points|[−40.26, +57.16] points|
| Raw minus AB |+4.69 percentage points|[−45.00, +52.79] points|

Both intervals cross zero; the raw-minus-AB point difference is itself below the
five-point margin. The intervals use the prespecified bounded-mean transformation
and KL constant log(80)/32, with outward endpoint rounding. Population coverage
requires independent/exchangeable board draws; consecutive fixed seeds do not
prove that assumption. This was an informative diagnostic count, not assurance
of power to detect a five-point advantage.

Descriptive paired bootstrap intervals also cross zero: raw−VF [−1.56, +22.66]
points and raw−AB [−6.25, +16.41] points. These use shared board resamples, 20,000
MT19937 draws, seed 202611064 and linear .0125/.9875 quantiles. They cannot override
the primary rule. Independent review regenerated the bounded-KL endpoints within
2e-14; optional descriptive bootstrap was not independently regenerated.

## What was verified

Logical boards 138200000..138200031, backend seed logical+1, balanced three-arm
order and all four seats were frozen before play. Source was clean commit
1cfaa4c0c570a53b39569aa60f0079e78b1b6b00. Raw used depth 6, nodes 2048, beam 24,
geometry 100000, seed 11, evaluator None and explicit margin 10000. That source
predates the later optional winning-prefix helper; no nonexistent flag was passed.
VF used authentic defaults; AB used depth 2, prunning=False and default weights.

Native references remain privileged, evaluation-only players with the declared
atomic-discard/reject-trade shims, upstream chance mismatch and
full_rule_equivalence=false. These results apply to that harness and offer regime;
they establish no broader negotiation or rule-equivalence claim. Native choices
never became learner or teacher inputs. Their helper exposes no public step
callback; this unavailable scope was preserved rather than replaced by hidden
state dumps.

The unchanged v4 guardians bounded each job at 120 seconds internally and 140
externally, with a 7200-second Root limit, at most two heavy workers, one sampled
OS thread and 2 GiB RSS per worker. Independent saved-record review checked all
384 joins, 11,345 raw public before/action/after callbacks (4,760 forced), all 49
source and 17,978 runtime opaque hashes, 12 native source pins, 192 saved pre/post
native identities, and process/argv/deadline evidence. All 5,397 live saved OS
samples had one thread; peak sampled RSS was 770,801,664 bytes and saved overlap
peaked at two. Recorded native work was AB 8,965 decisions plus 1,122 atomic-discard
shims, and VF 9,063 decisions plus 1,651 discard shims; reject shims were zero.

These are sampled saved records and opaque hashes. Independent review did not
reload tensors, execute a codec, replay games or prove live-object identity or
continuous OS behavior. Clean execution is not a strength advancement. There were
zero neural loads, optimizer updates or accepted training targets.

[Protocol](../runs/swarm-2026-11-04/agent-research/native-comparison32-001/freeze-001/protocol.json),
[summary](../runs/swarm-2026-11-04/agent-research/native-comparison32-001/analysis-001/summary.json),
[independent review](../runs/swarm-2026-11-04/agent-research/native-comparison32-001/saved-result-review-001/report.md)
and [Root acceptance](../runs/swarm-2026-11-04/agent-research/native-comparison32-001/Root-execution-acceptance-001.json)
retain the actual work and failed gate. The
[complete baseline Git bundle](../runs/swarm-2026-11-04/agent-research/native-comparison32-001/frozen-source-1cf.bundle)
was verified and locally fetched into the original repository without moving its
working branch. All 49 baseline source blobs matched the protocol hashes, so the
exact source remains recoverable after temporary directories disappear.

## What to improve next

A separate saved-public-data diagnosis reused only the first 64 TRAIN episodes
(5,448 own inputs) from the accepted mixed320 corpus. Actor source, all 49 source
hashes, search configuration, margin 10000, three H backgrounds and no-offer rules
match the current raw declaration. No VAL/EVAL episode or native choice was used
for that diagnosis, and no training targets or fits followed.

The most actionable finding is a purchase planning gap. H assigns a high direct
city score, but when its selected fallback is BUY_DEVELOPMENT_CARD, raw returns
that unsupported action before searching trade/build continuations. Of 310
recorded BUY choices, 142 had a legal bank trade. Eighteen states across 15 episodes
had an explicit one-bank-trade city-funding opportunity, a city piece and an owned
settlement to upgrade. All had known own VP at most eight, so they extend beyond
the retired near-win-only experiment. None retained direct or one-bank city
funding immediately after paying for the card. One still built a city later that
turn using two trades; 17 did not. There were 124 BUY-plus-bank negative controls
without that opportunity and no directly legal city/settlement in any BUY state.
Root checked the first eight witnesses' legal trade identities, cost arithmetic,
piece/context constraints and exact saved-after hash joins.

These facts identify an alternative, not the best move: development cards have
uncertain future value and spending has opportunity costs. Preserve the later-city
case as an adverse control. A separate Monopoly scan found two city-funding
alternatives, but one actual route built two settlements; a blanket ore/city rule
would discard a credible alternative. Do not manufacture labels from these facts.

Next, certify an exact TRAIN trade/city continuation in the engine, then test a
bounded city-funding-trade-before-purchase intervention with adverse controls and
fresh paired gameplay. Preserve the parent/default behavior until evidence earns
advancement. Do not respond with more unchanged benchmark games, automatic BC/PPO,
shaping or deployment of failed value heads.

[BUY diagnosis](../runs/swarm-2026-11-04/agent-research/raw-train-buy-diagnosis-001/NOTE.md),
[Root public-fact checks](../runs/swarm-2026-11-04/agent-research/raw-train-buy-diagnosis-001/Root-public-fact-acceptance-001.json)
and [Monopoly diagnosis](../runs/swarm-2026-11-04/agent-research/raw-train-card-diagnosis-001/NOTE.md)
retain their TRAIN-only scope and uncertainty. The goal remains active and unmet;
final boards 123100000..123100063 and legacy finals remain untouched.
