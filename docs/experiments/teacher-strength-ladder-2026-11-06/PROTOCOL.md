# Prospective raw-teacher strength ladder

Frozen 2026-11-06 **before any game in this study**. This evaluates whether the
unchanged public raw planner is a useful noisy demonstrator for a faster neural
student. It does not promote a playing policy or establish superiority to native
references. No PPO, reward tuning, retuning after a gate, or EVAL-derived label is
permitted.

## Identities and roles

- Gameplay source tree SHA-256: `eb7dd60c93a89ad7f30d715bb91041a4be9c01c22822d3f8e5b17ad2bb261dee`.
  The worktree begins at Git `121b985a2cb6ce8063f043079c8dc492e596ac94`;
  the final clean implementation commit is recorded in the handoff receipt.
  The raw planner source file is SHA-256
  `70bb8ff3f32c669bddea39fca08c80b951d01514f14255dc85e35a8937cdf55a`.
- Engine: Catanatron `3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`,
  `official_spiral`, backend seed `logical_board_seed+1`. Runtime identity digest
  `90789ec1a608f8ad24e5f6a17352c4ef6349da84e89477f99875daea738b6493`.
  Native source verification is enforced by `native_opponents.py` (SHA-256
  `dea2bf05588cc82668fa40b26ab6698421e98844f3bfd6f397689c74608e7cb0`).
- Raw: `PublicTurnPlannerPolicy`, depth 6, nodes 2048, beam 24, geometry 100000,
  seed 11, invalid-future pruning on, evaluator `None`, gain margin 10000,
  `search_build_win_before_buy=False` by the pinned constructor default. It uses
  only the focal player's permitted observation and exact current legal root;
  unsupported decisions use its public H-view fallback.
- Parent: unchanged seed-11 `graph_consequence_v1` / `hand_recipe`, hidden size 32,
  checkpoint
  `/Users/charlestang06/Documents/CS/colonist-rl/runs/swarm-2026-11-04/ppo-strength-preparation/seed-11-frozen-bc-3_3-start.pt`,
  byte SHA-256 `b3aae32ac36dc55f8ae26ec4676cc289243a6c540b959e1f8aa9786e5ddd0258`.
  Recurrent memory resets each game and advances on every own observation.
- H focal: persistent `HeuristicPolicy(logical_board_seed+31+focal_seat)` per game.
  H backgrounds use `logical_board_seed+absolute_seat`. GNT focal/background:
  unchanged strict `plain-greedy-no-trade-v1` (no maritime trade).
- Native opponent IDs are `native_vf_82aae93` and `native_ab2_82aae93`; AB uses
  depth 2, `prunning=False`, upstream default weights. They are privileged
  evaluation-only opponents with the documented atomic-discard and reject-trade
  shims, upstream chance mismatch, and `full_rule_equivalence=false`. No native
  decision, private state, or outcome enters training labels.

The previous zero-use local archive scan for `157510000..015` has SHA-256
`deeeaacc4adc91646d57b3ad44e3200cfbb92eebc8428b9ec4ebc7b0fecf144d`;
Root reconfirmed this range unspent and assigned it to Stage 1. Stage 2 seeds
`157540000..031` are **conditionally reserved, not approved for games** until
Root's new disjointness scan and CPU release. These EVAL boards are separate
from BC TRAIN `15741`, BC labelVAL `15742`, and BC fresh VAL/game `15752/15753`.

## Fixed games and work budget

All panels use standard four-player rules, all four focal seats per logical
board, no domestic offers, legal maritime trade, a 20,000-step game cap, and
the exact pinned engine. Within each board the manifest reference `raw` runs
first, then `parent`, then the focal public control if present. Every arm gets
an independent game on the same board/seat seed. Public focal before/action/
after traces are captured. Only raw's full permitted own observations are
retained for later EVAL decision inspection; the other traces are compact.

| Stage | Seeds | Focal arms | Three opponents | Games | Role |
| --- | --- | --- | --- | ---: | --- |
| 1 H | `157510000..007` | raw, parent, H | H, H, H | 96 | development EVAL |
| 1 GNT | `157510008..015` | raw, parent, GNT | GNT, GNT, GNT | 96 | development EVAL |
| 2 VF interim | `157540000..007` | raw, parent | VF, H, H | 64 | conditional validation EVAL |
| 2 AB interim | `157540016..023` | raw, parent | AB, H, H | 64 | conditional validation EVAL |
| 2 VF remainder | `157540008..015` | raw, parent | VF, H, H | 64 | conditional validation EVAL |
| 2 AB remainder | `157540024..031` | raw, parent | AB, H, H | 64 | conditional validation EVAL |

The native opponent occupies `(focal_seat+1)%4` in each native game, so the
four focal seats balance its absolute seat. Stage 2 tests raw facing a harder
native opponent; the prior 32-board native-*focal* comparison is historical
context only and is never pooled with this ladder.

Run one heavy worker at a time after the BC CPU release, with
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=NUMEXPR_NUM_THREADS=1`
and `PYTHONHASHSEED=0`. Planned process RSS ceiling is 2 GiB. Stage 1 manifests
allow 900 seconds soft gameplay / 1020 seconds external hard process time each;
Stage 2 manifests allow 1800 / 1920 seconds each. No automatic retry, partial
board substitution, or extension. Prior comparable jobs suggest roughly 3–8
minutes for Stage 1 and, if admitted, 10–25 additional minutes for Stage 2;
these are planning estimates, not measured results. Record actual wall time,
CPU/RSS samples, thread counts, and raw selection latency separately from
playing-strength gates.

## Fixed decisions

Every panel must have all scheduled games completed with resolved winner,
zero truncated/unresolved/invalid selections, and valid saved journals and
source/checkpoint/runtime hashes. Any failure stops advancement, with partial
records preserved. Wins and VP use exact board/seat pairs. Paired uncertainty
resamples whole four-seat boards 20,000 times with the existing fixed seed
`20261106`; the 95% percentile intervals are operational screen diagnostics,
not population-superiority proof.

**Stage 1 advance, evaluated once after all 192 games:** in *each* eight-board
panel, raw must win at least four more games of 32 than both the unchanged
parent and that panel's H/GNT focal control. Raw mean VP must exceed the
matching control by at least 0.25 and parent by at least 0.5. The lower endpoint
of each of these four raw-minus-control paired board bootstrap win intervals
must be at least zero. Failure of any item means no Stage 2, no rescue by more
boards, and `teacher_not_qualified_on_stage1`.

**Stage 2 fixed interim:** run exactly the first eight VF and first eight AB
boards, then inspect all 128 games. Stop before either remainder if any game is
incomplete/invalid, if in either lineup raw trails parent by at least four wins
of 32 *and* 0.5 mean VP, or if the native opponent wins at least eight more of
raw's 32 games than raw does. This is a downside stop only; it cannot confer
qualification. No replacement boards or automatic rerun.

**Stage 2 final qualification, only after all 256 native-opponent games:** in
*each* 16-board lineup, raw must beat parent by at least six wins of 64 and
0.5 mean VP; its raw-minus-parent paired board bootstrap win lower endpoint
must be at least zero; and raw must win at least as many of its 64 games as the
single native opponent in those same games. Full completion and provenance
remain required. Passing means `qualified_for_exploratory_noisy_BC`, not native
superiority or policy promotion. Otherwise report `teacher_not_qualified` and
the exact failed checks. Do not tune thresholds after seeing outcomes.

## Public decision inspection

Read only the saved, SHA-verified EVAL raw own observations. For each raw
episode, replay the unchanged parent deterministically on the same ordered
public own-observation prefix, including forced calls, with fresh recurrent
memory at the episode start. Compare exact current legal roots. In the fixed
manifest order and then `(board_seed, focal_seat, own_ordinal)`, report the
first three mismatches in each category where either root is maritime trade,
`END_TURN`, `BUY_DEVELOPMENT_CARD`, or a `PLAY_*` card action. Report each
category's total mismatch count, original menu identity, public hand/phase/
known VP, both actions, and raw same-turn continuation when captured. If a
category has fewer than three, report the missing slots. These examples are
diagnostic; neither parent counterfactual nor later EVAL outcome is a label.

Any verified illegal original root, missed directly legal current winning
city/settlement, or destructive same-turn repeated trade loop is a critical
failure and prevents qualification. Public fact replay may establish exact
feasibility of a suspected alternative, never optimality. Ambiguous BUY/card
choices are retained as ambiguous, not relabeled. Report raw/H/GNT/parent and
native winner counts, final VP, complete/invalid/unresolved counts, matched
board intervals, first changed cases, and latency even on a stopped path.
