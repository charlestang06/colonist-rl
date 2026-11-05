# Plain GreedyNoTrade v1

`colonist_rl.greedy_no_trade.GreedyNoTrade` is the stateless deterministic baseline
with immutable instance identity `plain-greedy-no-trade-v1`. It consumes only the
normalized public `Observation` (including the player's own hand) and returns an
original member of its legal-action tuple. It has no parameters, value head,
persistent game state, search, learned weights, opponent inference, or RNG use.
It does not delegate to `HeuristicPolicy`.

The v1 rules are frozen before benchmarking:

1. Select `REJECT_TRADE` whenever legal. Never accept a player offer, make a player
   proposal, or select `MARITIME_TRADE`, including profitable exchanges. Ordinary
   4:1 bank trading is legal without an owned port; this baseline deliberately
   declines it. `propose(view)` always returns `None`.
2. For mandatory `DISCARD_RESOURCE`, discard the resource with the largest own
   hand count among legal choices. Break count ties in shared order `WOOD`,
   `BRICK`, `SHEEP`, `WHEAT`, `ORE`. Legacy `DISCARD` uses the canonical legal
   choice without examining private information.
3. For `MOVE_ROBBER`, use the smallest legal tile ID, then the smallest victim ID;
   a missing victim sorts after real IDs. Select `ROLL` whenever legal before
   optional purchases and development plays.
4. Buy in exact order `BUILD_CITY` > `BUILD_SETTLEMENT` > `BUILD_ROAD` >
   `BUY_DEVELOPMENT_CARD`. Setup settlements, ordinary settlements, and cities
   maximize raw production pips: sum `6 - abs(7 - number)` over adjacent numbered
   productive tiles. Smallest vertex ID breaks ties. No resource, diversity,
   existing production, port, or robber weighting is applied. Roads always use
   the smallest legal edge ID, including setup and free roads. There is no
   destination or frontier search.
5. After purchases, play development cards in exact order `PLAY_ROAD_BUILDING` >
   `PLAY_KNIGHT_CARD` > `PLAY_YEAR_OF_PLENTY` > `PLAY_MONOPOLY`. Resource choices
   use the shared resource order lexicographically, first resource then second.
   Other ties use smallest target, then smallest victim, with missing values
   last. An exact tie keeps the original menu position. This same canonical
   ordering applies to legacy `DISCARD`; it does not infer any discard payload.
   Follow-up free roads and robber moves obey the rules above. There is no
   beneficial-use test.
6. Select `END_TURN` otherwise. Ignore unrecognized kinds. An empty menu or a menu
   containing only forbidden/unrecognized actions raises `ValueError`.

The policy is intentionally simple. It may spend resources poorly, build roads
without useful destinations, use development cards at unhelpful times, and miss
profitable trades. These rules and ties must not be tuned after seeing results.
A trading-disabled heuristic would be a separate diagnostic, with a different
identity.

## Opt-in evaluator integration

`GreedyNoTrade` is a `torch.nn.Module` solely for the existing actor adapter.
`select_action(view)` returns an `Action`; `act(view, memory=None, *,
deterministic=False)` always returns `(Action, None)`. Incoming memory and the
sampling flag do not influence selection. The evaluator owns the game loop,
resets memory per game, and restores module modes and external RNG state.

The standard evaluation routes accept `candidate="greedy_no_trade"` and
`baseline="greedy_no_trade"`. An explicit three-entry `opponent_lineup` may also
contain `"greedy_no_trade"`, alongside `"random"`, `"heuristic"`, or `"historical"`.
Entries map to the next, opposite, and previous seats relative to the candidate;
an explicit lineup overrides the baseline panel. Historical entries still
require a supplied frozen policy pool. The driver calls `select_action` for
greedy opponents, so it passes an `Action` to the simulator, never an actor tuple.
Pairing `baseline="greedy_no_trade"` with a historical pool requires an explicit
lineup; otherwise evaluation rejects it before creating a game. A greedy baseline
therefore cannot silently replace one of its three opponents with a historical
actor. Legacy random/heuristic panels retain their historical-pool behavior.

The default candidate and opponent panel remain heuristic. Choosing a greedy
candidate does not change its opponents. For a bounded protocol smoke:

```python
import time
import torch

from colonist_rl.evaluation import evaluate_blocks

torch.set_num_threads(1)
result = evaluate_blocks(
    board_seeds=[72010000],  # Explicit panel; all four candidate seats per board.
    candidate="greedy_no_trade",
    baseline="heuristic",  # Three fixed heuristic opponents.
    deterministic=True,
    max_game_steps=256,
    deadline=time.monotonic() + 30,
)
```

A supplied `GreedyNoTrade()` still uses `act(view, memory, deterministic=...)`
through the existing module interface. Both supplied and named greedy candidates
report `candidate="greedy_no_trade"`,
`action_mode="deterministic_greedy_no_trade"`, and
`candidate_identity="plain-greedy-no-trade-v1"`, even when `deterministic=False`
is requested. That flag remains recorded as requested. A conflicting supplied
actor with `candidate="greedy_no_trade"` is rejected before creating a game.

A greedy baseline panel adds `baseline_identity="plain-greedy-no-trade-v1"`.
Explicit lineups containing greedy add `opponent_lineup_identities`, parallel to
`opponent_lineup`, with the frozen identity for greedy entries and `None` for
other entries. With an explicit lineup, that field describes the actual panel
and `baseline_identity` is omitted. These extra fields are absent on existing
routes that do not use greedy. The legacy `training.evaluate` game-count route
also accepts the named candidate and baseline. Graph candidates retain their
existing stochastic/deterministic interfaces and metadata.

Both CLI commands accept the opt-in opponent baseline while retaining a
heuristic candidate when no checkpoint is supplied:

```sh
colonist-rl simulate --baseline greedy_no_trade --games 1 --max-game-steps 32
colonist-rl evaluate --baseline greedy_no_trade --games 1 --max-game-steps 32
```

The baseline itself enforces the no-trade restriction, including when the legacy
`reject_trades` flag is set; greedy never needs a model forward. Opponents may
still trade with each other and propose offers; candidate counts for
`ACCEPT_TRADE` and `MARITIME_TRADE` must remain zero. Default curriculum stages,
gates, rewards, teachers, and opponent panels are unchanged.

The completed legacy Catanatron 3.2.1 benchmark and its artifacts remain frozen.
New evaluation manifests must record their actual runtime, source/rule hashes,
caps, per-game rows, and candidate action counts. Runtime changes do not
reinterpret the historical result or make it comparable to a new run by default.

The same opponent panel compares candidates against fixed heuristic opponents;
it does not produce head-to-head games against this greedy agent. Keep
truncations, deadline interruptions, and partial blocks visible. Strict paired
comparisons require complete, uncensored matching panels. Boards, rather than
individual seats sharing a board, are the independent resampling units. Record
wins and VP by seed, all truncations, board-level differences, and uncertainty.
No performance claim follows from a short integration smoke. Independent review
must precede the actual benchmark; implementation tests do not select
checkpoints, train a model, or authorize policy tuning.
