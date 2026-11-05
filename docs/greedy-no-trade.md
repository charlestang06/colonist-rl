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

## Existing evaluator integration

`GreedyNoTrade` is a `torch.nn.Module` solely for the existing actor adapter.
`select_action(view)` returns an `Action`; `act(view, memory=None, *,
deterministic=False)` always returns `(Action, None)`. Incoming memory and the
sampling flag do not influence selection. The evaluator owns the game loop,
resets memory per game, and restores module modes and external RNG state.

After independent review, a bounded invocation can use the existing evaluator:

```python
import time
import torch

from colonist_rl.evaluation import evaluate_blocks
from colonist_rl.greedy_no_trade import GreedyNoTrade

torch.set_num_threads(1)
policy = GreedyNoTrade()
result = evaluate_blocks(
    policy,
    board_seeds=[72010000],  # Explicit panel; all four candidate seats per board.
    candidate="graph",
    baseline="heuristic",  # Three fixed heuristic opponents.
    deterministic=True,
    max_game_steps=20000,
    deadline=time.monotonic() + 30,
)
# Qualify the generic evaluator actor tag in the persisted run manifest.
manifest = {"policy_identity": policy.identity, "result": result}
```

Use the verified legacy Catanatron 3.2.1 runtime and record source/rule hashes,
actual caps, per-game rows, and candidate action counts. Do not use the
`reject_trades` compatibility route, which assumes a model forward. The baseline
itself enforces the no-trade restriction. Opponents may still trade with each
other and propose offers; candidate counts for `ACCEPT_TRADE` and
`MARITIME_TRADE` must remain zero.

The same opponent panel compares candidates against fixed heuristic opponents;
it does not produce head-to-head games against this greedy agent. Keep
truncations, deadline interruptions, and partial blocks visible. Strict paired
comparisons require complete, uncensored matching panels. Boards, rather than
individual seats sharing a board, are the independent resampling units. Record
wins and VP by seed, all truncations, board-level differences, and uncertainty.
No performance claim follows from a short integration smoke. Independent review
must precede the actual benchmark; implementation tests do not select
checkpoints, train a model, or authorize policy tuning.
