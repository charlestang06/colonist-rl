# Plain GreedyNoTrade baseline

User explicitly requested a strict plain greedy agent that does not trade. This
is not HeuristicPolicy with trades disabled. Implement one public-observation-only,
stateless deterministic baseline; no heuristic utility/goals, search, lookahead,
learned weights/value, belief inference, resource weighting or teacher delegation.

Root provides a separate legacy checkout from reviewed45e2c1. Use only verified
`/private/tmp/colonist-swarm-bxz1k39_/legacy-runtime/.venv/bin/python` and sibling
Ruff, PYTHONPATH=thischeckout/src, bytecode off and OMP/MKL/Torch1. Coordinator
verified its installed Catanatron is3.2.1. Original .venv is upgrading: never use it.
No migration, contract, driver, evaluator, teacher, checkpoint or roadmap edits.

## Frozen policy v1, before any benchmark

Class `GreedyNoTrade(torch.nn.Module)`, identity `plain-greedy-no-trade-v1`, no
parameters or persistent state. `select_action(view: Observation) -> Action`;
`act(view, memory=None, *, deterministic=False) -> (Action, None)`;
`propose(view) -> None`. Always deterministic; ignore/drop incoming memory and
never draw RNG. Return a member of the original legal tuple without mutation.

1. If REJECT_TRADE is legal, select it. NEVER ACCEPT_TRADE, MARITIME_TRADE or a
   player proposal, even if favorable or the only supplied action. Empty/no
   supported safe menu raises ValueError rather than trading or inventing action.
2. Mandatory discard: DISCARD_RESOURCE from largest own hand count; ties use
   shared resource order WOOD,BRICK,SHEEP,WHEAT,ORE. Mandatory robber: smallest
   legal tile target, then smallest victim ID; None victim after real IDs. No
   opponent-hand/VP utility or production search for robber. Legacy DISCARD kind,
   if exposed, selects canonical first legal action without private inference.
3. ROLL when legal (before optional purchases/development). Mandatory free/setup
   roads use the same smallest-edge rule below. Setup settlements maximize raw
   production pips: sum(6-abs(7-number)) over adjacent numbered productive tiles.
   No diversity/resource/port/robber weighting; ties choose smallest vertex ID.
4. Choose immediately legal purchases in exact order BUILD_CITY >
   BUILD_SETTLEMENT > BUILD_ROAD > BUY_DEVELOPMENT_CARD. City/settlement target
   maximizes the same raw pips, smallest vertex breaks ties. Road always smallest
   edge ID: no frontier/shortest-path/productive-destination lookahead.
5. After purchases, handle available development plays in frozen order
   PLAY_ROAD_BUILDING > PLAY_KNIGHT_CARD > PLAY_YEAR_OF_PLENTY > PLAY_MONOPOLY.
   Resource choices use lexicographic shared resource order, not hand/goal utility;
   other ties smallest target then victim, then original menu position. Legal
   follow-up free roads/robber choices follow rules above. No beneficial-use test.
6. END_TURN otherwise. Unrecognized legal kinds do not justify arbitrary fallback;
   raise clearly if there is no supported safe action.

Development cards are used by static priority; this deliberately simple handling
is not a claim of strategic optimality. Raw pip ties/development order/road rule
must not be tuned after results. Existing heuristic-no-trade is a later separate
diagnostic, never relabeled as this primary baseline.

## Integration and evidence

Pass the module directly to existing evaluate_blocks(policy, candidate='graph',
baseline='heuristic', deterministic=True). Evaluator resets memory per game;
GreedyNoTrade returns None and holds none. Its generic graph actor tag must be
qualified in the run manifest by the exact plain-baseline identity. No new loop
or fake critic/PolicyOutput is needed for this path. Do not use reject_trades
compatibility routes that require a model forward. Preserve evaluator RNG/modes.

One fresh worker implements new greedy_no_trade.py, tests/test_greedy_no_trade.py,
and docs/greedy-no-trade.md; independent review before actual evaluation. Unit
fixtures cover priorities, ties, prohibitions, privacy, immutability, RNG/modes and
actual evaluator adapter use. At most one <=30s bounded test smoke, no benchmark.

After review root declares a bounded old-engine run: plainGreedy plus fixed FULL15
hand_recipe checkpoints11/29, each72010000..72010015 ×4seats, versus three fixed
heuristics (192 games total). This compares candidate performance against a shared
opponent panel, not head-to-head games against Greedy. No checkpoint selection,
training, final/audit boards or cross-engine comparisons. Persist per-game rows,
source/checkpoint/rule hashes, action counts proving zero trades, actual caps and
all censored/partial evidence. Strict pairs use complete uncensored matching
panels. Report per-seed wins/VP/truncations and board-level differences/uncertainty;
16 boards are the independent resampling units, not64 independent seats.
