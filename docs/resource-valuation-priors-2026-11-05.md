# Resource valuation research and experiment boundary

November 5, 2026. These sources motivate a later, separate public-information
experiment. They do not change the currently running hand-recipe comparison,
its architecture, labels, reward or validation selection.

## What the sources provide

[Alex Cates](https://www.alexcates.com/post/board-game-breakdown-catan-resource-value)
analyzes 306 four-player Colonist games. His completed-trade count ratios change
when bank or bot trades are excluded. Bundled trades do not identify individual
resource utilities. These measurements describe accepted trades in that sample;
they are not universal strategic values or labels for rejected alternatives.

[The Reddit author](https://www.reddit.com/r/Catan/comments/1vgshcr/the_trade_value_of_each_resource_over_time_from/)
reports more than 40,000 games and changing resource demand over the game.
The post does not provide enough raw data and methodology to reproduce the
valuation here. Treat its plotted trends as hypotheses pending that evidence.

[The Nerdist interview](https://nerdist.com/article/the-geekonomics-of-catan/)
suggests public production, settlement coverage and trading outside options as
useful explanatory variables. Its examples are qualitative motivation.

## A useful separate experiment

Compare the existing model with explicit public production and scarcity inputs:
resource-specific dice probabilities across the board, occupied production by
player, own expected production, production diversity, and attainable bank rates.
During setup, distinguish total tile potential from production actually secured
by the candidate placement and from contested future settlement access. Preserve
these as distinct features rather than collapsing them into a fixed price list.

A weak auxiliary warm-start target may teach exact expected-production arithmetic
or recipe readiness on synthetic public states. Validate that it improves real
placement and trade choices at matched inputs and training work. Do not substitute
population trade prices for the unchanged heuristic action targets. Keep bank
rates, hand thresholds, connectivity, remaining pieces and public VP races in
context: losing three cards in a 4:1 trade can be useful if it enables a settlement.
Any added policy inputs require an explicit feature schema and checkpoint identity.

Use only information available at decision time. Percentage of eventual game
length uses a future outcome and must not become an online feature. Turn count,
public VP, piece counts and observed production are available alternatives.
Unknown opponent hands remain unknown. Synthetic counterfactual calculations must
not mutate verified observations, real resources or recurrent controller history.

## Bank access and evaluation denominators

Every player has the base 4:1 maritime option during their own trading/building
turn when they hold four matching cards. An owned port improves that rate to 3:1
or the resource-specific 2:1; no port is required for base access. See the
[official rules](https://www.catan.com/sites/default/files/2021-07/catan-25th-rules_eng-200313.pdf).

A replay audit found 1,169 saved observations with legal 4:1 trades and no owned
port. The raw audit is `runs/swarm-2026-11-04/verification/bank-no-port-audit.json`.
Report executable bank menus and choices separately from port ownership. A legal
exchange is not automatically strategically useful. Judge conversions by their
public continuation and distinguish sensible passes from missed opportunities;
raw trade frequency and phase balancing alone do not establish decision quality.

## Execution order

Finish current hand-recipe comparison, audited student-state aggregation versus
additional teacher trajectories, complete-game decision audit, then the reviewed
PPO curriculum. Keep reward unchanged for these comparisons. A richer potential
or resource prior is a separate later ablation, never a reward merely for trading.
