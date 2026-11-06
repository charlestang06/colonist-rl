# Unified public continuation v2: prospective redesign

Status: design frozen before reading the reused broad TRAIN public-view corpus
or any 15751/52/53 board. V1 remains a recorded negative engineering result.

## Why v1 stopped

The v1 archived preflight read only the public `raw-before.jsonl` member of the
older v3 archive. It passed exact-root parity on 128/128 views and incremental
p95 9.06 ms, but changed 0/128 first actions. All 141 eligible v3 views were
parent END states; the v2/forecast archives similarly contain only END states
(139/129). No v3 preflight route built a point this turn. The preflight receipt
is `runs/unified-public-continuation-2026-11-06/archived-preflight.json`, SHA-256
`6427e7f7bd687b5f09782c956515f376d937b437c0ac378eccd52b5a5241d148`.
No v1 fresh 15751/52/53 board or outcome was opened.

Two older public BUY fixtures expose a separate structural failure. Their
generic root-by-root search reaches the 100,000 geometry-work cap after
189/212 transition attempts, before comparing all ten exact legal roots. One
fixture has nine known points and a conservative
trade→trade→settlement winning route. Its public position score is 949.7m
versus the parent's BUY route at 844.9m, but v1's optimistic 100m unknown-card
endpoint can overturn the scalar rank. A known same-turn win must dominate an
unseen draw. These are public projection facts, not independent game outcomes.

## One structural change

V2 keeps the exact current legal root set, public lower/upper bank bounds,
four-action horizon, shared 512 transition-attempt cap, beam width 12, 100,000
geometry-work cap, four-roll future point option, 25 ms incremental p95 limit,
and 10m root margin. It changes the successor graph and the treatment of
unknown card purchases:

1. Search public point routes. Always consider a currently affordable city or
   connected settlement. Consider a future bank trade only when it receives a
   resource missing for a current city/settlement recipe or a road→settlement
   recipe reachable within the remaining action count; its payment must come
   from resources surplus to that recipe. This removes bank exchanges that
   cannot advance a known same-turn point route. Include roads only when a
   public unblocked vertex can become settlement-connected within at most two
   road additions; strict geometry and cost checks still decide feasibility.
   A root bank trade remains an exact candidate even if its continuation is
   unproductive, so END and trade-now are compared on the same score scale.
2. For each root, retain the best complete no-card route and, when the current
   legal menu contains BUY and no card was bought on that route, the best route
   that can pay for one later BUY within the four-action horizon. The later
   BUY projects only its known cost and a public deck receipt certified by
   the current BUY menu. The drawn card is never inspected or played. A BUY
   root already has one card; a second purchase is not certified by the menu.
   When a later BUY might be affordable but deck receipt is not certified,
   record the ambiguity and abstain if it could alter the first root.
3. At each lower/upper bank endpoint and 0/100m unknown-card value endpoint,
   take the better eligible no-card/one-card terminal route for **each** root,
   then compare roots with the unchanged common position + future point-option
   formula. This makes a publicly certified later BUY available to direct
   build and bank-trade roots, not only a BUY root. END cannot act again this
   turn; its four-roll bank-trade point option remains fully represented.
4. A complete route that reaches the standard victory threshold by known
   builds and conservative public lower-bank receipts takes priority over any
   nonwinning route or unseen-card draw. The first Action must be the exact
   current legal object, and each later step is reobserved before execution.
   A found complete winning route may stop root comparison early because its
   outcome is already certain within the public current-turn model. Any cap
   before finding such a route still returns the unchanged parent.

The card-value interval and score constants are unchanged from v1; this is a
candidate-set and outcome-ordering redesign, not a margin or weight retune.
The goal-directed search may miss strategic nonpoint current-turn routes,
including roads farther than two additions from a site. Such actions stay with
the unchanged raw parent unless a complete supported route wins the declared
comparison.
