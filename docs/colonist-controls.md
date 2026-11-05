# Colonist guest bot-game controls

These notes record controls exercised in a live four-player game against Easy
bots on November 4, 2026, as guest `Donni#0773`. The full game was completed with
a verified 10-point victory. The assistant operated the browser manually through
computer-use controls; the RL policy and browser adapter did not play the game.
Some card and forced-decision controls were not encountered and remain listed
below as unverified.

This guide and its linked screenshots are stored together under `docs/` so the
verified controls can be recovered without the chat history. The screenshots
are repository assets, separate from the working captures and traces in
`runs/`.

For the implementation boundaries, see [environment adapters](adapters.md) and
the [offline Colonist log reader](colonist-log-reader.md).

The in-app browser viewport was 594 × 827 pixels. The game canvas occupied
approximately `(x=0, y=195.75, width=594, height=435.5)`. Coordinates below are
illustrative for that capture only. Re-identify controls and recalibrate after
any viewport, zoom, or layout change.

## Verified result

The Victory overlay reported **67 turns and 27:27**, while the guest's score
display showed `8 (10)`: eight public points and ten including private cards.
The final score was accounted for as follows:

| Source | Points |
| --- | --- |
| Two settlements and one city, across three settlement sites | 4 |
| Longest Road | 2 |
| Largest Army | 2 |
| Two private Victory Point cards | 2 |
| Total | 10 |

The final Victory Point card purchase automatically triggered the game end and
Victory overlay. The overlay showed Continue, Map, and share controls. The
postgame overview and development-card statistics were subsequently reached;
Map and share controls were not exercised. The victory screenshot is
[16-victory.jpg](assets/colonist-walkthrough-2026-11-04/16-victory.jpg). This
verifies completion of this manually controlled game, not autonomous play or a
complete reconstruction of its resource ledger.

The final overview ranked Donni#0773 first with 10 points, followed by Kati with
4, Eilis with 3, and Almire with 2. The development-card statistics reported
these draws across all players:

| Development card | Drawn across all players |
| --- | --- |
| Knight | 7 |
| Victory Point | 3 |
| Monopoly | 2 |
| Road Building | 0 |
| Year of Plenty | 2 |

No player drew Road Building in this game. Both Year of Plenty cards were used
by bots; their own-card controls were not exercised by the guest.

## Starting the game

Select Bots mode, then use the **Easy checkbox** to select the bot difficulty.
Clicking the card title opens an information page rather than selecting that
difficulty. A four-player guest bot game was started successfully.

## Initial placement

Both initial settlement and initial road placement use a separate confirmation:

1. Click the desired legal board vertex or edge.
2. Inspect the floating house or road icon that appears above the selected
   location.
3. Click that icon to commit the placement.
4. Verify the board and game log before proceeding.

Clicking the board location alone selects a preview. The complete snake-order
setup was exercised, including both guest settlements and both guest roads.
The second settlement produced a visible starting-resource log of Grain,
Lumber, and Ore.

## Ordinary turns

| Control | Verified behavior |
| --- | --- |
| Large dice above the bottom-right board area | Rolls the dice. |
| Second control in the bottom toolbar | Buys a development card; approximately `(225, 609)`. A purchase deducted one Wool, one Grain, and one Ore and produced a purchase log entry. |
| Third control in the bottom toolbar | Selects the road-building tool. Clicking a legal edge then committed an ordinary paid road directly, with a matching log entry and piece-count change. |
| Settlement control, approximately `(311, 609)` | A click showed the cost tooltip. Dragging the control onto the board entered `Build Settlement`; a subsequent click on the intended vertex committed the settlement directly. |
| City control, approximately `(354, 609)` | Clicking to focus/show the tooltip, then dragging the control to an owned settlement entered `Build City`; a subsequent click on the settlement committed the upgrade directly. |
| Double-arrow at the bottom right | Ends the turn. |

The ordinary road did not require the floating confirmation used during setup.
The ordinary settlement also committed without a floating confirmation once its
build mode was active. At approximately `(183, 375)`, placement deducted the four
required resources, logged `built a Settlement (+1 VP)`, and increased the
guest's score from two to three.

Upgrading the owned settlement at approximately `(210, 329)` to a city cost three
Ore and two Grain. The log reported a city build worth one additional victory
point, and the guest's score increased from three to four. The settlement piece
returned to supply, leaving three settlements and three cities available.

Click-to-focus followed by drag was a tested activation pattern for the
settlement and city controls. The observed tooltip-only clicks do not establish
that direct clicks can never activate a tool. Verify the displayed build mode
before clicking the target.

After the verified development-card purchase, a purple Knight card appeared in
the hand. The turn was ended after purchase before the card was played.

## Victory-point cards and automatic awards

The third development-card purchase drew a Victory Point card shown with a
trophy. It stayed in the hand and required no activation. The guest's score
display changed from four public points to `4 (5)`: four public points and five
points known to the guest, including the private card.

Clicking the Victory Point card opened an information/acknowledgement panel
stating that it secretly awards one point and is automatically used. The check
mark dismissed that panel. Opening or dismissing this information panel was not
an activation required to earn the point.

Building two connecting roads extended the guest's existing network to length
five. The game immediately displayed a Longest Road banner, logged receipt of
Longest Road worth two victory points, and raised the public score from four to
six. Including the private Victory Point card, the guest's display was `6 (7)`.
The road-length indicator showed five.

Later, Almire also reached road length five while the guest's length remained
five. The guest retained Longest Road through that tie. A later loss of an award
was not observed.

Playing the guest's third Knight awarded Largest Army at a played-Knight count
of three. The two-point award occurred immediately after card confirmation,
before the resulting robber placement and victim selection were resolved. The
guest's score then displayed `8 (9)`.

These awards are automatic observed state changes, not additional actions to
click. Across the completed game, the guest drew seven development cards: three
Knights, two Monopolies, and two Victory Point cards. Both private Victory Point
cards counted automatically; the second triggered victory.

## Settings, pause, and keyboard controls

The gear opens Settings. Opening Settings automatically paused this bot game.
Closing Settings left a **Paused** overlay with a **Resume** button. Clicking
Resume resumed play and produced a `Game Resumed` log entry.

The Controls page showed **Keyboard Shortcuts On** and these mappings:

| Action | Displayed shortcut |
| --- | --- |
| Roll dice | Spacebar |
| End turn | Spacebar |
| Pause | P |
| Fullscreen | F |
| Mute | M |
| Adjust volume | Arrow keys |

Spacebar successfully ended a turn after the canvas received focus. The other
listed shortcuts were inspected, not all exercised. Focus matters: pressing
Space while the Resume button retained focus retriggered pause instead of
performing the intended gameplay action. Coordinate clicks on the dice were
used afterward. A keyboard controller must verify both the current game phase
and focused control before sending a shortcut.

## Knight and robber sequence

Ordinary clicks and a double-click on the Knight card showed its tooltip without
playing it. Dragging an unfocused card sometimes produced no visible response.
The card's rendered DOM bounding box was used to locate the exact card; clicking
it to focus/show the tooltip and then dragging it opened the confirmation panel.
The panel, rather than completion of the drag gesture, establishes that this
step succeeded. The following multi-stage sequence was exercised:

1. Drag the Knight card from the hand, approximately `(123, 609)`, onto the board.
   This opens a card confirmation panel at the lower left.
2. Click the panel's check mark, approximately `(193, 569)`; its **X** is near
   `(166, 569)`. Confirmation logged use of the Knight and entered `Place Robber`.
3. Select a destination tile. Selecting approximately `(129, 344)` displayed a
   floating robber icon above it, around `(129, 309)`.
4. Click that robber icon to confirm the move. The log identified the destination
   as the Ore tile numbered 3.
5. With two adjacent eligible victims, a lower-left `Select Player To Rob` dialog
   appeared. Select a player avatar, then click the dialog's check mark. The
   exercised avatar was around `(30, 552)` and the check around `(175, 571)`.
6. Verify the steal result. The guest's log reported `You stole Lumber from Eilis`,
   the hand gained one Lumber, and the played-Knight count was one.

Card confirmation, robber placement confirmation, and victim confirmation are
distinct steps. The final steal must not be inferred from the earlier Knight or
robber clicks. The two-victim dialog path and a single-victim automatic-steal path
were both exercised. With one eligible victim, confirming the robber destination
stole from Eilis automatically and showed no player-selection dialog.

A later Knight was played after resolving a seven and its robber move in that
same turn. The focus-then-drag and confirmation sequence entered robber placement;
selecting and confirming the Grain tile numbered 6 automatically stole Ore from
the sole eligible victim. The guest's played-Knight count then showed two. This
was the second Knight played over the game, not evidence of two development-card
plays in one turn.

## Monopoly and development-card timing

For Monopoly, clicking to focus/show its tooltip and then dragging the card onto
the board opened `Select Resource` directly. It did not require the preliminary
Knight confirmation panel. The five resource choices were around
`x=17, 39, 63, 86, 108` at `y=527`; cancel was around `(166, 568)` and confirmation
around `(193, 568)` in this viewport.

Selecting Ore and confirming logged use of Monopoly and collection of seven
Ore, verified against the guest's hand change. A later Monopoly selecting Wool
collected five Wool. Resource choice and successful card use must be verified
against the selected resource and reported transfer.

Attempting to click/drag a newly bought Monopoly during its purchase turn was
blocked by the toast `Cannot play a development card on the turn you bought it`.
The card could be played on the guest's next turn. This restriction was verified
for Monopoly; Victory Point cards counted immediately without a play action.

## Rolling seven and visible bot actions

A seven rolled on the guest's turn required an over-limit bot to discard before
the guest could place the robber. The guest's hand held at most seven cards, so
the guest's own discard interface was not exercised. In this observed UI, the
bot discard log exposed the resource types: Grain, Wool, Brick, Ore, and Grain.
Resource identities are known here because that visible log disclosed them;
do not assume every discard or UI configuration exposes the same information.

After the discard, selecting the Grain tile numbered 11 and clicking the
floating robber confirmation automatically stole from Eilis, the single
eligible victim. A separate victim chooser was unnecessary for this path.

Both Year of Plenty cards were seen used by bots. Eilis's earlier log included
`Eilis used Year of Plenty` followed by a bank withdrawal of Lumber and Brick;
Kati later also took Lumber and Brick after using that card. The guest drew
neither Year of Plenty nor Road Building, so their own-card controls remain
unverified.

## Reading visible resource logs

The rendered game log contains resource information that plain `innerText` can
omit: inline image icons carry their labels in `alt` attributes. Expanding those
icons in the current log produced messages such as:

```text
Donni#0773 got [Ore] [Ore] [Ore]
Eilis got [Brick] [Ore]
```

The same visible log included trade, build, and Longest Road messages. The
observed message selector was `#ui-game .messagePart-XeUsOgLX`. It is an
implementation-specific selector, not a stable interface; verify it against the
rendered log if the site changes. This read-only Playwright JavaScript example
reads mounted message nodes and expands inline image labels:

```javascript
const messages = await page
  .locator("#ui-game .messagePart-XeUsOgLX")
  .evaluateAll(rows => {
    const expand = node => {
      if (node.nodeType === Node.TEXT_NODE) return node.textContent ?? "";
      if (node.nodeType !== Node.ELEMENT_NODE) return "";
      if (node.tagName === "IMG") {
        return ` [${node.getAttribute("alt") || "unknown icon"}] `;
      }
      return [...node.childNodes].map(expand).join("");
    };
    return rows.map(row => expand(row).replace(/\s+/g, " ").trim());
  });
```

Normalize only labels actually present in the visible UI:

| Visible label | Observation resource or value |
| --- | --- |
| Lumber | `WOOD` |
| Brick | `BRICK` |
| Wool | `SHEEP` |
| Grain | `WHEAT` |
| Ore | `ORE` |
| `dice_N` | Visible die face `N`; use the roll's complete visible dice to establish its total. |
| Card back or unknown icon | Unknown identity; retain the visible count and uncertainty. |

Only eight message nodes were mounted when this extraction was checked. The
log is virtualized: one DOM snapshot is not the complete game history. An
ongoing historian must capture continuously, reconcile ordered overlap between
captures, and scroll the visible log to recover any missing accessible history.
Use a stable message identifier if the rendered UI supplies one; do not dedupe
solely by text because distinct rolls or transfers can produce identical text.
Record gaps instead of inventing missing events.

The ordered captures numbered 001–008 retained only partial log windows. Later
attempts, starting with 009, did not recover complete earlier history. During
postgame scroll recovery the source game was left, the room became unavailable,
and the browser returned to a guest landing modal. Empty/failed recovery captures
are not included as evidence. The full log was not recovered.

With a known starting hand and a complete visible event sequence, this can
support exact resource accounting for changes whose identities are disclosed.
It cannot guarantee exact opponent hands after hidden steals, unrevealed card
backs, or missing history. Keep those outcomes as joint resource uncertainty,
even when public card totals are known. An offer, a willingness response, or
composer contents are not a transfer; apply ledger changes only after visible
evidence of the completed exchange.

This recipe uses rendered DOM text and icon labels. It does not inspect internal
game objects, hidden opponent state, storage, or private game APIs, and it is not
yet integrated into the capture reader or event ledger. The separate
[offline log reader](colonist-log-reader.md) now normalizes supplied text/icon
tokens and conservatively compares captured log windows; it does not infer game
events, resource transfers, or complete history.

## Trading

The first bottom-toolbar control opens the trade composer, approximately
`(182, 609)` in the observed viewport. While the composer is open, a close **X**
replaces that opener.

In the composer, select resources to receive from the upper resource palette
and resources to give from the guest's hand along the bottom. The players icon
submits a player-trade offer. The bank icon is a separate control.

The composer retained its selected give and receive quantities after a completed
trade. Clicking a selected give or receive card removes one unit. Inspect both
sides before submitting the next offer rather than assuming a blank composer.

### Bank trade

A four-Lumber-for-one-Grain bank trade was exercised: select four Lumber from
the hand to give, select one Grain to receive, then click the bank icon,
approximately `(181, 523)` in the observed composer. The exchange committed
immediately, with the log reporting four Lumber given to the bank and one Grain
taken. The hand and displayed bank counts updated.

Unlike the completed player trade, this bank trade cleared the composer's
selected quantities. Port-specific exchanges have not yet been exercised.

### Offering a player trade

An offer giving one Ore for one Wool was submitted. Other players displayed
colored rejection crosses and acceptance check marks. The green player rejected
the offer; clicking the accepting orange player's check mark finalized the
exchange. The resource transfer was verified against the guest's hand and the
game log, which identified the final partner.

An acceptance check therefore identifies a willing partner; the proposer still
selects the partner to finalize the trade. Submitting an offer does not itself
transfer resources.

Submitting revised quantities can leave multiple outgoing offers stacked at
once. Each offer row has its own responses and accepting-partner checks. A
two-Lumber-for-two-Grain offer was rejected, followed by a successful
one-for-one offer. Later, a two-Lumber-for-one-Grain offer and a new
one-Lumber-for-one-Grain offer were visible simultaneously. The newer offer was
finalized using the green check in its own row, approximately `(379, 341)`, rather
than a check in the upper row around `y=274`. Bind every finalization click to
the exact current offer row, terms, and accepting partner.

### Responding to another player's offer

The incoming offer panel displayed an edit pencil, rejection cross, and
acceptance check. An incoming offer giving the guest one Lumber in exchange for
one Grain was accepted. This signalled willingness without immediately changing
the guest's hand. The proposing bot could subsequently select a different
partner or cancel the offer.

Rejecting an incoming offer with its **X** was also exercised and returned the
UI to the normal bot turn. The edit pencil was observed, but its complete
interaction sequence is not yet verified. An acceptance click or a disappearing
panel must not be recorded as a completed resource transfer without
corresponding visible evidence.

## Saved captures

Durable screenshot copies are under
`docs/assets/colonist-walkthrough-2026-11-04/`. The original working captures
remain under `runs/colonist-walkthrough-2026-11-04/`.

| File | Evidence |
| --- | --- |
| [01-setup-complete.jpg](assets/colonist-walkthrough-2026-11-04/01-setup-complete.jpg) | Setup board after the guest's placements; bots may still be completing their setup. The filename alone does not establish that every player had finished. |
| [02-incoming-trade.jpg](assets/colonist-walkthrough-2026-11-04/02-incoming-trade.jpg) | Incoming trade panel. |
| [03-trade-composer.jpg](assets/colonist-walkthrough-2026-11-04/03-trade-composer.jpg) | Resource-selection composer. |
| [04-trade-completed.jpg](assets/colonist-walkthrough-2026-11-04/04-trade-completed.jpg) | State after the finalized outgoing player trade. |
| [05-keyboard-controls.jpg](assets/colonist-walkthrough-2026-11-04/05-keyboard-controls.jpg) | Settings Controls page and displayed keyboard shortcuts. |
| [06-knight-confirmation.jpg](assets/colonist-walkthrough-2026-11-04/06-knight-confirmation.jpg) | Confirmation panel after dragging the Knight card onto the board. |
| [07-robber-victim-selection.jpg](assets/colonist-walkthrough-2026-11-04/07-robber-victim-selection.jpg) | Player selection after confirming the robber destination. |
| [08-settlement-vp.jpg](assets/colonist-walkthrough-2026-11-04/08-settlement-vp.jpg) | Ordinary settlement and its resulting victory-point increase. |
| [09-city-upgrade.jpg](assets/colonist-walkthrough-2026-11-04/09-city-upgrade.jpg) | City upgrade and its resulting victory-point increase. |
| [10-bank-trade-composer.jpg](assets/colonist-walkthrough-2026-11-04/10-bank-trade-composer.jpg) | Composer quantities for the exercised four-to-one bank trade. |
| [11-longest-road-and-private-vp.jpg](assets/colonist-walkthrough-2026-11-04/11-longest-road-and-private-vp.jpg) | Longest Road award, five-road length, and the guest's public/private score display. |
| [12-vp-card-panel.jpg](assets/colonist-walkthrough-2026-11-04/12-vp-card-panel.jpg) | Victory Point card information stating that its point is secret and automatic. |
| [13-largest-army.jpg](assets/colonist-walkthrough-2026-11-04/13-largest-army.jpg) | Largest Army awarded on the third Knight, before its robber sequence was resolved. |
| [14-new-dev-card-timing.jpg](assets/colonist-walkthrough-2026-11-04/14-new-dev-card-timing.jpg) | Toast preventing a newly bought Monopoly from being played in its purchase turn. |
| [15-monopoly-resource-selection.jpg](assets/colonist-walkthrough-2026-11-04/15-monopoly-resource-selection.jpg) | Monopoly's direct resource-selection panel. |
| [16-victory.jpg](assets/colonist-walkthrough-2026-11-04/16-victory.jpg) | Final Victory overlay: 67 turns, 27:27, and the guest's `8 (10)` score display. |
| [17-final-overview.jpg](assets/colonist-walkthrough-2026-11-04/17-final-overview.jpg) | Final rankings: Donni#0773 10, Kati 4, Eilis 3, Almire 2. |
| [18-final-development-card-stats.jpg](assets/colonist-walkthrough-2026-11-04/18-final-development-card-stats.jpg) | All-player draws: seven Knights, three Victory Points, two Monopolies, no Road Building, two Year of Plenty. |

These snapshots supplement the observed interactions. A single image does not
establish a complete before-and-after transition.

## Adapter implications — recommendations, not implemented

The current `BrowserAdapter.execute` in
`src/colonist_rl/adapters/colonist/execution.py` (also exported through
`colonist_rl.perception`) performs one click and immediately requires a newer
observation with a committed
action-specific change. Its transport protocol exposes screenshots and clicks,
without a drag gesture. Its placement tests assume one-click commitment. Its
trade-acceptance verifier requires an immediate exact hand change and reported
`DOMESTIC_TRADE` event, matching the simulator's atomic trade abstraction. These
interfaces do not cover all of the live controls above.

A browser-specific UI state should accompany the game observation. It should
record a UI revision, setup or ordinary-turn phase, active tool, exact selected
typed action, required click or drag gesture, visible confirmation control,
current dialog stage, and all visible offer rows with their identities, terms,
responses, and partner controls. UI acknowledgement can advance this revision
while the game observation and resource ledger remain unchanged.

Placement execution should follow the verified current UI: select a target,
verify any action-bound confirmation, click that confirmation once, and verify
the committed placement. Ordinary direct-placement controls should retain their
shorter sequence. A confirmation must be tied to the exact selected action,
including its target, rather than inferred solely from its action kind. Tool
activation may require a drag even when a click on the same control only opens
a tooltip. Verify entry into the intended build mode before selecting a target.
Knight execution needs separate verified stages for card confirmation, robber
target confirmation, and any victim-selection dialog. An unfocused drag may
have no visible effect; inspect the resulting tool or dialog state before
continuing. Robber completion can either expose a victim chooser or perform a
visible automatic steal, depending on the eligible victims.

Development-card dialogs differ by card: Monopoly proceeds directly to resource
selection, while Knight has an initial confirmation. The reader must also
recognize purchase-turn timing rejections, automatic award changes before a
forced robber sequence completes, and automatic terminal victory after a
Victory Point draw.

Trade execution should distinguish a pending response from a completed exchange.
Accepting an incoming offer should verify the same live offer and its accepted
marker, then return a pending result. Later observations should establish either
the actual resource transfer or cancellation/selection of another partner. The
offer fingerprint should bind proposer, terms, identity, and current UI revision
so that a changed offer cannot consume an old response. Multiple offer rows can
coexist, so the exact row and accepting partner must also be bound to a
finalization control. Composer selections persist across exchanges and must be
read before composing a subsequent offer. The existing policy's binary accept
decision can express willingness; the simulator's immediate commitment remains a
documented simplification.

Bounded waiting for stable UI updates is also needed. A partial click sequence,
unknown confirmation, or ambiguous outcome should preserve the existing manual
review lockout without repeating clicks or predicting ledger changes.

## Controls not verified in this game

- Port trades and the incoming-offer edit sequence.
- The guest's own discard controls and robber destinations with no eligible victim.
- The guest's own Year of Plenty and Road Building controls.
- Loss of Longest Road or Largest Army.
- Postgame Map/share controls and full resource-ledger reconciliation.
