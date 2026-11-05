# Offline Colonist log semantic candidates

`colonist_rl.adapters.colonist.semantics` projects explicitly supplied normalized
`LogRow` and `LogSnapshot` evidence. It reads ordered visible tokens only. It
imports the shared resource contract and log record types; it starts no browser,
queries no service, and reads no simulator, policy, private cards, or ledger.

## API

```python
from colonist_rl.adapters.colonist.logs import normalize_snapshot
from colonist_rl.adapters.colonist.semantics import (
    VisiblePlayerName, project_row, project_snapshot,
)

snapshot = normalize_snapshot([{"tokens": [
    {"kind": "text", "text": "Eilis got "},
    {"kind": "icon", "alt": "Ore"},
    {"kind": "icon", "alt": "Lumber"},
]}])
players = [VisiblePlayerName(name="Eilis", player_id=2)]
projection = project_row(snapshot.rows[0], players)
assert projection.source is snapshot.rows[0]
assert projection.candidate.kind == "production"
assert projection.candidate.actor_name == "Eilis"
assert projection.candidate.actor == 2
assert projection.candidate.receive.known == (1, 0, 0, 0, 1)
assert projection.candidate.receive.unknown_count == 0
assert project_snapshot(snapshot, players) == (projection,)
```

All five new record types are frozen:

- `VisiblePlayerName(name, player_id)` supplies an explicit visible-name binding.
  Names must be nonblank strings; IDs are nonnegative integers, excluding bools.
  A sequence can contain duplicate names so ambiguity remains representable.
- `ResourceEvidence(known, unknown_count)` holds five nonnegative integer counts
  ordered `WOOD, BRICK, SHEEP, WHEAT, ORE`, plus a nonnegative unknown count.
- `SemanticCandidate(kind, actor_name, actor, give, receive, dice)` holds visible
  operation evidence. `actor` can be `None`. Resource fields are
  `ResourceEvidence`; dice is an immutable pair for rolls and `()` otherwise.
- `RowProjection(source, candidate, issues)` retains the exact original source
  row, an optional candidate, and immutable issue codes.

Resource vectors, dice sequences, and issue sequences are detached into tuples.
Invalid kinds, invalid dice faces, empty production/bank amounts, roll resource
amounts, production give amounts, and dice on resource operations are rejected.
`project_snapshot` returns one projection per supplied row in display order,
including empty placeholders and identical repeated rows. No deduplication or
new identity is introduced.

## Complete-row grammar

Matching is case-sensitive and consumes every icon and every nonwhitespace text
fragment. Adjacent text fragments are joined without inserting text, then
whitespace is folded for matching only. The candidate's `actor_name` is the
matched visible name with whitespace folded; bindings are compared with the same
folding. Source whitespace, token boundaries, image-alt values, row IDs, and
repetitions remain untouched in `source`.

| Supplied visible grammar | Candidate kind | Evidence |
| --- | --- | --- |
| `<name> got <resource icons>` | `production` | Receive only; no cause of production or turn is inferred. |
| `<name> gave bank <resource icons> and took <resource icons>` | `bank_trade` | Give and receive from the named actor's perspective. |
| `<name> rolled <die> <die>` | `roll` | Exactly two recognized faces, each from 1 through 6. |

A resource slot requires at least one supplied icon. `Lumber`, `Brick`, `Wool`,
`Grain`, and `Ore` icons contribute known resource counts. Repeated icons count
independently. Resource names in body text never become resources. Card-back
icons (`CardBack`, `card back`, `cardback`, as recognized by the normalizer)
contribute one unknown resource each; no identity is inferred. A non-resource or
unknown icon in a resource slot rejects the candidate, retaining all raw
information in `source`. Extra text, punctuation, or icons cannot be silently
ignored. Incomplete rolls, three dice, text digits, and unknown die labels do not
produce a roll candidate.

Offers, acceptances, completed player trades, builds, steals, discards, and all
other wording remain unparsed. In particular, willingness to trade never
produces a resource transfer candidate.

## Issue codes

| Code | Meaning |
| --- | --- |
| `unresolved_actor` | No supplied binding matches the visible actor name; the candidate retains `actor=None`. |
| `ambiguous_actor` | More than one binding matches, even if their IDs are equal; the candidate retains `actor=None`. |
| `unknown_resources` | A parsed resource slot contains card backs; inspect `unknown_count` rather than treating the known vector as complete. |
| `unsupported_resource_icon` | A resource slot contains an icon other than a recognized resource or card back; no candidate is produced. |
| `invalid_dice` | A roll prefix is visible but its entire tail is not exactly two recognized dice; no candidate is produced. |
| `empty_row` | No nonwhitespace text or icon is present. |
| `unparsed_row` | The whole row does not satisfy a supported grammar, including missing slots or unaccounted text. |
| `unsupported_icon` | An unparsed row also contains an unknown-category icon. |

Issue tuples follow deterministic parser order. A candidate can carry both an
actor-resolution issue and `unknown_resources`. No player ID is guessed from
row order, previous rows, a guest name, or a presumed opponent list. Unsupported
rows still have their original source evidence available to a future parser.

## Evidence provenance and remaining integration

The unchanged real DOM captures `tests/fixtures/colonist-log-001.json` and
`colonist-log-002.json` support production and complete `(1, 2)` and `(3, 2)`
rolls, as well as the unsupported trade-offer/player-trade/build examples.
They contain no ordered DOM bank-exchange row.

`tests/fixtures/colonist-semantic-synthetic.json` is explicitly **synthetic
ordered token data**, not a DOM capture. Its completed four-Lumber-for-one-Grain
bank grammar follows the visible bank log wording recorded in the
[controls guide](colonist-controls.md#bank-trade) and screenshot evidence; it
also supplies resource repetition, card-back, and unsupported-icon examples.
A synthetic ordering does not establish that a live collector will emit that
ordering. Live collection and bank DOM capture verification remain outstanding.

A semantic candidate describes one row's visible meaning. An established game
event additionally needs trustworthy identity and ordering evidence. Ledger
application additionally needs validated event semantics, actor resolution,
resource completeness, history continuity, and application policy. This module
creates no `VisibleEvent`, sequence, event timestamp, resource delta, or ledger
update, and never marks history complete. Capture metadata is not event time.
Equal rows are independent supplied evidence, not proof of equal event identity.

The [offline log reader](colonist-log-reader.md) remains unchanged. Integrating
these projections with its conservative history results, establishing event
identity, handling gaps/replays, and deciding ledger application are future work.
