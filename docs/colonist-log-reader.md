# Offline Colonist log reader

## Implementation plan

**Goal:** Preserve explicitly supplied visible DOM log rows and their ordered text
and image-alt tokens, then report conservative overlap between captures.

**Boundary:** This module reads supplied Python data only. It imports no browser,
Playwright, simulator, policy, or resource ledger. It does not create game events,
infer private cards, assign event times, or fabricate stable row identities.

**Files:** `src/colonist_rl/adapters/colonist/logs.py` contains the typed offline
API; `tests/test_colonist_logs.py` covers normalization and history behavior; this
document specifies the input and limitations. Existing modules stay untouched.

- [x] Write failing normalization tests for ordered raw text and image-alt tokens,
  known resource and dice labels, unknown labels, and repeated identical rows.
- [x] Implement immutable normalized token, row, and snapshot records plus
  `normalize_snapshot`; keep every raw token value and optional explicit row ID.
- [x] Write failing history tests for initial incomplete history, exact snapshot
  idempotence, unique overlap, disjoint windows, repeated-pattern ambiguity, and
  explicit stable IDs. Implement `LogHistory.ingest` and typed `HistoryUpdate`.
- [x] Verify the dedicated tests and lint checks, document usage, and save command
  results to `/tmp/colonist-logs-implementation-report.md`. Do not commit.

## Source and limitations

The observed selector is `#ui-game .messagePart-XeUsOgLX`. Its CSS class is an
observed site detail, not a stable upstream API. Only about eight visible log
rows were mounted in an early walkthrough capture; later structured captures
contained 9–12 rows including blank placeholders. The first capture must always
report incomplete history. The caller supplies rows in displayed order and
text/image-alt tokens in DOM order; the reader does not parse HTML or request
earlier rows.

Resource labels map as follows: `Lumber` to `WOOD`, `Brick` to `BRICK`, `Wool` to
`SHEEP`, `Grain` to `WHEAT`, and `Ore` to `ORE`. `dice_1` through `dice_6` describe
visible die faces. Development-card, road, longest-road, and card-back labels
remain icon categories, never resource identities. Unknown labels remain raw
unknown tokens. These are token descriptions, not parsed game operations.

Row order describes capture order and display order, not simulator versions.
Capture timestamps describe when the caller captured the visible rows, not when
the underlying game events occurred. Equal row text is not a stable event ID.

## Input and normalization

```python
from colonist_rl.adapters.colonist.logs import LogHistory, normalize_snapshot

snapshot = normalize_snapshot(
    [
        {"row_id": None, "tokens": [
            {"kind": "text", "text": "Eilis"},
            {"kind": "text", "text": " got "},
            {"kind": "icon", "alt": "Ore"},
        ]},
    ],
    capture_id="caller-capture-001",
    captured_at="2026-11-05T03:01:11.107Z",
)
history = LogHistory()
update = history.ingest(snapshot)
assert update.status == "initial"
assert update.issues == ("initial_incomplete",)
assert update.history_complete is False
assert snapshot.rows[0].tokens[2].resource == "ORE"
```

`LogSnapshot`, `LogRow`, `LogToken`, and `HistoryUpdate` are frozen records.
Normalization copies input sequences into tuples, preserving token boundaries,
whitespace, repeated icons, repeated rows, and empty placeholder rows. A token's
`value` is its original text or `img.alt`; `category`, `resource`, and `dice`
describe only explicitly recognized labels. Matching uses exact labels, so
unrecognized variants retain `category="unknown"` and their original value.

Each row permits only `tokens` and an optional `row_id`; tokens permit `kind`
plus `text` or `alt` according to their kind. Unsupported token kinds, malformed
records, and duplicate explicit row IDs within a snapshot are rejected. Empty
and unknown icon labels remain valid evidence. `LogRow.uncertain` flags blank
rows and unknown icons; a false value does not verify an event or its meaning.

`row_id` is optional and must be an actual stable identifier explicitly supplied
by the source. Do not use DOM position, an inferred game version, or a content
hash as a row ID. `capture_id` optionally identifies a complete caller capture;
it does not identify events. Both IDs must be nonempty strings when supplied.
`captured_at` is optional text retained as capture metadata without sorting,
comparison, or conversion to event time. Ingestion order is caller order.

## Conservative history

`LogHistory.ingest(snapshot)` compares the latest window's suffix with the new
window's prefix. Every possible positive overlap length is returned; repeated
patterns do not justify selecting the longest candidate.

| Status | Result |
| --- | --- |
| `initial` | All initially observed rows appear in `new_rows`; `initial_incomplete` records the unknown earlier history. |
| `unchanged` | An exact capture replay adds no rows or stored capture and does not move the latest window backward. |
| `overlap` | One matching overlap with a nonblank anchor yields the remaining suffix as `new_rows`. |
| `gap` | No overlap: `history_gap` is reported and the entire current window remains in `unresolved_rows`. |
| `ambiguous` | Multiple overlaps, blank-only anchors, conflicting stable IDs, or previously seen IDs reappearing as new rows leave the current window unresolved. |

Explicit stable row IDs distinguish events with identical text. If IDs are
missing, matching content remains an overlap hypothesis, never proof of event
identity. Supplying an ID in only one window does not justify silently matching
it to an unidentified row in another. Reused row IDs with changed tokens report
`row_id_conflict`; IDs appearing again in an inferred new suffix report
`row_id_reappeared`. A blank-only overlap without an explicit ID reports
`empty_overlap`.

Reusing an explicit `capture_id` with exactly the same snapshot is idempotent,
including replay after newer captures. Reusing it with changed data raises
`ValueError` without changing history. Without a capture ID, only an identical
latest complete snapshot is treated as a replay. A genuinely new capture should
carry a fresh capture ID or capture timestamp so repeated windows can retain
their uncertainty instead of appearing to be the same capture.

`history.snapshots` and `history.updates` retain accepted captures in ingestion
order, including gaps and ambiguity; `history.issues` retains encountered issue
codes. Exact capture replays are excluded from those stored sequences.
`new_rows` is relative to the immediately preceding captured window, which may
itself be unresolved. Concatenating these rows cannot reconstruct complete game
history. `history_complete` stays false, including after unique overlaps. No
`VisibleEvent`, belief ledger, resource delta, or simulator version is produced.

## Reproducible captures and verification

`tests/fixtures/colonist-log-001.json` and `colonist-log-002.json` are unchanged
copies of the corresponding `log-rows-001.json` and `log-rows-002.json` files in
`runs/colonist-walkthrough-2026-11-04/`. Their source supplied no stable row IDs.
The first contains ten rows including two empty placeholders; the second has
one candidate overlap of eight rows followed by one new trade-offer row.

The dedicated tests cover unknown labels, all five resources, dice, token and
row repetition, blank rows, exact replays, gaps, ambiguous repeated patterns,
stable IDs and conflicts, and the actual consecutive DOM captures:

```sh
.venv/bin/python -m pytest tests/test_colonist_logs.py -q
.venv/bin/ruff check src/colonist_rl/adapters/colonist/logs.py tests/test_colonist_logs.py
.venv/bin/ruff format --check src/colonist_rl/adapters/colonist/logs.py tests/test_colonist_logs.py
```
