# Colonist adapter evidence design

The approved swarm wave keeps learning primary and advances the adapter through two bounded offline capabilities: A4 UI contracts and A3 semantic candidates. The existing one-click BrowserAdapter.execute remains the explicit legacy atomic execution path. A6/A7 gesture wiring follows review of these contracts in a later wave.

## Global constraints

- Work only in `/private/tmp/colonist-swarm-bxz1k39_/adapter`; original repository is read-only.
- Preserve shared `Observation`, `Action`, simulator atomic trade behavior, legacy aliases, and browser origin protections.
- Read only supplied player-visible evidence; no simulator objects, hidden DOM fields, storage, private APIs, inferred opponent identities, or invented history.
- Issue no UI gestures, start no browser, contact no external service, install no dependencies, and never update a resource or belief ledger.
- New runtime files belong only under `src/colonist_rl/adapters/colonist/`; new tests and adapter documents may accompany them.
- Use original `.venv/bin/python` and `.venv/bin/ruff` with workspace `PYTHONPATH=src` and `PYTHONDONTWRITEBYTECODE=1`; never run uv sync.

## UI evidence contracts

Add frozen, defensively detached records in `ui.py` for an exact action-bound control, visible trade response, visible trade row, UI state, and execution result. UI state records UI revision, game observation version, phase, focused control, active tool, selected complete Action, dialog stage, confirmation, and all visible rows. Controls record caller-supplied visible control identity, revision, complete Action, gesture (`click` or `drag`), and optional row identity/partner. Trade rows record caller-supplied row identity, offer identity, proposer, give/receive terms, current revision, response markers and partner controls. Do not derive stable row IDs from positions, terms, or hashes.

UI acknowledgement may advance revision while the complete game Observation stays equal. An action-bound confirmation must reject a different target, resource, partner, offer, or terms, even when action.kind matches. Trade validation binds offer ID, row ID, proposer, terms, revision, and finalization partner. A willingness marker means pending and does not certify a transfer. Duplicate rows or ambiguous duplicate response/control bindings are rejected. Constructor validation must reject negative/non-integer revisions, malformed vectors, unknown statuses, missing identities, and mutable nested payloads or detach valid input sequences to immutable tuples.

Pure validators validate confirmation, incoming response binding, outgoing finalization binding, and acknowledgement. Invalid input raises `PerceptionError` (or constructor `TypeError`/`ValueError`); failed validation produces no mutation. Acknowledgement requires newer UI revision, equal game Observation and matching stored observation versions, same exact action selection, and returns pending. A result can represent committed, pending, blocked, or ambiguous; committed requires an explicitly supplied later game Observation and nonempty caller-supplied commitment evidence. The result is an evidence record, not an independent commit verifier. Pending cannot carry a committed observation. No gesture or result record updates beliefs.

Outgoing finalization uses `Action('FINALIZE_TRADE', partner=chosen_partner, give=row.give, receive=row.receive, offer_id=row.offer_id)` only as a browser UI evidence descriptor, with proposer-perspective terms. It is not a shared policy action: do not add it to policy kinds, observation legal actions, simulator actions, codecs, or shared schemas. Incoming response descriptors remain canonical `Action('ACCEPT_TRADE'/'REJECT_TRADE', offer_id=...)`; their exact terms and proposer bind through the supplied `TradeOffer`, and the control partner is its recipient. All three control-binding validators return the exact `UIControl`.

Fixtures are synthetic replay descriptions, explicitly labelled as such, with links to the saved controls and screenshots. Coverage includes target preview, tooltip/focus, failed drag, Knight confirmation, robber target/victim stages, Monopoly choice, incoming willingness, simultaneous rows and finalization. Do not label screenshots as captured transitions. Existing execute is documented as the unchanged compatibility path and still requires immediate committed changes.

## Semantic projection

Add `semantics.py` consuming normalized immutable `LogRow`/`LogSnapshot` values and an explicit sequence of visible name/player ID bindings (allow duplicate visible names so ambiguity is representable). Output frozen per-row projections with source row, optional candidate, and issue codes; each candidate retains visible actor name, uniquely resolved actor ID or None, visible resource counts plus explicit unknown counts, and exact dice faces only for a complete pair.

Support exact visible grammars: `<name> got <resource icons>`, `<name> gave bank <resource icons> and took <resource icons>`, and `<name> rolled <die> <die>`. Text may be split across adjacent text tokens. Whitespace normalization applies only for grammar matching; preserve source tokens exactly. Production remains a candidate `production` operation, not an established event. The bank grammar is supported by the controls screenshot log and synthetic ordered token fixtures; no claim of an actual ordered DOM bank fixture. Only resource icons contribute known identities. Card backs contribute explicit unknown resource count in a resource slot; other unknown/non-resource icons retain issues and must not silently establish complete resource amounts. Exactly two recognized dice are required, with no unaccounted nonwhitespace material. No offer or acceptance row yields a transfer candidate. Player trade/build/steal/discard and all unsupported rows remain unparsed.

Missing/ambiguous player mappings retain candidate actor=None plus an issue; no guessed player ID. Preserve repeated identical rows independently. Do not create VisibleEvent, assign event sequence/time, mark history complete, reconcile LogHistory, or infer private cards. Existing normalizer/historian stays unchanged.

## Evidence and acceptance

Inputs reviewed: roadmap A4/A3, `docs/colonist-controls.md`, all eighteen saved screenshot assets as a contact sheet, `docs/colonist-log-reader.md`, current contracts/executor/normalizer/tests, both actual DOM fixtures. No applicable AGENTS.md found in the isolated repository or original repository ancestors.

Tests must demonstrate conservative rejection, retained ambiguity, immutability, exact binding, no side effects/imports, and real DOM fixture projection. Existing adapter/browser/perception/log tests remain passing. New docs distinguish delivered offline capabilities from future capture/execution/history integration.
