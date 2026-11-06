# Fast public route fact replay

`tools/public_fact_replay.py` checks a proposed immediate action sequence in a
public Catan observation. It is meant for the factual stage of a small decision
experiment: settle whether a named trade, build, or card action is possible
before paying for another game panel or asking for a strategic judgment.

The checker uses the current checkout's simulator. Run it with that checkout's
pinned Python environment. First capture its source and engine binding:

```sh
.venv/bin/python tools/public_fact_replay.py --identity
```

Put that output's `engine`, `resource_order`, and complete
`source_files_sha256` mapping in the input JSON. `source_head` is optional; if
included, it must match this checkout's commit. The input has a `cases` array.
Each case names `packet_id`, a versioned `before` observation export,
`observation_sha256` of that export, `menu_sha256` of
`before.observation.legal_actions`, `root_index`, and the exact `root_action`
at that zero-based menu index. For a named immediate route, add
`projected_steps`: the first step is the root action, followed by exact
actions in intended order. The checker accepts the action fields consumed by
`Action`; the [compact fixture](../tests/fixtures/public-fact-replay-compact.json)
shows a root bank trade, a second bank trade and a settlement.

Hash the completed input file *before* review or evaluation, then run:

```sh
shasum -a 256 tests/fixtures/public-fact-replay-compact.json

.venv/bin/python tools/public_fact_replay.py \
  --input tests/fixtures/public-fact-replay-compact.json \
  --expected-sha256 05a4ebd3d1e0c7039c94040f21ad77dc012df3126c8c160102254f69c7a53ca6 \
  --output /tmp/public-fact-replay-compact-report.json --write
```

The command checks the input digest, engine revision, and five source-file
hashes. It refuses imports from another checkout in the same Python process.
For each case it constructs a synthetic state consistent with the public
observation, requires the **entire ordered current legal menu** and core public
view to match, then executes the exact indexed root. Each follow-up must be in
its own resulting menu. The report records resource and piece accounting,
public victory points, Longest Road, and whether bank or development-deck
availability follows from public information. A repeated `--write` succeeds
only when the saved report bytes match a fresh replay.

`engine_verified` means this exact route is feasible in a public-consistent
synthetic witness. `contradicted` marks a concrete conflict with the current
menu or rule check; `unverified` means the witness, public availability premise,
or claim could not be proved. Neither status ranks the action against passing,
buying a development card, or another trade. Opponent hands, deck contents and
post-root menus in the replay are synthetic; do not treat them as the source
game's hidden state or future events. If a strategic reviewer consumes the
report, first project only the public route evidence and omit synthetic witness
hands and bank counts.

The supported replay scope is four-player ordinary `PLAY_TURN`. It handles
named immediate bank trades, road/settlement/city builds, development-card
purchase, and a Knight root. Prefer structured `projected_steps`; the included
English claim parser is a legacy convenience and rejects wording it cannot
parse exactly. A reconstruction failure leaves the case unverified.
