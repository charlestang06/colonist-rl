# Second-settlement outcome study: TRAIN_FIT stopped by backend topology bug

Status: **failed mechanically; no model or policy-strength result**. The frozen
two-worker `TRAIN_FIT` role planned 6,400 matched full-game continuations on
logical boards `157650008`–`157650047`. It wrote 5,325 append-only game journals
before worker 0 failed during board `157650040`, focal seat 2, world slot 6,
original candidate index 30. There is no complete `RESULT.json`. The partial
labels were not fit or scored; label validation and H/native panels did not
open.

The failed source commit was `8d0d375fa11211cf23241e694b04e9372ee10198`
(tree `b92c8eb4f57bdf8f88cee302fd2646506cfee100`), with frozen packet
`MANIFEST.json` SHA-256
`c0683ef86138d35f342dcfcb0ab4ee966271bcc78c0d8da3082f96fbb11bdc6a`.
The parent `FAILED.json` SHA-256 is
`ba08ceabb1224c32199e49d42f5ca935f80a7011712736f273b0dbbd018b355e`;
worker 0 `FAILED.json` SHA-256 is
`22ab1c0d5cc47ee5aff3dfba1a5f8b4fbdfd1c66c73b14eb7054845b8d3c6a6f`.
The worker traceback ends in pinned Catanatron 3.3.0
`Board.build_settlement` at `catanatron/models/board.py:132`, where a legal
settlement between two opposing roads asks for a connected-component index and
gets `None`.

An excluded, 19-action pure-Board reproduction confirms the cause. The backend
retains an opponent settlement vertex in a player's cached road component;
`buildable_edges` then offers a road through that enemy settlement. A later
enemy settlement cuts a different pair of roads and the backend's DFS drops
road endpoints behind the first blocker while the roads remain on the board.
The crash is a delayed symptom of illegal road expansion and stale longest-road
topology, so catching the `None` index alone would not repair game outcomes.
The failing role's diagnostic action history confirms the same sequence: RED
road 1–2 at game version 19, ORANGE settlement at node 2 at version 41, RED
road 2–3 through that enemy node at version 119, a later component-losing
settlement at version 234, and the crash at version 401.

Root and Original ignored archives each preserve the entire partial packet,
all 5,325 worker game journals, 160 public cases, the manifest, all run intents,
both failure receipts, both logs, and the excluded reproduction at
`runs/swarm-2026-11-04/agent-research/second-settlement-fit-failed-001/`.
`BYTE-MANIFEST.json` SHA-256 is
`da1b19b8d66cd0449a5cd812032de1d7df67bca86d56c5b2b79e1e2e71dfe2f2`,
covering 5,493 files and 6,363,692 source bytes. The verified compressed
archive SHA-256 is
`618ba8b0a89245a4d30f6f315bd7bd83a531aefc3c62f9a42bb0914645073751`.

The earlier 640/640 feasibility gate remains a historical, correctly reported
diagnostic under the old backend. It **does not qualify** any repaired-engine
FIT: even games that resolved may have included illegal through-blocker roads
or wrong longest-road awards. No opened board or world seed will be reused.

## Corrected study ordering

1. Seal and independently review the opt-in local topology/award shim, its
   exact source hash and backend identity, excluded mechanics regressions,
   unaffected-game parity, and bounded throughput.
2. Reserve entirely fresh logical board/world seed roles and scan archives and
   local artifacts for exact overlap before opening any corrected game.
3. Freeze a new corrected-engine feasibility packet and gate before outcomes.
   Only a positive gate can authorize a new corrected-engine `TRAIN_FIT` packet.
4. Fit on complete, verified corrected-engine TRAIN journals only. Freeze the
   model and its public choices before independent label validation.
5. If label validation passes, run the corrected-engine H pilot with a sealed
   public-only blind decision audit, then fresh H confirmation; native and final
   roles remain conditional on their prior gates.

Every role must call the shared corrected-game factory, assert its mode after
reset, and bind the shim source SHA-256 and effective backend version in its
manifest. No role outcome is opened or compared across the two engine versions
as if they were one experiment.
