# Second-settlement outcome study: TRAIN feasibility result

Status: the prospective TRAIN feasibility gate passed. This is an eight-board training diagnostic, not a held-out policy-strength claim. No `TRAIN_FIT`, label-validation, pilot, confirmation, native or final-quarantine board was used in this result.

## Frozen run and integrity

The clean runner commit was `d2d545fbf804fa5c1b1dc6a8d50a205cd64cea0d` (tree `4c66dc364913dcf1f162ba38d9d120225cc83f54`), with the exact audited seed clearance receipt. It froze logical boards `157650000`–`157650007`, all four focal seats, five original legal second-settlement candidates and four paired sampled worlds per root. The unchanged raw focal and three persistent H opponents played every continuation, without domestic offers. All **640/640** counterfactual games resolved; the single-worker run exited zero in **351.32 seconds**, below the 900-second cap. No failed or replacement game was recorded. The source preflight also passed the exact public-root/menu identity, hidden-deck-order independence and candidate-order invariance tests.

The byte-verified role manifest, complete runner result and pre-outcome-frozen readout are:

| Artifact | SHA-256 |
| --- | --- |
| `train-feasibility/MANIFEST.json` | `3a64bc149dde5cba8ff9b486e1fedc93e1526a9f8e89f9bee17b653059c73d26` |
| `train-feasibility/run/RESULT.json` | `80626a279da034c37d2d3f58e54919b906c0b53f12a54c12cbd7cbc772e8610a` |
| `FEASIBILITY-READOUT.json` | `7bea0f4ba877f2deb1e075ec0f438605fa464ec066b6df57e1f905fa96f0dffb` |

The readout source was committed before reading the complete panel at `f7dbe20cee817edd6224e058c9d5ca4cef2611b6` (module SHA-256 `e4bfa5e96e039e778a0915522ad443ba14fd19b8f08cb83042c06b82d1dc444e`). It verified every saved game SHA and expected board, seat, world, original action index, public-root digest, outcome identity and resolution. It aggregates matched-world candidate-minus-raw contrasts within each root and treats the logical board as the independent unit.

## Prospective gate

All **32/32** second-settlement roots across **8/8** boards had at least one alternative that changed terminal win versus raw in a matched world, exceeding the required eight roots on four boards. The fixed λ=10 public pairwise ridge head, evaluated leave one board out, changed **19/32** roots and gained **7 net wins across 128 paired world comparisons** (+5.47 percentage points). Its board-mean relative final-VP margin was +0.320. A descriptive 20,000-resample whole-board 95% bootstrap interval for win contrast was **−0.78 to +11.72 percentage points**. Board net-win contrasts were `0, 0, +1, +1, +1, −2, +3, +3` in seed order.

The within-root shuffled-label control preserved each board/world outcome multiset but broke its candidate association. It changed 22 roots and lost 5 net matched wins (−3.91 points); its board-mean relative-VP margin was −0.750. Fixed public H-production ranking chose the raw root in all 32 candidate menus, yielding zero changes and zero contrast. These are diagnostics only; the eight-board interval includes zero and there is no promotion based on this panel.

The predeclared feasibility disposition is **`CONTINUE_NORMAL`**: the headroom condition passed, and the unshuffled leave-one-board-out head changed at least four roots with nonnegative net paired wins. The next role is the separate, still fresh `TRAIN_FIT` range `157650008`–`157650047`; model selection remains confined to TRAIN data, followed by independent label validation and full-game H panels if their own gates pass.

## Recovery

Root and Original each contain identical ignored copies at `runs/swarm-2026-11-04/agent-research/second-settlement-outcome-2026-11-06/`: the full `train-feasibility/` packet (675 files), `FEASIBILITY-READOUT.json`, and verified complete-history `source.bundle`/`readout.bundle`/`model-fit.bundle`. All 675 packet files and the readout were SHA-matched byte for byte in both destinations. The source bundle SHA-256 is `620cc1eed367ad9bd29943cc0c752a9787726689f4708e343c9cf5a15d9a2860`; the feasibility readout bundle SHA-256 is `2162358ef8a91bc6b40b8f7d9d1ed4d6c386a91826924ab81834e801cf3a2461`. Bundle verification passed in both repositories.
