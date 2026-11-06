# More search nodes did not reach more states in the first fast screen

The manifest-driven runner completed a fixed exploratory comparison of the raw
public planner at 2,048 and 8,192 attempted nodes. The larger budget changed
no actions or outcomes on four fresh boards. It spent 3,341 extra attempts,
all on invalid transitions. Keep the 2,048-node setting; this screen gives no
reason to run an unchanged larger-budget panel.

## Frozen comparison

Both arms used depth 6, beam 24, geometry budget 100,000, seed 11, gain margin
10,000 and the same raw public score. Only `max_nodes` differed. Four boards
(`138230000`–`138230003`) were run in all four focal seats against three
persistent H opponents with domestic offers disabled; bank trades remained
legal. The separate smoke seed `138229999` was excluded. This was a development
screen with no training labels or strength-promotion rule.

The [frozen protocol](../runs/swarm-2026-11-04/agent-research/search-budget-001/protocol.json),
[input manifest](../runs/swarm-2026-11-04/agent-research/search-budget-001/screen-manifest.json)
and [pre-run bindings](../runs/swarm-2026-11-04/agent-research/search-budget-001/freeze.json)
pin source commit `874fe91ca8fea18b31db862bc5d6aba3235aee08`, project
source-tree SHA `429912d6d2e44338c35cd64d1c9eb3e136d1af938d099023b3eac833eddae650`,
engine identity and the input manifest SHA. The run verified matching source and
version digests after gameplay. Every one of 32 planned games resolved, with no
truncation. The command completed in 20.25 seconds and recorded 2,690 public
candidate steps across both arms.

| Measure, over 16 complete games per arm | 2,048 nodes | 8,192 nodes |
|---|---:|---:|
| Wins | 5 | 5 |
| Own decisions | 1,345 | 1,345 |
| Attempted search transitions | 211,929 | 215,270 |
| Feasible transitions | 17,810 | 17,810 |
| Unique successors | 17,017 | 17,017 |
| Duplicate successors | 793 | 793 |
| Beam-truncated successors | 6,213 | 6,213 |
| Node-limit stops | 13 | 0 |
| Empty-beam stops | 547 | 560 |
| Geometry-limit stops | 36 | 36 |
| Mean / p95 selection time | 1.45 / 8.05 ms | 1.50 / 8.21 ms |

The 13 small-budget node-limit stops became empty-beam stops with the larger
budget. Root attempts (3,068), maximum feasible-depth histograms, selected
actions, and every public action trace matched. All 16 paired seat games had
identical traces and outcomes; their paired win-rate and mean-VP differences
were both zero. The maximum feasible depth histogram was 749 at depth 0
(unsupported/no search), 410 at depth 1, 100 at depth 2, 63 at depth 3, and
23 at depth 4 for **each** arm. No feasible state reached depth 5 or 6 in
this screen. The extra 3,341 attempts did not add a feasible transition.

The [saved game/step journals and summary](../runs/swarm-2026-11-04/agent-research/search-budget-001/screen-runs/paired-20261106T155431Z-12f6fbe9/summary.json)
and [digest-checked analysis](../runs/swarm-2026-11-04/agent-research/search-budget-001/screen-analysis.json)
are in the [byte-verified archive](../runs/swarm-2026-11-04/agent-research/search-budget-001/archive-manifest.json).
The analyzer only forms win-rate contrasts when every matched game resolves;
search work remains descriptive over attempted selections. This four-board
screen can reject the unchanged budget idea for now; it cannot prove that a
larger budget never helps on another board.

## Runner checks and next decision

The same manifest command also completed a one-board raw-versus-8,192-node
smoke: eight resolved games, valid game/step journals and identical four-seat
traces. Five 2,048-node stops and 924 additional 8,192-node attempts again
added no feasible state. A separate one-board smoke loaded the pinned frozen
neural checkpoint through the graph arm, completed eight games with 817 public
steps, and preserved source/checkpoint provenance. Its raw 2/4 versus neural
0/4 wins are an integration observation, not strength evidence. The CLI and
analyzer passed 47 focused tests after review, including explicit reference
orientation, trace durability, hashes, censored outcomes, and malformed traces.

Next diagnose the large invalid-transition share with an exact public candidate
filter or ordering study, keeping source selection semantics and latency visible.
The economic track can separately compare uncertain card purchase value with
certain bank/build/pass outcomes. Neither track should spend more nodes or add
epochs solely because a bigger budget is available.
