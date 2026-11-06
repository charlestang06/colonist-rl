# Frozen midgame public-world probe: negative result

The one-shot TRAIN probe launched under source commit
`103eb6e870f71c932a1c41be39bb2231b36d6ae4` and protocol SHA-256
`a61266c487967ee7e9b5316bcff57c9c0991bab9baea1a36237f80a140b8d30d`.
Its previously frozen 32-case selection, exact proposals, four-world
reconstruction preflight, and independent seed schedule passed their
pre-outcome gates. The candidate ceiling was 660 continuations and the shared
wall limit was 900 seconds.

The runner stopped at the first case with
`continuation_error: resource bounds disagree with public hand total` after
0.441 seconds total wall time (case elapsed 0.409 seconds). It recorded zero
completed continuations, zero resolved games, zero teacher labels, and zero
ranked changes. No case, world, or chance seed was replaced or rerun.

The three files in this directory are byte-for-byte copies of the only probe
output. Their SHA-256 digests are:

| File | SHA-256 |
| --- | --- |
| `case-00-journal.json.gz` | `0ce920bf76db9499eac4c92ff219443ec4e784a414afaad4ba40f2a41486f160` |
| `case-00-receipt.json` | `b105279eb890d06733b5355e4c8ac9edd57bf52f2f2527923ee70fa23e91ee8c` |
| `probe-index.json` | `2082d4c943f63e3290f1825b6dc547554625f33865811617a12634108850cf3f` |

The observed defect is a missing joint historical resource constraint in the
world sampler: it accepted hands within each focal marginal belief interval
and a physically conserved bank, without requiring their aggregate resource
vector to match the observer ledgers' public-event history. A synthetic
hidden-transfer regression and a prospectively frozen new probe are required
before another outcome run. This archived panel is engineering evidence only.
