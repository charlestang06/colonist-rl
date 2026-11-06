# Raw public teacher Stage 1: stopped before native opponents

The [prospectively frozen protocol](PROTOCOL.md) tested the unchanged raw public
planner, unchanged seed-11 neural parent, and H or strict GreedyNoTrade focal
control on the same board and seat schedule. Both eight-board panels completed
all 96 games: 192/192 resolved, no truncated or invalid games, no domestic
offers, valid journals, and zero provenance errors. The raw planner received
only its permitted public observation and exact legal root.

| Stage 1 panel, 32 games per arm | Raw wins / mean VP | Parent wins / mean VP | Control wins / mean VP |
| --- | ---: | ---: | ---: |
| Versus three H opponents | 9 / 7.344 | 3 / 5.844 | H: 7 / 6.625 |
| Versus three no-trade opponents | 31 / 10.063 | 20 / 8.531 | No-trade: 8 / 5.531 |

**Frozen gate: `advance=false`.** In the H panel, raw beat H by only two wins
of 32 versus the required four. The paired whole-board bootstrap 95% raw–H
win interval was [−0.09375, 0.21875], and raw–parent was
[−0.03125, 0.375]; both lower endpoints missed the required zero. Raw's
mean-VP gains (+0.719 over H, +1.500 over parent) passed. All GNT win, VP and
interval checks passed, but the protocol required **both** panels. This is an
operational eight-board screen, not a population estimate of policy strength.
The raw planner is `teacher_not_qualified_on_stage1`; no VF/AB Stage 2 game ran,
no model was fitted or promoted, and the conditional `157540000..031` seeds
remain untouched by this study.

A deterministic parent replay on every saved raw own public prefix found 634
exact-root mismatches among 2,959 H-panel raw decisions and 830 among 3,318
GNT-panel decisions. Trade-related mismatches were 166/259, END-related
108/164, card-related 20/17, and BUY-related zero in H/GNT respectively;
categories can overlap. A scoped public screen found no raw root outside its
exact menu, no missed directly legal point win at known own VP ≥9, and no
obvious inverse same-turn bank-trade cycle. It does not certify the strategic
quality of a trade or exhaustively audit other card continuations. The zero
BUY mismatch count does not establish that the common BUY choice was good.

The negative H result argues against scaling raw-planner behavioral cloning
as if every root were an approved action. Next, test a stronger public-only
teacher through paired continuations over several public-consistent sampled
worlds, with hard observation/menu parity, private-information invariance and
abstention on unstable trade/pass/BUY/card rankings. Opening placement is a
separate prospective search track. PPO remains deferred.

The [complete local archive](../../../runs/swarm-2026-11-04/agent-research/teacher-strength-stage1-001/README.md)
is copied and file-by-file verified in both project repositories: 19 files,
109,395,818 bytes, `artifact-manifest.json` SHA-256
`78e96ed4b22afc08f2761459ba15431a2e15ddabef348f253c6b3054544e341a`.
It includes the source-commit bundle, protocol/manifests, parent checkpoint,
full journals and public observations, and idempotent gate and decision-audit
scripts. Source commit: `36053752199b3f1d06e4045cfded7efc91a99b81`;
protocol SHA-256 `9f7bc4bf542ed0c4aa727e9894f1a6639d35a0b77cc212318466f5e66c537262`;
H/GNT summary SHA-256 values
`142ce92cfd602dc383de4b92bdec02740bb1293d6d17a08746a8db4b089240bc`
and `2f8f556c026606ef7c8a42b5a72232a76db97e933c49b45da02699b712cad2f9`;
joint readout SHA-256 `b7bcd062dba6b2064826a8b009508aecb2332058f4d26ffa95e3433fd747a11d`;
public-prefix audit SHA-256 `36e885142b6605b817a98930bfe48a8b12077b74705f9a23e2f3450764fe8c05`.
