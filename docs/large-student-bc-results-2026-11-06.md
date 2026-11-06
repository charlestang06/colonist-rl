# Large student-state behavioral cloning: stopped before games

The frozen base actor fit used 6,215 parent-reached TRAIN observations and 3,809
nonforced raw public-planner roots. Its binder checked exact menu, action, row and
recurrent-prefix identities. A nine-class gradient preflight passed, followed by
832 full-actor Adam updates over two epochs with current-weight prefix replay.
The candidate-successor treatment remained on hold after code review; it was
neither loaded nor fitted. No pairwise judgments, prior label-validation data,
private outcomes or EVAL decisions entered the loss.

The fresh eight-board label-validation panel had 1,843 nonforced roots and
prefit headroom: 236 cases where the parent chose a nontrade and the raw planner
chose a bank trade. **The fitted actor failed both frozen behavior gates:**

| Fresh label-validation metric | Parent | Fitted | Required change |
| --- | ---: | ---: | ---: |
| Mean raw-root cross-entropy | 1.06858 | 1.09915 | decrease at least 0.03 |
| Nine-class macro exact-root agreement | 0.67395 | 0.67740 | increase at least 0.02 |

It changed 233 roots, including **100 bank trade → END** and seven END → bank
trade. On the 345 raw-bank targets, exact-root agreement fell from 65/345
(18.84%) to 19/345 (5.51%); the number choosing *any* bank action fell from
109/345 (31.6%) to 31/345. Raw-END exact agreement rose from 141/156 to
156/156. The same bank decline appeared on TRAIN (21.20% → 14.89%), and the
raw-bank-versus-END logit margin fell on both splits. The fixed per-row END
loss weight was 2.223 times the bank weight; the observations support an
objective/representation conflict but do not isolate its cause. Raw planner
matching is not a strategic-quality label.

The held-out gate stopped the study. **No fitted smoke, opponent panel or
complete-game evaluation ran, and no checkpoint was promoted.** A separate
public check of three changed high-VP settlement cases certified three legal
winning settlement roots and found no fitted missed win in that limited scope;
it did not change the frozen gate. The unchanged raw teacher now faces a
separate H/GNT strength ladder. If it qualifies, a prospectively frozen
family/within-family policy study with reviewed public successor features can
address trade-versus-END mass on fresh roles. If it fails, develop a stronger
public-information teacher before more cloning. PPO remains deferred.

The [local evidence archive](../runs/swarm-2026-11-04/agent-research/large-student-bc-001/README.md)
contains complete VAL public histories and private files, raw overlays, the
checkpoint, frozen source bundle, fit and score receipts, and the read-only
postmortem. Its 88-file `archive-manifest.json` SHA-256 is
`95ab3c7adbd0d07eccb4c2c4e84b581d8477b0791e31fca094b04b23d24f0c9f`.
The frozen protocol SHA-256 is
`0a7bc1f09cb0c75868a92fdaf95cfedf3d9060cd5e14eb791ef61cbfa5ec74ae`;
the fit and held-out score SHA-256 values are
`0c49de7e458c7c8fc2636b7eaf928b47588270463b1a96416907e9326a499346`
and `3075bd86c4d8a9b5e9c6dcbbee317acf5faf1512da3167b85101b9abbefe5bfc`.
