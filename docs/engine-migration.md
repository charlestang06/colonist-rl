# Exact upstream engine migration

The simulation backend is the repository-root Catanatron package at Git commit
`82aae93ab1f7c267218be0566df573ce477ec3d8`, whose distribution version is `3.3.0`.
The dependency uses the full SHA and does not request Catanatron's gym extra;
the project's `gymnasium>=1.0,<2` requirement remains in effect. Hatch permits
this direct reference so ordinary project wheels retain the exact dependency.

`colonist_rl.engine_identity.verify_engine_installation()` checks both the
installed distribution version and the PEP 610 Git `commit_id`. Missing,
malformed, non-Git, or mismatched provenance fails closed, including another
commit reporting the same release version. Its primitive result records the
source URL, full commit, artifact identity, number placement, and seed offset.
A verified local Git install may supply a `file://` source URL; a plain directory
install without VCS provenance does not qualify.

`FourPlayerGame` verifies that installation on construction. The simulator
re-exports the artifact compatibility token
`3.3.0+git.82aae93ab1f7c267218be0566df573ce477ec3d8`, so existing checkpoint,
demonstration, and student-label producers and compatibility checks use the exact
engine identity without changing their interfaces. Legacy checkpoint and dataset
loads remain incompatible by default. This migration does not convert artifacts
or transfer optimizer/RNG state. Existing legacy files retain their original
identity and bytes.

Reset explicitly requests `official_spiral` number placement and keeps backend
seed `user_seed + 1`, including user seed zero. Replays within this new cohort
are deterministic and isolate the process random state. The new board
distribution and engine behavior mean identical numeric seeds do not establish
comparability to legacy games or evaluation metrics.

The adapter uses `Game.playable_actions`, `(coordinate, color)` robber intents,
and `ActionRecord.result` for stochastic execution outcomes. Native development
age flags, pre-roll cards, one-card-per-turn behavior, and both free roads replace
the legacy timing/free-road patches. Shared observations continue to hide stolen
resources from uninvolved players, discarded identities from other players, and
purchased development-card identities. Resource conservation and the existing
one-card discard interface remain enforced.

Directed atomic wrapper trades retain their existing incoming acceptance and
rejection behavior. Native `DECIDE_TRADE` / `DECIDE_ACCEPTEES` phases and native
`OFFER_TRADE`, `ACCEPT_TRADE`, `REJECT_TRADE`, `CONFIRM_TRADE`, and `CANCEL_TRADE`
actions raise an explicit unsupported-boundary error. Native table-wide
negotiation is not exposed to learner policies; the shared action vocabulary is
unchanged.

Validation runs in the isolated side clone's new `.venv`, with exact installed
provenance and import-origin checks, mechanics/privacy/conservation regressions,
a complete game, the full suite, and a separate wheel installation with a tiny
rollout/PPO/checkpoint smoke. Temporary environments and caches are not committed.
The root integration owner controls the original repository's runtime transition
after active legacy runs end; the original runtime and artifacts stay untouched
by this isolated migration.
