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
loads remain incompatible by default. The upgrade does not implicitly convert
artifacts or transfer optimizer/RNG state. Existing legacy files retain their original
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

An explicitly requested policy-weight transfer is available through the Python
API. Inspection reads the source's stored labels and SHA-256 from one immutable
byte snapshot with Torch's restricted `weights_only=True` loader. It constructs
no policy, restores no RNG, and does not certify source compatibility. For
example, with an existing legacy source and a new destination:

```python
from colonist_rl.checkpoint_migration import inspect_checkpoint, migrate_legacy_weights
from colonist_rl.training import load_checkpoint

identity = inspect_checkpoint("legacy-policy.pt")
provenance = migrate_legacy_weights("legacy-policy.pt", "new-engine-policy.pt", seed=17)
policy, state = load_checkpoint("new-engine-policy.pt")
assert provenance["source_sha256"] == identity["source_sha256"]
assert state["steps"] == 0 and state["optimizer_state"] is None
```

The bridge accepts only a checkpoint labelled `3.2.1`, schema 1, the exact current
ordered action kinds, a supported exact policy architecture, and matching tensor
keys, shapes, dtypes, and layouts. Both the base graph policy and the two explicit
consequence feature sets use the existing architecture factory. The exact target
installation must pass its normal distribution and Git-provenance guard. Ordinary
checkpoint and dataset loaders retain their strict cross-engine rejection.

The source remains byte-identical. The output is a new ordinary checkpoint;
`metadata["migration"]` records `kind="weights_only_cross_engine"`,
`source_sha256`, `source_backend_version`, `target_backend_version` (the full Git
token), `source_steps`, and `seed`. Only policy tensors are transferred. Optimizer
state is `None`, steps restart at zero, and Python, NumPy, and CPU Torch RNG start
freshly from the supplied integer seed, independent of policy initialization.
NumPy uses `seed % 2**32`; Torch's supported integer-seed range applies. Caller RNG
is preserved, including on failure, and CUDA RNG is untouched. Legacy optimizer,
RNG, metadata, metrics, and evaluation seed labels are not inherited.

The destination must be absent. Same paths, existing files, hard-link aliases,
symlinks, and dangling symlinks are rejected. The ordinary checkpoint is first
saved in a private temporary directory alongside the destination and published
with an exclusive hard-link creation. A concurrent destination creator wins;
the bridge fails without replacing that entry and cleans its staging directory.
This requires a filesystem supporting hard links and does not provide a resume
guarantee or establish old/new performance or seed-result comparability. No
corpus conversion is provided: legacy datasets and historical results retain
their original identities and remain readable through their frozen legacy
source/runtime.
