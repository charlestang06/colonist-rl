# Catan RL implementation plan

Approved scope: local four-player recurrent PPO; variable boards/player sets;
player-visible event ledger and joint resource beliefs; incoming trade response;
optional independent trade generator/critic interfaces; VP potential shaping;
offline board/log ingestion and a guarded browser transport boundary. Six-player
representation is required now; complete six-player simulation is a later milestone.

1. State contracts, variable topology, rule configuration, resource beliefs.
2. Validated and bounded trade plug-ins, incoming accept/reject and fallbacks.
3. Pinned Catanatron adapter, visible event projection, proposing opponents.
4. Graph/recurrent policy, sequence PPO, checkpoint/resume, evaluation CLI.
5. Offline observation ingestion, coordinate calibration and UI transport.
6. Independent review, integration checks, delivery to requested repository path.

Acceptance: hidden-state noninterference; real four-player games; nonnegative
resource conservation; stale/duplicate trade rejection; trajectory-level shaping;
variable board inference; PPO parameter update and checkpoint roundtrip; held-out
seat-rotated evaluation with confidence intervals. Long training strength is
measured rather than assumed. Live Colonist access is not authorized by this
implementation request; the transport remains disabled until explicitly enabled
for an authorized environment.
