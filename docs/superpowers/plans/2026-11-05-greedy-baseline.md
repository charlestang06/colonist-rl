# Plain greedy baseline implementation

User explicitly requested a plain greedy agent that never trades. Full binding
rules in ../specs/2026-11-05-greedy-no-trade-spec.md and task brief in SDD workspace.
Legacy3.2.1 source/runtime isolated. No algorithm, contract, evaluator or data edits.

### Task1: Plain greedy candidate module

Implement exact frozen priorities and legal no-trade policy in greedy_no_trade.py,
focused tests and docs, then independent review. Baseline benchmark192fixedgames
is a separate root-owned bounded run after acceptance. No training/modelselection.

Ruling: Baseline clone uses accepted db146ed7, a legacy45 descendant with only
byte-identical persistence optimization. This does not change engine/actor behavior;
cost if wrong: evaluation source identity is separately recorded and reviewed.
