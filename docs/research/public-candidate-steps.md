# Immediate candidate public transitions

`evaluation.evaluate_blocks(..., on_candidate_step=record)` calls `record(step)`
right after each executed candidate action, before the next actor can act or
propose a trade. `step` is a frozen `CandidatePublicStep` with exactly:

- `before: Observation`, the public view used to choose the action;
- `action: Action`, the executed candidate action;
- `after: Observation`, explicitly observed from that same candidate seat;
- `terminated: bool` and `truncated: bool`, the immediate step flags.

The callback also receives terminal candidate actions. The after view belongs to
the candidate even if its context names a different current actor. It contains
no `StepResult`, environment, native objects, diagnostic `info`, rewards, or
winner label. Persist true game outcomes separately through `on_game_end` and
join them outside feature encoding; unresolved outcomes remain unresolved.

```python
from colonist_rl.evaluation import CandidatePublicStep, evaluate_blocks


def record(step: CandidatePublicStep) -> None:
    persist_public_pair(step.before, step.action, step.after,
                        step.terminated, step.truncated)


evaluate_blocks(policy, board_seeds=audited_board_seeds,
                on_candidate_step=record, on_game_end=persist_game_outcome)
```

The persistence functions, policy, and audited seed list are caller supplied.
The callback is synchronous: a persistence failure propagates after evaluation
restores external Python, NumPy, and Torch RNG states and policy module modes.
With the default `None`, evaluation takes no additional observations. Training
entry points and their ordinary positional arguments are unchanged.
