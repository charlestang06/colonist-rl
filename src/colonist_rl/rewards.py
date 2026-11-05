"""Win objective with potential shaping, including off-turn point losses."""


def potential_rewards(previous, current, target, *, winner=None, terminated=False, scale=0.2):
    if target <= 0 or len(previous) != len(current):
        raise ValueError("invalid victory threshold or player counts")
    if winner is not None and (not terminated or winner not in range(len(current))):
        raise ValueError("winner requires a terminal outcome and valid player")
    return tuple(
        float(terminated and winner == i)
        + scale * ((0.0 if terminated else current[i] / target) - previous[i] / target)
        for i in range(len(current))
    )
