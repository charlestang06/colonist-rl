"""Opt-in road continuation after a single unchanged parent policy choice."""

from torch import nn

from .road_continuation import choose_road_continuation


class RoadContinuationPolicy(nn.Module):
    """Preserve the parent and its memory while applying the pure road helper."""

    def __init__(self, parent: nn.Module):
        super().__init__()
        self.parent = parent
        self.last_decision = None

    @property
    def device(self):
        return self.parent.device

    def forward(self, view, memory=None):
        return self.parent(view, memory)

    def act(self, view, memory=None, *, deterministic=False):
        selected, returned_memory = self.parent.act(
            view, memory, deterministic=deterministic
        )
        decision = choose_road_continuation(view, selected)
        self.last_decision = decision
        return decision.action, returned_memory
