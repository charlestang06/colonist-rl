"""Variable-size graph actor, recurrent memory and context-specific heads.

Only immutable player observations enter this module. Legal candidates carry
their target node embeddings, so adding vertices or players adds no output units.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .contracts import RESOURCES, Observation
from .graph_inputs import NODE_DIM, graph_features

KINDS = (
    "ROLL",
    "MOVE_ROBBER",
    "DISCARD",
    "DISCARD_RESOURCE",
    "BUILD_ROAD",
    "BUILD_SETTLEMENT",
    "BUILD_CITY",
    "BUY_DEVELOPMENT_CARD",
    "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY",
    "PLAY_MONOPOLY",
    "PLAY_ROAD_BUILDING",
    "MARITIME_TRADE",
    "END_TURN",
    "ACCEPT_TRADE",
    "REJECT_TRADE",
)
PHASES = (
    "TURN",
    "SETUP_SETTLEMENT",
    "SETUP_ROAD",
    "DISCARD",
    "ROBBER",
    "TRADE_RESPONSE",
    "TRADE_CONFIRM",
    "BUILD_INITIAL_SETTLEMENT",
    "BUILD_INITIAL_ROAD",
    "PLAY_TURN",
    "MOVE_ROBBER",
)
CONTEXT_DIM = 48
ACTION_DIM = len(KINDS) + 20


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    memory: torch.Tensor


def context_features(view, device):
    values = torch.zeros(CONTEXT_DIM, device=device)
    context = view.context
    if context.kind in PHASES:
        values[PHASES.index(context.kind)] = 1
    values[12:17] = torch.tensor(
        [
            float(context.has_rolled),
            float(context.has_played_development_card),
            context.discard_remaining / 10,
            context.free_roads / 2,
            view.own_victory_points / view.rules.victory_points,
        ],
        device=device,
    )
    values[17:22] = torch.tensor(view.development, device=device) / 5
    if view.bank is not None:
        values[22:27] = torch.tensor(view.bank, device=device) / 24
    event_names = (
        "ROLL",
        "BUILD_ROAD",
        "BUILD_SETTLEMENT",
        "BUILD_CITY",
        "BUY_DEVELOPMENT_CARD",
        "MOVE_ROBBER",
        "DISCARD",
        "TRADE",
    )
    for event in view.events[-32:]:
        if event.kind in event_names:
            values[27 + event_names.index(event.kind)] += 1 / 16
    if context.offer is not None and view.player_id in (
        context.offer.proposer,
        context.offer.recipient,
    ):
        give, receive = context.offer.for_player(view.player_id)
        values[38:43] = torch.tensor(give, device=device) / 5
        values[43:48] = torch.tensor(receive, device=device) / 5
    return values


class GraphPolicy(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_in = nn.Linear(NODE_DIM, hidden_size)
        self.self_layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(2))
        self.neighbor_layers = nn.ModuleList(nn.Linear(hidden_size, hidden_size) for _ in range(2))
        self.state_in = nn.Linear(hidden_size * 2 + CONTEXT_DIM, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.action_in = nn.Linear(ACTION_DIM, hidden_size)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
                )
                for name in ("turn", "setup", "forced")
            }
        )
        self.trade_response = nn.Linear(hidden_size, 2)
        self.value = nn.Linear(hidden_size, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def _recurrent(self, view: Observation, memory=None):
        features, links, offsets = graph_features(view, self.device)
        nodes = torch.tanh(self.node_in(features))
        source, target = links
        degree = torch.bincount(target, minlength=len(nodes)).clamp(min=1).unsqueeze(1)
        for own, neighbors in zip(self.self_layers, self.neighbor_layers):
            messages = torch.zeros_like(nodes).index_add(0, target, nodes[source]) / degree
            nodes = torch.tanh(own(nodes) + neighbors(messages))
        _, _, player_nodes = offsets
        state = torch.cat(
            (
                nodes.mean(0),
                nodes[player_nodes[view.player_id]],
                context_features(view, self.device),
            )
        )
        previous = torch.zeros(self.hidden_size, device=self.device) if memory is None else memory
        hidden = self.gru(torch.tanh(self.state_in(state)), previous)
        return nodes, offsets, hidden

    def forward(self, view: Observation, memory=None):
        nodes, offsets, hidden = self._recurrent(view, memory)
        return self._output(view, nodes, offsets, hidden)

    def recurrent_step(self, view: Observation, memory=None) -> torch.Tensor:
        """The identical recurrent path, without computing unused actor/value heads."""
        return self._recurrent(view, memory)[2]

    def _recurrent_many(self, views, memories):
        if len(views) != len(memories):
            raise ValueError("views and memories must have equal lengths")
        if not views:
            return ()
        if len(views) == 1:
            return (self._recurrent(views[0], memories[0]),)
        graphs = [graph_features(view, self.device) for view in views]
        features, links, offsets = zip(*graphs)
        sizes = [len(f) for f in features]
        shifted, start = [], 0
        for indices, size in zip(links, sizes):
            shifted.append(indices + start)
            start += size
        nodes = torch.tanh(self.node_in(torch.cat(features)))
        source, target = torch.cat(shifted, dim=1)
        # Ownership edges are observation-dependent; degrees are never cached.
        degree = torch.bincount(target, minlength=len(nodes)).clamp(min=1).unsqueeze(1)
        for own, neighbors in zip(self.self_layers, self.neighbor_layers):
            messages = torch.zeros_like(nodes).index_add(0, target, nodes[source]) / degree
            nodes = torch.tanh(own(nodes) + neighbors(messages))
        separated = nodes.split(sizes)
        states = torch.stack(
            [
                torch.cat(
                    (
                        graph.mean(0),
                        graph[off[2][view.player_id]],
                        context_features(view, self.device),
                    )
                )
                for view, graph, off in zip(views, separated, offsets)
            ]
        )
        previous = torch.stack(
            [nodes.new_zeros(self.hidden_size) if memory is None else memory for memory in memories]
        )
        hidden = self.gru(torch.tanh(self.state_in(states)), previous)
        return tuple(zip(separated, offsets, hidden.unbind(0)))

    def forward_many(
        self, views: tuple[Observation, ...], memories: tuple[torch.Tensor | None, ...]
    ) -> tuple[PolicyOutput, ...]:
        """Disconnected graphs, per-graph pooling and independent ordered GRU rows."""
        encoded = self._recurrent_many(views, memories)
        return tuple(self._output(view, *values) for view, values in zip(views, encoded))

    def recurrent_many(
        self, views: tuple[Observation, ...], memories: tuple[torch.Tensor | None, ...]
    ) -> tuple[torch.Tensor, ...]:
        return tuple(values[2] for values in self._recurrent_many(views, memories))

    def _output(self, view, nodes, offsets, hidden):
        vertex_offset, edge_offset, player_nodes = offsets
        if not view.legal_actions:
            return PolicyOutput(hidden.new_empty((0,)), self.value(hidden).squeeze(), hidden)
        if view.context.kind == "TRADE_RESPONSE":
            binary = self.trade_response(hidden)
            logits = torch.stack(
                [binary[int(a.kind == "ACCEPT_TRADE")] for a in view.legal_actions]
            )
        else:
            candidates = []
            for action in view.legal_actions:
                f = torch.zeros(ACTION_DIM, device=self.device)
                if action.kind not in KINDS:
                    raise ValueError(f"unsupported action kind {action.kind}")
                f[KINDS.index(action.kind)] = 1
                base = len(KINDS)
                f[base : base + 5] = torch.tensor(action.give, device=self.device) / 4
                f[base + 5 : base + 10] = torch.tensor(action.receive, device=self.device) / 4
                if action.resource in RESOURCES:
                    f[base + 10 + RESOURCES.index(action.resource)] = 1
                if action.second_resource in RESOURCES:
                    f[base + 15 + RESOURCES.index(action.second_resource)] = 1
                candidate = self.action_in(f)
                if action.target is not None:
                    offset = (
                        edge_offset
                        if action.kind in ("BUILD_ROAD", "PLAY_ROAD_BUILDING")
                        else 0
                        if action.kind == "MOVE_ROBBER"
                        else vertex_offset
                    )
                    candidate = candidate + nodes[offset + action.target]
                if action.partner in player_nodes:
                    candidate = candidate + nodes[player_nodes[action.partner]]
                candidates.append(candidate)
            phase = view.context.kind
            head = (
                "setup"
                if "INITIAL" in phase or "SETUP" in phase
                else "forced"
                if phase in ("DISCARD", "ROBBER", "MOVE_ROBBER")
                else "turn"
            )
            embeddings = torch.stack(candidates)
            logits = self.heads[head](
                torch.cat((hidden.expand(len(embeddings), -1), embeddings), 1)
            ).squeeze(1)
        return PolicyOutput(logits, self.value(hidden).squeeze(), hidden)

    @torch.no_grad()
    def act(self, view, memory=None, *, deterministic=False):
        if not view.legal_actions:
            raise ValueError("a policy decision requires at least one legal action")
        output = self(view, memory)
        distribution = Categorical(logits=output.logits)
        index = output.logits.argmax() if deterministic else distribution.sample()
        return view.legal_actions[index.item()], output.memory.detach()
