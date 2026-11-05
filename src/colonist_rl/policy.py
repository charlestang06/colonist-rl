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
NODE_DIM = 40
CONTEXT_DIM = 48
ACTION_DIM = len(KINDS) + 20


@dataclass
class PolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    memory: torch.Tensor


def graph_features(view: Observation, device):
    board = view.board
    nt, nv, ne = len(board.tiles), len(board.vertices), len(board.edges)
    vertex_offset, edge_offset, player_offset = nt, nt + nv, nt + nv + ne
    player_nodes = {p.player_id: player_offset + i for i, p in enumerate(view.players)}
    player_seats = {p.player_id: i for i, p in enumerate(view.players)}
    own_seat = player_seats[view.player_id]
    features = torch.zeros(player_offset + len(view.players), NODE_DIM, device=device)
    links = []

    def link(a, b):
        links.extend(((a, b), (b, a)))

    for tile in board.tiles:
        f = features[tile.id]
        f[0] = 1
        f[5 + RESOURCES.index(tile.resource) if tile.resource in RESOURCES else 10] = 1
        f[11] = tile.number / 12
        f[12] = (6 - abs(7 - tile.number)) / 36 if tile.number else 0
        f[13] = float(tile.id == view.robber_tile)
        for vertex in board.tile_vertices[tile.id]:
            link(tile.id, vertex_offset + vertex)
    features[vertex_offset:edge_offset, 1] = 1
    features[edge_offset:player_offset, 2] = 1
    for index, (a, b) in enumerate(board.edges):
        link(edge_offset + index, vertex_offset + a)
        link(edge_offset + index, vertex_offset + b)
    for building in view.buildings:
        node = vertex_offset + building.vertex
        features[node, 14] = building.level / 2
        features[node, 15] = float(building.owner == view.player_id)
        link(node, player_nodes[building.owner])
    for road in view.roads:
        node = edge_offset + road.edge
        features[node, 14] = 1
        features[node, 15] = float(road.owner == view.player_id)
        link(node, player_nodes[road.owner])
    for port in board.ports:
        for vertex in port.vertices:
            features[vertex_offset + vertex, 16] = port.ratio / 4
            if port.resource in RESOURCES:
                features[vertex_offset + vertex, 5 + RESOURCES.index(port.resource)] = 1
    beliefs = {b.player_id: b for b in view.beliefs}
    for player in view.players:
        f = features[player_nodes[player.player_id]]
        f[3] = 1
        f[14] = player.victory_points / view.rules.victory_points
        f[15] = float(player.player_id == view.player_id)
        f[16:25] = torch.tensor(
            [
                player.resource_count / 20,
                player.development_count / 10,
                player.roads_left / 15,
                player.settlements_left / 5,
                player.cities_left / 4,
                player.longest_road / 15,
                player.played_knights / 3,
                float(player.has_longest_road),
                float(player.has_largest_army),
            ],
            device=device,
        )
        if player.player_id == view.player_id:
            f[25:30] = torch.tensor(view.hand, device=device) / 10
        elif player.player_id in beliefs:
            belief = beliefs[player.player_id]
            f[25:30] = torch.tensor(belief.expected, device=device) / 10
            f[30:35] = torch.tensor(belief.variance, device=device) / 100
            f[35] = float(not belief.exact)
        else:
            f[35] = 1
        # Relative turn position preserves seat order without a fixed player vocabulary.
        f[36] = ((player_seats[player.player_id] - own_seat) % len(view.players)) / len(
            view.players
        )
        f[37] = float(player.player_id == view.context.turn_owner)
    indices = torch.tensor(links, dtype=torch.long, device=device).T
    return features, indices, (vertex_offset, edge_offset, player_nodes)


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

    def forward(self, view: Observation, memory=None):
        features, links, offsets = graph_features(view, self.device)
        nodes = torch.tanh(self.node_in(features))
        source, target = links
        degree = torch.bincount(target, minlength=len(nodes)).clamp(min=1).unsqueeze(1)
        for own, neighbors in zip(self.self_layers, self.neighbor_layers):
            messages = torch.zeros_like(nodes).index_add(0, target, nodes[source]) / degree
            nodes = torch.tanh(own(nodes) + neighbors(messages))
        vertex_offset, edge_offset, player_nodes = offsets
        state = torch.cat(
            (
                nodes.mean(0),
                nodes[player_nodes[view.player_id]],
                context_features(view, self.device),
            )
        )
        previous = torch.zeros(self.hidden_size, device=self.device) if memory is None else memory
        hidden = self.gru(torch.tanh(self.state_in(state)), previous)
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
