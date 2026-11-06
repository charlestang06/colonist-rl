"""Variable-size graph actor, recurrent memory and context-specific heads.

Only immutable player observations enter this module. Legal candidates carry
their target node embeddings, so adding vertices or players adds no output units.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from .consequences import CONSEQUENCE_DIM, CONSEQUENCE_SCHEMA, consequence_features
from .contracts import RESOURCES, Observation
from .graph_inputs import NODE_DIM, graph_features
from .successor_features import SUCCESSOR_DIM, SUCCESSOR_SCHEMA, successor_features_for_menu
from .trading_context import (
    TRADING_CONTEXT_DIM,
    TRADING_CONTEXT_SCHEMA,
    trading_context_features,
)

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


class ConsequenceGraphPolicy(GraphPolicy):
    """Explicit opt-in candidate scorer; recurrence and value remain inherited."""

    architecture_kind = "graph_consequence_v1"

    def __init__(self, hidden_size=64, *, feature_set="hand_recipe"):
        if feature_set not in ("hand_recipe", "hand_recipe_and_connectivity"):
            raise ValueError("unsupported consequence feature set")
        super().__init__(hidden_size)
        self._feature_set = feature_set
        self.consequence = nn.Linear(CONSEQUENCE_DIM, 1, bias=False)
        nn.init.zeros_(self.consequence.weight)

    @property
    def feature_set(self):
        return self._feature_set

    def _output(self, view, nodes, offsets, hidden):
        base = super()._output(view, nodes, offsets, hidden)
        if not view.legal_actions:
            return base
        features = torch.stack(
            [
                consequence_features(view, action, device=self.device)
                for action in view.legal_actions
            ]
        ).to(dtype=self.consequence.weight.dtype)
        if self.feature_set == "hand_recipe":
            features[:, 30:34] = 0
        logits = base.logits + self.consequence(features).squeeze(1)
        return PolicyOutput(logits, base.value, base.memory)


class CandidateSuccessorPolicy(ConsequenceGraphPolicy):
    """Opt-in public candidate scorer with zero-start linear and nonlinear branches."""

    architecture_kind = "graph_candidate_successor_v3"

    def __init__(self, hidden_size=64, *, residual_hidden_size=16):
        if type(residual_hidden_size) is not int or residual_hidden_size != 16:
            raise ValueError("residual_hidden_size must be the exact integer 16")
        super().__init__(hidden_size, feature_set="hand_recipe")
        self.residual_hidden_size = residual_hidden_size
        self.successor_linear = nn.Linear(SUCCESSOR_DIM, 1, bias=False)
        self.successor_hidden = nn.Linear(SUCCESSOR_DIM + hidden_size, residual_hidden_size)
        self.successor_out = nn.Linear(residual_hidden_size, 1, bias=False)
        nn.init.zeros_(self.successor_linear.weight)
        nn.init.zeros_(self.successor_out.weight)

    def _output(self, view, nodes, offsets, hidden):
        base = super()._output(view, nodes, offsets, hidden)
        if not view.legal_actions:
            return base
        features = successor_features_for_menu(view, device=self.device).to(
            dtype=self.successor_linear.weight.dtype
        )
        linear = self.successor_linear(features).squeeze(1)
        interaction = torch.tanh(
            self.successor_hidden(
                torch.cat((features, hidden.expand(len(features), -1)), dim=1)
            )
        )
        residual = linear + self.successor_out(interaction).squeeze(1)
        return PolicyOutput(base.logits + residual, base.value, base.memory)


class _TradingResidualConsequencePolicy(ConsequenceGraphPolicy):
    """Preserve the trained 34D reduction before adding a supported-only branch."""

    def _output(self, view, nodes, offsets, hidden):
        base = super()._output(view, nodes, offsets, hidden)
        if not view.legal_actions:
            return base
        context = trading_context_features(view, device=self.device).to(
            dtype=self.consequence.weight.dtype
        )
        residual = self._trading_residual(view, context)
        supported = torch.tensor(
            [
                a.kind in ("MARITIME_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "END_TURN")
                for a in view.legal_actions
            ],
            device=self.device,
        )
        logits = base.logits + torch.where(supported, residual, 0)
        return PolicyOutput(logits, base.value, base.memory)


class ContextLinearConsequencePolicy(_TradingResidualConsequencePolicy):
    """Same-input linear control: the context term is common to supported choices."""

    architecture_kind = "graph_consequence_context_linear_v1"

    def __init__(self, hidden_size=64):
        super().__init__(hidden_size, feature_set="hand_recipe")
        self.context_linear = nn.Linear(TRADING_CONTEXT_DIM, 1, bias=False)
        nn.init.zeros_(self.context_linear.weight)

    def _trading_residual(self, view, context):
        return self.context_linear(context).expand(len(view.legal_actions))


class NonlinearConsequencePolicy(_TradingResidualConsequencePolicy):
    """Zero-output nonlinear hand/context interaction, with a nonzero hidden map."""

    architecture_kind = "graph_consequence_nonlinear_v1"

    def __init__(self, hidden_size=64, *, residual_hidden_size=16):
        if type(residual_hidden_size) is not int or residual_hidden_size != 16:
            raise ValueError("residual_hidden_size must be the exact integer 16")
        super().__init__(hidden_size, feature_set="hand_recipe")
        self.residual_hidden_size = residual_hidden_size
        self.residual = nn.Sequential(
            nn.Linear(CONSEQUENCE_DIM + TRADING_CONTEXT_DIM, residual_hidden_size),
            nn.Tanh(),
            nn.Linear(residual_hidden_size, 1, bias=False),
        )
        nn.init.zeros_(self.residual[2].weight)

    def _trading_residual(self, view, context):
        features = torch.stack(
            [
                consequence_features(view, action, device=self.device)
                for action in view.legal_actions
            ]
        ).to(dtype=self.consequence.weight.dtype)
        features[:, 30:34] = 0
        return self.residual(
            torch.cat((features, context.expand(len(features), -1)), dim=1)
        ).squeeze(1)


def candidate_successor_from_consequence(
    policy: ConsequenceGraphPolicy, *, seed=0
) -> CandidateSuccessorPolicy:
    """Copy an exact hand-recipe parent without advancing Torch RNG."""
    if type(policy) is not ConsequenceGraphPolicy:
        raise ValueError("candidate successor upgrade requires an exact consequence parent")
    if policy.feature_set != "hand_recipe":
        raise ValueError("candidate successor upgrade requires hand_recipe parent")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            variant = CandidateSuccessorPolicy(policy.hidden_size)
    parameter = next(policy.parameters())
    variant.to(device=parameter.device, dtype=parameter.dtype)
    weights = {name: value.detach().clone() for name, value in variant.state_dict().items()}
    weights.update({name: value.detach().clone() for name, value in policy.state_dict().items()})
    variant.load_state_dict(weights, strict=True, assign=True)
    parameters = dict(variant.named_parameters())
    for name, old in policy.named_parameters():
        parameters[name].requires_grad_(old.requires_grad)
    old_modules = dict(policy.named_modules())
    for name, module in variant.named_modules():
        module.training = old_modules[name].training if name in old_modules else policy.training
    return variant


def trading_residual_from_consequence(
    policy: ConsequenceGraphPolicy, *, kind, seed=0
) -> ContextLinearConsequencePolicy | NonlinearConsequencePolicy:
    """Independently copy an exact hand-recipe parent without changing caller RNG."""
    if type(policy) is not ConsequenceGraphPolicy or policy.feature_set != "hand_recipe":
        raise ValueError("trading upgrade requires an exact hand_recipe consequence parent")
    if kind not in ("context_linear", "nonlinear"):
        raise ValueError("unsupported trading residual kind")
    cls = ContextLinearConsequencePolicy if kind == "context_linear" else NonlinearConsequencePolicy
    with torch.random.fork_rng(devices=[]):
        # Construction is on CPU; seed only its generator, leaving accelerator RNG alone.
        torch.random.default_generator.manual_seed(seed)
        variant = cls(policy.hidden_size)
    parameter = next(policy.parameters())
    variant.to(device=parameter.device, dtype=parameter.dtype)
    variant.load_state_dict({**variant.state_dict(), **policy.state_dict()}, strict=True)
    parameters = dict(variant.named_parameters())
    for name, old in policy.named_parameters():
        parameters[name].requires_grad_(old.requires_grad)
    old_modules = dict(policy.named_modules())
    for name, module in variant.named_modules():
        # Preserve mixed modes; newly added modules follow the overall parent mode.
        module.training = old_modules[name].training if name in old_modules else policy.training
    return variant


def consequence_from_legacy(
    policy: GraphPolicy, *, feature_set="hand_recipe"
) -> ConsequenceGraphPolicy:
    """Copy a legacy policy exactly without advancing the caller's Torch RNG."""
    if type(policy) is not GraphPolicy:
        raise ValueError("consequence upgrade requires an unmodified legacy GraphPolicy")
    with torch.random.fork_rng(devices=[]):
        variant = ConsequenceGraphPolicy(policy.hidden_size, feature_set=feature_set)
    parameter = next(policy.parameters())
    variant.to(device=parameter.device, dtype=parameter.dtype)
    variant.load_state_dict(
        {**policy.state_dict(), "consequence.weight": variant.consequence.weight.detach()},
        strict=True,
    )
    variant.train(policy.training)
    return variant


def policy_architecture(policy: GraphPolicy) -> dict:
    """Explicit checkpoint identities; legacy metadata stays unchanged."""
    if type(policy) is GraphPolicy:
        return {"hidden_size": policy.hidden_size}
    if type(policy) is CandidateSuccessorPolicy:
        return {
            "kind": policy.architecture_kind,
            "hidden_size": policy.hidden_size,
            "feature_schema": CONSEQUENCE_SCHEMA,
            "feature_set": policy.feature_set,
            "successor_schema": SUCCESSOR_SCHEMA,
            "successor_dim": SUCCESSOR_DIM,
            "residual_hidden_size": policy.residual_hidden_size,
        }
    if type(policy) in (ContextLinearConsequencePolicy, NonlinearConsequencePolicy):
        architecture = {
            "kind": policy.architecture_kind,
            "hidden_size": policy.hidden_size,
            "feature_schema": CONSEQUENCE_SCHEMA,
            "feature_set": policy.feature_set,
            "context_schema": TRADING_CONTEXT_SCHEMA,
        }
        if type(policy) is NonlinearConsequencePolicy:
            architecture["residual_hidden_size"] = policy.residual_hidden_size
        return architecture
    if type(policy) is ConsequenceGraphPolicy:
        return {
            "kind": policy.architecture_kind,
            "hidden_size": policy.hidden_size,
            "feature_schema": CONSEQUENCE_SCHEMA,
            "feature_set": policy.feature_set,
        }
    raise ValueError("unsupported policy architecture")


def policy_from_architecture(architecture: dict) -> GraphPolicy:
    """Reject ambiguous metadata rather than silently dropping policy branches."""
    if (
        not isinstance(architecture, dict)
        or type(architecture.get("hidden_size")) is not int
        or architecture["hidden_size"] < 1
    ):
        raise ValueError("invalid policy architecture")
    if set(architecture) == {"hidden_size"}:
        return GraphPolicy(architecture["hidden_size"])
    if architecture.get("kind") == CandidateSuccessorPolicy.architecture_kind:
        if (
            set(architecture) != {
                "kind", "hidden_size", "feature_schema", "feature_set",
                "successor_schema", "successor_dim", "residual_hidden_size",
            }
            or architecture["feature_schema"] != CONSEQUENCE_SCHEMA
            or architecture["feature_set"] != "hand_recipe"
            or architecture["successor_schema"] != SUCCESSOR_SCHEMA
            or type(architecture["successor_dim"]) is not int
            or architecture["successor_dim"] != SUCCESSOR_DIM
        ):
            raise ValueError("unsupported candidate successor architecture or feature schema")
        return CandidateSuccessorPolicy(
            architecture["hidden_size"],
            residual_hidden_size=architecture["residual_hidden_size"],
        )
    if architecture.get("kind") in (
        ContextLinearConsequencePolicy.architecture_kind,
        NonlinearConsequencePolicy.architecture_kind,
    ):
        nonlinear = architecture["kind"] == NonlinearConsequencePolicy.architecture_kind
        keys = {"kind", "hidden_size", "feature_schema", "feature_set", "context_schema"}
        if nonlinear:
            keys.add("residual_hidden_size")
        if (
            set(architecture) != keys
            or architecture["feature_schema"] != CONSEQUENCE_SCHEMA
            or architecture["feature_set"] != "hand_recipe"
            or architecture["context_schema"] != TRADING_CONTEXT_SCHEMA
        ):
            raise ValueError("unsupported trading architecture or feature/context schema")
        if nonlinear:
            return NonlinearConsequencePolicy(
                architecture["hidden_size"],
                residual_hidden_size=architecture["residual_hidden_size"],
            )
        return ContextLinearConsequencePolicy(architecture["hidden_size"])
    if (
        set(architecture) != {"kind", "hidden_size", "feature_schema", "feature_set"}
        or architecture["kind"] != ConsequenceGraphPolicy.architecture_kind
        or architecture["feature_schema"] != CONSEQUENCE_SCHEMA
    ):
        raise ValueError("unsupported policy architecture or feature schema")
    return ConsequenceGraphPolicy(
        architecture["hidden_size"], feature_set=architecture["feature_set"]
    )
