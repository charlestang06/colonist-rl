"""Trusted evaluation-only access to pinned, privileged native opponents.

This is deliberately not an Observation policy or teacher. Only the match driver
may pass its simulator here. Candidate inputs and serialized observations remain
public. The native 82aae93 chance model is delegated unchanged, including its
Action.value / ActionRecord.result mismatch.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import catanatron

from .contracts import RESOURCES
from .engine_identity import ENGINE_REVISION as ENGINE_REVISION  # noqa: PLC0414
from .engine_identity import verify_engine_installation
from .simulator import _RNG_LOCK

NATIVE_IDS = ("native_vf_82aae93", "native_ab2_82aae93")
_SOURCE_SHA256 = {
    "players/value.py": "e25e6e15ff38141568119b912474761fee47a289ad3735a25913b8635f58a56b",
    "players/minimax.py": "97a6bf10e5fac7f6655ad5b1614f268926ed86976ee59815d93f24a1b0e048dd",
    "players/tree_search_utils.py": "fa3329dde569dc0448b5dd2ddaea9ed5f36b3d1ff944b6c95a4caf67a82f22c4",
    "features.py": "cc593434fb30f04d2c4f82c627eaf4f7ef13b220086fd47a0e50a7357e7dbee5",
    "game.py": "f94980006ac8597d9fe65bd20d325bf4d80c30ea84d8ff4190c48bd1928b56b7",
    "state.py": "998b2cecc1e0c8a8143216740863b6662e300dafdad2f0910af3b4afe9290b29",
    "state_functions.py": "ed9aa7a4fa858bb4645a163f31660288cc64fb6430f7577886aa1ce8aea3ca6b",
    "apply_action.py": "96fee2ba287ac7987843d9a3860f862056e9f0beeb3be0c634d095e5c2490712",
    "models/actions.py": "f5fdea72ca90e15b7960e469d33b8a41046f6de61dad21bdc8a455a6fbe140b1",
    "models/player.py": "bfcb6223582a88e664f4bf07f355b52b5bb81ca34919f0253b47f4532bab47ac",
    "models/enums.py": "cf09b59a17750ad9c7c799ea149fa818ea2fb9e2e4b86b2da161a5293c95c728",
    "models/decks.py": "cdb92c55ef0b52448f3970915f2bcf132749ece942aaca3701843ebc66ba0e1a",
}


def _native_player_class(identity):
    if identity == NATIVE_IDS[0]:
        from catanatron.players.value import ValueFunctionPlayer

        return ValueFunctionPlayer
    if identity == NATIVE_IDS[1]:
        from catanatron.players.minimax import AlphaBetaPlayer

        return AlphaBetaPlayer
    raise ValueError("unknown native opponent identity")


def verify_native_installation():
    """Read metadata/source only; never construct a game or query a policy."""
    identity = verify_engine_installation()
    package = Path(catanatron.__file__).resolve().parent
    actual = {}
    for relative, expected in _SOURCE_SHA256.items():
        actual[relative] = hashlib.sha256((package / relative).read_bytes()).hexdigest()
        if actual[relative] != expected:
            raise RuntimeError(f"native source identity mismatch: {relative}")
    for name in NATIVE_IDS:
        player = _native_player_class(name)
        if not callable(getattr(player, "decide", None)):
            raise RuntimeError(f"native class missing decide: {name}")  # noqa: TRY004
    return {**identity, "source_sha256": actual, "license": "GPL-3.0-or-later"}


class NativeOpponentRunner:
    """One privileged opponent, bound after reset to a single simulator seat.

    ``select_action()`` returns an actual member of the current wrapper legal
    tuple. It never executes the real game. ``metadata`` / ``metrics`` contain
    primitives only; no native player, state, action record, or hidden labels.
    """

    def __init__(self, env, seat, identity, *, planning_seed):
        if identity not in NATIVE_IDS or type(seat) is not int or seat not in range(4):
            raise ValueError("known native identity and valid opponent seat required")
        if type(planning_seed) is not int:
            raise ValueError("planning_seed must be an explicit independent integer")
        self._env, self.seat, self.identity = env, seat, identity
        self._source = verify_native_installation()
        self._planning_seed = planning_seed
        self._rng_state = random.Random(planning_seed).getstate()
        self._discard = None
        self._counts = Counter()
        self._phase_seconds = Counter()
        self._phase_counts = Counter()
        self._configuration = (
            {"depth": 2, "prunning": False, "value_fn_builder_name": None, "epsilon": None}
            if identity == NATIVE_IDS[1]
            else {"value_fn_builder_name": None, "params": None, "epsilon": None}
        )
        with self._planning_randomness():
            self._player = _native_player_class(identity)(
                env.game.state.colors[seat], **self._configuration
            )

    @contextmanager
    def _planning_randomness(self):
        # Never enter env._randomness or read its private future RNG stream.
        with _RNG_LOCK:
            external = random.getstate()
            random.setstate(self._rng_state)
            try:
                yield
            finally:
                self._rng_state = random.getstate()
                random.setstate(external)

    def select_action(self):
        view = self._env.observe()
        if view.player_id != self.seat or view.context.actor != self.seat:
            raise ValueError("native runner invoked for another seat")
        if not view.legal_actions:
            raise ValueError("native opponent has no wrapper legal actions")
        phase = view.context.kind
        start = time.perf_counter()
        self._phase_counts[phase] += 1
        try:
            if phase == "TRADE_RESPONSE":
                self._counts["trade_reject_shim_calls"] += 1
                return next(a for a in view.legal_actions if a.kind == "REJECT_TRADE")
            if phase == "DISCARD":
                self._counts["atomic_discard_shim_calls"] += 1
                return self._discard_action(view)
            self._discard = None
            # Deep copy also isolates nested action records and mutable action
            # lists, which upstream Game.copy intentionally shares in part.
            with self._planning_randomness():
                copied = deepcopy(self._env.game)
                root_actions = list(copied.playable_actions)
                copied.playable_actions = list(root_actions)  # AB ignores passed list
                legal = tuple(deepcopy(self._env.game.playable_actions))
                self._counts["native_decisions"] += 1
                self._counts["native_nonforced_decisions"] += int(len(root_actions) > 1)
                proposal = self._player.decide(copied, root_actions)
            if proposal not in legal:
                raise ValueError("native return is not a current native legal action")
            converted = self._env._convert(proposal)
            for action in view.legal_actions:
                if action == converted:
                    return action
            raise ValueError("native return has no current wrapper legal member")
        finally:
            self._phase_seconds[phase] += time.perf_counter() - start

    def _discard_action(self, view):
        remaining, version = view.context.discard_remaining, view.context.version
        if remaining <= 0:
            raise ValueError("unsupported discard without a positive remaining count")
        if self._discard is None or version >= self._discard[0] + self._discard[1]:
            hand = [r for r, n in zip(RESOURCES, view.hand) for _ in range(n)]
            with self._planning_randomness():
                resources = tuple(random.sample(hand, k=remaining))
            self._discard = (version, remaining, resources)
        start, count, resources = self._discard
        index = count - remaining
        if not (0 <= index < count) or version != start + index:
            raise ValueError("unsupported noncontiguous native discard sequence")
        for action in view.legal_actions:
            if action.kind == "DISCARD_RESOURCE" and action.resource == resources[index]:
                return action
        raise ValueError("native discard shim has no current wrapper legal member")

    def metadata(self):
        return {
            "identity": self.identity,
            "adapter_identity": self.identity + "+atomic_discard_reject_trade_v1",
            "engine": deepcopy(self._source),
            "seat": self.seat,
            "configuration": deepcopy(self._configuration),
            "effective_weights": "upstream_base_DEFAULT_WEIGHTS",
            "discard_semantic_difference": "native_search_bulk_discard; wrapper_executes_one_card_per_step",
            "planning_seed": self._planning_seed,
            "information_regime": "public_only_candidate_vs_privileged_native_opponents",
            "discard_adapter": "native_random_bulk_to_per_card",
            "trade_adapter": "reject_synthetic_trade_response",
            "proposal_support": "none",
            "root_action_order": "native_original",
            "chance_model": "native82_unmodified_Action.value_ActionRecord.result_mismatch",
            "internal_deadline_seconds": 20 if self.identity == NATIVE_IDS[1] else None,
            "deadline_kind": "soft_recursion_entry_only; worker_must_enforce_hard_cap",
            "full_rule_equivalence": False,
        }

    def metrics(self):
        return {
            **dict(self._counts),
            "native_decisions": self._counts["native_decisions"],
            "native_nonforced_decisions": self._counts["native_nonforced_decisions"],
            "phase_counts": dict(self._phase_counts),
            "phase_planning_seconds": dict(self._phase_seconds),
            "total_planning_seconds": sum(self._phase_seconds.values()),
            "search_nodes": None,
            "search_expansions": None,
            "internal_timeout_count": None,
        }
