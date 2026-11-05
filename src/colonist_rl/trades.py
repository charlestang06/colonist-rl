"""Optional trade models over immutable player-visible records.

Models suggest or assess; they never execute a transfer. The engine must call
``validate_trade_decision`` against its latest observation before committing.
The trade critic is independent of a policy's PPO value-function critic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from queue import Empty, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Literal, Protocol, TypeVar

from .contracts import DecisionContext, Observation, TradeOffer, resource_vector

TradeKind = Literal["ACCEPT", "REJECT", "PROPOSE", "NO_OFFER"]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class TradeAssessment:
    """Benefit uses the actor's perspective and includes rejection probability.

    ``offer=None`` denotes no trade, whose benefit is always zero. Optional
    ``benefit_if_accepted`` must not be substituted for expected benefit by the
    controller; a very attractive offer can still be unlikely to be accepted.
    """

    offer: TradeOffer | None
    benefit_relative_to_no_trade: float
    uncertainty: float
    acceptance_probability: float | None = None
    benefit_if_accepted: float | None = None

    def __post_init__(self) -> None:
        if self.offer is not None and not isinstance(self.offer, TradeOffer):
            raise ValueError("assessment offer must be a TradeOffer or None")
        for name in ("benefit_relative_to_no_trade", "uncertainty"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))
        if self.uncertainty < 0:
            raise ValueError("uncertainty must be nonnegative")
        for name in ("acceptance_probability", "benefit_if_accepted"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_number(value, name))
        if self.acceptance_probability is not None and not 0 <= self.acceptance_probability <= 1:
            raise ValueError("acceptance_probability must be between zero and one")
        if self.offer is None and self.benefit_relative_to_no_trade != 0:
            raise ValueError("the no-trade benefit must be zero")


@dataclass(frozen=True)
class TradeDecision:
    kind: TradeKind
    offer: TradeOffer | None = None
    context_version: int | None = None
    assessments: tuple[TradeAssessment, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("ACCEPT", "REJECT", "PROPOSE", "NO_OFFER"):
            raise ValueError("unknown trade decision kind")
        if self.kind in ("ACCEPT", "PROPOSE") and not isinstance(self.offer, TradeOffer):
            raise ValueError("accepting or proposing requires an offer")
        if self.kind == "NO_OFFER" and self.offer is not None:
            raise ValueError("NO_OFFER cannot contain an offer")
        if self.offer is not None and not isinstance(self.offer, TradeOffer):
            raise ValueError("decision offer must be a TradeOffer")
        if self.context_version is not None and type(self.context_version) is not int:
            raise ValueError("context_version must be an integer")
        object.__setattr__(self, "assessments", tuple(self.assessments))
        if any(not isinstance(item, TradeAssessment) for item in self.assessments):
            raise ValueError("assessments must contain TradeAssessment records")


class TradeGenerator(Protocol):
    def propose(
        self, player_view: Observation, context: DecisionContext, max_candidates: int
    ) -> Iterable[TradeOffer]: ...


class TradeCritic(Protocol):
    def score(
        self, player_view: Observation, context: DecisionContext, offers: Sequence[TradeOffer]
    ) -> list[TradeAssessment]: ...


class TradeController(Protocol):
    def choose(self, player_view: Observation, context: DecisionContext) -> TradeDecision: ...


BinaryResponse = Callable[[Observation, DecisionContext], bool]
_NO_TRADE = TradeAssessment(None, 0.0, 0.0)
_T = TypeVar("_T")


def _live_context(player_view: Observation, context: DecisionContext) -> bool:
    return (
        isinstance(player_view, Observation)
        and isinstance(context, DecisionContext)
        and context == player_view.context
        and type(context.version) is int
        and context.actor == player_view.player_id
        and type(context.actor) is int
        and any(player.player_id == context.actor for player in player_view.players)
    )


def validate_trade_offer(
    offer: TradeOffer, player_view: Observation, context: DecisionContext
) -> bool:
    """Validate public legality and only the actor's exact resource inventory.

    Incoming vectors are normalized with ``offer.for_player(actor)``. Opponent
    affordability is deliberately left to the engine's atomic commit boundary.
    """
    if not _live_context(player_view, context) or not isinstance(offer, TradeOffer):
        return False
    try:
        give = resource_vector(offer.give)
        receive = resource_vector(offer.receive)
        hand = resource_vector(player_view.hand)
        if not any(give) or not any(receive) or any(a and b for a, b in zip(give, receive)):
            return False
        if not isinstance(offer.offer_id, str) or not offer.offer_id:
            return False
        if type(offer.proposer) is not int or type(offer.recipient) is not int:
            return False
        players = {player.player_id for player in player_view.players}
        if (
            offer.proposer not in players
            or offer.recipient not in players
            or offer.proposer == offer.recipient
        ):
            return False
        if context.kind in ("TURN", "PLAY_TURN"):
            if not (
                context.has_rolled
                and context.actor == context.turn_owner == offer.proposer
                and offer.recipient in context.eligible_partners
                and context.offer is None
            ):
                return False
        elif context.kind == "TRADE_RESPONSE":
            if not (
                context.offer == offer
                and offer.recipient == context.actor
                and offer.proposer == context.turn_owner
            ):
                return False
        else:
            return False
        own_give, _ = offer.for_player(player_view.player_id)
        return all(needed <= available for needed, available in zip(own_give, hand))
    except (TypeError, ValueError, AttributeError):
        return False


def validate_trade_decision(
    decision: TradeDecision, player_view: Observation, context: DecisionContext
) -> bool:
    """Revalidate immediately before execution using the latest player view.

    Matching IDs alone are insufficient: incoming terms and parties must also
    equal the live offer. A stale result, including a late rejection, is invalid.
    """
    if not isinstance(decision, TradeDecision) or not _live_context(player_view, context):
        return False
    if decision.context_version != context.version:
        return False
    if decision.kind == "NO_OFFER":
        return decision.offer is None
    if decision.kind in ("ACCEPT", "REJECT"):
        if (
            context.kind != "TRADE_RESPONSE"
            or context.offer is None
            or decision.offer != context.offer
            or context.offer.recipient != context.actor
        ):
            return False
        # Rejection does not transfer resources and remains valid if unaffordable.
        return decision.kind == "REJECT" or validate_trade_offer(
            decision.offer, player_view, context
        )
    return (
        decision.kind == "PROPOSE"
        and context.kind in ("TURN", "PLAY_TURN")
        and validate_trade_offer(decision.offer, player_view, context)
    )


class TradeOrchestrator:
    """A conservative controller with independently enabled optional plug-ins.

    By default models are disabled. ``enabled=True`` enables supplied models;
    ``generator_enabled`` / ``critic_enabled`` override the respective setting.
    Generating proposals requires both models. Incoming responses need only a
    critic, with an injected binary policy as fallback (otherwise reject).

    A total decision budget defaults to 0.1 seconds. Daemon workers prevent a
    blocking plug-in from hanging shutdown, and one in-flight worker per model
    stage prevents repeated timeouts from creating unbounded threads. Python
    cannot terminate a stuck thread: use a process boundary for untrusted code.
    """

    def __init__(
        self,
        generator: TradeGenerator | None = None,
        critic: TradeCritic | None = None,
        *,
        enabled: bool = False,
        generator_enabled: bool | None = None,
        critic_enabled: bool | None = None,
        max_candidates: int = 32,
        max_input_items: int | None = None,
        timeout_seconds: float = 0.1,
        binary_callback: BinaryResponse | None = None,
        include_assessments: bool = False,
    ) -> None:
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        if max_input_items is None:
            max_input_items = 4 * max_candidates
        if type(max_input_items) is not int or max_input_items < 1:
            raise ValueError("max_input_items must be a positive integer")
        timeout_seconds = _finite_number(timeout_seconds, "timeout_seconds")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.generator = generator
        self.critic = critic
        self.generator_enabled = enabled if generator_enabled is None else generator_enabled
        self.critic_enabled = enabled if critic_enabled is None else critic_enabled
        self.max_candidates = max_candidates
        self.max_input_items = max_input_items
        self.timeout_seconds = timeout_seconds
        self.binary_callback = binary_callback
        self.include_assessments = include_assessments
        self._model_lock = Lock()
        self._binary_lock = Lock()

    @staticmethod
    def _invoke(function: Callable[[], _T], deadline: float, lock: Lock) -> _T:
        remaining = deadline - monotonic()
        if remaining <= 0 or not lock.acquire(blocking=False):
            raise TimeoutError("trade model unavailable within decision budget")
        result: Queue[tuple[bool, object]] = Queue(maxsize=1)

        def worker() -> None:
            try:
                outcome = (True, function())
            except BaseException as error:  # noqa: BLE001 - contain failures inside the daemon
                outcome = (False, error)
            finally:
                lock.release()
            # Publish only after releasing the lock, so a following stage can start.
            result.put(outcome)

        try:
            Thread(target=worker, name="trade-model", daemon=True).start()
        except RuntimeError:
            lock.release()
            raise
        try:
            success, value = result.get(timeout=max(0.0, deadline - monotonic()))
        except Empty as error:
            raise TimeoutError("trade model exceeded decision budget") from error
        if not success:
            raise RuntimeError("trade plug-in failed") from value
        return value  # type: ignore[return-value]

    def _collect(
        self, player_view: Observation, context: DecisionContext
    ) -> tuple[TradeOffer, ...]:
        assert self.generator is not None
        source = iter(self.generator.propose(player_view, context, self.max_candidates))
        candidates: list[TradeOffer] = []
        identities: set[str] = set()
        terms: set[tuple[object, ...]] = set()
        for _ in range(self.max_input_items):
            try:
                candidate = next(source)
            except StopIteration:
                break
            if not validate_trade_offer(candidate, player_view, context):
                continue
            key = (candidate.proposer, candidate.recipient, candidate.give, candidate.receive)
            if candidate.offer_id in identities or key in terms:
                continue
            identities.add(candidate.offer_id)
            terms.add(key)
            candidates.append(candidate)
            if len(candidates) == self.max_candidates:
                break
        return tuple(candidates)

    def _score(
        self, player_view: Observation, context: DecisionContext, offers: tuple[TradeOffer, ...]
    ) -> tuple[TradeAssessment, ...]:
        assert self.critic is not None
        scored = self.critic.score(player_view, context, offers)
        if not isinstance(scored, list) or len(scored) != len(offers):
            raise ValueError("critic must return one assessment per offer")
        remaining = set(offers)
        validated: list[TradeAssessment] = []
        for item in scored:
            if not isinstance(item, TradeAssessment) or item.offer not in remaining:
                raise ValueError("critic returned an unknown or duplicated offer")
            # Reconstruct to enforce numeric validation even for malformed plug-in records.
            validated.append(
                TradeAssessment(
                    item.offer,
                    item.benefit_relative_to_no_trade,
                    item.uncertainty,
                    item.acceptance_probability,
                    item.benefit_if_accepted,
                )
            )
            remaining.remove(item.offer)
        return (_NO_TRADE, *validated)

    def _decision(
        self,
        kind: TradeKind,
        context: DecisionContext,
        offer: TradeOffer | None = None,
        assessments: tuple[TradeAssessment, ...] = (_NO_TRADE,),
    ) -> TradeDecision:
        return TradeDecision(
            kind,
            offer,
            context.version,
            assessments if self.include_assessments else (),
        )

    def _fallback(
        self, player_view: Observation, context: DecisionContext, deadline: float
    ) -> TradeDecision:
        accepted = False
        if self.binary_callback is not None:
            try:
                accepted = (
                    self._invoke(
                        lambda: self.binary_callback(player_view, context),
                        deadline,
                        self._binary_lock,
                    )
                    is True
                )
            except (RuntimeError, TimeoutError):
                accepted = False
        decision = self._decision("ACCEPT" if accepted else "REJECT", context, context.offer)
        if not validate_trade_decision(decision, player_view, context):
            return self._decision("REJECT", context, context.offer)
        return decision

    def choose(self, player_view: Observation, context: DecisionContext) -> TradeDecision:
        deadline = monotonic() + self.timeout_seconds
        incoming = context.kind == "TRADE_RESPONSE"
        if not _live_context(player_view, context):
            return self._decision(
                "REJECT" if incoming else "NO_OFFER", context, context.offer if incoming else None
            )
        if incoming:
            if context.offer is None or not validate_trade_offer(
                context.offer, player_view, context
            ):
                return self._decision("REJECT", context, context.offer)
            if not self.critic_enabled or self.critic is None:
                return self._fallback(player_view, context, deadline)
            # Reserve budget so a timed-out critic can still use the binary policy.
            model_deadline = deadline
            if self.binary_callback is not None:
                model_deadline -= self.timeout_seconds * 0.25
            try:
                assessments = self._invoke(
                    lambda: self._score(player_view, context, (context.offer,)),
                    model_deadline,
                    self._model_lock,
                )
            except (RuntimeError, TimeoutError):
                return self._fallback(player_view, context, deadline)
        else:
            if not (
                context.kind in ("TURN", "PLAY_TURN")
                and context.has_rolled
                and context.actor == context.turn_owner
                and context.offer is None
                and self.generator_enabled
                and self.critic_enabled
                and self.generator is not None
                and self.critic is not None
            ):
                return self._decision("NO_OFFER", context)
            try:
                offers = self._invoke(
                    lambda: self._collect(player_view, context), deadline, self._model_lock
                )
                if not offers:
                    return self._decision("NO_OFFER", context, assessments=(_NO_TRADE,))
                assessments = self._invoke(
                    lambda: self._score(player_view, context, offers), deadline, self._model_lock
                )
            except (RuntimeError, TimeoutError):
                return self._decision("NO_OFFER", context)
        best = max(assessments, key=lambda item: item.benefit_relative_to_no_trade)
        kind: TradeKind = "REJECT" if incoming else "NO_OFFER"
        selected = context.offer if incoming else None
        if best.offer is not None:
            kind = "ACCEPT" if incoming else "PROPOSE"
            selected = best.offer
        decision = self._decision(kind, context, selected, assessments)
        if validate_trade_decision(decision, player_view, context):
            return decision
        return self._decision(
            "REJECT" if incoming else "NO_OFFER", context, context.offer if incoming else None
        )
