"""Player-visible, joint resource beliefs with conservative uncertainty bounds.

No engine or simulator state is accepted. Exact public operations update every
joint hypothesis; hidden draws branch with card-proportional probabilities. A
bounded particle approximation supplies moments while independently propagated
bounds prevent sample agreement from becoming a false claim of certainty.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache

from .contracts import RESOURCES, ResourceBelief, ResourceVector, resource_vector

Hand = tuple[int, ...]
Joint = tuple[Hand, ...]
Bounds = tuple[tuple[Hand, Hand], ...]


class BeliefInconsistency(ValueError):
    """Visible evidence contradicts the ledger's public resource constraints."""


def _count(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("card counts must be nonnegative integers")
    return value


def _compositions(total: int, lower: Hand, upper: Hand):
    """Enumerate bounded integer hands; each returned hand has the given total."""

    def visit(i, remaining, prefix):
        if i == 4:
            if lower[i] <= remaining <= upper[i]:
                yield tuple(prefix + [remaining])
            return
        first = max(lower[i], remaining - sum(upper[i + 1 :]))
        last = min(upper[i], remaining - sum(lower[i + 1 :]))
        for n in range(first, last + 1):
            yield from visit(i + 1, remaining - n, prefix + [n])

    yield from visit(0, total, [])


class BeliefLedger:
    """A deterministic, bounded joint particle ledger.

    ``known_hands`` contains only hands actually observed by this player.
    Unknown players must have public ``totals``. Passing just ``player_count``
    initializes a new game's empty hands. Player ids may be any nonnegative
    integers and need not be contiguous.

    ``particles`` and ``weights`` are diagnostic snapshots, not hidden truth.
    When support is sampled, summary bounds are conservative rather than
    extrema of the sampled hands. Rejuvenation after particle depletion keeps
    public interval constraints and known global conservation; its posterior
    moments are approximate (as are all capped particle-filter moments).
    """

    def __init__(
        self,
        known_hands: Mapping[int, ResourceVector] | None = None,
        totals: Mapping[int, int] | Sequence[int] | None = None,
        *,
        player_count: int | None = None,
        seed: int = 0,
        max_particles: int = 256,
    ):
        if type(max_particles) is not int or max_particles < 1:
            raise ValueError("max_particles must be positive")
        if player_count is not None and (type(player_count) is not int or player_count < 1):
            raise ValueError("player_count must be positive")
        self.max_particles = max_particles
        self._rng = random.Random(seed)
        known = {p: resource_vector(hand) for p, hand in (known_hands or {}).items()}
        public = self._total_map(totals) if totals is not None else {}
        for p, hand in known.items():
            if p in public and public[p] != sum(hand):
                raise BeliefInconsistency("known hand disagrees with public total")
            public[p] = sum(hand)
        if player_count is not None:
            for p in range(player_count):
                public.setdefault(p, 0)
        if not public:
            raise ValueError("supply known hands, totals, or player_count")
        if any(type(p) is not int or p < 0 for p in public):
            raise ValueError("player ids must be nonnegative integers")
        self.player_ids = tuple(sorted(public))
        self._indices = {p: i for i, p in enumerate(self.player_ids)}
        self._totals = tuple(public[p] for p in self.player_ids)
        self._bounds: Bounds = tuple(
            (known[p], known[p]) if p in known else ((0,) * 5, (public[p],) * 5)
            for p in self.player_ids
        )
        self._global: Hand | None = None
        if all(lo == hi for lo, hi in self._bounds):
            self._global = tuple(sum(lo[r] for lo, _ in self._bounds) for r in range(5))
        self._states, self._weights, self._exhaustive = self._generate(
            self._bounds, self._totals, self._global
        )
        self.rejuvenations = 0

    @staticmethod
    def _total_map(totals):
        items = totals.items() if isinstance(totals, Mapping) else enumerate(totals)
        return {p: _count(n) for p, n in items}

    @property
    def particles(self) -> tuple[dict[int, Hand], ...]:
        return tuple(dict(zip(self.player_ids, state)) for state in self._states)

    @property
    def weights(self) -> tuple[float, ...]:
        return self._weights

    @property
    def exhaustive(self) -> bool:
        return self._exhaustive

    def _index(self, player: int) -> int:
        try:
            return self._indices[player]
        except KeyError as exc:
            raise ValueError(f"unknown player {player}") from exc

    @staticmethod
    def _tighten(bounds: Bounds, totals: tuple[int, ...], global_hand: Hand | None) -> Bounds:
        lower = [list(lo) for lo, _ in bounds]
        upper = [list(hi) for _, hi in bounds]
        changed = True
        while changed:
            changed = False
            for p, total in enumerate(totals):
                if total < 0 or sum(lower[p]) > total or sum(upper[p]) < total:
                    raise BeliefInconsistency("resource bounds disagree with public hand total")
                for r in range(5):
                    lo = max(0, lower[p][r], total - sum(upper[p][s] for s in range(5) if s != r))
                    hi = min(upper[p][r], total - sum(lower[p][s] for s in range(5) if s != r))
                    if global_hand is not None:
                        lo = max(
                            lo,
                            global_hand[r] - sum(upper[q][r] for q in range(len(totals)) if q != p),
                        )
                        hi = min(
                            hi,
                            global_hand[r] - sum(lower[q][r] for q in range(len(totals)) if q != p),
                        )
                    if lo > hi:
                        raise BeliefInconsistency("visible resources cannot satisfy the ledger")
                    if lo != lower[p][r] or hi != upper[p][r]:
                        lower[p][r], upper[p][r] = lo, hi
                        changed = True
        return tuple((tuple(lo), tuple(hi)) for lo, hi in zip(lower, upper))

    def _enumerate(self, bounds, totals, global_hand):
        def visit(p, remaining, prefix):
            if p == len(totals):
                if remaining is None or not any(remaining):
                    yield tuple(prefix)
                return
            lo, hi = bounds[p]
            if remaining is not None:
                lo = tuple(
                    max(
                        lo[r],
                        remaining[r] - sum(bounds[q][1][r] for q in range(p + 1, len(totals))),
                    )
                    for r in range(5)
                )
                hi = tuple(
                    min(
                        hi[r],
                        remaining[r] - sum(bounds[q][0][r] for q in range(p + 1, len(totals))),
                    )
                    for r in range(5)
                )
            for hand in _compositions(totals[p], lo, hi):
                residual = (
                    None if remaining is None else tuple(a - b for a, b in zip(remaining, hand))
                )
                yield from visit(p + 1, residual, prefix + [hand])

        yield from visit(0, global_hand, [])

    def _sample_hand(self, total, lower, upper):
        @lru_cache(None)
        def ways(i, remaining):
            if i == 5:
                return int(remaining == 0)
            return sum(
                ways(i + 1, remaining - n)
                for n in range(
                    max(lower[i], remaining - sum(upper[i + 1 :])),
                    min(upper[i], remaining - sum(lower[i + 1 :])) + 1,
                )
            )

        hand = []
        remaining = total
        for i in range(5):
            choices = list(
                range(
                    max(lower[i], remaining - sum(upper[i + 1 :])),
                    min(upper[i], remaining - sum(lower[i + 1 :])) + 1,
                )
            )
            counts = [ways(i + 1, remaining - n) for n in choices]
            total_ways = sum(counts)
            if not total_ways:
                return None
            draw = self._rng.randrange(total_ways)
            for n, count in zip(choices, counts):
                draw -= count
                if draw < 0:
                    hand.append(n)
                    remaining -= n
                    break
        return tuple(hand)

    def _generate(self, bounds, totals, global_hand):
        bounds = self._tighten(bounds, totals, global_hand)
        candidates = []
        for state in self._enumerate(bounds, totals, global_hand):
            candidates.append(state)
            if len(candidates) > self.max_particles:
                break
        if not candidates:
            raise BeliefInconsistency("no joint hands satisfy the visible constraints")
        if len(candidates) <= self.max_particles:
            return tuple(candidates), (1 / len(candidates),) * len(candidates), True
        sampled: dict[Joint, float] = defaultdict(float)
        # Start with a guaranteed feasible joint state, then sample bounded
        # compositions. All cross-player conservation constraints are retained.
        sampled[candidates[self._rng.randrange(len(candidates))]] += 1
        for _ in range(self.max_particles - 1):
            state = []
            remaining = global_hand
            for p, total in enumerate(totals):
                lo, hi = bounds[p]
                if remaining is not None:
                    lo = tuple(
                        max(
                            lo[r],
                            remaining[r] - sum(bounds[q][1][r] for q in range(p + 1, len(totals))),
                        )
                        for r in range(5)
                    )
                    hi = tuple(
                        min(
                            hi[r],
                            remaining[r] - sum(bounds[q][0][r] for q in range(p + 1, len(totals))),
                        )
                        for r in range(5)
                    )
                hand = self._sample_hand(total, lo, hi)
                if hand is None:
                    break
                state.append(hand)
                if remaining is not None:
                    remaining = tuple(a - b for a, b in zip(remaining, hand))
            if len(state) == len(totals) and (remaining is None or not any(remaining)):
                sampled[tuple(state)] += 1
        return self._normalize(sampled), self._normalized_weights(sampled), False

    @staticmethod
    def _normalize(weighted):
        return tuple(weighted)

    @staticmethod
    def _normalized_weights(weighted):
        total = sum(weighted.values())
        return tuple(w / total for w in weighted.values())

    def _reduce(self, weighted, exhaustive):
        if len(weighted) > self.max_particles:
            states = list(weighted)
            selected = self._rng.choices(
                states, weights=list(weighted.values()), k=self.max_particles
            )
            weighted = defaultdict(float)
            for state in selected:
                weighted[state] += 1
            exhaustive = False
        return tuple(weighted), self._normalized_weights(weighted), exhaustive

    def _commit(self, states, weights, bounds, totals, global_hand, exhaustive):
        bounds = self._tighten(bounds, totals, global_hand)
        compatible = defaultdict(float)
        for state, weight in zip(states, weights):
            if all(
                all(lo[r] <= hand[r] <= hi[r] for r in range(5))
                for hand, (lo, hi) in zip(state, bounds)
            ):
                compatible[state] += weight
        rejuvenated = False
        if not compatible:
            if exhaustive:
                raise BeliefInconsistency("visible evidence excludes every possible joint hand")
            states, weights, _ = self._generate(bounds, totals, global_hand)
            exhaustive = False
            rejuvenated = True
        else:
            states, weights, exhaustive = self._reduce(compatible, exhaustive)
        if exhaustive:
            bounds = tuple(
                (
                    tuple(min(s[p][r] for s in states) for r in range(5)),
                    tuple(max(s[p][r] for s in states) for r in range(5)),
                )
                for p in range(len(totals))
            )
        self._states, self._weights = states, weights
        self._bounds, self._totals, self._global = bounds, totals, global_hand
        self._exhaustive = exhaustive
        self.rejuvenations += rejuvenated

    def apply_delta(self, player: int, delta: Iterable[int]) -> None:
        """Apply a publicly known gain/cost vector (negative entries are costs)."""
        p = self._index(player)
        change = tuple(delta)
        if len(change) != 5 or any(type(n) is not int for n in change):
            raise ValueError("delta must contain five signed integer counts")
        lo, hi = self._bounds[p]
        bounds = list(self._bounds)
        bounds[p] = (
            tuple(max(lo[r], -change[r]) + change[r] for r in range(5)),
            tuple(hi[r] + change[r] for r in range(5)),
        )
        totals = list(self._totals)
        totals[p] += sum(change)
        global_hand = (
            None if self._global is None else tuple(a + b for a, b in zip(self._global, change))
        )
        weighted = defaultdict(float)
        for state, weight in zip(self._states, self._weights):
            hand = tuple(a + b for a, b in zip(state[p], change))
            if min(hand) >= 0:
                updated = list(state)
                updated[p] = hand
                weighted[tuple(updated)] += weight
        self._commit(
            tuple(weighted),
            tuple(weighted.values()),
            tuple(bounds),
            tuple(totals),
            global_hand,
            self._exhaustive,
        )

    def transfer(self, source: int, target: int, count: int, resource: str | None = None) -> None:
        """Transfer known cards or a uniformly random card from the victim per draw."""
        self._take(source, target, _count(count), resource)

    def _take(self, source, target, count, resource):
        p = self._index(source)
        q = None if target is None else self._index(target)
        if q == p:
            raise ValueError("source and target must differ")
        if count > self._totals[p]:
            raise BeliefInconsistency("cannot remove more cards than the public total")
        if resource is not None and resource not in RESOURCES:
            raise ValueError("unknown resource")
        if count == 0:
            return
        r = None if resource is None else RESOURCES.index(resource)
        totals = list(self._totals)
        totals[p] -= count
        if q is not None:
            totals[q] += count
        bounds = list(self._bounds)
        lo, hi = bounds[p]
        if r is None:
            bounds[p] = (tuple(max(0, n - count) for n in lo), tuple(min(n, totals[p]) for n in hi))
        else:
            bounds[p] = (
                tuple(max(n, count) - count if i == r else n for i, n in enumerate(lo)),
                tuple(n - count if i == r else n for i, n in enumerate(hi)),
            )
        if q is not None:
            lo, hi = bounds[q]
            bounds[q] = (
                tuple(n + (count if i == r else 0) for i, n in enumerate(lo)),
                tuple(n + (count if r is None or i == r else 0) for i, n in enumerate(hi)),
            )
        global_hand = self._global
        if q is None:
            global_hand = (
                None
                if r is None or global_hand is None
                else tuple(n - (count if i == r else 0) for i, n in enumerate(global_hand))
            )
        states, weights, exhaustive = self._states, self._weights, self._exhaustive
        for _ in range(count):
            weighted = defaultdict(float)
            for state, weight in zip(states, weights):
                total = sum(state[p])
                for i, available in enumerate(state[p]):
                    if not available or (r is not None and i != r):
                        continue
                    updated = list(state)
                    hand = list(updated[p])
                    hand[i] -= 1
                    updated[p] = tuple(hand)
                    if q is not None:
                        hand = list(updated[q])
                        hand[i] += 1
                        updated[q] = tuple(hand)
                    weighted[tuple(updated)] += weight * (available / total if r is None else 1)
            if not weighted:
                states, weights = (), ()
                break
            states, weights, exhaustive = self._reduce(weighted, exhaustive)
        self._commit(states, weights, tuple(bounds), tuple(totals), global_hand, exhaustive)

    def discard(self, player: int, count: int, known: Iterable[int] | None = None) -> None:
        """Remove a visible discard vector or a hidden card-proportional sample."""
        _count(count)
        if known is not None:
            hand = resource_vector(known)
            if sum(hand) != count:
                raise ValueError("known discard must sum to its public count")
            self.apply_delta(player, tuple(-n for n in hand))
        else:
            self._take(player, None, count, None)

    def reconcile(
        self,
        totals: Mapping[int, int] | Sequence[int],
        own_player: int | None = None,
        own_hand: Iterable[int] | None = None,
    ) -> None:
        """Condition on public totals and the observer's actually visible hand.

        A changed public total requires a corresponding ledger event. This
        catches missed history rather than silently fabricating a card flow.
        """
        public = self._total_map(totals)
        if set(public) != set(self.player_ids):
            raise BeliefInconsistency("reconciliation must cover the same players")
        if any(public[p] != self._totals[i] for i, p in enumerate(self.player_ids)):
            raise BeliefInconsistency("public total changed without a matching visible event")
        if (own_player is None) != (own_hand is None):
            raise ValueError("own_player and own_hand must be supplied together")
        bounds = list(self._bounds)
        if own_player is not None:
            p = self._index(own_player)
            hand = resource_vector(own_hand)
            if sum(hand) != self._totals[p]:
                raise BeliefInconsistency("own hand disagrees with public total")
            lo, hi = bounds[p]
            if any(not lo[r] <= hand[r] <= hi[r] for r in range(5)):
                raise BeliefInconsistency("own hand contradicts known resource bounds")
            bounds[p] = (hand, hand)
        self._commit(
            self._states, self._weights, tuple(bounds), self._totals, self._global, self._exhaustive
        )

    def summaries(self) -> tuple[ResourceBelief, ...]:
        result = []
        for p, player in enumerate(self.player_ids):
            mean = tuple(
                sum(w * state[p][r] for state, w in zip(self._states, self._weights))
                for r in range(5)
            )
            variance = tuple(
                sum(
                    w * (state[p][r] - mean[r]) ** 2
                    for state, w in zip(self._states, self._weights)
                )
                for r in range(5)
            )
            lo, hi = self._bounds[p]
            result.append(ResourceBelief(player, mean, lo, hi, variance, exact=lo == hi))
        return tuple(result)
