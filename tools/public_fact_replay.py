"""Public-only factual route checker against pinned Catan.

Read one SHA-bound public fact input, construct synthetic states consistent
with its before views, require equality of the entire ordered current legal
menu, then execute only each named root and explicit immediate continuations.
Synthetic private values are witnesses, not the source game's hidden state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import permutations
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("simulator.py", "contracts.py", "board.py", "beliefs.py", "engine_identity.py")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from catanatron.models.enums import ActionPrompt

from colonist_rl.beliefs import BeliefLedger
from colonist_rl.board import RulesConfig
from colonist_rl.contracts import Action
from colonist_rl.engine_identity import verify_engine_installation
from colonist_rl.simulator import (
    DEVELOPMENT_CARDS,
    PORT_DIRECTION_TO_NODEREFS,
    RESOURCES,
    FourPlayerGame,
)

ZERO = [0, 0, 0, 0, 0]
COSTS = {
    "BUILD_ROAD": [1, 1, 0, 0, 0],
    "BUILD_SETTLEMENT": [1, 1, 1, 1, 0],
    "BUILD_CITY": [0, 0, 0, 2, 3],
    "BUY_DEVELOPMENT_CARD": [0, 0, 1, 1, 1],
}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
RESOURCE_PATTERN = r"(?:WOOD|BRICK|SHEEP|WHEAT|ORE)"
COUNT_PATTERN = r"(?:\d+|one|two|three|four)"
TRADE_RE = re.compile(
    rf"\b(?:trade|trading|the)\s+(?P<give_count>{COUNT_PATTERN})[-\s]+"
    rf"(?P<give_resource>{RESOURCE_PATTERN})[-\s]+for[-\s]+"
    rf"(?P<receive_count>{COUNT_PATTERN})[-\s]+"
    rf"(?P<receive_resource>{RESOURCE_PATTERN})(?:\s+trade)?\b",
    re.IGNORECASE,
)
SETTLEMENT_PATTERNS = [
    re.compile(r"\bBUILD_SETTLEMENT\s+at\s+vertex\s+(\d+)\b", re.IGNORECASE),
    re.compile(
        r"\bbuild\s+(?:immediately\s+)?(?:a\s+)?settlement\s+at\s+vertex\s+(\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsettlement\s+can\s+be\s+built(?:\s+immediately)?\s+at\s+vertex\s+(\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:an?\s+)?(?:immediate\s+)?settlement\s+at\s+vertex\s+(\d+)"
        r"(?:\s+can\s+be\s+built)?\b",
        re.IGNORECASE,
    ),
]
CITY_PATTERNS = [
    re.compile(
        r"\bBUILD_CITY\s+at\s+(?:the\s+owned\s+settlement\s+on\s+)?vertex\s+(\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bupgrade\s+the\s+settlement\s+at\s+vertex\s+(\d+)\s+to\s+a\s+city\b",
        re.IGNORECASE,
    ),
]
ROAD_PATTERNS = [
    re.compile(
        r"\b(?:build|building|built)\s+(?:a\s+)?road\s+edge\s+(\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:an?\s+)?road\s+can\s+be\s+built(?:\s+immediately)?\s+on\s+edge\s+(\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\broad\s+(?:at\s+)?edge\s+(\d+)\b", re.IGNORECASE),
]
BUY_RE = re.compile(r"\b(?:BUY_DEVELOPMENT_CARD|buy\s+a\s+development[\s-]+card)\b", re.IGNORECASE)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize(value: Any) -> Any:
    return json.loads(json.dumps(asdict(value)))


def as_action(data: dict[str, Any]) -> Action:
    return Action(
        data["kind"],
        target=data.get("target"),
        resource=data.get("resource"),
        second_resource=data.get("second_resource"),
        partner=data.get("partner"),
        give=tuple(data.get("give", ZERO)),
        receive=tuple(data.get("receive", ZERO)),
        offer_id=data.get("offer_id"),
    )


def action_data(
    kind: str,
    *,
    target: int | None = None,
    give: list[int] | None = None,
    receive: list[int] | None = None,
) -> dict[str, Any]:
    return normalize(
        Action(kind, target=target, give=tuple(give or ZERO), receive=tuple(receive or ZERO))
    )


def resource_names(vector: list[int] | tuple[int, ...]) -> dict[str, int]:
    return {name: count for name, count in zip(RESOURCES, vector) if count}


def environment_identity() -> dict[str, Any]:
    for name in SOURCE_FILES:
        module = sys.modules.get(f"colonist_rl.{name.removesuffix('.py')}")
        expected_path = (PROJECT_ROOT / "src" / "colonist_rl" / name).resolve()
        actual_path = Path(getattr(module, "__file__", "")).resolve()
        if actual_path != expected_path:
            raise RuntimeError(f"imported {name} from another checkout: {actual_path}")
    head = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    engine = verify_engine_installation()
    return {
        "source_head": head,
        "engine": engine,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "source_files_sha256": {
            name: hashlib.sha256(
                (PROJECT_ROOT / "src" / "colonist_rl" / name).read_bytes()
            ).hexdigest()
            for name in SOURCE_FILES
        },
    }


def public_bank_bounds(observation: dict[str, Any]) -> tuple[list[int], list[int]]:
    actor = observation["player_id"]
    others = [b for b in observation["beliefs"] if b["player_id"] != actor]
    supply = observation["rules"]["bank_resources"]
    lower = [
        supply[j] - observation["hand"][j] - sum(b["upper"][j] for b in others) for j in range(5)
    ]
    upper = [
        supply[j] - observation["hand"][j] - sum(b["lower"][j] for b in others) for j in range(5)
    ]
    return lower, upper


def root_required_bank(observation: dict[str, Any]) -> list[int]:
    required = [0] * 5
    for action in observation["legal_actions"]:
        if action["kind"] == "MARITIME_TRADE":
            for j, count in enumerate(action["receive"]):
                required[j] = max(required[j], count)
        elif action["kind"] == "PLAY_YEAR_OF_PLENTY":
            requested = [action["resource"], action["second_resource"]]
            for j, resource in enumerate(RESOURCES):
                required[j] = max(required[j], requested.count(resource))
    return required


def candidate_hands(observation: dict[str, Any]):
    """Yield hidden-hand witnesses within public totals and belief intervals."""
    actor = observation["player_id"]
    players = {p["player_id"]: p for p in observation["players"]}
    beliefs = {b["player_id"]: b for b in observation["beliefs"]}
    opponents = tuple(q for q in sorted(players) if q != actor)
    supply = observation["rules"]["bank_resources"]
    lower = {q: list(beliefs[q]["lower"]) for q in opponents}
    upper = {q: list(beliefs[q]["upper"]) for q in opponents}
    remaining = {q: players[q]["resource_count"] - sum(lower[q]) for q in opponents}
    if any(n < 0 or players[q]["resource_count"] > sum(upper[q]) for q, n in remaining.items()):
        return
    reserve = root_required_bank(observation)
    capacity = [
        supply[j] - observation["hand"][j] - sum(lower[q][j] for q in opponents) - reserve[j]
        for j in range(5)
    ]
    if min(capacity) < 0:
        return
    seen = set()
    player_orders = (opponents,) + tuple(
        order for order in permutations(opponents) if order != opponents
    )
    for order in player_orders:
        for strategy in range(5):
            hands = {q: lower[q].copy() for q in opponents}
            available = capacity.copy()
            possible = True
            for q in order:
                need = remaining[q]
                belief = beliefs[q]
                if strategy == 0:
                    resources = sorted(
                        range(5), key=lambda j: (-(belief["expected"][j] - hands[q][j]), j)
                    )
                elif strategy == 1:
                    resources = sorted(range(5), key=lambda j: (-(upper[q][j] - hands[q][j]), j))
                elif strategy == 2:
                    resources = list(range(5))
                elif strategy == 3:
                    resources = list(reversed(range(5)))
                else:
                    resources = sorted(
                        range(5), key=lambda j: (belief["expected"][j] - hands[q][j], j)
                    )
                for j in resources:
                    n = min(need, available[j], upper[q][j] - hands[q][j])
                    if n > 0:
                        hands[q][j] += n
                        available[j] -= n
                        need -= n
                if need:
                    possible = False
                    break
            if not possible:
                continue
            full = tuple(
                tuple(observation["hand"]) if q == actor else tuple(hands[q])
                for q in sorted(players)
            )
            if full not in seen:
                seen.add(full)
                yield [list(hand) for hand in full]


def synthetic_development_inventory(
    observation: dict[str, Any],
) -> tuple[dict[int, list[int]], list[str], dict[str, Any]]:
    """Conserve the finite development deck without guessing private card types.

    Known played Knights and the actor's exact hand consume their card supply.
    Opponent card types are assigned within their public hand totals.  If the
    actor can pay but BUY is absent from the exact root menu, the deck must be
    empty; every unplayable VP card must then be in a current hand.  Remaining
    non-VP cards may have been played earlier.
    """
    actor = observation["player_id"]
    supply = list(observation["rules"]["development_cards"])
    supply[0] -= sum(p["played_knights"] for p in observation["players"])
    for j, count in enumerate(observation["development"]):
        supply[j] -= count
    if min(supply) < 0:
        raise ValueError("public played Knights or own cards exceed finite deck supply")
    affordable = all(observation["hand"][j] >= COSTS["BUY_DEVELOPMENT_CARD"][j] for j in range(5))
    root_has_buy = any(
        action["kind"] == "BUY_DEVELOPMENT_CARD" for action in observation["legal_actions"]
    )
    deck_must_be_empty = affordable and not root_has_buy
    opponents = [p for p in observation["players"] if p["player_id"] != actor]
    assignments = {p["player_id"]: [0] * 5 for p in opponents}
    slots = {p["player_id"]: p["development_count"] for p in opponents}
    if deck_must_be_empty:
        vp_to_place = supply[4]
        if sum(slots.values()) < vp_to_place:
            raise ValueError(
                "empty deck inferred from menu conflicts with unplayable VP card supply"
            )
        for p in opponents:
            q = p["player_id"]
            n = min(slots[q], vp_to_place)
            assignments[q][4] += n
            slots[q] -= n
            supply[4] -= n
            vp_to_place -= n
    card_order = (0, 1, 2, 3, 4)
    for p in opponents:
        q = p["player_id"]
        for j in card_order:
            n = min(slots[q], supply[j])
            assignments[q][j] += n
            slots[q] -= n
            supply[j] -= n
        if slots[q]:
            raise ValueError("public opponent development counts exceed remaining card supply")
    if deck_must_be_empty:
        if supply[4]:
            raise ValueError("unplayable VP cards cannot be outside hands and empty deck")
        deck = []
    else:
        deck = [card for card, count in zip(DEVELOPMENT_CARDS, supply) for _ in range(count)]
        if root_has_buy and not deck:
            raise ValueError("saved BUY action conflicts with conserved empty deck")
    return (
        assignments,
        deck,
        {
            "public_development_counts": [p["development_count"] for p in observation["players"]],
            "known_played_knights": [p["played_knights"] for p in observation["players"]],
            "synthetic_opponent_card_types": {
                str(player_id): counts for player_id, counts in assignments.items()
            },
            "synthetic_deck_count": len(deck),
            "deck_forced_empty_by_current_menu": deck_must_be_empty,
            "unplayable_vp_cards_deck_lower_bound": max(
                0,
                observation["rules"]["development_cards"][4]
                - sum(p["development_count"] for p in observation["players"]),
            ),
        },
    )


def make_synthetic_game(observation: dict[str, Any]) -> FourPlayerGame:
    if observation["context"]["kind"] != "PLAY_TURN":
        raise ValueError("synthetic constructor currently supports PLAY_TURN only")
    if observation["rules"]["player_count"] != 4:
        raise ValueError("pinned engine requires four players")
    rules = RulesConfig(**observation["rules"])
    game = FourPlayerGame(rules)
    game.reset(seed=0)
    state = game.game.state
    board = state.board
    backend_map = board.map
    public_board = observation["board"]
    if len(backend_map.land_tiles) != len(public_board["tiles"]):
        raise ValueError("public tile count differs from pinned board")
    for i, tile in enumerate(backend_map.land_tiles.values()):
        source = public_board["tiles"][i]
        tile.resource = None if source["resource"] == "DESERT" else source["resource"]
        tile.number = source["number"] or None
    port_by_vertices = {tuple(sorted(port["vertices"])): port for port in public_board["ports"]}
    backend_map.port_nodes = defaultdict(set)
    for port in backend_map.ports_by_id.values():
        nodes = [port.nodes[ref] for ref in PORT_DIRECTION_TO_NODEREFS[port.direction]]
        vertices = tuple(sorted(game._node_to_vertex[node] for node in nodes))
        if vertices not in port_by_vertices:
            raise ValueError(f"public port is missing backend vertices {vertices}")
        port.resource = port_by_vertices[vertices]["resource"]
        backend_map.port_nodes[port.resource].update(nodes)
    backend_map.node_production = {}
    for node in backend_map.land_nodes:
        production = Counter()
        for tile in backend_map.land_tiles.values():
            if node in tile.nodes.values() and tile.resource is not None:
                production[tile.resource] += (6 - abs(7 - tile.number)) / 36
        backend_map.node_production[node] = production
    game._make_board()
    board.robber_coordinate = game._tile_to_coordinate[observation["robber_tile"]]
    board.buildings = {}
    state.buildings_by_color = {color: defaultdict(list) for color in state.colors}
    for building in observation["buildings"]:
        color = state.colors[building["owner"]]
        node = game._vertex_to_node[building["vertex"]]
        kind = "CITY" if building["level"] == 2 else "SETTLEMENT"
        board.buildings[node] = (color, kind)
        state.buildings_by_color[color][kind].append(node)
    board.roads = {}
    road_nodes = {color: set() for color in state.colors}
    for road in observation["roads"]:
        color = state.colors[road["owner"]]
        edge = game._id_to_edge[road["edge"]]
        board.roads[edge] = color
        board.roads[edge[::-1]] = color
        road_nodes[color].update(edge)
    board.board_buildable_ids = set(backend_map.land_nodes)
    for building in observation["buildings"]:
        vertex = building["vertex"]
        board.board_buildable_ids.discard(game._vertex_to_node[vertex])
        for edge in public_board["edges"]:
            if vertex in edge:
                neighbor = edge[0] if edge[1] == vertex else edge[1]
                board.board_buildable_ids.discard(game._vertex_to_node[neighbor])
    board.connected_components = defaultdict(list)
    for color in state.colors:
        nodes = road_nodes[color] | {
            node for node, (owner, _) in board.buildings.items() if owner == color
        }
        if nodes:
            board.connected_components[color] = [nodes]
    board.buildable_edges_cache = {}
    board.player_port_resources_cache = {}
    board.road_lengths = defaultdict(
        int,
        {state.colors[p["player_id"]]: p["longest_road"] for p in observation["players"]},
    )
    holders = [
        state.colors[p["player_id"]] for p in observation["players"] if p["has_longest_road"]
    ]
    if len(holders) > 1:
        raise ValueError("public view names multiple Longest Road holders")
    board.road_color = holders[0] if holders else None
    board.road_length = max(board.road_lengths.values(), default=0)
    opponent_cards, synthetic_deck, inventory_evidence = synthetic_development_inventory(
        observation
    )
    game._synthetic_development_inventory_evidence = inventory_evidence
    state.development_listdeck = synthetic_deck
    for player in observation["players"]:
        q = player["player_id"]
        prefix = f"P{q}_"
        state.player_state[prefix + "VICTORY_POINTS"] = player["victory_points"]
        state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] = player["victory_points"] + (
            observation["development"][4] if q == observation["player_id"] else opponent_cards[q][4]
        )
        for field, key in (
            ("ROADS_AVAILABLE", "roads_left"),
            ("SETTLEMENTS_AVAILABLE", "settlements_left"),
            ("CITIES_AVAILABLE", "cities_left"),
            ("LONGEST_ROAD_LENGTH", "longest_road"),
            ("PLAYED_KNIGHT", "played_knights"),
            ("HAS_ROAD", "has_longest_road"),
            ("HAS_ARMY", "has_largest_army"),
        ):
            state.player_state[prefix + field] = player[key]
        for card in DEVELOPMENT_CARDS:
            state.player_state[prefix + card + "_IN_HAND"] = 0
            state.player_state[prefix + card + "_OWNED_AT_START"] = False
        if q == observation["player_id"]:
            for j, card in enumerate(DEVELOPMENT_CARDS):
                state.player_state[prefix + card + "_IN_HAND"] = observation["development"][j]
                play_kind = "PLAY_KNIGHT_CARD" if card == "KNIGHT" else "PLAY_" + card
                state.player_state[prefix + card + "_OWNED_AT_START"] = any(
                    a["kind"] == play_kind for a in observation["legal_actions"]
                )
            state.player_state[prefix + "HAS_ROLLED"] = observation["context"]["has_rolled"]
            state.player_state[prefix + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = observation[
                "context"
            ]["has_played_development_card"]
        else:
            for j, card in enumerate(DEVELOPMENT_CARDS):
                state.player_state[prefix + card + "_IN_HAND"] = opponent_cards[q][j]
    actor = observation["player_id"]
    state.current_player_index = actor
    state.current_turn_index = observation["context"]["turn_owner"]
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.num_turns = observation["context"]["turn_number"]
    state.is_initial_build_phase = False
    state.is_road_building = False
    state.is_discarding = False
    state.is_moving_knight = False
    state.is_resolving_trade = False
    game.version = observation["context"]["version"]
    return game


def set_synthetic_hands(game: FourPlayerGame, hands: list[list[int]]) -> None:
    state = game.game.state
    for q, hand in enumerate(hands):
        for j, resource in enumerate(RESOURCES):
            state.player_state[f"P{q}_{resource}_IN_HAND"] = hand[j]
    state.resource_freqdeck = [
        game.rules.bank_resources[j] - sum(hand[j] for hand in hands) for j in range(5)
    ]
    if min(state.resource_freqdeck) < 0:
        raise ValueError("synthetic hands overdraw bank")
    game._ledgers = [
        BeliefLedger(
            player_count=4,
            known_hands={q: tuple(hands[q]) for q in range(4)},
            totals=[sum(hand) for hand in hands],
            seed=0,
        )
        for _ in range(4)
    ]
    game._refresh_actions()
    game.assert_resource_conservation()


def reconstruct(observation: dict[str, Any]) -> tuple[FourPlayerGame | None, dict[str, Any]]:
    evidence: dict[str, Any] = {"attempted": True, "candidate_count": 0}
    try:
        game = make_synthetic_game(observation)
        evidence["development_inventory"] = game._synthetic_development_inventory_evidence
        public_board_equal = normalize(game.board) == observation["board"]
        evidence["full_public_board_equal"] = public_board_equal
        if not public_board_equal:
            evidence["reason"] = "pinned board differs from the public board"
            return None, evidence
        expected_menu = observation["legal_actions"]
        best = None
        for hands in candidate_hands(observation):
            evidence["candidate_count"] += 1
            set_synthetic_hands(game, hands)
            view = normalize(game.observe())
            actual_menu = view["legal_actions"]
            differences = [
                key
                for key in observation
                if key not in ("beliefs", "events", "legal_actions")
                and view.get(key) != observation[key]
            ]
            menu_equal = actual_menu == expected_menu
            candidate = {
                "core_public_view_equal": not differences,
                "core_public_view_differences": differences,
                "exact_ordered_root_menu_equal": menu_equal,
                "actual_root_menu_count": len(actual_menu),
                "actual_root_menu_sha256": canonical_sha256(actual_menu),
                "synthetic_hands": hands,
                "synthetic_bank": list(game.game.state.resource_freqdeck),
            }
            if menu_equal and not differences:
                evidence.update(candidate)
                return game, evidence
            distance = (
                len(differences)
                + len([a for a in expected_menu if a not in actual_menu])
                + len([a for a in actual_menu if a not in expected_menu])
            )
            if best is None or distance < best[0]:
                best = (distance, candidate, actual_menu)
        evidence["exact_ordered_root_menu_equal"] = False
        if best:
            evidence.update(best[1])
            evidence["root_menu_missing"] = [a for a in expected_menu if a not in best[2]]
            evidence["root_menu_extra"] = [a for a in best[2] if a not in expected_menu]
            evidence["reason"] = (
                "no tested public-consistent synthetic state has the exact ordered root menu"
            )
        else:
            evidence["reason"] = "no public-consistent opponent hand assignment was found"
        return None, evidence
    except Exception as exc:  # noqa: BLE001 - failed witness is unverified, never certified
        evidence["reason"] = f"synthetic reconstruction failed: {type(exc).__name__}: {exc}"
        return None, evidence


def count_word(value: str) -> int:
    return int(value) if value.isdigit() else NUMBER_WORDS[value.lower()]


def parse_action_mentions(text: str) -> list[dict[str, Any]]:
    """Extract only the narrow, explicitly named immediate actions."""
    matches: list[tuple[int, int, dict[str, Any], str]] = []
    for match in TRADE_RE.finditer(text):
        give = [0] * 5
        receive = [0] * 5
        give[RESOURCES.index(match["give_resource"].upper())] = count_word(match["give_count"])
        receive[RESOURCES.index(match["receive_resource"].upper())] = count_word(
            match["receive_count"]
        )
        matches.append(
            (
                match.start(),
                match.end(),
                action_data("MARITIME_TRADE", give=give, receive=receive),
                match.group(),
            )
        )
    for kind, patterns in (
        ("BUILD_SETTLEMENT", SETTLEMENT_PATTERNS),
        ("BUILD_CITY", CITY_PATTERNS),
    ):
        for pattern in patterns:
            for match in pattern.finditer(text):
                matches.append(
                    (
                        match.start(),
                        match.end(),
                        action_data(kind, target=int(match[1])),
                        match.group(),
                    )
                )
    for pattern in ROAD_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(
                (
                    match.start(),
                    match.end(),
                    action_data("BUILD_ROAD", target=int(match[1])),
                    match.group(),
                )
            )
    for match in BUY_RE.finditer(text):
        matches.append(
            (match.start(), match.end(), action_data("BUY_DEVELOPMENT_CARD"), match.group())
        )
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected = []
    for start, end, action, phrase in matches:
        if selected and start < selected[-1]["end"]:
            continue
        selected.append({"start": start, "end": end, "action": action, "phrase": phrase})
    return selected


def parse_claim(
    text: str, root_action: dict[str, Any], *, legacy_steps: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A route parser that refuses to invent missing action targets or order."""
    errors = []
    branches = []
    if legacy_steps is not None:
        if not legacy_steps:
            errors.append("structured route is empty")
        else:
            first = legacy_steps[0]
            if not all(root_action.get(key) == value for key, value in first.items()):
                errors.append("structured first step does not match indexed root")
            followups = [normalize(as_action(step)) for step in legacy_steps[1:]]
            branches.append(
                {"name": "main", "actions": followups, "source": "structured_projected_steps"}
            )
    else:
        segments = [text]
        if ";" in text and "alternative" in text.split(";", 1)[1].lower():
            left, right = text.split(";", 1)
            segments = [left, right]
        for index, segment in enumerate(segments):
            mentions = parse_action_mentions(segment)
            actions = [item["action"] for item in mentions]
            masked = list(segment)
            for item in mentions:
                masked[item["start"] : item["end"]] = " " * (item["end"] - item["start"])
            remainder = "".join(masked)
            remainder = re.sub(
                r"\blongest\s+road\s+becomes\s+\w+\b",
                " ",
                remainder,
                flags=re.IGNORECASE,
            )
            remainder = re.sub(
                r"\bLongest\s+Road\s+award\b",
                " ",
                remainder,
                flags=re.IGNORECASE,
            )
            remainder = re.sub(
                r"\bfrom\s+vertex\s+\d+\s+to\s+vertex\s+\d+\b",
                " ",
                remainder,
                flags=re.IGNORECASE,
            )
            if re.search(
                r"\b(?:build|building|built|upgrade|trade|trading|buy|purchase|"
                r"settlement|city|road|edge|vertex)\b",
                remainder,
                re.IGNORECASE,
            ):
                errors.append(f"branch {index}: unparsed action or target language remains")
            root_positions = [i for i, action in enumerate(actions) if action == root_action]
            if root_positions:
                if root_positions[0] != 0:
                    errors.append(f"branch {index}: indexed root appears after another action")
                actions.pop(root_positions[0])
            elif index == 0:
                root_word = {
                    "MARITIME_TRADE": "trad",
                    "BUILD_ROAD": "road",
                    "BUILD_SETTLEMENT": "settlement",
                    "BUILD_CITY": "city",
                    "PLAY_KNIGHT_CARD": "knight",
                }.get(root_action["kind"])
                if root_word and root_word in segment.lower():
                    errors.append(
                        f"branch {index}: root action is mentioned but not exactly parsed"
                    )
            if index > 0 and "alternative" not in segment.lower():
                errors.append(f"branch {index}: alternative branch marker is missing")
            if not mentions and segment.strip():
                errors.append(f"branch {index}: no exact action was parsed")
            branches.append(
                {
                    "name": "main" if index == 0 else f"alternative_{index}",
                    "actions": actions,
                    "parsed_mentions": mentions,
                    "source": "explicit_claim",
                }
            )
    claimed_gain = None
    match = re.search(
        r"\bgains?\s+(one|two|three|\d+)\s+public\s+victory\s+points?\b",
        text,
        re.IGNORECASE,
    )
    if match:
        claimed_gain = count_word(match[1])
    elif re.search(r"\bgains?\s+the\s+two-point\s+Longest\s+Road\s+award\b", text, re.IGNORECASE):
        claimed_gain = 2
    longest_match = re.search(
        r"\blongest\s+road\s+becomes\s+(one|two|three|four|five|six|\d+)\b",
        text,
        re.IGNORECASE,
    )
    port_match = re.search(
        rf"\b(\d+):1\s+({RESOURCE_PATTERN})\s+port\b",
        text,
        re.IGNORECASE,
    )
    endpoints_match = re.search(
        r"\bfrom\s+vertex\s+(\d+)\s+to\s+vertex\s+(\d+)\b", text, re.IGNORECASE
    )
    if legacy_steps is None and not branches:
        errors.append("no route branch")
    return {
        "text": text,
        "parse_errors": errors,
        "branches": branches,
        "claimed_public_vp_gain": claimed_gain,
        "claimed_longest_road_length": (count_word(longest_match[1]) if longest_match else None),
        "claimed_longest_road_award": bool(
            re.search(r"\bLongest\s+Road\s+award\b", text, re.IGNORECASE)
        ),
        "claimed_port": (
            {"ratio": int(port_match[1]), "resource": port_match[2].upper()} if port_match else None
        ),
        "claimed_edge_endpoints": (
            [int(endpoints_match[1]), int(endpoints_match[2])] if endpoints_match else None
        ),
    }


def claims_for_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    if "projected_steps" in case:
        return [
            parse_claim(
                "Structured immediate route from older public fixture",
                case["root_action"],
                legacy_steps=case["projected_steps"],
            )
        ]
    return [parse_claim(text, case["root_action"]) for text in case.get("explicit_claims", [])]


def public_geometry(observation: dict[str, Any], kind: str, target: int | None) -> dict[str, Any]:
    actor = observation["player_id"]
    board = observation["board"]
    buildings = {
        building["vertex"]: (building["owner"], building["level"])
        for building in observation["buildings"]
    }
    owned_road_edges = {road["edge"] for road in observation["roads"] if road["owner"] == actor}
    occupied_edges = {road["edge"] for road in observation["roads"]}
    owned_road_endpoints = {vertex for edge in owned_road_edges for vertex in board["edges"][edge]}
    player = next(p for p in observation["players"] if p["player_id"] == actor)
    if kind == "BUILD_SETTLEMENT":
        if type(target) is not int or target not in range(len(board["vertices"])):
            return {"passes": False, "reason": "vertex is outside public board"}
        adjacent = {
            other
            for a, b in board["edges"]
            for other in ([b] if a == target else [a] if b == target else [])
        }
        checks = {
            "site_unoccupied": target not in buildings,
            "distance_rule": not adjacent.intersection(buildings),
            "connected_to_own_road": target in owned_road_endpoints,
            "settlement_piece_available": player["settlements_left"] > 0,
        }
        return {
            **checks,
            "passes": all(checks.values()),
            "adjacent_tiles": [
                {"tile": tile["id"], "resource": tile["resource"], "number": tile["number"]}
                for tile, vertices in zip(board["tiles"], board["tile_vertices"])
                if target in vertices
            ],
        }
    if kind == "BUILD_CITY":
        checks = {
            "owned_settlement": buildings.get(target) == (actor, 1),
            "city_piece_available": player["cities_left"] > 0,
        }
        return {**checks, "passes": all(checks.values())}
    if kind == "BUILD_ROAD":
        if type(target) is not int or target not in range(len(board["edges"])):
            return {"passes": False, "reason": "edge is outside public board"}
        endpoints = board["edges"][target]
        checks = {
            "edge_unoccupied": target not in occupied_edges,
            "road_piece_available": player["roads_left"] > 0
            or observation["context"]["free_roads"] > 0,
            "incident_to_own_network": any(
                buildings.get(vertex, (None, None))[0] == actor
                or (
                    vertex in owned_road_endpoints
                    and buildings.get(vertex, (actor, None))[0] == actor
                )
                for vertex in endpoints
            ),
        }
        return {**checks, "passes": all(checks.values()), "edge_vertices": endpoints}
    return {"passes": True, "reason": "no site or piece geometry for this action kind"}


def active_port_at_vertex(observation: dict[str, Any], vertex: int, port: dict[str, Any]) -> bool:
    actor = observation["player_id"]
    own_building = any(
        b["vertex"] == vertex and b["owner"] == actor for b in observation["buildings"]
    )
    return own_building and any(
        vertex in p["vertices"]
        and p["ratio"] == port["ratio"]
        and p["resource"] == port["resource"]
        for p in observation["board"]["ports"]
    )


def public_trade_ratio(observation: dict[str, Any], give: list[int]) -> int | None:
    positions = [j for j, count in enumerate(give) if count]
    if len(positions) != 1:
        return None
    resource = RESOURCES[positions[0]]
    actor = observation["player_id"]
    owned_vertices = {
        building["vertex"] for building in observation["buildings"] if building["owner"] == actor
    }
    return min(
        [4]
        + [
            port["ratio"]
            for port in observation["board"]["ports"]
            if owned_vertices.intersection(port["vertices"])
            and port["resource"] in (None, resource)
        ]
    )


def public_action_checks(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    kind = action["kind"]
    if kind in ("BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD"):
        return public_geometry(observation, kind, action["target"])
    hand = observation["hand"]
    if kind == "MARITIME_TRADE":
        ratio = public_trade_ratio(observation, action["give"])
        receive = action["receive"]
        checks = {
            "one_received_resource": sum(receive) == 1,
            "give_matches_public_port_ratio": ratio is not None and sum(action["give"]) == ratio,
            "public_hand_can_pay": all(hand[j] >= action["give"][j] for j in range(5)),
        }
        return {
            **checks,
            "passes": all(checks.values()),
            "effective_exchange_ratio": ratio,
            "give_named": resource_names(action["give"]),
            "receive_named": resource_names(receive),
        }
    if kind in COSTS:
        checks = {
            "public_hand_can_pay": all(hand[j] >= COSTS[kind][j] for j in range(5)),
        }
        return {**checks, "passes": all(checks.values()), "cost": COSTS[kind]}
    if kind == "PLAY_KNIGHT_CARD":
        checks = {
            "knight_card_in_hand": observation["development"][0] > 0,
            "no_development_card_already_played_this_turn": not observation["context"][
                "has_played_development_card"
            ],
        }
        return {**checks, "passes": all(checks.values())}
    return {"passes": True, "reason": "exact legal menu supplies this unsupported action predicate"}


def deterministic_deltas(
    action: dict[str, Any], before: dict[str, Any]
) -> tuple[list[int] | None, list[int] | None]:
    kind = action["kind"]
    if kind == "MARITIME_TRADE":
        hand_delta = [action["receive"][j] - action["give"][j] for j in range(5)]
        return hand_delta, [-n for n in hand_delta]
    if kind in COSTS:
        cost = COSTS[kind]
        if kind == "BUILD_ROAD" and before["context"]["free_roads"] > 0:
            cost = ZERO
        return [-n for n in cost], list(cost)
    if kind == "PLAY_KNIGHT_CARD":
        return ZERO.copy(), ZERO.copy()
    return None, None


def resource_premise(
    action: dict[str, Any],
    *,
    ordinal: int,
    root_menu: list[dict[str, Any]],
    bank_lower_before: list[int],
    cumulative_bank_delta: list[int],
    earlier_buys: int,
    public_vp_deck_lower_bound: int,
) -> dict[str, Any]:
    kind = action["kind"]
    if kind == "MARITIME_TRADE":
        receive = action["receive"]
        receipt_indices = [j for j, count in enumerate(receive) if count]
        root_legal = ordinal == 0 and action in root_menu
        checks = []
        for j in receipt_indices:
            known_at_root = any(
                candidate["kind"] == "MARITIME_TRADE" and candidate["receive"][j] >= receive[j]
                for candidate in root_menu
            )
            root_menu_still_guarantees = known_at_root and cumulative_bank_delta[j] >= 0
            public_bound_guarantees = bank_lower_before[j] >= receive[j]
            checks.append(
                {
                    "resource": RESOURCES[j],
                    "requested": receive[j],
                    "public_bank_lower_before": bank_lower_before[j],
                    "saved_root_menu_has_receipt": known_at_root,
                    "prior_steps_bank_delta_for_resource": cumulative_bank_delta[j],
                    "root_menu_still_guarantees": root_menu_still_guarantees,
                    "public_lower_bound_guarantees": public_bound_guarantees,
                    "guaranteed": root_legal
                    or root_menu_still_guarantees
                    or public_bound_guarantees,
                }
            )
        return {
            "kind": "bank_receipt",
            "checks": checks,
            "independently_proven": all(check["guaranteed"] for check in checks),
        }
    if kind == "BUY_DEVELOPMENT_CARD":
        root_menu_proves_nonempty = (
            "BUY_DEVELOPMENT_CARD" in {a["kind"] for a in root_menu} and earlier_buys == 0
        )
        # VP cards cannot be played or discarded in this pinned engine.
        # Every VP card missing from all current hands must still be in deck.
        deck_known_nonempty = root_menu_proves_nonempty or public_vp_deck_lower_bound > 0
        return {
            "kind": "development_deck",
            "saved_root_menu_proves_nonempty_deck": root_menu_proves_nonempty,
            "unplayable_vp_cards_remaining_lower_bound": public_vp_deck_lower_bound,
            "independently_proven": ordinal == 0 or deck_known_nonempty,
        }
    return {"kind": "none", "independently_proven": True}


def execute_exact_step(
    game: FourPlayerGame,
    action: dict[str, Any],
    *,
    ordinal: int,
    bank_lower_before: list[int],
    bank_upper_before: list[int],
    cumulative_bank_delta: list[int],
    root_menu: list[dict[str, Any]],
    earlier_buys: int,
) -> tuple[dict[str, Any], bool]:
    pre_view = game.observe()
    pre = normalize(pre_view)
    vp_deck_lower_bound = max(
        0,
        pre["rules"]["development_cards"][4]
        - sum(player["development_count"] for player in pre["players"]),
    )
    menu = [normalize(candidate) for candidate in pre_view.legal_actions]
    exact_indices = [i for i, candidate in enumerate(menu) if candidate == action]
    record: dict[str, Any] = {
        "ordinal": ordinal,
        "action": action,
        "pre_step_ordered_menu_count": len(menu),
        "pre_step_ordered_menu_sha256": canonical_sha256(menu),
        "exact_action_match_count": len(exact_indices),
        "public_checks": public_action_checks(pre, action),
        "resource_premise": resource_premise(
            action,
            ordinal=ordinal,
            root_menu=root_menu,
            bank_lower_before=bank_lower_before,
            cumulative_bank_delta=cumulative_bank_delta,
            earlier_buys=earlier_buys,
            public_vp_deck_lower_bound=vp_deck_lower_bound,
        ),
        "hand_before": pre["hand"],
        "public_bank_lower_before": bank_lower_before,
        "public_bank_upper_before": bank_upper_before,
        "synthetic_bank_before": list(game.game.state.resource_freqdeck),
    }
    if len(exact_indices) != 1:
        record["error"] = "exact action is absent or duplicated in its own ordered legal menu"
        return record, False
    hand_delta, bank_delta = deterministic_deltas(action, pre)
    if hand_delta is None or bank_delta is None:
        record["error"] = "checker has no deterministic resource accounting for this action"
        return record, False
    actor = pre["player_id"]
    pre_player = pre["players"][actor]
    pre_dev = pre["development"]
    expected_hand = [pre["hand"][j] + hand_delta[j] for j in range(5)]
    expected_bank = [game.game.state.resource_freqdeck[j] + bank_delta[j] for j in range(5)]
    try:
        result = game.step(pre_view.legal_actions[exact_indices[0]])
    except Exception as exc:  # noqa: BLE001 - failed engine step is unverified, never certified
        record["error"] = f"pinned simulator execution failed: {type(exc).__name__}: {exc}"
        return record, False
    post = normalize(result.observation)
    post_menu = post["legal_actions"]
    post_player = post["players"][actor]
    actual_bank = list(game.game.state.resource_freqdeck)
    accounting_equal = post["hand"] == expected_hand and actual_bank == expected_bank
    record.update(
        {
            "expected_hand_after": expected_hand,
            "hand_after": post["hand"],
            "hand_delta": [post["hand"][j] - pre["hand"][j] for j in range(5)],
            "expected_bank_after": expected_bank,
            "synthetic_bank_after": actual_bank,
            "bank_delta": [actual_bank[j] - record["synthetic_bank_before"][j] for j in range(5)],
            "resource_accounting_equal": accounting_equal,
            "public_bank_lower_after": [bank_lower_before[j] + bank_delta[j] for j in range(5)],
            "public_bank_upper_after": [bank_upper_before[j] + bank_delta[j] for j in range(5)],
            "public_vp_before": pre_player["victory_points"],
            "public_vp_after": post_player["victory_points"],
            "public_vp_delta": post_player["victory_points"] - pre_player["victory_points"],
            "roads_left_delta": post_player["roads_left"] - pre_player["roads_left"],
            "settlements_left_delta": post_player["settlements_left"]
            - pre_player["settlements_left"],
            "cities_left_delta": post_player["cities_left"] - pre_player["cities_left"],
            "longest_road_before": pre_player["longest_road"],
            "longest_road_after": post_player["longest_road"],
            "had_longest_road_before": pre_player["has_longest_road"],
            "has_longest_road_after": post_player["has_longest_road"],
            "played_knights_before": pre_player["played_knights"],
            "played_knights_after": post_player["played_knights"],
            "own_development_before": pre_dev,
            "own_development_after": post["development"],
            "post_step_ordered_legal_menu": post_menu,
            "post_step_ordered_legal_menu_count": len(post_menu),
            "post_step_ordered_legal_menu_sha256": canonical_sha256(post_menu),
            "post_step_context": post["context"],
            "terminated": result.terminated,
            "winner": result.winner,
        }
    )
    piece_and_card_equal = True
    kind = action["kind"]
    if kind == "BUILD_ROAD":
        piece_and_card_equal = record["roads_left_delta"] == -1
    elif kind == "BUILD_SETTLEMENT":
        piece_and_card_equal = record["settlements_left_delta"] == -1
    elif kind == "BUILD_CITY":
        piece_and_card_equal = (
            record["cities_left_delta"] == -1 and record["settlements_left_delta"] == 1
        )
    elif kind == "BUY_DEVELOPMENT_CARD":
        piece_and_card_equal = sum(post["development"]) == sum(pre_dev) + 1
    elif kind == "PLAY_KNIGHT_CARD":
        piece_and_card_equal = (
            post["development"][0] == pre_dev[0] - 1
            and record["played_knights_after"] == record["played_knights_before"] + 1
            and post["context"]["has_played_development_card"]
        )
    record["piece_and_card_accounting_equal"] = piece_and_card_equal
    if not accounting_equal:
        record["error"] = "simulator hand or bank differs from deterministic resource arithmetic"
        return record, False
    if not piece_and_card_equal:
        record["error"] = "simulator piece or card delta differs from the action rule"
        return record, False
    if not record["public_checks"]["passes"]:
        record["error"] = "engine-legal action conflicts with checker public predicate"
        return record, False
    return record, True


def audit_claim(
    claim: dict[str, Any],
    *,
    post_root_game: FourPlayerGame,
    root_step: dict[str, Any],
    root_action: dict[str, Any],
    root_menu: list[dict[str, Any]],
    before_public_vp: int,
) -> dict[str, Any]:
    contradictions: list[str] = []
    unverified: list[str] = list(claim["parse_errors"])
    branch_records = []
    for branch in claim["branches"]:
        game = copy.deepcopy(post_root_game)
        lower = list(root_step["public_bank_lower_after"])
        upper = list(root_step["public_bank_upper_after"])
        cumulative = list(root_step["bank_delta"])
        earlier_buys = int(root_action["kind"] == "BUY_DEVELOPMENT_CARD")
        steps = []
        branch_contradictions = []
        branch_unverified = []
        for ordinal, action in enumerate(branch["actions"], start=1):
            if game._terminated or game._truncated:
                branch_contradictions.append(
                    f"continuation step {ordinal} follows a terminated game"
                )
                break
            step, succeeded = execute_exact_step(
                game,
                action,
                ordinal=ordinal,
                bank_lower_before=lower,
                bank_upper_before=upper,
                cumulative_bank_delta=cumulative,
                root_menu=root_menu,
                earlier_buys=earlier_buys,
            )
            steps.append(step)
            if not step["resource_premise"]["independently_proven"]:
                branch_unverified.append(
                    f"continuation step {ordinal} lacks a public bank or deck availability premise"
                )
            if not succeeded:
                if (
                    step["exact_action_match_count"] == 0
                    and step["resource_premise"]["independently_proven"]
                    and step["public_checks"]["passes"]
                ):
                    branch_contradictions.append(
                        f"continuation step {ordinal} is absent from its exact post-step legal menu"
                    )
                elif not step["public_checks"]["passes"]:
                    branch_contradictions.append(
                        f"continuation step {ordinal} fails public hand, site, or piece checks"
                    )
                else:
                    branch_unverified.append(step.get("error", "continuation step did not replay"))
                break
            lower = list(step["public_bank_lower_after"])
            upper = list(step["public_bank_upper_after"])
            cumulative = [cumulative[j] + step["bank_delta"][j] for j in range(5)]
            earlier_buys += int(action["kind"] == "BUY_DEVELOPMENT_CARD")
        completed = len(steps) == len(branch["actions"]) and all(
            step.get("resource_accounting_equal")
            and step.get("piece_and_card_accounting_equal")
            and step["exact_action_match_count"] == 1
            for step in steps
        )
        if not completed and not branch_contradictions and not branch_unverified:
            branch_unverified.append("continuation did not finish")
        final_view = normalize(game.observe())
        actor = final_view["player_id"]
        final_player = final_view["players"][actor]
        gain = final_player["victory_points"] - before_public_vp
        checks: dict[str, Any] = {
            "final_public_vp": final_player["victory_points"],
            "public_vp_gain_from_before_root": gain,
            "final_longest_road_length": final_player["longest_road"],
            "final_has_longest_road": final_player["has_longest_road"],
        }
        if branch["name"] == "main" and completed:
            build_count = sum(
                action["kind"] in ("BUILD_SETTLEMENT", "BUILD_CITY")
                for action in [root_action, *branch["actions"]]
            )
            checks["intrinsic_point_build_count"] = build_count
            if gain < build_count:
                branch_contradictions.append(
                    f"completed point builds imply at least {build_count} public points, observed {gain}"
                )
            claimed_gain = claim["claimed_public_vp_gain"]
            if claimed_gain is not None:
                checks["claimed_public_vp_gain"] = claimed_gain
                checks["claimed_public_vp_gain_equal"] = gain == claimed_gain
                if gain != claimed_gain:
                    branch_contradictions.append(
                        f"claimed public victory-point gain {claimed_gain}, observed {gain}"
                    )
            claimed_length = claim["claimed_longest_road_length"]
            if claimed_length is not None:
                checks["claimed_longest_road_length"] = claimed_length
                checks["claimed_longest_road_length_equal"] = (
                    final_player["longest_road"] == claimed_length
                )
                if final_player["longest_road"] != claimed_length:
                    branch_contradictions.append(
                        f"claimed Longest Road length {claimed_length}, observed {final_player['longest_road']}"
                    )
            if claim["claimed_longest_road_award"]:
                checks["claimed_longest_road_award_gained"] = (
                    not root_step["had_longest_road_before"] and final_player["has_longest_road"]
                )
                if not checks["claimed_longest_road_award_gained"]:
                    branch_contradictions.append("claimed Longest Road award was not gained")
            if claim["claimed_port"] is not None:
                settlement_targets = [
                    action["target"]
                    for action in [root_action, *branch["actions"]]
                    if action["kind"] == "BUILD_SETTLEMENT"
                ]
                if len(settlement_targets) != 1:
                    branch_unverified.append(
                        "claimed port cannot be assigned to one exact settlement vertex"
                    )
                else:
                    vertex = settlement_targets[0]
                    has_port = active_port_at_vertex(final_view, vertex, claim["claimed_port"])
                    checks["claimed_port"] = claim["claimed_port"]
                    checks["claimed_port_vertex"] = vertex
                    checks["claimed_port_acquired"] = has_port
                    if not has_port:
                        branch_contradictions.append(
                            f"claimed port is not held at settlement vertex {vertex}"
                        )
            if claim["claimed_edge_endpoints"] is not None:
                road_targets = [
                    action["target"]
                    for action in [root_action, *branch["actions"]]
                    if action["kind"] == "BUILD_ROAD"
                ]
                if len(road_targets) != 1:
                    branch_unverified.append(
                        "claimed road endpoints cannot be assigned to one exact road edge"
                    )
                else:
                    edge = road_targets[0]
                    actual_endpoints = final_view["board"]["edges"][edge]
                    equal = set(actual_endpoints) == set(claim["claimed_edge_endpoints"])
                    checks["claimed_edge_endpoints"] = claim["claimed_edge_endpoints"]
                    checks["actual_edge_endpoints"] = actual_endpoints
                    checks["claimed_edge_endpoints_equal"] = equal
                    if not equal:
                        branch_contradictions.append(
                            f"claimed road edge {edge} endpoints disagree with public board"
                        )
        branch_status = (
            "contradicted"
            if branch_contradictions
            else "unverified"
            if branch_unverified
            else "engine_verified"
        )
        branch_records.append(
            {
                "name": branch["name"],
                "status": branch_status,
                "expected_followup_actions": branch["actions"],
                "all_followups_replayed": completed,
                "followup_steps": steps,
                "checks": checks,
                "contradictions": branch_contradictions,
                "unverified_reasons": branch_unverified,
            }
        )
        contradictions.extend(branch_contradictions)
        unverified.extend(branch_unverified)
    status = "contradicted" if contradictions else "unverified" if unverified else "engine_verified"
    return {
        "text": claim["text"],
        "status": status,
        "parse_errors": claim["parse_errors"],
        "claimed_public_vp_gain": claim["claimed_public_vp_gain"],
        "claimed_longest_road_length": claim["claimed_longest_road_length"],
        "claimed_longest_road_award": claim["claimed_longest_road_award"],
        "claimed_port": claim["claimed_port"],
        "claimed_edge_endpoints": claim["claimed_edge_endpoints"],
        "branches": branch_records,
        "contradictions": contradictions,
        "unverified_reasons": unverified,
    }


def audit_case(case: dict[str, Any]) -> dict[str, Any]:
    observation = case["before"]["observation"]
    root_menu = observation["legal_actions"]
    root_index = case["root_index"]
    before_hash_equal = canonical_sha256(case["before"]) == case["observation_sha256"]
    menu_hash_equal = canonical_sha256(root_menu) == case["menu_sha256"]
    index_valid = type(root_index) is int and 0 <= root_index < len(root_menu)
    action_equal = index_valid and root_menu[root_index] == case["root_action"]
    bank_lower, bank_upper = public_bank_bounds(observation)
    record: dict[str, Any] = {
        "packet_id": case["packet_id"],
        "root_index": root_index,
        "root_action": case["root_action"],
        "before_observation_sha256": case["observation_sha256"],
        "saved_root_menu_sha256": case["menu_sha256"],
        "saved_before_hash_equal": before_hash_equal,
        "saved_root_menu_hash_equal": menu_hash_equal,
        "root_index_valid": index_valid,
        "root_action_equals_indexed_menu_action": action_equal,
        "saved_root_menu_count": len(root_menu),
        "exact_root_menu_equal": False,
        "engine_menu_verified": False,
        "public_bank_bounds_before": {
            "lower": bank_lower,
            "upper": bank_upper,
        },
        "synthetic_setup": None,
        "root_step": None,
        "replayed_root_menu_sha256": None,
        "post_root_menu_sha256": None,
        "post_root_menu": None,
        "post_root_menu_scope": "synthetic public-consistent state, not source hidden-state replay",
        "explicit_claim_checks": [],
        "contradictions": [],
        "unverified_reasons": [],
        "scope": "Factual feasibility of this exact root only; no strategic preference or original private-state replay.",
    }
    if not before_hash_equal or not menu_hash_equal:
        record["unverified_reasons"].append("saved public observation or root-menu hash mismatch")
    if not index_valid:
        record["contradictions"].append("root index is outside the saved ordered legal menu")
    elif not action_equal:
        record["contradictions"].append("root action differs from saved action at exact index")
    if record["contradictions"] or record["unverified_reasons"]:
        record["status"] = "contradicted" if record["contradictions"] else "unverified"
        return record
    game, setup = reconstruct(observation)
    record["synthetic_setup"] = setup
    if game is None:
        record["unverified_reasons"].append(
            setup.get("reason", "exact synthetic state could not be constructed")
        )
        record["status"] = "unverified"
        return record
    record["exact_root_menu_equal"] = setup["exact_ordered_root_menu_equal"]
    record["replayed_root_menu_sha256"] = setup["actual_root_menu_sha256"]
    record["engine_menu_verified"] = (
        setup["exact_ordered_root_menu_equal"]
        and setup["full_public_board_equal"]
        and setup["core_public_view_equal"]
    )
    if not record["engine_menu_verified"]:
        record["unverified_reasons"].append("full ordered root menu or core public view differs")
        record["status"] = "unverified"
        return record
    root_step, root_ok = execute_exact_step(
        game,
        case["root_action"],
        ordinal=0,
        bank_lower_before=bank_lower,
        bank_upper_before=bank_upper,
        cumulative_bank_delta=ZERO.copy(),
        root_menu=root_menu,
        earlier_buys=0,
    )
    record["root_step"] = root_step
    if not root_ok:
        record["unverified_reasons"].append(
            root_step.get("error", "indexed root failed pinned simulator execution")
        )
        record["status"] = "unverified"
        return record
    record["post_root_menu_sha256"] = root_step["post_step_ordered_legal_menu_sha256"]
    record["post_root_menu"] = root_step["post_step_ordered_legal_menu"]
    record["root_hand_delta"] = root_step["hand_delta"]
    record["root_public_vp_delta"] = root_step["public_vp_delta"]
    record["root_longest_road_delta"] = (
        root_step["longest_road_after"] - root_step["longest_road_before"]
    )
    record["root_played_knights_delta"] = (
        root_step["played_knights_after"] - root_step["played_knights_before"]
    )
    if not root_step["resource_premise"]["independently_proven"]:
        record["unverified_reasons"].append("root action has an unproven resource premise")
    if case["root_action"]["kind"] == "PLAY_KNIGHT_CARD":
        pre_player = observation["players"][observation["player_id"]]
        post_player = normalize(game.observe())["players"][observation["player_id"]]
        knight_checks = {
            "knight_card_decremented": root_step["own_development_after"][0]
            == root_step["own_development_before"][0] - 1,
            "played_knight_count_incremented": root_step["played_knights_after"]
            == root_step["played_knights_before"] + 1,
            "post_root_context_is_move_robber": root_step["post_step_context"]["kind"]
            == "MOVE_ROBBER",
            "public_vp_delta": root_step["public_vp_delta"],
            "has_largest_army_before": pre_player["has_largest_army"],
            "has_largest_army_after": post_player["has_largest_army"],
            "rival_largest_army_holders": [
                p["player_id"] for p in observation["players"] if p["has_largest_army"]
            ],
        }
        record["played_card_checks"] = knight_checks
        if not all(
            knight_checks[key]
            for key in (
                "knight_card_decremented",
                "played_knight_count_incremented",
                "post_root_context_is_move_robber",
            )
        ):
            record["unverified_reasons"].append("played Knight transition differs from pinned rule")
    before_public_vp = observation["players"][observation["player_id"]]["victory_points"]
    for claim in claims_for_case(case):
        result = audit_claim(
            claim,
            post_root_game=game,
            root_step=root_step,
            root_action=case["root_action"],
            root_menu=root_menu,
            before_public_vp=before_public_vp,
        )
        record["explicit_claim_checks"].append(result)
        record["contradictions"].extend(result["contradictions"])
        record["unverified_reasons"].extend(result["unverified_reasons"])
    record["status"] = (
        "contradicted"
        if record["contradictions"]
        else "unverified"
        if record["unverified_reasons"]
        else "engine_verified"
    )
    return record


def build_report(input_path: Path, expected_sha256: str) -> dict[str, Any]:
    input_bytes = input_path.read_bytes()
    actual_sha256 = hashlib.sha256(input_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"public fact input SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    source = json.loads(input_bytes)
    identity = environment_identity()
    if source["resource_order"] != list(RESOURCES):
        raise ValueError("public input resource order differs from pinned simulator")
    if source.get("source_files_sha256") != identity["source_files_sha256"]:
        raise ValueError("public input source file hashes differ from current checkout")
    if "source_head" in source and source["source_head"] != identity["source_head"]:
        raise ValueError("public input source commit differs from current checkout")
    if source.get("engine") != identity["engine"]["backend_version"]:
        raise ValueError("public input names a different Catanatron revision")
    cases = source["cases"]
    keys = [(case["packet_id"], case["root_index"]) for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("public input repeats a packet_id/root_index pair")
    records = [audit_case(case) for case in cases]
    if [(r["packet_id"], r["root_index"]) for r in records] != keys:
        raise AssertionError("fact report lost or reordered an input root")
    statuses = Counter(record["status"] for record in records)
    report: dict[str, Any] = {
        "format": "colonist-rl.public-fact-replay/1",
        "fact_input_path": str(input_path.resolve()),
        "fact_input_sha256": actual_sha256,
        "checker_path": str(Path(__file__).resolve()),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "identity": identity,
        "resource_order": list(RESOURCES),
        "status_vocabulary": [
            "engine_verified",
            "rule_derived_feasible",
            "contradicted",
            "unverified",
        ],
        "method": (
            "For each exact root: reconstruct a public-consistent synthetic pinned-engine "
            "state; require full board, core public view, and entire ordered current legal "
            "menu equality; execute that indexed root; capture and hash its complete "
            "post-root legal menu; replay only explicitly named immediate actions from "
            "their own menus; check resource, piece, point, award, and public bank/deck "
            "premises. A synthetic witness establishes feasibility only."
        ),
        "summary": {
            "case_count": len(records),
            "distinct_packet_count": len({r["packet_id"] for r in records}),
            "exact_root_menu_equal_count": sum(r["exact_root_menu_equal"] for r in records),
            "post_root_menu_recorded_count": sum(
                r["root_step"] is not None and "post_step_ordered_legal_menu" in r["root_step"]
                for r in records
            ),
            "status_counts": dict(statuses),
            "unverified_pairs": [
                [r["packet_id"], r["root_index"]] for r in records if r["status"] == "unverified"
            ],
            "contradicted_pairs": [
                [r["packet_id"], r["root_index"]] for r in records if r["status"] == "contradicted"
            ],
        },
        "cases": records,
    }
    report["canonical_payload_sha256"] = canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity", action="store_true", help="print current source and engine binding"
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.identity:
        if (
            args.input is not None
            or args.expected_sha256 is not None
            or args.output is not None
            or args.write
        ):
            parser.error("--identity takes no report arguments")
        identity = environment_identity()
        print(
            json.dumps(
                {
                    "engine": identity["engine"]["backend_version"],
                    "resource_order": list(RESOURCES),
                    "source_files_sha256": identity["source_files_sha256"],
                    "source_head": identity["source_head"],
                },
                sort_keys=True,
            )
        )
        return
    if args.input is None or args.expected_sha256 is None:
        parser.error("--input and --expected-sha256 are required for a report")
    if args.write and args.output is None:
        parser.error("--write requires --output")
    report = build_report(args.input, args.expected_sha256.lower())
    if args.output is not None and args.write:
        if args.output.is_symlink():
            raise ValueError("report output cannot be a symbolic link")
        rendered = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if args.output.exists():
            if args.output.read_bytes() != rendered:
                raise ValueError("existing report differs from fresh public-only replay")
        else:
            with args.output.open("xb") as file:
                file.write(rendered)
        print("REPORT_SHA256", hashlib.sha256(args.output.read_bytes()).hexdigest())
    print("FACT_INPUT_SHA256", report["fact_input_sha256"])
    print("CHECKER_SHA256", report["checker_sha256"])
    print("SUMMARY", json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
