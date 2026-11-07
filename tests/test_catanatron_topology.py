"""Corrected road topology against the pinned Catanatron board."""

import copy
import hashlib
import importlib.util
from collections import defaultdict
from pathlib import Path

import pytest
from catanatron.models.board import Board
from catanatron.models.player import Color
from catanatron.state_functions import maintain_longest_road


def corrected_board():
    assert importlib.util.find_spec("colonist_rl.catanatron_topology") is not None
    from colonist_rl.catanatron_topology import CorrectedBoard

    return CorrectedBoard.from_board(Board())


def _place(board, kind, color, target, initial=False):
    if kind == "settlement":
        assert target in board.buildable_node_ids(color, initial_build_phase=initial)
        board.build_settlement(color, target, initial_build_phase=initial)
    else:
        assert target in board.buildable_edges(color)
        board.build_road(color, target)


def test_enemy_settlement_blocks_road_expansion_from_cached_endpoint():
    board = corrected_board()
    for kind, color, target, initial in (
        ("settlement", Color.RED, 10, True),
        ("road", Color.RED, (9, 10), False),
        ("settlement", Color.ORANGE, 4, True),
        ("road", Color.ORANGE, (3, 4), False),
        ("road", Color.RED, (2, 9), False),
        ("road", Color.ORANGE, (2, 3), False),
        ("settlement", Color.ORANGE, 2, False),
    ):
        _place(board, kind, color, target, initial)
    assert board.is_enemy_node(2, Color.RED)
    assert (1, 2) not in board.buildable_edges(Color.RED)

    # The original pinned backend offers this through-opponent extension.
    old = Board()
    for kind, color, target, initial in (
        ("settlement", Color.RED, 10, True),
        ("road", Color.RED, (9, 10), False),
        ("settlement", Color.ORANGE, 4, True),
        ("road", Color.ORANGE, (3, 4), False),
        ("road", Color.RED, (2, 9), False),
        ("road", Color.ORANGE, (2, 3), False),
        ("settlement", Color.ORANGE, 2, False),
    ):
        _place(old, kind, color, target, initial)
    assert (1, 2) in old.buildable_edges(Color.RED)


def test_roads_arriving_at_both_sides_of_enemy_settlement_remain_separate():
    board = corrected_board()
    for kind, color, target, initial in (
        ("settlement", Color.RED, 10, True),
        ("road", Color.RED, (9, 10), False),
        ("settlement", Color.RED, 0, True),
        ("road", Color.RED, (0, 1), False),
        ("settlement", Color.ORANGE, 4, True),
        ("road", Color.ORANGE, (3, 4), False),
        ("road", Color.RED, (2, 9), False),
        ("road", Color.ORANGE, (2, 3), False),
        ("settlement", Color.ORANGE, 2, False),
        ("road", Color.RED, (1, 2), False),
    ):
        _place(board, kind, color, target, initial)
    red = [set(component) for component in board.connected_components[Color.RED]]
    assert len(red) == 2
    assert any({0, 1}.issubset(component) and 9 not in component for component in red)
    assert any({9, 10}.issubset(component) and 0 not in component for component in red)


def test_conversion_repairs_stale_component_before_a_second_enemy_cut():
    # This is the excluded 19-action backend failure, stopped before the
    # second cut. The repaired board must preserve the remote road segment.
    source = Board()
    for kind, color, target, initial in (
        ("settlement", Color.RED, 10, True),
        ("road", Color.RED, (9, 10), False),
        ("settlement", Color.ORANGE, 4, True),
        ("road", Color.ORANGE, (3, 4), False),
        ("settlement", Color.BLUE, 22, True),
        ("road", Color.BLUE, (22, 23), False),
        ("road", Color.RED, (2, 9), False),
        ("road", Color.ORANGE, (2, 3), False),
        ("settlement", Color.ORANGE, 2, False),
        ("road", Color.RED, (1, 2), False),
        ("road", Color.RED, (1, 6), False),
        ("road", Color.RED, (6, 7), False),
        ("road", Color.RED, (10, 11), False),
        ("road", Color.RED, (11, 12), False),
        ("road", Color.RED, (12, 13), False),
        ("road", Color.BLUE, (6, 23), False),
    ):
        _place(source, kind, color, target, initial)
    assert source._get_connected_component_index(
        12, Color.RED
    ) == source._get_connected_component_index(6, Color.RED)

    from colonist_rl.catanatron_topology import CorrectedBoard

    board = CorrectedBoard.from_board(source)
    assert board._get_connected_component_index(
        12, Color.RED
    ) != board._get_connected_component_index(6, Color.RED)
    _place(board, "settlement", Color.BLUE, 6)
    assert board._get_connected_component_index(12, Color.RED) is not None
    assert board.roads[(11, 12)] == board.roads[(12, 13)] == Color.RED


def test_longest_road_counts_final_edge_into_enemy_settlement():
    board = corrected_board()
    for kind, color, target, initial in (
        ("settlement", Color.RED, 10, True),
        ("road", Color.RED, (9, 10), False),
        ("settlement", Color.ORANGE, 25, True),
        ("road", Color.ORANGE, (24, 25), False),
        ("road", Color.ORANGE, (7, 24), False),
        ("settlement", Color.ORANGE, 7, False),
        ("road", Color.RED, (2, 9), False),
        ("road", Color.RED, (1, 2), False),
        ("road", Color.RED, (1, 6), False),
        ("road", Color.RED, (6, 7), False),
    ):
        _place(board, kind, color, target, initial)
    assert board.road_lengths[Color.RED] == 5
    assert board.road_color == Color.RED


def test_enemy_settlement_cut_recomputes_lengths_and_removes_subfive_award():
    board = corrected_board()
    for kind, color, target, initial in (
        ("settlement", Color.RED, 10, True),
        ("road", Color.RED, (9, 10), False),
        ("settlement", Color.BLUE, 22, True),
        ("road", Color.BLUE, (22, 23), False),
        ("road", Color.RED, (2, 9), False),
        ("road", Color.RED, (1, 2), False),
        ("road", Color.RED, (1, 6), False),
        ("road", Color.RED, (6, 7), False),
        ("road", Color.BLUE, (6, 23), False),
    ):
        _place(board, kind, color, target, initial)
    assert board.road_lengths[Color.RED] == 5
    assert board.road_color == Color.RED
    _place(board, "settlement", Color.BLUE, 6)
    assert board.road_lengths[Color.RED] == 4
    assert board.road_color is None


def test_longest_road_tie_retains_incumbent_and_unheld_tie_stays_unheld():
    source = Board()
    red_path = ((9, 10), (2, 9), (1, 2), (1, 6), (6, 7))
    blue_path = ((22, 23), (23, 52), (52, 53), (24, 53), (24, 25))
    for color, path in ((Color.RED, red_path), (Color.BLUE, blue_path)):
        for edge in path:
            source.roads[edge] = color
            source.roads[edge[::-1]] = color
    from colonist_rl.catanatron_topology import CorrectedBoard

    source.road_color = Color.RED
    tied = CorrectedBoard.from_board(source)
    assert tied.road_lengths[Color.RED] == tied.road_lengths[Color.BLUE] == 5
    assert tied.road_color == Color.RED

    source.road_color = None
    unheld = CorrectedBoard.from_board(source)
    assert unheld.road_color is None

    source.road_color = Color.RED
    source.roads[(25, 26)] = source.roads[(26, 25)] = Color.BLUE
    overtaken = CorrectedBoard.from_board(source)
    assert overtaken.road_lengths[Color.BLUE] == 6
    assert overtaken.road_color == Color.BLUE


def test_reconciliation_revokes_backend_award_when_no_one_qualifies():
    from colonist_rl.simulator import FourPlayerGame

    env = FourPlayerGame()
    env.reset(seed=7)
    state = env.game.state
    color = state.colors[0]
    key = f"P{state.color_to_index[color]}"
    baseline = (
        state.player_state[f"{key}_VICTORY_POINTS"],
        state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"],
    )
    state.player_state[f"{key}_HAS_ROAD"] = True
    state.player_state[f"{key}_VICTORY_POINTS"] += 2
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 2
    state.board.road_color = None
    state.board.road_lengths = defaultdict(int, {color: 4})
    maintain_longest_road(state, color, None, state.board.road_lengths)
    assert state.player_state[f"{key}_HAS_ROAD"]

    from colonist_rl.catanatron_topology import reconcile_road_award

    reconcile_road_award(state)
    assert not state.player_state[f"{key}_HAS_ROAD"]
    assert state.player_state[f"{key}_VICTORY_POINTS"] == baseline[0]
    assert state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] == baseline[1]
    assert state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] == 4


def test_backend_transfer_is_not_double_counted_by_reconciliation():
    from colonist_rl.catanatron_topology import reconcile_road_award
    from colonist_rl.simulator import FourPlayerGame

    env = FourPlayerGame()
    env.reset(seed=7)
    state = env.game.state
    red, blue = state.colors[:2]
    red_key = f"P{state.color_to_index[red]}"
    blue_key = f"P{state.color_to_index[blue]}"
    before = {
        key: (
            state.player_state[f"{key}_VICTORY_POINTS"],
            state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"],
        )
        for key in (red_key, blue_key)
    }
    state.player_state[f"{red_key}_HAS_ROAD"] = True
    state.player_state[f"{red_key}_VICTORY_POINTS"] += 2
    state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"] += 2
    state.board.road_color = blue
    state.board.road_lengths = defaultdict(int, {red: 5, blue: 6})
    maintain_longest_road(state, red, blue, state.board.road_lengths)
    reconcile_road_award(state)
    reconcile_road_award(state)
    assert not state.player_state[f"{red_key}_HAS_ROAD"]
    assert state.player_state[f"{blue_key}_HAS_ROAD"]
    assert (
        state.player_state[f"{red_key}_VICTORY_POINTS"],
        state.player_state[f"{red_key}_ACTUAL_VICTORY_POINTS"],
    ) == before[red_key]
    assert (
        state.player_state[f"{blue_key}_VICTORY_POINTS"],
        state.player_state[f"{blue_key}_ACTUAL_VICTORY_POINTS"],
    ) == tuple(value + 2 for value in before[blue_key])


def test_opt_in_simulator_has_distinct_identity_and_copyable_board():
    from colonist_rl.catanatron_topology import CorrectedBoard
    from colonist_rl.simulator import BACKEND_VERSION, FourPlayerGame

    legacy = FourPlayerGame()
    corrected = FourPlayerGame(corrected_topology=True)
    original_view = legacy.reset(seed=7)
    corrected_view = corrected.reset(seed=7)
    assert original_view == corrected_view
    assert corrected.backend_version != BACKEND_VERSION
    assert legacy.backend_version == BACKEND_VERSION
    assert isinstance(corrected.game.state.board, CorrectedBoard)
    clone = copy.deepcopy(corrected)
    assert isinstance(clone.game.state.board, CorrectedBoard)
    assert clone.observe() == corrected_view
    assert isinstance(corrected.game.state.copy().board, CorrectedBoard)
    assert isinstance(corrected.game.copy().state.board, CorrectedBoard)


def test_corrected_initial_road_length_is_public_immediately():
    from colonist_rl.simulator import FourPlayerGame

    env = FourPlayerGame(corrected_topology=True)
    view = env.reset(seed=7)
    player = view.player_id
    view = env.step(view.legal_actions[0]).observation
    assert view.context.kind == "BUILD_INITIAL_ROAD"
    view = env.step(view.legal_actions[0]).observation
    color = env.game.state.colors[player]
    length = env.game.state.board.road_lengths[color]
    assert length == 1
    assert env.game.state.player_state[f"P{player}_LONGEST_ROAD_LENGTH"] == length
    assert view.players[player].longest_road == length


def test_simulator_step_cut_revokes_public_and_actual_longest_road_points():
    from catanatron.models.enums import ActionPrompt
    from catanatron.state_functions import build_road as record_road
    from catanatron.state_functions import build_settlement as record_settlement

    from colonist_rl.contracts import RESOURCES
    from colonist_rl.simulator import FourPlayerGame

    env = FourPlayerGame(corrected_topology=True)
    env.reset(seed=7)
    state = env.game.state
    board = state.board
    for kind, color, target, initial in (
        ("settlement", Color.RED, 10, True),
        ("road", Color.RED, (9, 10), False),
        ("settlement", Color.BLUE, 22, True),
        ("road", Color.BLUE, (22, 23), False),
        ("road", Color.RED, (2, 9), False),
        ("road", Color.RED, (1, 2), False),
        ("road", Color.RED, (1, 6), False),
        ("road", Color.RED, (6, 7), False),
        ("road", Color.BLUE, (6, 23), False),
    ):
        _place(board, kind, color, target, initial)
        if kind == "settlement":
            record_settlement(state, color, target, True)
        else:
            record_road(state, color, target, True)
    assert board.road_color == Color.RED
    maintain_longest_road(state, None, board.road_color, board.road_lengths)
    red = state.color_to_index[Color.RED]
    blue = state.color_to_index[Color.BLUE]
    assert state.player_state[f"P{red}_ACTUAL_VICTORY_POINTS"] == 3

    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.current_player_index = state.current_turn_index = blue
    state.player_state[f"P{blue}_HAS_ROLLED"] = True
    before = [env._hand(player) for player in range(4)]
    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT"):
        state.player_state[f"P{blue}_{resource}_IN_HAND"] += 1
        state.resource_freqdeck[RESOURCES.index(resource)] -= 1
    env._record_known_changes(before, "TEST_PUBLIC_GRANT", blue)
    env._refresh_actions()
    view = env.observe()
    selected = next(
        action
        for action in view.legal_actions
        if action.kind == "BUILD_SETTLEMENT" and action.target == env._node_to_vertex[6]
    )
    result = env.step(selected)
    red_public = result.observation.players[red]
    assert red_public.longest_road == 4
    assert not red_public.has_longest_road
    assert red_public.victory_points == 1
    assert env._actual_vp()[red] == 1


def test_corrected_factory_assertion_and_source_digest_bind_role_engine():
    from colonist_rl import catanatron_topology
    from colonist_rl.simulator import FourPlayerGame, corrected_game, require_corrected_game

    corrected = corrected_game(max_steps=123)
    assert corrected.max_steps == 123
    corrected.reset(seed=7)
    assert require_corrected_game(corrected) is corrected
    expected = hashlib.sha256(Path(catanatron_topology.__file__).read_bytes()).hexdigest()
    assert catanatron_topology.topology_shim_sha256() == expected
    assert len(expected) == 64

    legacy = FourPlayerGame()
    legacy.reset(seed=7)
    with pytest.raises(ValueError, match="corrected topology"):
        require_corrected_game(legacy)
    corrected.game.state.board = Board()
    with pytest.raises(ValueError, match="corrected topology"):
        require_corrected_game(corrected)
