"""Topology tests catch duplicated corners, missing shared edges, and fixed-size rules."""

from dataclasses import FrozenInstanceError

import pytest

from colonist_rl.board import BoardSpec, RulesConfig


def test_standard_board_shares_vertices_and_edges():
    board = BoardSpec.standard(seed=7)
    assert (len(board.tiles), len(board.vertices), len(board.edges)) == (19, 54, 72)
    assert len({t.resource for t in board.tiles}) == 6
    assert sum(t.resource == "DESERT" for t in board.tiles) == 1
    assert all(len(vs) == len(es) == 6 for vs, es in zip(board.tile_vertices, board.tile_edges))
    for vs, es in zip(board.tile_vertices, board.tile_edges):
        for i, edge in enumerate(es):
            assert set(board.edges[edge]) == {vs[i], vs[(i + 1) % 6]}
    assert len(board.ports) == 9
    assert all(
        p.vertices in board.edges or tuple(reversed(p.vertices)) in board.edges for p in board.ports
    )
    assert board == BoardSpec.standard(seed=7)


def test_two_neighboring_hexes_share_one_edge_and_keep_input_order():
    board = BoardSpec.from_hexes([(0, 0), (1, 0)], ["WOOD", "BRICK"], [6, 8])
    assert (len(board.vertices), len(board.edges)) == (10, 11)
    assert [(t.q, t.r, t.resource, t.number) for t in board.tiles] == [
        (0, 0, "WOOD", 6),
        (1, 0, "BRICK", 8),
    ]
    assert len(set(board.tile_vertices[0]) & set(board.tile_vertices[1])) == 2
    assert len(set(board.tile_edges[0]) & set(board.tile_edges[1])) == 1
    with pytest.raises(FrozenInstanceError):
        board.tiles = ()
    with pytest.raises(TypeError):
        board.tile_vertices[0][0] = 12


def test_extended_board_has_thirty_tiles_and_variable_topology():
    board = BoardSpec.extended(seed=2)
    assert (len(board.tiles), len(board.vertices), len(board.edges)) == (30, 80, 109)
    assert sorted(sum(t.r == r for t in board.tiles) for r in {t.r for t in board.tiles}) == [
        3,
        3,
        4,
        4,
        5,
        5,
        6,
    ]
    assert sum(t.resource == "DESERT" for t in board.tiles) == 2
    assert len(board.ports) == 11


def test_malformed_topology_and_tile_metadata_are_rejected():
    with pytest.raises(ValueError):
        BoardSpec.from_hexes([(0, 0), (0, 0)])
    with pytest.raises(ValueError):
        BoardSpec.from_hexes([(0, 0)], resources=["WOOD", "BRICK"])
    with pytest.raises(ValueError):
        BoardSpec.from_hexes([(0, 0)], resources=["GOLD"])
    with pytest.raises(ValueError):
        BoardSpec.from_hexes([(0, 0)], resources=["WOOD"], numbers=[7])


def test_rule_profiles_support_player_counts_and_extended_supply():
    standard = RulesConfig.standard(player_count=3)
    extended = RulesConfig.extended(player_count=5)
    assert standard.player_count == 3
    assert standard.bank_resources == (19, 19, 19, 19, 19)
    assert sum(standard.development_cards) == 25
    assert not standard.special_build
    assert extended.player_count == 5
    assert extended.bank_resources == (24, 24, 24, 24, 24)
    assert sum(extended.development_cards) == 34
    assert extended.special_build
    assert "special_build" in extended.supported_phases
    assert extended.version
    with pytest.raises(ValueError):
        RulesConfig.standard(player_count=0)


def test_direct_board_construction_copies_mutable_inputs():
    original = BoardSpec.from_hexes([(0, 0)])
    vertices = [list(v) for v in original.vertices]
    tile_vertices = [list(v) for v in original.tile_vertices]
    clone = BoardSpec(
        list(original.tiles),
        vertices,
        list(original.edges),
        tile_vertices,
        list(original.tile_edges),
    )
    vertices[0][0] = 999
    tile_vertices[0][0] = 999
    assert clone == original


def test_generated_board_red_tokens_do_not_touch():
    for factory in (BoardSpec.standard, BoardSpec.extended):
        board = factory(seed=11)
        red_tiles = [t.id for t in board.tiles if t.number in (6, 8)]
        for i, tile in enumerate(red_tiles):
            for other in red_tiles[i + 1 :]:
                assert not (set(board.tile_edges[tile]) & set(board.tile_edges[other]))


def test_port_endpoints_cannot_change_after_construction():
    from colonist_rl.board import Port

    endpoints = [0, 1]
    port = Port(endpoints)
    endpoints[0] = 2
    assert port.vertices == (0, 1)
