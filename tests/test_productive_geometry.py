"""Hand-checked synthetic public geometry; no games or recorded EVAL states."""

import importlib.util
import math
from dataclasses import replace

import pytest
from test_public_turn_planner import chain_view, replay, view

from colonist_rl.board import BoardSpec, Port, Tile
from colonist_rl.contracts import Action, Building, Road
from colonist_rl.public_turn_planner import PlannedStep


def encode(observation, projected=None):
    assert importlib.util.find_spec("colonist_rl.productive_geometry") is not None, (
        "productive geometry runtime missing"
    )
    from colonist_rl.productive_geometry import encode_productive_geometry

    return encode_productive_geometry(observation, projected)


def test_connected_sites_obey_spacing_and_road_extensions_respect_opponent_blocking():
    observation = replace(view(), roads=(Road(0, 10),))
    result = encode(observation)
    assert result["pg_connected_sites"] == 0
    assert result["pg_connected_best_pips"] == 0
    assert result["pg_one_road_new_sites"] == 1  # vertex 2; vertex 5 neighbors own 0
    assert result["pg_one_road_best_pips"] == 5
    assert result["pg_expansion_edges_without_site"] == 1
    blocked = replace(observation, buildings=(*observation.buildings, Building(1, 20)))
    assert encode(blocked)["pg_one_road_new_sites"] == 0
    assert encode(blocked)["pg_expansion_edges_without_site"] == 1
    connected = replace(observation, roads=(Road(0, 10), Road(1, 10)))
    assert encode(connected)["pg_connected_sites"] == 1
    assert encode(connected)["pg_one_road_new_sites"] == 1  # vertex 3, not existing 2
    occupied = replace(connected, buildings=(*connected.buildings, Building(3, 20)))
    assert encode(occupied)["pg_connected_sites"] == 0
    assert encode(occupied)["pg_one_road_new_sites"] == 0


def test_roads_and_settlement_stock_gate_future_geometry_without_reusing_root_menu():
    observation = replace(view(), roads=(Road(0, 10), Road(1, 10)))
    result = encode(observation)
    assert encode(replace(observation, legal_actions=())) == result
    no_roads = replace(observation.players[0], roads_left=0)
    after = encode(replace(observation, players=(no_roads, *observation.players[1:])))
    assert after["pg_connected_sites"] == 1
    assert after["pg_one_road_new_sites"] == after["pg_expansion_edges_without_site"] == 0
    no_settlements = replace(observation.players[0], settlements_left=0)
    after = encode(replace(observation, players=(no_settlements, *observation.players[1:])))
    assert after["pg_connected_sites"] == after["pg_one_road_new_sites"] == 0
    assert after["pg_connected_best_pips"] == after["pg_one_road_best_pips"] == 0
    assert after["pg_expansion_edges_without_site"] == 2
    enemy_road = replace(observation, roads=(*observation.roads, Road(2, 20)))
    assert encode(enemy_road)["pg_one_road_new_sites"] == 0


def test_ports_need_own_production_and_potential_is_incremental_over_owned_ratio():
    observation = replace(view(), roads=(Road(0, 10),))
    ports = (Port((0, 5), None, 3), Port((2, 3), "WHEAT", 2))
    observation = replace(observation, board=replace(observation.board, ports=ports))
    result = encode(observation)
    assert result["pg_owned_port_utility"] == pytest.approx(5 / 432)
    assert result["pg_connected_port_gain"] == 0
    assert result["pg_one_road_port_gain"] == pytest.approx(5 / 216)
    connected = replace(observation, roads=(*observation.roads, Road(1, 10)))
    assert encode(connected)["pg_connected_port_gain"] == pytest.approx(5 / 216)
    assert encode(connected)["pg_one_road_port_gain"] == pytest.approx(5 / 216)
    mismatch = replace(
        observation, board=replace(observation.board, ports=(Port((2, 3), "ORE", 2),))
    )
    assert encode(mismatch)["pg_one_road_port_gain"] == 0
    # Nominal production/pips intentionally retain blocked-tile long-term utility.
    assert encode(replace(observation, robber_tile=0)) == result


def test_projection_and_actual_overlay_match_and_every_feature_is_finite():
    observation = chain_view()
    steps = (
        PlannedStep("MARITIME_TRADE", give=(2, 0, 0, 0, 0), receive=(0, 0, 0, 1, 0)),
        PlannedStep("BUILD_ROAD", 1),
        PlannedStep("BUILD_SETTLEMENT", 2),
        PlannedStep("END_TURN"),
    )
    for state in replay(observation, steps[:-1]):
        own = replace(
            observation.players[0],
            roads_left=state.roads_left,
            settlements_left=state.settlements_left,
            cities_left=state.cities_left,
        )
        actual = replace(
            observation,
            buildings=state.buildings,
            roads=state.roads,
            players=(own, *observation.players[1:]),
            hand=state.hand,
            legal_actions=(Action("END_TURN"),),
        )
        encoded = encode(observation, state)
        assert encoded == encode(actual)
        assert len(encoded) == 8
        assert all(type(x) is float and math.isfinite(x) for x in encoded.values())


def test_best_pips_combine_all_adjacent_tiles_and_one_road_sites_are_a_union():
    board = BoardSpec(
        (Tile(0, 0, 0, "WOOD", 6), Tile(1, 1, 0, "ORE", 12), Tile(2, 0, 1, "SHEEP", 8)),
        tuple((i, 0) for i in range(7)),
        tuple((i, i + 1) for i in range(6)),
        ((0, 1, 2), (3, 4, 5, 6), (2, 3)),
        ((), (), ()),
    )
    observation = replace(view(), board=board, roads=(Road(0, 10), Road(1, 10), Road(2, 10)))
    result = encode(observation)
    assert result["pg_connected_sites"] == 2
    assert result["pg_connected_best_pips"] == 10  # vertex 2: two five-pip tiles
    assert result["pg_one_road_new_sites"] == 1
    assert result["pg_one_road_best_pips"] == 1  # vertex 4: twelve token
    triangle = BoardSpec((), ((0, 0), (1, 0), (2, 0)), ((0, 1), (1, 2), (0, 2)), (), ())
    result = encode(replace(observation, board=triangle, buildings=(), roads=(Road(0, 10),)))
    assert result["pg_connected_sites"] == 2
    assert result["pg_one_road_new_sites"] == 1  # vertex 2, reachable by either edge


def test_city_projection_doubles_production_supported_owned_port_utility():
    observation = replace(
        view(hand=(0, 0, 0, 2, 3)), board=replace(view().board, ports=(Port((0, 5), "WHEAT", 2),))
    )
    state = replay(observation, (PlannedStep("BUILD_CITY", 0),))[-1]
    assert encode(observation)["pg_owned_port_utility"] == pytest.approx(5 / 144)
    assert encode(observation, state)["pg_owned_port_utility"] == pytest.approx(5 / 72)


def test_nonfinite_productive_values_and_invalid_port_ratios_reject():
    observation = view()
    tile = replace(observation.board.tiles[0], number=math.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        encode(replace(observation, board=replace(observation.board, tiles=(tile,))))
    with pytest.raises(ValueError, match="port ratio"):
        encode(
            replace(observation, board=replace(observation.board, ports=(Port((0, 5), None, 0),)))
        )
