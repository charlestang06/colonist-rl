"""Cached immutable inputs must never leak dynamic state between observations."""

import importlib
from dataclasses import replace

import torch
from test_policy import view

from colonist_rl.board import BoardSpec
from colonist_rl.contracts import Building, Road
from colonist_rl.policy import graph_features


def inputs():
    assert importlib.util.find_spec("colonist_rl.graph_inputs") is not None, (
        "bounded immutable graph-input cache is missing"
    )
    return importlib.import_module("colonist_rl.graph_inputs")


def test_cache_clones_static_features_and_rebuilds_dynamic_links_degrees():
    module = inputs()
    original = view()
    expected, expected_links, offsets = graph_features(original, "cpu")
    changed = replace(
        original,
        robber_tile=1,
        hand=(3, 2, 1, 0, 4),
        buildings=(Building(0, 1, 2),),
        roads=(Road(0, 0),),
    )
    dynamic, links, (_, _, players) = graph_features(changed, "cpu")
    vo, eo, _ = offsets
    assert dynamic[0, 13] == 0 and dynamic[1, 13] == 1
    assert dynamic[vo, 14] == 1 and dynamic[vo, 15] == 0
    assert dynamic[eo, 14] == 1 and dynamic[eo, 15] == 1
    torch.testing.assert_close(dynamic[players[0], 25:30], torch.tensor([0.3, 0.2, 0.1, 0, 0.4]))
    assert links.shape[1] == expected_links.shape[1] + 4
    assert torch.bincount(links[1])[players[0]] == 1
    again, again_links, _ = graph_features(original, "cpu")
    torch.testing.assert_close(again, expected, rtol=0, atol=0)
    assert torch.equal(again_links, expected_links)
    assert not again.requires_grad and again.grad_fn is None
    assert module._static_graph.cache_info().maxsize == 32


def test_cache_keys_full_board_schema_dtype_and_player_order_and_is_bounded():
    module = inputs()
    module._static_graph.cache_clear()
    original = view()
    baseline, _, _ = graph_features(original, "cpu")
    tile = next(t for t in original.board.tiles if t.resource != "DESERT")
    changed = replace(
        original,
        board=replace(
            original.board,
            tiles=tuple(
                replace(t, number=12 if t.number != 12 else 2) if t.id == tile.id else t
                for t in original.board.tiles
            ),
        ),
    )
    encoded, _, _ = graph_features(changed, "cpu")
    assert baseline[tile.id, 11] != encoded[tile.id, 11]
    reordered = replace(original, players=tuple(reversed(original.players)))
    features, _, (_, _, nodes) = graph_features(reordered, "cpu")
    assert nodes[0] > nodes[1]
    assert features[nodes[0], 15] == 1
    graph_features(replace(original, schema_version=2), "cpu")
    double, _, _ = graph_features(original, "cpu", dtype=torch.float64)
    assert double.dtype == torch.float64
    assert module._static_graph.cache_info().currsize == 5
    for seed in range(36):
        graph_features(replace(original, board=BoardSpec.standard(seed)), "cpu")
    assert module._static_graph.cache_info().currsize == 32


def test_cache_can_be_built_during_inference_then_used_for_training():
    module = inputs()
    module._static_graph.cache_clear()
    from colonist_rl.policy import GraphPolicy

    policy = GraphPolicy(8)
    with torch.inference_mode():
        policy(view())
    output = policy(view())
    output.value.backward()
    assert torch.isfinite(policy.node_in.weight.grad).all()
