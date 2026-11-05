"""A frozen player-visible regression; no engine state, game replay or strength claim."""

import hashlib
import json
from pathlib import Path

from colonist_rl.adapters.observation_io import view_from_dict
from colonist_rl.public_turn_planner import PlannedStep, TurnPlannerConfig, plan_turn


def test_completed_known_win_survives_geometry_limit():
    fixture = json.loads(
        (Path(__file__).parent / 'fixtures/public_trade_settlement_win.json').read_text()
    )
    envelope = fixture['observation']
    canonical = json.dumps(envelope, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == fixture['provenance'][
        'observation_sha256'
    ]
    assert fixture['provenance']['source_case_sha256'] == (
        '7e84bf4388f4382f584e640e6c0c1f162b1908d8f59d05afea2f36d89b2a7b60'
    )
    assert fixture['provenance']['observation_sha256'] == (
        '2053475fa8954491dbbc0611daea0fe615449fff32a65eb705bbfdf034183315'
    )
    observation = view_from_dict(envelope)
    assert observation.hand == (3, 5, 1, 0, 1)
    assert observation.own_victory_points == 9
    assert next(p for p in observation.players if p.player_id == 0).victory_points == 8

    config = TurnPlannerConfig()
    plan = plan_turn(observation, config)

    assert plan.reason == 'planned'
    assert plan.search_stop == 'geometry_limit'
    assert plan.first_action is observation.legal_actions[plan.root_index]
    assert plan.root_index == 12
    assert plan.steps == (
        PlannedStep('MARITIME_TRADE', give=(0, 3, 0, 0, 0), receive=(0, 0, 0, 1, 0)),
        PlannedStep('BUILD_SETTLEMENT', 31),
    )
    # The known building adds the last point; the existing road award adds none.
    assert plan.nodes <= config.max_nodes
    assert plan.geometry_visits == config.max_geometry_visits
