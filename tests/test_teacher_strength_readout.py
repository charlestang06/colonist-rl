"""Frozen teacher-strength readout on hand-counted paired board fixtures."""

from __future__ import annotations

from colonist_rl.teacher_strength_readout import (
    paired_panel_summary,
    stage1_advance,
    stage2_interim_stop,
    stage2_qualification,
)


def _row(arm, seed, seat, *, won, vp, native_identity=None, native_winner=False):
    row = {
        "arm": arm, "board_seed": seed, "seat": seat,
        "won": won, "winner": seat if won else (seat + 2) % 4,
        "victory_points": vp, "terminated": True, "truncated": False,
    }
    if native_identity is not None:
        row["native_opponents"] = [{
            "metadata": {"identity": native_identity, "seat": (seat + 1) % 4,
                         "full_rule_equivalence": False},
            "metrics": {"native_decisions": 1},
        }]
        if arm == "raw" and native_winner:
            row["winner"] = (seat + 1) % 4
    return row


def _panel(start, count, control, *, raw_wins=2, parent_wins=0,
           control_wins=0, native_identity=None, native_wins=0,
           raw_vp=8, parent_vp=6):
    seeds = list(range(start, start + count))
    rows = []
    for seed in seeds:
        for seat in range(4):
            rows.append(_row("raw", seed, seat, won=seat < raw_wins, vp=raw_vp,
                             native_identity=native_identity,
                             native_winner=(native_identity is not None
                                            and raw_wins <= seat < raw_wins + native_wins)))
            rows.append(_row("parent", seed, seat, won=seat < parent_wins, vp=parent_vp,
                             native_identity=native_identity))
            if control is not None:
                rows.append(_row(control, seed, seat, won=seat < control_wins, vp=7))
    return paired_panel_summary(
        seeds, ("raw", "parent") + ((control,) if control else ()), rows,
        native_identity=native_identity,
    )


def test_stage1_advances_only_on_complete_board_paired_win_vp_and_uncertainty_gates():
    h = _panel(100, 8, "H", raw_wins=2, parent_wins=0, control_wins=1)
    gnt = _panel(200, 8, "GNT", raw_wins=2, parent_wins=0, control_wins=1)

    assert h["raw_minus"]["H"]["win_count_difference"] == 8
    assert h["raw_minus"]["H"]["board_bootstrap_95"][0] == 0.25
    assert h["raw_minus"]["parent"]["win_count_difference"] == 16
    assert stage1_advance(h, gnt)["advance"] is True

    weak_h = _panel(100, 8, "H", raw_wins=1, parent_wins=0, control_wins=1)
    decision = stage1_advance(weak_h, gnt)
    assert decision["advance"] is False
    assert "H:raw_minus_H_wins_below_4" in decision["reasons"]


def test_native_ladder_interim_and_full_gate_use_raw_games_native_winners():
    vf8 = _panel(300, 8, None, raw_wins=2, parent_wins=0,
                 native_identity="native_vf_82aae93", native_wins=1)
    ab8 = _panel(400, 8, None, raw_wins=2, parent_wins=0,
                 native_identity="native_ab2_82aae93", native_wins=1)
    assert vf8["native_opponent_wins_in_raw_games"] == 8
    assert stage2_interim_stop(vf8, ab8)["stop"] is False

    dominated = _panel(400, 8, None, raw_wins=0, parent_wins=1,
                       native_identity="native_ab2_82aae93", native_wins=2,
                       raw_vp=5)
    stop = stage2_interim_stop(vf8, dominated)
    assert stop["stop"] is True
    assert "AB:raw_trails_parent_wins_and_vp" in stop["reasons"]

    vf16 = _panel(300, 16, None, raw_wins=2, parent_wins=0,
                  native_identity="native_vf_82aae93", native_wins=1)
    ab16 = _panel(400, 16, None, raw_wins=2, parent_wins=0,
                  native_identity="native_ab2_82aae93", native_wins=1)
    assert stage2_qualification(vf16, ab16)["qualified"] is True
    native_leads = _panel(400, 16, None, raw_wins=1, parent_wins=0,
                          native_identity="native_ab2_82aae93", native_wins=2)
    final = stage2_qualification(vf16, native_leads)
    assert final["qualified"] is False
    assert "AB:native_wins_exceed_raw" in final["reasons"]
