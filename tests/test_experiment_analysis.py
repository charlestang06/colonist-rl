"""Read-only comparisons of synthetic saved paired-run artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256

import pytest

from colonist_rl import experiment_analysis as analysis


def _action(kind="END_TURN", target=None):
    return {
        "kind": kind, "target": target, "resource": None,
        "second_resource": None, "partner": None,
        "give": [0] * 5, "receive": [0] * 5, "offer_id": None,
    }


def _step(arm, seed, seat):
    return {
        "arm": arm, "board_seed": seed, "seat": seat,
        "before": {
            "version": 1, "phase": "PLAY_TURN", "turn_number": 3,
            "turn_owner": seat, "hand": [1, 0, 0, 0, 0], "known_own_vp": 3,
        },
        "action": _action(),
        "after": {
            "version": 2, "phase": "PLAY_TURN", "hand": [1, 0, 0, 0, 0],
            "known_own_vp": 3, "target_building_level": None,
        },
        "terminated": False, "truncated": False,
    }


def _game(arm, seed, seat, won=False, vp=5):
    return {"arm": arm, "board_seed": seed, "seat": seat,
            "won": won, "victory_points": vp, "truncated": won is None}


def _arm_report(boards, *, nodes=0, available=0, unavailable=0):
    return {
        "boards": [
            {"board_seed": seed, "status": status, "games": 4 if status == "complete" else 0}
            for seed, status in boards
        ],
        "planning_metrics_scope": "attempted_policy_selections",
        "selection_latency_ms": {"count": available + unavailable, "mean": 1.0, "p95": 2.0},
        "plan_nodes": nodes, "geometry_visits": nodes * 5,
        "wrapper_nodes": 0, "wrapper_geometry_visits": 0,
        "search_stop_counts": {"empty_beam": available},
        "search_diagnostics_scope": "attempted_policy_selections_with_search_diagnostics",
        "search_diagnostics_record_count": available,
        "search_diagnostics_unavailable_count": unavailable,
        "search_diagnostics_totals": {"attempted_transitions": nodes},
        "search_diagnostics_depth_histograms": {
            "selected_plan_depth": {"0": available},
            "max_feasible_depth": {"1": available},
        },
    }


def _write_run(path, *, seeds, arm_reports, games, capture, steps=(), reference="z_raw"):
    path.mkdir()
    manifest = {
        "format": "colonist-rl.paired-run", "version": 1,
        "validated_manifest": {
            "reference_arm": reference, "board_seeds": seeds,
            "arms": {name: {"kind": "public_planner"} for name in arm_reports},
            "capture_candidate_steps": capture,
        },
    }
    summary = {
        "format": "colonist-rl.paired-summary", "version": 1,
        "status": (
            "complete" if all(
                board["status"] == "complete"
                for arm in arm_reports.values() for board in arm["boards"]
            ) else "incomplete"
        ),
        "journal_valid": True,
        "journal_games": len(games), "candidate_steps": len(steps) if capture else 0,
        "paired_summary": {"reference_arm": reference}, "arms": arm_reports,
    }
    # The writer sorts keys, so saved arms are alphabetized regardless of reference.
    manifest_path = path / "manifest.json"
    games_path = path / "games.jsonl"
    steps_path = path / "candidate_steps.jsonl"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    games_path.write_text("".join(json.dumps(row) + "\n" for row in games))
    if capture:
        steps_path.write_text("".join(json.dumps(row) + "\n" for row in steps))
    summary["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    summary["games_jsonl_sha256"] = sha256(games_path.read_bytes()).hexdigest()
    summary["candidate_steps_jsonl_sha256"] = (
        sha256(steps_path.read_bytes()).hexdigest() if capture else None
    )
    (path / "summary.json").write_text(json.dumps(summary, sort_keys=True))
    return path


def _complete_run(path, *, seats_with_steps=range(4)):
    seed = 101
    arms = ("z_raw", "a_city")
    return _write_run(
        path, seeds=[seed],
        arm_reports={name: _arm_report([(seed, "complete")]) for name in arms},
        games=[_game(name, seed, seat) for name in arms for seat in range(4)],
        capture=True,
        steps=[_step(name, seed, seat) for name in arms for seat in seats_with_steps],
    )


def _rewrite_steps_with_hash(run_dir, steps):
    steps_path = run_dir / "candidate_steps.jsonl"
    steps_path.write_text("".join(json.dumps(step) + "\n" for step in steps))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["candidate_steps"] = len(steps)
    summary["candidate_steps_jsonl_sha256"] = sha256(steps_path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary, sort_keys=True))


def test_paired_outcomes_work_and_first_trace_divergences(tmp_path, capsys):
    seed = 101
    games = [
        _game("z_raw", seed, seat, won, vp)
        for seat, (won, vp) in enumerate(((True, 8), (False, 5), (False, 5), (False, 4)))
    ] + [
        _game("a_city", seed, seat, won, vp)
        for seat, (won, vp) in enumerate(((True, 9), (True, 7), (False, 5), (False, 3)))
    ]
    steps = []
    for seat in range(4):
        raw, city = _step("z_raw", seed, seat), _step("a_city", seed, seat)
        if seat == 0:
            raw["hidden"] = "reference"
            city["hidden"] = "candidate"
        elif seat == 1:
            city["action"] = _action("BUILD_ROAD", 9)
        elif seat == 2:
            city["before"]["hand"] = [2, 0, 0, 0, 0]
        steps.extend((raw, city))
        if seat == 3:
            extra = _step("a_city", seed, seat)
            extra["before"]["version"] = 2
            extra["after"]["version"] = 3
            steps.append(extra)
    run_dir = _write_run(
        tmp_path / "run", seeds=[seed],
        arm_reports={
            "z_raw": _arm_report([(seed, "complete")], nodes=100,
                                 available=3, unavailable=1),
            "a_city": _arm_report([(seed, "complete")], nodes=120,
                                  available=2, unavailable=2),
        },
        games=games, capture=True, steps=steps,
    )
    result = analysis.analyze_run(run_dir, detail_limit=4)
    assert result["reference_arm"] == "z_raw"  # Its name sorts after a_city.
    assert result["trace_capture"] is True
    work = result["search_work"]["z_raw"]
    assert work["scope"] == "attempted_policy_selections"
    assert work["plan_nodes"] == 100 and work["geometry_visits"] == 500
    assert work["diagnostics"]["record_count"] == 3
    assert work["diagnostics"]["unavailable_count"] == 1
    assert work["diagnostics"]["coverage_fraction"] == 0.75
    assert work["diagnostics"]["totals"]["attempted_transitions"] == 100
    comparison = result["comparisons"]["a_city"]
    assert comparison["matched_board_seeds"] == [seed]
    assert comparison["matched_game_count"] == 4
    assert comparison["resolved_win_pair_count"] == 4
    assert comparison["paired_win_rate_difference"] == 0.25
    assert comparison["win_difference_counts"] == {
        "arm_only_win": 1, "reference_only_win": 0, "same_result": 3}
    assert comparison["victory_points_pair_count"] == 4
    assert comparison["mean_victory_points_difference"] == 0.5
    trace = comparison["trace_divergence"]
    assert trace["category_counts"] == {
        "same_trace": 1, "same_before_state_different_action": 1,
        "prior_public_state_divergence": 1, "unequal_trace_length": 1,
    }
    assert [(detail["seat"], detail["category"]) for detail in trace["details"]] == [
        (1, "same_before_state_different_action"),
        (2, "prior_public_state_divergence"),
        (3, "unequal_trace_length"),
    ]
    assert trace["details"][0]["arm_action"]["kind"] == "BUILD_ROAD"
    assert trace["details"][1]["stage"] == "before"
    assert trace["details"][2]["reference_length"] == 1
    assert trace["details"][2]["arm_length"] == 2
    assert "hidden" not in json.dumps(result)
    bounded = analysis.analyze_run(run_dir, detail_limit=2)["comparisons"]["a_city"]
    assert len(bounded["trace_divergence"]["details"]) == 2
    assert bounded["trace_divergence"]["omitted_detail_count"] == 1
    assert analysis.main(["--run-dir", str(run_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["reference_arm"] == "z_raw"


def test_saved_search_work_keeps_searched_latency_scope_and_reason_mix(tmp_path):
    seed = 101
    scope = "attempted_planner_selections_with_search_stop_other_than_not_searched"
    reports = {
        "z_raw": _arm_report([(seed, "complete")], nodes=8, available=4, unavailable=4),
        "a_pruned": _arm_report([(seed, "complete")], nodes=6, available=5, unavailable=3),
    }
    searched = {
        "z_raw": {"count": 2, "mean": 12.0, "p95": 19.2},
        "a_pruned": {"count": 3, "mean": 4.0, "p95": 5.8},
    }
    reasons = {
        "z_raw": {"planned": 1, "insufficient_gain": 1},
        "a_pruned": {"planned": 2, "insufficient_gain": 1},
    }
    for name, report in reports.items():
        report["searched_selection_latency_scope"] = scope
        report["searched_selection_latency_ms"] = searched[name]
        report["searched_plan_reason_counts"] = reasons[name]
        report["search_diagnostics_totals"]["pruned_invalid_candidates"] = (
            0 if name == "z_raw" else 7
        )
    run_dir = _write_run(
        tmp_path / "run", seeds=[seed], arm_reports=reports,
        games=[_game(name, seed, seat) for name in reports for seat in range(4)],
        capture=False,
    )

    result = analysis.analyze_run(run_dir)
    for name, report in reports.items():
        work = result["search_work"][name]
        assert work["scope"] == report["planning_metrics_scope"]
        assert work["selection_latency_ms"] == report["selection_latency_ms"]
        assert work["searched_selection_latency_scope"] == scope
        assert work["searched_selection_latency_ms"] == searched[name]
        assert work["searched_plan_reason_counts"] == reasons[name]
        assert work["diagnostics"]["totals"] == report["search_diagnostics_totals"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("artifact", ("manifest.json", "games.jsonl", "candidate_steps.jsonl"))
def test_same_record_count_saved_artifact_mutation_is_rejected_by_digest(tmp_path, artifact):
    run_dir = _complete_run(tmp_path / "run")
    path = run_dir / artifact
    original_lines = path.read_text().splitlines()
    if artifact == "manifest.json":
        manifest = json.loads(path.read_text())
        manifest["tampered"] = True
        path.write_text(json.dumps(manifest, sort_keys=True))
    else:
        rows = [json.loads(line) for line in original_lines]
        if artifact == "games.jsonl":
            rows[0]["victory_points"] += 1
        else:
            rows[0]["action"] = _action("BUILD_ROAD", 9)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(path.read_text().splitlines()) == len(original_lines)
    with pytest.raises(ValueError, match="digest mismatch"):
        analysis.analyze_run(run_dir)


def test_invalid_provenance_status_is_rejected_even_with_valid_artifact_hashes(tmp_path):
    run_dir = _complete_run(tmp_path / "run")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "invalid_provenance"
    summary["provenance_errors"] = ["source_changed"]
    summary_path.write_text(json.dumps(summary, sort_keys=True))
    with pytest.raises(ValueError, match="invalid_provenance"):
        analysis.analyze_run(run_dir)


def test_missing_both_traces_for_complete_matched_game_is_rejected(tmp_path):
    # The saved count and digest agree with the six-line trace file.
    run_dir = _complete_run(tmp_path / "run", seats_with_steps=range(3))
    with pytest.raises(ValueError, match="missing captured trace"):
        analysis.analyze_run(run_dir)


def test_one_sided_missing_trace_for_complete_matched_game_is_rejected(tmp_path):
    run_dir = _complete_run(tmp_path / "run")
    steps_path = run_dir / "candidate_steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    _rewrite_steps_with_hash(
        run_dir,
        [step for step in steps if not (
            step["arm"] == "a_city" and step["board_seed"] == 101 and step["seat"] == 3
        )],
    )
    with pytest.raises(ValueError, match="missing captured trace"):
        analysis.analyze_run(run_dir)


@pytest.mark.parametrize("section,field,bad", [
    ("before", "hand", None),
    ("before", "hand", [1, 0, 0, 0]),
    ("before", "hand", [1, 0, False, 0, 0]),
    ("after", "hand", [1, -1, 0, 0, 0]),
    ("before", "version", "1"),
    ("before", "phase", None),
    ("action", "kind", 7),
    ("action", "give", [0, 0, 0, 0]),
])
def test_malformed_captured_public_field_is_rejected_before_classification(
    tmp_path, section, field, bad
):
    run_dir = _complete_run(tmp_path / "run")
    steps_path = run_dir / "candidate_steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    for step in steps:
        if step["seat"] == 0:
            step[section][field] = bad
    _rewrite_steps_with_hash(run_dir, steps)
    with pytest.raises(ValueError, match="invalid captured public"):
        analysis.analyze_run(run_dir)


def test_later_malformed_trace_step_is_rejected_before_an_earlier_divergence():
    reference = [_step("z_raw", 101, 0), _step("z_raw", 101, 0)]
    candidate = [_step("a_city", 101, 0), _step("a_city", 101, 0)]
    candidate[0]["action"] = _action("BUILD_ROAD", 9)
    candidate[1]["before"]["hand"] = None
    with pytest.raises(ValueError, match="invalid captured public"):
        analysis._first_trace_divergence(reference, candidate)


def test_pairwise_complete_boards_exclude_partial_rows_and_skip_uncaptured_traces(tmp_path):
    seeds = [101, 102]
    statuses = {
        "z_raw": [(101, "complete"), (102, "complete")],
        "a_alt": [(101, "complete"), (102, "incomplete")],
        "b_third": [(101, "incomplete"), (102, "complete")],
    }
    games = [
        _game("z_raw", seed, seat, None if seed == 101 and seat == 0 else False)
        for seed in seeds for seat in range(4)
    ] + [
        _game("a_alt", 101, seat, seat == 1, 6) for seat in range(4)
    ] + [
        _game("b_third", 102, seat) for seat in range(4)
    ] + [
        _game("a_alt", 102, 0, True, 99),
        _game("b_third", 101, 0, True, 99),
    ]
    run_dir = _write_run(
        tmp_path / "run", seeds=seeds,
        arm_reports={name: _arm_report(boards) for name, boards in statuses.items()},
        games=games, capture=False,
    )
    result = analysis.analyze_run(run_dir)
    assert result["trace_capture"] is False
    alt = result["comparisons"]["a_alt"]
    third = result["comparisons"]["b_third"]
    assert alt["matched_board_seeds"] == [101]
    assert third["matched_board_seeds"] == [102]
    assert alt["matched_game_count"] == third["matched_game_count"] == 4
    assert alt["resolved_win_pair_count"] == 3
    assert alt["paired_win_rate_difference"] is None
    assert alt["victory_points_pair_count"] == 4
    assert alt["mean_victory_points_difference"] == 1.0
    assert "trace_divergence" not in alt and "trace_divergence" not in third
    assert not (run_dir / "candidate_steps.jsonl").exists()


def test_same_action_with_different_public_after_state_is_prior_state_divergence():
    reference = _step("z_raw", 101, 0)
    candidate = deepcopy(reference)
    candidate["after"]["hand"] = [0, 1, 0, 0, 0]
    divergence = analysis._first_trace_divergence([reference], [candidate])
    assert divergence["category"] == "prior_public_state_divergence"
    assert divergence["stage"] == "after" and divergence["step_index"] == 0


def test_reference_mismatch_and_invalid_journal_are_rejected(tmp_path):
    run_dir = _write_run(
        tmp_path / "run", seeds=[101],
        arm_reports={
            "z_raw": _arm_report([(101, "complete")]),
            "a_city": _arm_report([(101, "complete")]),
        },
        games=[], capture=False,
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["paired_summary"]["reference_arm"] = "a_city"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="reference_arm mismatch"):
        analysis.analyze_run(run_dir)
    summary["paired_summary"]["reference_arm"] = "z_raw"
    summary["journal_valid"] = False
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="validated game journal"):
        analysis.analyze_run(run_dir)
