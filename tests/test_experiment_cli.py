"""The manifest CLI is exercised with a synthetic panel, never an engine game."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from colonist_rl import experiment_cli as cli
from colonist_rl.contracts import Action


def _manifest(seed=880001):
    return {
        "format": "colonist-rl.paired-experiment",
        "version": 1,
        "role": "smoke",
        "reference_arm": "raw",
        "board_seeds": [seed],
        "arms": {
            "raw": {"kind": "public_planner", "config": {"max_nodes": 2048},
                    "gain_margin_micro": 250000},
            "city": {"kind": "city_before_purchase", "config": {"max_nodes": 2048},
                     "gain_margin_micro": 250000},
        },
        "opponent_lineup": ["heuristic", "heuristic", "heuristic"],
        "no_offers": True,
        "max_game_steps": 100,
        "wall_seconds": 30,
        "capture_candidate_steps": True,
    }


def _write_manifest(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _fake_panel(factories, *, board_seeds, on_game_end, **kwargs):
    assert board_seeds == [880001]
    assert len(kwargs["opponent_lineup"]) == 3
    assert kwargs["max_game_steps"] == 100 and kwargs["no_offers"] is True
    assert kwargs["deadline"] > 0
    reports = {}
    for name in factories:
        rows = []
        for seat in range(4):
            if kwargs["on_candidate_step"] is not None:
                before = NS(player_id=seat, context=NS(version=7, kind="PLAY_TURN",
                            turn_number=2, turn_owner=seat), hand=(1, 2, 0, 0, 0),
                            own_victory_points=3)
                after = NS(context=NS(version=8, kind="PLAY_TURN", turn_number=2,
                                      turn_owner=seat), hand=(1, 2, 0, 0, 0),
                           own_victory_points=3, buildings=())
                kwargs["on_candidate_step"](NS(
                    arm=name, board_seed=board_seeds[0],
                    step=NS(before=before, action=Action("END_TURN"), after=after,
                            terminated=False, truncated=False),
                ))
            row = {
                "board_seed": board_seeds[0], "seat": seat, "won": seat == 0,
                "winner": 0, "victory_points": 8 if seat == 0 else 5,
                "terminated": True, "truncated": False,
                "action_kind_counts": {"END_TURN": 1},
            }
            on_game_end({"arm": name, **deepcopy(row)})
            rows.append(row)
        reports[name] = {
            "status": "complete", "wins": 1, "games": 4,
            "boards": [{"board_seed": board_seeds[0], "status": "complete",
                        "rows": rows, "partial_rows": [], "wins": 1, "games": 4,
                        "changed_decision_details": []}],
        }
    return {
        "status": "complete", "incomplete_reason": None,
        "requested_board_seeds": board_seeds,
        "paired_complete_board_seeds": board_seeds,
        "paired_resolved_board_seeds": board_seeds,
        "paired_summary": {"status": "complete", "paired_resolved": True,
                           "board_seeds": board_seeds,
                           "reference_arm": next(iter(factories))},
        "arms": reports,
    }


def _patch_runtime(monkeypatch):
    monkeypatch.setattr(cli, "run_paired_panel", _fake_panel)
    monkeypatch.setattr(cli, "_source_tree_sha256", lambda: "a" * 64)
    monkeypatch.setattr(cli, "_version_identity", lambda: {
        "protocol_version": 1, "package_version": "0.1.0", "backend_version": "fixture",
    })


def test_public_option_arm_validates_and_factory_wraps_fresh_raw_parent(tmp_path):
    from colonist_rl.public_option_purchase import PublicOptionPurchasePolicy
    from colonist_rl.public_turn_planner import PublicTurnPlannerPolicy

    manifest = _manifest()
    manifest["arms"]["option"] = {
        "kind": "public_option_purchase",
        "config": {"max_nodes": 111},
        "gain_margin_micro": 375000,
    }
    validated, checkpoints = cli._validate_manifest(manifest, base=tmp_path)
    assert checkpoints == {}
    assert validated["arms"]["option"] == {
        "kind": "public_option_purchase",
        "config": {**validated["arms"]["raw"]["config"], "max_nodes": 111},
        "gain_margin_micro": 375000,
    }
    factories = cli._factories(validated)
    first, second = factories["option"](), factories["option"]()
    assert isinstance(first, PublicOptionPurchasePolicy) and first.enabled is True
    assert isinstance(first.parent, PublicTurnPlannerPolicy)
    assert first.parent is not second.parent and first is not second
    assert first.parent.gain_margin_micro == 375000
    assert first.parent.config.max_nodes == 111
    assert type(factories["raw"]()) is PublicTurnPlannerPolicy


def test_one_board_two_arm_smoke_writes_fresh_validated_artifacts(tmp_path, monkeypatch, capsys):
    _patch_runtime(monkeypatch)
    manifest_path = tmp_path / "smoke.json"
    _write_manifest(manifest_path, _manifest())
    root = tmp_path / "runs"

    assert cli.main(["--manifest", str(manifest_path), "--output-root", str(root)]) == 0
    notice = json.loads(capsys.readouterr().out)
    run_dir = Path(notice["run_dir"])
    assert notice["status"] == "complete" and run_dir.is_dir()
    saved_manifest = json.loads((run_dir / "manifest.json").read_text())
    config = saved_manifest["validated_manifest"]
    assert config["role"] == "smoke"
    assert list(json.loads(manifest_path.read_text())["arms"]) == ["city", "raw"]
    assert config["reference_arm"] == "raw"
    assert config["arms"]["raw"]["config"]["max_nodes"] == 2048
    assert config["arms"]["raw"]["gain_margin_micro"] == 250000
    assert saved_manifest["input_manifest_sha256"] == sha256(manifest_path.read_bytes()).hexdigest()
    assert saved_manifest["source_sha256"] == "a" * 64
    assert len(saved_manifest["versions_sha256"]) == 64
    assert saved_manifest["wall_budget_scope"] == "gameplay_after_preflight"
    assert saved_manifest["deadline_enforcement"] == "between_evaluator_decisions"
    games = [json.loads(line) for line in (run_dir / "games.jsonl").read_text().splitlines()]
    assert len(games) == 8
    assert {(row["arm"], row["board_seed"], row["seat"]) for row in games} == {
        (arm, 880001, seat) for arm in ("raw", "city") for seat in range(4)}
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["journal_valid"] is True
    assert summary["paired_summary"]["paired_resolved"] is True
    assert "rows" not in summary["arms"]["raw"]["boards"][0]
    assert summary["journal_games"] == 8
    steps = [
        json.loads(line)
        for line in (run_dir / "candidate_steps.jsonl").read_text().splitlines()
    ]
    assert len(steps) == 8
    assert steps[0]["arm"] == "raw" and steps[0]["board_seed"] == 880001
    assert steps[0]["seat"] == 0 and steps[0]["before"]["version"] == 7
    assert steps[0]["before"]["phase"] == "PLAY_TURN"
    assert steps[0]["action"]["kind"] == "END_TURN"
    assert steps[0]["after"]["version"] == 8
    assert "board" not in steps[0] and "legal_actions" not in steps[0]

    assert cli.main(["--manifest", str(manifest_path), "--output-root", str(root)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["run_dir"] != notice["run_dir"]


def test_candidate_trace_is_synced_before_each_completed_game_commit(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    manifest_path = tmp_path / "smoke.json"
    _write_manifest(manifest_path, _manifest())
    synchronized_inodes = []
    monkeypatch.setattr(cli.os, "fsync", lambda fd: synchronized_inodes.append(os.fstat(fd).st_ino))

    summary = cli.run_manifest(manifest_path, tmp_path / "runs")
    run_dir = Path(summary["run_dir"])
    steps_inode = (run_dir / "candidate_steps.jsonl").stat().st_ino
    games_inode = (run_dir / "games.jsonl").stat().st_ino
    assert synchronized_inodes == [steps_inode, games_inode] * 8


def test_two_raw_planner_budgets_remain_explicit_and_independent(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    manifest = _manifest()
    manifest["arms"] = {
        "nodes2048": {"kind": "public_planner", "config": {"max_nodes": 2048},
                      "gain_margin_micro": 250000},
        "nodes8192": {"kind": "public_planner", "config": {"max_nodes": 8192},
                      "gain_margin_micro": 250000},
    }
    manifest["reference_arm"] = "nodes2048"
    manifest_path = tmp_path / "budgets.json"
    _write_manifest(manifest_path, manifest)

    def inspect(factories, **kwargs):
        assert factories["nodes2048"]().config.max_nodes == 2048
        assert factories["nodes8192"]().config.max_nodes == 8192
        assert factories["nodes2048"]().gain_margin_micro == 250000
        return _fake_panel(factories, **kwargs)

    monkeypatch.setattr(cli, "run_paired_panel", inspect)
    summary = cli.run_manifest(manifest_path, tmp_path / "runs")
    assert summary["status"] == "complete"
    saved = json.loads((Path(summary["run_dir"]) / "manifest.json").read_text())
    assert saved["validated_manifest"]["arms"]["nodes8192"]["config"]["max_nodes"] == 8192


def test_optional_trace_defaults_off_and_supported_lineup_is_preserved(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    manifest = _manifest()
    manifest.pop("capture_candidate_steps")
    manifest["opponent_lineup"] = ["greedy_no_trade", "random", "native_vf_82aae93"]
    manifest_path = tmp_path / "mixed.json"
    _write_manifest(manifest_path, manifest)
    observed = []

    def inspect(factories, **kwargs):
        observed.append((kwargs["opponent_lineup"], kwargs["on_candidate_step"]))
        return _fake_panel(factories, **kwargs)

    monkeypatch.setattr(cli, "run_paired_panel", inspect)
    summary = cli.run_manifest(manifest_path, tmp_path / "runs")
    assert observed == [(tuple(manifest["opponent_lineup"]), None)]
    assert not (Path(summary["run_dir"]) / "candidate_steps.jsonl").exists()
    saved = json.loads((Path(summary["run_dir"]) / "manifest.json").read_text())
    assert saved["validated_manifest"]["opponent_lineup"] == manifest["opponent_lineup"]
    assert saved["validated_manifest"]["capture_candidate_steps"] is False


def test_reference_arm_is_first_when_its_name_sorts_after_treatment(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    manifest = _manifest()
    manifest["arms"] = {"z_raw": manifest["arms"]["raw"],
                        "a_city": manifest["arms"]["city"]}
    manifest["reference_arm"] = "z_raw"
    manifest_path = tmp_path / "reference.json"
    _write_manifest(manifest_path, manifest)
    assert list(json.loads(manifest_path.read_text())["arms"]) == ["a_city", "z_raw"]
    seen = []

    def inspect(factories, **kwargs):
        seen.append(list(factories))
        return _fake_panel(factories, **kwargs)

    monkeypatch.setattr(cli, "run_paired_panel", inspect)
    summary = cli.run_manifest(manifest_path, tmp_path / "runs")
    assert seen == [["z_raw", "a_city"]]
    assert summary["paired_summary"]["reference_arm"] == "z_raw"
    saved = json.loads((Path(summary["run_dir"]) / "manifest.json").read_text())
    assert saved["validated_manifest"]["reference_arm"] == "z_raw"


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(role="unknown"),
    lambda m: m.update(reference_arm="missing"),
    lambda m: m.pop("reference_arm"),
    lambda m: m.update(board_seeds=[880001, 880001]),
    lambda m: m.update(board_seeds=[True]),
    lambda m: m.update(no_offers=False),
    lambda m: m.update(opponent_lineup=["heuristic", "bogus", "heuristic"]),
    lambda m: m.update(opponent_lineup=["heuristic", "historical", "heuristic"]),
    lambda m: m.update(opponent_lineup=["heuristic", "heuristic"]),
    lambda m: m.update(max_game_steps=0),
    lambda m: m.update(wall_seconds=-1),
    lambda m: m.update(capture_candidate_steps=1),
    lambda m: m.update(arms={"raw": m["arms"]["raw"]}),
    lambda m: m.update(arms={"city": m["arms"]["city"], "city2": m["arms"]["city"]}),
    lambda m: m["arms"]["city"].update(kind="unrecognized"),
    lambda m: m["arms"]["raw"].update(gain_margin_micro=-1),
    lambda m: m["arms"]["raw"]["config"].update(max_nodes=-1),
    lambda m: m.update(extra="silent config drift"),
    lambda m: m.update(expected_source_sha256="b" * 64),
])
def test_invalid_manifest_is_rejected_before_run_directory(tmp_path, monkeypatch, mutation):
    _patch_runtime(monkeypatch)
    manifest = _manifest()
    mutation(manifest)
    manifest_path = tmp_path / "invalid.json"
    _write_manifest(manifest_path, manifest)
    root = tmp_path / "runs"
    with pytest.raises(ValueError):
        cli.run_manifest(manifest_path, root)
    assert not root.exists()


def test_graph_checkpoint_hash_is_pinned_and_factory_returns_fresh_copies(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    manifest = _manifest()
    manifest["arms"]["graph"] = {
        "kind": "graph_checkpoint", "checkpoint": "student.pt",
        "checkpoint_sha256": "0" * 64,
    }
    manifest_path = tmp_path / "graph.json"
    _write_manifest(manifest_path, manifest)
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="checkpoint.*sha256"):
        cli.run_manifest(manifest_path, root)
    assert not root.exists()

    manifest["arms"]["graph"]["checkpoint_sha256"] = sha256(checkpoint.read_bytes()).hexdigest()
    _write_manifest(manifest_path, manifest)
    loaded = []

    def load(path):
        assert path == checkpoint.resolve()
        template = {"weights": [1, 2, 3]}
        loaded.append(template)
        return template

    monkeypatch.setattr(cli, "_load_graph_policy", load)

    def inspect_factories(factories, **kwargs):
        first, second = factories["graph"](), factories["graph"]()
        assert first == second == loaded[0]
        assert first is not second and first is not loaded[0]
        return _fake_panel(factories, **kwargs)

    monkeypatch.setattr(cli, "run_paired_panel", inspect_factories)
    summary = cli.run_manifest(manifest_path, root)
    assert len(loaded) == 1 and summary["status"] == "complete"
    saved = json.loads((Path(summary["run_dir"]) / "manifest.json").read_text())
    assert saved["checkpoints"]["graph"]["sha256"] == manifest["arms"]["graph"]["checkpoint_sha256"]


def test_source_change_after_gameplay_marks_summary_invalid(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(cli, "_source_tree_sha256", lambda: next(hashes))
    manifest_path = tmp_path / "smoke.json"
    _write_manifest(manifest_path, _manifest())
    summary = cli.run_manifest(manifest_path, tmp_path / "runs")
    assert summary["status"] == "invalid_provenance"
    assert summary["provenance_errors"] == ["source_changed"]
    assert summary["runner_status"] == "complete"
    assert summary["journal_valid"] is True


def test_runner_reference_mismatch_is_rejected_after_journaling(tmp_path, monkeypatch):
    _patch_runtime(monkeypatch)
    manifest_path = tmp_path / "smoke.json"
    _write_manifest(manifest_path, _manifest())

    def mismatch(factories, **kwargs):
        result = _fake_panel(factories, **kwargs)
        result["paired_summary"]["reference_arm"] = "city"
        return result

    monkeypatch.setattr(cli, "run_paired_panel", mismatch)
    root = tmp_path / "runs"
    with pytest.raises(ValueError, match="reference_arm"):
        cli.run_manifest(manifest_path, root)
    run_dir, = root.iterdir()
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "failed"
    assert len((run_dir / "games.jsonl").read_text().splitlines()) == 8


def test_incomplete_panel_exits_nonzero_and_keeps_partial_game_records(
    tmp_path, monkeypatch, capsys
):
    _patch_runtime(monkeypatch)
    manifest_path = tmp_path / "smoke.json"
    _write_manifest(manifest_path, _manifest())

    def incomplete(factories, *, board_seeds, on_game_end, **kwargs):
        reports = {}
        for name in factories:
            seats = range(4) if name == "raw" else range(1)
            rows = []
            for seat in seats:
                row = {"board_seed": board_seeds[0], "seat": seat, "won": seat == 0,
                       "victory_points": 6, "truncated": False,
                       "action_kind_counts": {"END_TURN": 1}}
                on_game_end({"arm": name, **deepcopy(row)})
                rows.append(row)
            reports[name] = {
                "status": "complete" if name == "raw" else "incomplete",
                "boards": [{"board_seed": board_seeds[0],
                            "rows": rows if name == "raw" else [],
                            "partial_rows": [] if name == "raw" else rows}],
            }
        return {
            "status": "incomplete", "incomplete_reason": "deadline",
            "requested_board_seeds": board_seeds,
            "paired_complete_board_seeds": [], "paired_resolved_board_seeds": [],
            "paired_summary": {"status": "incomplete", "paired_resolved": False,
                               "reference_arm": "raw"},
            "arms": reports,
        }

    monkeypatch.setattr(cli, "run_paired_panel", incomplete)
    assert cli.main(["--manifest", str(manifest_path),
                     "--output-root", str(tmp_path / "runs")]) == 2
    notice = json.loads(capsys.readouterr().out)
    run_dir = Path(notice["run_dir"])
    assert notice["status"] == "incomplete"
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "incomplete"
    assert len((run_dir / "games.jsonl").read_text().splitlines()) == 5
