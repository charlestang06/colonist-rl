"""The reusable checker certifies exact public routes, not strategic choices."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import colonist_rl.simulator as simulator_module
from tools.public_fact_replay import audit_case, build_report, canonical_sha256

FIXTURE = Path(__file__).parent / "fixtures" / "public-fact-replay-compact.json"
FIXTURE_SHA256 = "05a4ebd3d1e0c7039c94040f21ad77dc012df3126c8c160102254f69c7a53ca6"


def test_structured_trade_trade_settlement_route_replays_from_own_menus():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    report = build_report(FIXTURE, FIXTURE_SHA256)
    checkout = Path(__file__).resolve().parents[1]
    current_head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    assert report["identity"]["source_head"] == current_head
    assert (
        report["identity"]["source_files_sha256"]
        == json.loads(FIXTURE.read_text())["source_files_sha256"]
    )
    assert report["summary"]["status_counts"] == {"engine_verified": 1}
    assert report["summary"]["exact_root_menu_equal_count"] == 1
    case = report["cases"][0]
    assert case["exact_root_menu_equal"] is True
    assert case["root_step"]["resource_accounting_equal"] is True
    assert case["root_step"]["piece_and_card_accounting_equal"] is True
    assert case["root_step"]["hand_delta"] == [0, 0, 0, 1, -4]
    claim = case["explicit_claim_checks"][0]
    assert claim["status"] == "engine_verified"
    branch = claim["branches"][0]
    assert branch["all_followups_replayed"] is True
    assert [step["action"]["kind"] for step in branch["followup_steps"]] == [
        "MARITIME_TRADE",
        "BUILD_SETTLEMENT",
    ]
    assert branch["followup_steps"][0]["exact_action_match_count"] == 1
    assert branch["followup_steps"][0]["resource_premise"]["independently_proven"]
    assert branch["followup_steps"][0]["hand_delta"] == [1, -4, 0, 0, 0]
    assert branch["followup_steps"][1]["hand_delta"] == [-1, -1, -1, -1, 0]
    assert branch["followup_steps"][1]["public_vp_delta"] == 1
    assert branch["checks"]["public_vp_gain_from_before_root"] == 1


def test_identity_command_emits_runtime_binding_without_an_input_packet():
    checkout = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(checkout / "tools" / "public_fact_replay.py"), "--identity"],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    identity = json.loads(result.stdout)
    assert identity["resource_order"] == ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    assert identity["engine"] == json.loads(FIXTURE.read_text())["engine"]
    assert identity["source_files_sha256"] == json.loads(FIXTURE.read_text())["source_files_sha256"]
    assert (
        identity["source_head"]
        == subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    )


def test_changed_full_current_menu_blocks_a_route_even_with_new_hashes():
    case = copy.deepcopy(json.loads(FIXTURE.read_text())["cases"][0])
    menu = case["before"]["observation"]["legal_actions"]
    menu.append(copy.deepcopy(menu[-1]))
    case["observation_sha256"] = canonical_sha256(case["before"])
    case["menu_sha256"] = canonical_sha256(menu)
    result = audit_case(case)
    assert result["status"] == "unverified"
    assert result["exact_root_menu_equal"] is False
    assert result["root_step"] is None


def test_later_trade_must_be_legal_in_its_own_post_root_menu():
    case = copy.deepcopy(json.loads(FIXTURE.read_text())["cases"][0])
    case["projected_steps"][1] = copy.deepcopy(case["projected_steps"][0])
    result = audit_case(case)
    assert result["status"] == "contradicted"
    step = result["explicit_claim_checks"][0]["branches"][0]["followup_steps"][0]
    assert step["exact_action_match_count"] == 0


def test_wrong_digest_or_source_file_binding_stops_before_route_replay(tmp_path):
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_report(FIXTURE, "0" * 64)

    packet = json.loads(FIXTURE.read_text())
    packet["source_files_sha256"]["simulator.py"] = "0" * 64
    changed = tmp_path / "changed-source.json"
    changed.write_text(json.dumps(packet, sort_keys=True))
    digest = hashlib.sha256(changed.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source file hashes differ"):
        build_report(changed, digest)


def test_preloaded_simulator_from_another_checkout_is_rejected(monkeypatch):
    monkeypatch.setattr(simulator_module, "__file__", "/tmp/other-checkout/simulator.py")
    with pytest.raises(RuntimeError, match="imported simulator.py"):
        build_report(FIXTURE, FIXTURE_SHA256)


def test_cli_write_is_idempotent_and_never_replaces_a_changed_report(tmp_path):
    checkout = Path(__file__).resolve().parents[1]
    output = tmp_path / "route-report.json"
    command = [
        sys.executable,
        str(checkout / "tools" / "public_fact_replay.py"),
        "--input",
        str(FIXTURE),
        "--expected-sha256",
        FIXTURE_SHA256,
        "--output",
        str(output),
        "--write",
    ]
    first = subprocess.run(command, cwd=checkout, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    first_bytes = output.read_bytes()
    second = subprocess.run(command, cwd=checkout, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes

    output.write_text('{"bad":true}\n')
    changed_bytes = output.read_bytes()
    third = subprocess.run(command, cwd=checkout, capture_output=True, text=True, check=False)
    assert third.returncode != 0
    assert "existing report differs" in third.stderr
    assert output.read_bytes() == changed_bytes


def test_cli_rejects_byte_changed_report_even_when_json_is_equal(tmp_path):
    checkout = Path(__file__).resolve().parents[1]
    output = tmp_path / "route-report.json"
    command = [
        sys.executable,
        str(checkout / "tools" / "public_fact_replay.py"),
        "--input",
        str(FIXTURE),
        "--expected-sha256",
        FIXTURE_SHA256,
        "--output",
        str(output),
        "--write",
    ]
    first = subprocess.run(command, cwd=checkout, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    output.write_bytes(output.read_bytes() + b"\n")
    altered = output.read_bytes()
    second = subprocess.run(command, cwd=checkout, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "existing report differs" in second.stderr
    assert output.read_bytes() == altered


def test_cli_rejects_symlinked_report_output(tmp_path):
    checkout = Path(__file__).resolve().parents[1]
    real_output = tmp_path / "real-report.json"
    link_output = tmp_path / "linked-report.json"
    command = [
        sys.executable,
        str(checkout / "tools" / "public_fact_replay.py"),
        "--input",
        str(FIXTURE),
        "--expected-sha256",
        FIXTURE_SHA256,
        "--output",
        str(real_output),
        "--write",
    ]
    first = subprocess.run(command, cwd=checkout, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    link_output.symlink_to(real_output)
    command[command.index(str(real_output))] = str(link_output)
    second = subprocess.run(command, cwd=checkout, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "symbolic link" in second.stderr
