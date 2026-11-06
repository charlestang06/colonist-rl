"""Manifest-driven paired gameplay, with durable per-game evidence.

This module chooses policies and persists records; paired_gameplay owns every
game, seat, deadline check, and public policy decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import tomllib
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


_FORMAT = "colonist-rl.paired-experiment"
_VERSION = 1
_ROLES = frozenset(("smoke", "development", "validation", "final_test"))
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def run_paired_panel(*args, **kwargs):
    """Import the game runner only after manifest preflight."""
    from .paired_gameplay import run_paired_panel as run

    return run(*args, **kwargs)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256():
    root = Path(__file__).resolve().parents[2]
    paths = [root / "pyproject.toml", *(root / "src" / "colonist_rl").rglob("*.py")]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{_file_sha256(path)}\n".encode())
    return digest.hexdigest()


def _version_identity():
    from .engine_identity import BACKEND_VERSION, verify_engine_installation

    root = Path(__file__).resolve().parents[2]
    package_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    return {
        "protocol_version": _VERSION,
        "package_version": package_version,
        "backend_version": BACKEND_VERSION,
        "engine_installation": verify_engine_installation(),
    }


def _keys(value, required, optional=(), *, label):
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{label} has missing {sorted(missing)} or unknown {sorted(extra)} keys")


def _hex(value, *, label):
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return value


def _load_json(source_bytes):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate manifest key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(source_bytes, object_pairs_hook=unique_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("manifest must be valid JSON") from exc


def _validate_manifest(raw, *, base):
    _keys(raw,
          ("format", "version", "role", "reference_arm", "board_seeds", "arms", "no_offers",
           "max_game_steps", "wall_seconds"),
          ("opponent_lineup", "capture_candidate_steps", "expected_source_sha256",
           "expected_versions_sha256"), label="manifest")
    if raw["format"] != _FORMAT or type(raw["version"]) is not int or raw["version"] != _VERSION:
        raise ValueError("unsupported experiment manifest format/version")
    if type(raw["role"]) is not str or raw["role"] not in _ROLES:
        raise ValueError("role must be smoke, development, validation, or final_test")
    seeds = raw["board_seeds"]
    if type(seeds) is not list or not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("board_seeds must be explicit nonempty integer list")
    if len(set(seeds)) != len(seeds):
        raise ValueError("board_seeds must be unique")
    if raw["no_offers"] is not True:
        raise ValueError("this public paired runner requires no_offers=true")
    steps = raw["max_game_steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("max_game_steps must be a positive integer")
    wall = raw["wall_seconds"]
    if type(wall) not in (int, float) or not math.isfinite(wall) or not 0 < wall <= 3600:
        raise ValueError("wall_seconds must be finite and in (0, 3600]")
    capture = raw.get("capture_candidate_steps", False)
    if type(capture) is not bool:
        raise ValueError("capture_candidate_steps must be a boolean")
    lineup_input = raw.get("opponent_lineup", ["heuristic"] * 3)
    if type(lineup_input) is not list or len(lineup_input) != 3:
        raise ValueError("opponent_lineup must be exactly three names")
    from .training import _lineup

    lineup = _lineup(lineup_input, allow_native=True)
    if "historical" in lineup:
        raise ValueError("historical opponents require a frozen pool not configured here")

    from .public_turn_planner import PublicTurnPlannerPolicy, TurnPlannerConfig

    arms_input = raw["arms"]
    if type(arms_input) is not dict or len(arms_input) < 2:
        raise ValueError("arms must name at least two policies")
    arms, checkpoints = {}, {}
    for name, spec in arms_input.items():
        if (type(name) is not str or not name or len(name) > 64
                or name.strip() != name or not name.isprintable()):
            raise ValueError("arm names must be nonempty printable labels up to 64 chars")
        if type(spec) is not dict or spec.get("kind") not in (
            "public_planner", "city_before_purchase", "public_option_purchase",
            "graph_checkpoint"
        ):
            raise ValueError(f"unsupported arm kind: {name}")
        kind = spec["kind"]
        if kind in ("public_planner", "city_before_purchase", "public_option_purchase"):
            _keys(spec, ("kind",), ("config", "gain_margin_micro"), label=f"arm {name}")
            config_input = spec.get("config", {})
            if type(config_input) is not dict:
                raise ValueError(f"arm {name} config must be an object")
            try:
                config = TurnPlannerConfig(**config_input)
                options = ({"gain_margin_micro": spec["gain_margin_micro"]}
                           if "gain_margin_micro" in spec else {})
                planner = PublicTurnPlannerPolicy(config, **options)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid planner configuration for arm {name}") from exc
            arms[name] = {
                "kind": kind,
                "config": asdict(config),
                "gain_margin_micro": planner.gain_margin_micro,
            }
        else:
            _keys(spec, ("kind", "checkpoint", "checkpoint_sha256"), label=f"arm {name}")
            if type(spec["checkpoint"]) is not str or not spec["checkpoint"]:
                raise ValueError(f"arm {name} checkpoint must be a path")
            expected = _hex(spec["checkpoint_sha256"], label=f"arm {name} checkpoint sha256")
            try:
                path = (base / spec["checkpoint"]).resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError(f"arm {name} checkpoint does not exist") from exc
            if not path.is_file():
                raise ValueError(f"arm {name} checkpoint must be a file")
            actual = _file_sha256(path)
            if actual != expected:
                raise ValueError(f"arm {name} checkpoint sha256 mismatch")
            arms[name] = {"kind": kind, "checkpoint": str(path), "checkpoint_sha256": actual}
            checkpoints[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    if not any(spec["kind"] == "public_planner" for spec in arms.values()):
        raise ValueError("at least one raw public_planner arm is required")
    reference = raw["reference_arm"]
    if type(reference) is not str or reference not in arms:
        raise ValueError("reference_arm must name a declared arm")
    arms = {reference: arms[reference], **{name: spec for name, spec in arms.items()
                                         if name != reference}}

    validated = {
        "format": _FORMAT, "version": _VERSION, "role": raw["role"],
        "reference_arm": reference, "board_seeds": list(seeds),
        "arms": arms, "opponent_lineup": list(lineup),
        "no_offers": True, "max_game_steps": steps, "wall_seconds": float(wall),
        "capture_candidate_steps": capture,
    }
    for key in ("expected_source_sha256", "expected_versions_sha256"):
        if key in raw:
            validated[key] = _hex(raw[key], label=key)
    return validated, checkpoints


def _load_graph_policy(path):
    """Checkpoint loading may restore saved RNG; leave the caller's RNG intact."""
    import random

    import numpy as np
    import torch

    from .training import load_checkpoint

    python_state, numpy_state, torch_state = (
        random.getstate(), np.random.get_state(), torch.get_rng_state()
    )
    try:
        return load_checkpoint(path)[0]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def _factories(validated):
    from .city_before_purchase import CityBeforePurchasePolicy
    from .public_option_purchase import PublicOptionPurchasePolicy
    from .public_turn_planner import PublicTurnPlannerPolicy, TurnPlannerConfig

    result = {}
    for name, spec in validated["arms"].items():
        kind = spec["kind"]
        if kind == "graph_checkpoint":
            path = Path(spec["checkpoint"])
            template = _load_graph_policy(path)
            if _file_sha256(path) != spec["checkpoint_sha256"]:
                raise ValueError(f"arm {name} checkpoint changed while loading")
            result[name] = lambda template=template: deepcopy(template)
        else:
            config = TurnPlannerConfig(**spec["config"])
            gain = spec["gain_margin_micro"]

            def make(kind=kind, config=config, gain=gain):
                parent = PublicTurnPlannerPolicy(config, gain_margin_micro=gain)
                if kind == "city_before_purchase":
                    return CityBeforePurchasePolicy(parent, enabled=True)
                if kind == "public_option_purchase":
                    return PublicOptionPurchasePolicy(parent, enabled=True)
                return parent

            result[name] = make
    return result


def _new_run_dir(root):
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        name = datetime.now(timezone.utc).strftime("paired-%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        candidate = root / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError("could not allocate a fresh run directory")


def _write_json(path, value):
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def _json_line(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _line_count(path):
    with path.open(encoding="utf-8") as source:
        return sum(1 for _ in source)


def _candidate_step_record(event):
    step = event.step
    before, after, action = step.before, step.after, step.action
    target_level = next((building.level for building in after.buildings
                         if building.owner == before.player_id
                         and building.vertex == action.target), None)
    return {
        "arm": event.arm,
        "board_seed": event.board_seed,
        "seat": before.player_id,
        "before": {
            "version": before.context.version,
            "phase": before.context.kind,
            "turn_number": before.context.turn_number,
            "turn_owner": before.context.turn_owner,
            "hand": list(before.hand),
            "known_own_vp": before.own_victory_points,
        },
        "action": asdict(action),
        "after": {
            "version": after.context.version,
            "phase": after.context.kind,
            "hand": list(after.hand),
            "known_own_vp": after.own_victory_points,
            "target_building_level": target_level,
        },
        "terminated": bool(step.terminated),
        "truncated": bool(step.truncated),
    }


def _journal_valid(result, journal):
    expected = [
        {"arm": name, **row}
        for name, arm in result["arms"].items()
        for board in arm["boards"]
        for row in board["rows"] + board["partial_rows"]
    ]

    def keyed(rows):
        mapped = {}
        for row in rows:
            try:
                key = (row["arm"], row["board_seed"], row["seat"])
            except (KeyError, TypeError):
                return None
            if key in mapped:
                return None
            mapped[key] = row
        return mapped

    return len(journal) == len(expected) and keyed(journal) == keyed(expected)


def _compact_arms(result):
    return {
        name: {
            **{key: value for key, value in arm.items() if key != "boards"},
            "boards": [{key: value for key, value in board.items()
                        if key not in ("rows", "partial_rows")} for board in arm["boards"]],
        }
        for name, arm in result["arms"].items()
    }


def run_manifest(manifest_path, output_root):
    """Validate inputs, create a fresh run, and persist matched public gameplay.

    ``wall_seconds`` starts after manifest/source/checkpoint preflight and is a
    soft evaluator deadline checked between decisions, not a process hard cap.
    """
    path = Path(manifest_path).expanduser().resolve(strict=True)
    source_bytes = path.read_bytes()
    input_sha256 = hashlib.sha256(source_bytes).hexdigest()
    validated, checkpoints = _validate_manifest(_load_json(source_bytes), base=path.parent)
    source_before = _source_tree_sha256()
    versions_before = _version_identity()
    versions_sha256 = _digest(versions_before)
    if validated.get("expected_source_sha256", source_before) != source_before:
        raise ValueError("expected_source_sha256 does not match current source")
    if validated.get("expected_versions_sha256", versions_sha256) != versions_sha256:
        raise ValueError("expected_versions_sha256 does not match runtime versions")
    factories = _factories(validated)

    run_dir = _new_run_dir(Path(output_root).expanduser().resolve())
    deadline = time.monotonic() + validated["wall_seconds"]
    manifest = {
        "format": "colonist-rl.paired-run", "version": _VERSION,
        "run_id": run_dir.name, "started_unix_seconds": time.time(),
        "deadline_monotonic": deadline,
        "input_manifest_path": str(path), "input_manifest_sha256": input_sha256,
        "validated_manifest": validated, "validated_manifest_sha256": _digest(validated),
        "source_sha256": source_before,
        "versions": versions_before, "versions_sha256": versions_sha256,
        "checkpoints": checkpoints,
        "board_freshness": "caller_declared",
        "wall_budget_scope": "gameplay_after_preflight",
        "deadline_enforcement": "between_evaluator_decisions",
    }
    manifest_path_out = run_dir / "manifest.json"
    _write_json(manifest_path_out, manifest)
    saved_manifest_sha256 = _file_sha256(manifest_path_out)
    games_path = run_dir / "games.jsonl"
    steps_path = run_dir / "candidate_steps.jsonl"
    try:
        with games_path.open("w", encoding="utf-8", buffering=1) as game_file:
            step_context = (steps_path.open("w", encoding="utf-8", buffering=1)
                            if validated["capture_candidate_steps"] else nullcontext(None))
            with step_context as step_file:
                def game_end(row):
                    if step_file is not None:
                        step_file.flush()
                        os.fsync(step_file.fileno())
                    game_file.write(_json_line(row))
                    game_file.flush()
                    os.fsync(game_file.fileno())

                def candidate_step(event):
                    step_file.write(_json_line(_candidate_step_record(event)))

                result = run_paired_panel(
                    factories,
                    board_seeds=validated["board_seeds"],
                    opponent_lineup=tuple(validated["opponent_lineup"]),
                    max_game_steps=validated["max_game_steps"],
                    deadline=deadline,
                    no_offers=True,
                    on_game_end=game_end,
                    on_candidate_step=(candidate_step
                                       if validated["capture_candidate_steps"] else None),
                )
                if result["paired_summary"].get("reference_arm") != validated["reference_arm"]:
                    raise ValueError("runner paired_summary reference_arm mismatch")
    except Exception as exc:
        _write_json(run_dir / "summary.json", {
            "status": "failed", "run_dir": str(run_dir),
            "error_type": type(exc).__name__, "error": str(exc),
            "manifest_sha256": saved_manifest_sha256,
            "journal_games": _line_count(games_path) if games_path.exists() else 0,
        })
        raise

    journal = [json.loads(line) for line in games_path.read_text().splitlines()]
    valid_journal = _journal_valid(result, journal)
    source_after = _source_tree_sha256()
    versions_after = _version_identity()
    input_after = _file_sha256(path)
    checkpoint_after = {name: _file_sha256(ref["path"]) for name, ref in checkpoints.items()}
    errors = []
    if source_after != source_before:
        errors.append("source_changed")
    if _digest(versions_after) != versions_sha256:
        errors.append("versions_changed")
    if input_after != input_sha256:
        errors.append("input_manifest_changed")
    if _file_sha256(manifest_path_out) != saved_manifest_sha256:
        errors.append("saved_manifest_changed")
    for name, ref in checkpoints.items():
        if checkpoint_after[name] != ref["sha256"]:
            errors.append(f"checkpoint_changed:{name}")
    status = (
        "invalid_provenance" if errors else
        "invalid_journal" if not valid_journal else result["status"]
    )
    summary = {
        "format": "colonist-rl.paired-summary", "version": _VERSION,
        "status": status, "runner_status": result["status"],
        "run_dir": str(run_dir), "summary_path": str(run_dir / "summary.json"),
        "role": validated["role"], "manifest_sha256": saved_manifest_sha256,
        "validated_manifest_sha256": manifest["validated_manifest_sha256"],
        "source_sha256_start": source_before, "source_sha256_end": source_after,
        "versions_sha256_start": versions_sha256,
        "versions_sha256_end": _digest(versions_after),
        "checkpoint_sha256_after": checkpoint_after,
        "provenance_errors": errors,
        "journal_valid": valid_journal, "journal_games": len(journal),
        "games_jsonl_sha256": _file_sha256(games_path),
        "candidate_steps": (_line_count(steps_path)
                            if validated["capture_candidate_steps"] else 0),
        "candidate_steps_jsonl_sha256": (_file_sha256(steps_path)
                                         if validated["capture_candidate_steps"] else None),
        "requested_board_seeds": result["requested_board_seeds"],
        "paired_complete_board_seeds": result["paired_complete_board_seeds"],
        "paired_resolved_board_seeds": result["paired_resolved_board_seeds"],
        "paired_summary": result["paired_summary"],
        "arms": _compact_arms(result),
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manifest-driven paired public gameplay")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_manifest(args.manifest, args.output_root)
    print(json.dumps({key: summary[key] for key in ("status", "run_dir", "summary_path")},
                     sort_keys=True))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
