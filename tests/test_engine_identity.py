"""Exact Git provenance is required even when the release version matches."""

import gzip
import importlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SHA = "82aae93ab1f7c267218be0566df573ce477ec3d8"
TOKEN = f"3.3.0+git.{SHA}"


def identity():
    assert importlib.util.find_spec("colonist_rl.engine_identity") is not None, (
        "exact engine provenance verifier is missing"
    )
    return importlib.import_module("colonist_rl.engine_identity")


def test_real_installation_has_exact_identity_and_isolated_import_origin():
    engine = identity()
    import catanatron

    from colonist_rl.simulator import BACKEND_VERSION

    direct_url = json.loads(importlib.metadata.distribution("catanatron").read_text("direct_url.json"))
    provenance = engine.verify_engine_installation()
    assert provenance == {
        "distribution": "catanatron",
        "version": "3.3.0",
        "commit": SHA,
        "source_url": direct_url["url"],
        "backend_version": TOKEN,
        "number_placement": "official_spiral",
        "seed_offset": 1,
    }
    assert BACKEND_VERSION == engine.BACKEND_VERSION == TOKEN
    assert Path(catanatron.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert Path(sys.executable).parent == Path(sys.prefix) / "bin"
    assert Path(sys.prefix).name == ".venv"


def install_metadata(monkeypatch, *, version="3.3.0", record=None):
    distribution = SimpleNamespace(version=version, read_text=lambda name: record)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)


def test_exact_git_commit_accepts_local_vcs_url(monkeypatch):
    engine = identity()
    record = {"url": "file:///verified/upstream", "vcs_info": {"vcs": "git", "commit_id": SHA}}
    install_metadata(monkeypatch, record=json.dumps(record))
    assert engine.verify_engine_installation()["source_url"] == record["url"]


@pytest.mark.parametrize(
    "record",
    [
        None,
        "not json",
        "[]",
        "null",
        "{}",
        '{"url":"https://github.com/bcollazo/catanatron.git","dir_info":{}}',
        json.dumps({"url": "x", "vcs_info": {"vcs": "hg", "commit_id": SHA}}),
        json.dumps({"url": "x", "vcs_info": {"vcs": "git", "commit_id": "0" * 40}}),
        json.dumps({"url": "x", "vcs_info": {"vcs": "git", "commit_id": SHA[:7]}}),
        json.dumps({"url": "x", "vcs_info": []}),
        json.dumps({"url": None, "vcs_info": {"vcs": "git", "commit_id": SHA}}),
    ],
)
def test_missing_malformed_non_vcs_or_wrong_commit_is_rejected(monkeypatch, record):
    engine = identity()
    install_metadata(monkeypatch, record=record)
    with pytest.raises(RuntimeError, match="catanatron.*provenance|catanatron.*commit"):
        engine.verify_engine_installation()


def test_wrong_distribution_version_is_rejected_even_at_exact_commit(monkeypatch):
    engine = identity()
    record = {"url": "file:///verified/upstream", "vcs_info": {"vcs": "git", "commit_id": SHA}}
    install_metadata(monkeypatch, version="3.2.1", record=json.dumps(record))
    with pytest.raises(RuntimeError, match="3.3.0"):
        engine.verify_engine_installation()


def test_missing_distribution_fails_clearly(monkeypatch):
    engine = identity()

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    with pytest.raises(RuntimeError, match="catanatron.*installed"):
        engine.verify_engine_installation()


def test_exact_token_flows_through_ordinary_checkpoint_and_dataset(tmp_path):
    import torch

    from colonist_rl.imitation import (
        collect_demonstrations,
        load_demonstrations,
        save_demonstrations,
    )
    from colonist_rl.policy import GraphPolicy
    from colonist_rl.training import load_checkpoint, save_checkpoint

    path = tmp_path / "new.pt"
    save_checkpoint(path, GraphPolicy(hidden_size=16), steps=0)
    _, state = load_checkpoint(path)
    assert state["backend_version"] == TOKEN
    legacy = torch.load(path, weights_only=True)
    legacy["backend_version"] = "3.2.1"
    torch.save(legacy, tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="backend|incompatible"):
        load_checkpoint(tmp_path / "legacy.pt")
    dataset = collect_demonstrations(board_seeds=[7], max_decisions=1, max_game_steps=1)
    assert dataset["backend_version"] == TOKEN
    save_demonstrations(tmp_path / "new.json.gz", dataset)
    assert load_demonstrations(tmp_path / "new.json.gz")["backend_version"] == TOKEN
    legacy_dataset = json.loads(gzip.decompress((tmp_path / "new.json.gz").read_bytes()))
    legacy_dataset["backend_version"] = "3.2.1"
    (tmp_path / "legacy.json.gz").write_bytes(gzip.compress(json.dumps(legacy_dataset).encode()))
    with pytest.raises(ValueError, match="backend|incompatible"):
        load_demonstrations(tmp_path / "legacy.json.gz")
