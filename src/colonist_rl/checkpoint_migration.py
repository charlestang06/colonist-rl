"""Explicit, audited policy-weight transfer from the legacy 3.2.1 engine."""

from __future__ import annotations

import io
import os
import random
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .engine_identity import BACKEND_VERSION, verify_engine_installation
from .policy import KINDS, policy_from_architecture
from .training import save_checkpoint


def _read_checkpoint(path):
    # Hash exactly the immutable byte snapshot that the restricted loader sees.
    source = Path(path).read_bytes()
    state = torch.load(io.BytesIO(source), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        # Malformed on-disk payloads share the ordinary loader's ValueError boundary.
        raise ValueError("checkpoint must contain labelled checkpoint data")  # noqa: TRY004
    return state, sha256(source).hexdigest()


def inspect_checkpoint(path) -> dict:
    """Read labels and a content hash without constructing a policy or restoring RNG.

    Labels describe the source as stored; inspection does not certify that the
    checkpoint is compatible with this runtime or eligible for migration.
    """
    state, digest = _read_checkpoint(path)
    return {
        "source_sha256": digest,
        **{
            key: state.get(key)
            for key in (
                "schema_version",
                "backend_version",
                "action_kinds",
                "architecture",
                "steps",
            )
        },
    }


def migrate_legacy_weights(source, destination, *, seed=0) -> dict:
    """Create a new-engine checkpoint from exact legacy policy tensors only.

    This is a fresh training start, not a resume or a conversion of historical
    results. The destination must be absent, including dangling symlinks.
    Python/NumPy/CPU Torch RNG are freshly seeded in the output; caller RNG is
    restored even on failure. CUDA RNG is never touched by this CPU-only bridge.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different paths")
    if os.path.lexists(destination):
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    if type(seed) is not int:
        raise ValueError("migration seed must be an integer")
    verify_engine_installation()
    state, digest = _read_checkpoint(source)
    if type(state.get("schema_version")) is not int or state["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema")
    if state.get("backend_version") != "3.2.1":
        raise ValueError("weight migration requires a labelled legacy 3.2.1 checkpoint")
    if state.get("action_kinds") != list(KINDS):
        raise ValueError("checkpoint action schema is incompatible")
    if type(state.get("steps")) is not int or state["steps"] < 0:
        raise ValueError("checkpoint source steps must be a nonnegative integer")
    provenance = {
        "kind": "weights_only_cross_engine",
        "source_sha256": digest,
        "source_backend_version": "3.2.1",
        "target_backend_version": BACKEND_VERSION,
        "source_steps": state["steps"],
        "seed": seed,
    }
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            policy = policy_from_architecture(state.get("architecture"))
            tensors, expected = state.get("policy_state"), policy.state_dict()
            if not isinstance(tensors, dict) or set(tensors) != set(expected):
                raise ValueError("checkpoint policy state keys are incompatible")
            for name, target in expected.items():
                tensor = tensors[name]
                if (
                    type(tensor) is not torch.Tensor
                    or tensor.shape != target.shape
                    or tensor.dtype != target.dtype
                    or tensor.layout != target.layout
                ):
                    raise ValueError(f"checkpoint policy tensor is incompatible: {name}")
            policy.load_state_dict(tensors, strict=True)
            # Seed after construction so the saved RNG starts exactly at seed,
            # independent of architecture initialization or the caller's state.
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            random.seed(seed)
            np.random.seed(seed % (2**32))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(
                prefix=".checkpoint-migration-", dir=destination.parent
            ) as staging:
                staged = Path(staging) / "checkpoint.pt"
                save_checkpoint(staged, policy, steps=0, metadata={"migration": provenance})
                # link() atomically creates a new entry and fails if ANY entry
                # already exists. Never publish with replace() or a check/write.
                os.link(staged, destination, follow_symlinks=False)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    return provenance
