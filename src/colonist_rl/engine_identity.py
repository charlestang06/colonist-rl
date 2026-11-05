"""Exact engine identity shared by simulation and persisted artifacts."""

from __future__ import annotations

import importlib.metadata
import json

ENGINE_DISTRIBUTION_VERSION = "3.3.0"
ENGINE_REVISION = "82aae93ab1f7c267218be0566df573ce477ec3d8"
NUMBER_PLACEMENT = "official_spiral"
SEED_OFFSET = 1
BACKEND_VERSION = f"{ENGINE_DISTRIBUTION_VERSION}+git.{ENGINE_REVISION}"


def verify_engine_installation() -> dict[str, str | int]:
    """Require the pinned release and its full PEP 610 Git commit provenance."""
    try:
        distribution = importlib.metadata.distribution("catanatron")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("catanatron is not installed") from exc
    if distribution.version != ENGINE_DISTRIBUTION_VERSION:
        raise RuntimeError(f"catanatron=={ENGINE_DISTRIBUTION_VERSION} is required")
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("catanatron has malformed PEP 610 provenance") from exc
    if not isinstance(direct_url, dict):
        # Metadata format errors are installation failures at this public boundary.
        raise RuntimeError("catanatron is missing PEP 610 VCS provenance")  # noqa: TRY004
    vcs = direct_url.get("vcs_info")
    source_url = direct_url.get("url")
    if (
        not isinstance(vcs, dict)
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != ENGINE_REVISION
        or not isinstance(source_url, str)
        or not source_url
    ):
        raise RuntimeError(f"catanatron requires Git commit {ENGINE_REVISION} provenance")
    return {
        "distribution": "catanatron",
        "version": distribution.version,
        "commit": vcs["commit_id"],
        "source_url": source_url,
        "backend_version": BACKEND_VERSION,
        "number_placement": NUMBER_PLACEMENT,
        "seed_offset": SEED_OFFSET,
    }
