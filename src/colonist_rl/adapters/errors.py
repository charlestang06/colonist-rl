"""Errors shared by observation ingestion and environment adapters."""


class PerceptionError(ValueError):
    """Input or a UI transition cannot be established unambiguously."""
