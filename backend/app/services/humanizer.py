"""Thin wrapper around ml.humanize so tests can monkeypatch one place."""

from ml import humanize as _ml_humanize


def run(text: str, tone: str, strength: str, target: str = "external") -> dict:
    return _ml_humanize.humanize(text, tone=tone, strength=strength, target=target)
