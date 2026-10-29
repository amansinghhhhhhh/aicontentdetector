"""Thin wrapper around the ml package so tests can monkeypatch one place."""

from typing import Optional

from ml import detect as _ml_detect
from ml import model as _ml_model


def run_analysis(text: str) -> dict:
    return _ml_detect.analyze_text(text)


def is_model_loaded() -> bool:
    return _ml_model.model_loaded()


def model_source() -> Optional[str]:
    return _ml_model.model_source()


def model_backend() -> Optional[str]:
    return _ml_model.model_backend()


def zerogpt_available() -> bool:
    try:
        from ml import zerogpt_style

        return zerogpt_style.available()
    except Exception:
        return False


def warmup() -> Optional[str]:
    return _ml_detect.warmup()
