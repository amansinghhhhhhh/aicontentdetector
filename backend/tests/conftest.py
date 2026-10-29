import os
import tempfile

_TMP_DIR = tempfile.mkdtemp(prefix="aidetector-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DIR}/test.db"
os.environ["MODEL_WARMUP"] = "0"

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import detector


def fake_analysis(text: str) -> dict:
    is_ai = "SYNTHETIC_MARKER" in text
    score = 92.0 if is_ai else 8.0
    words = len(text.split())
    return {
        "text": text,
        "ai_percent": score,
        "verdict": "ai" if is_ai else "human",
        "zerogpt": {"pct": 88.0 if is_ai else 21.0, "prob": 0.88 if is_ai else 0.21},
        "confidence": 84.0,
        "class_percent": {"human": 8.0, "ai": 92.0, "ai_edited": 0.0, "humanized": 0.0},
        "stats": {
            "characters": len(text),
            "words": words,
            "sentences": 2,
            "chunks": 1,
            "ai_sentences": 2 if is_ai else 0,
        },
        "sentences": [
            {
                "index": 0,
                "text": text[:60],
                "start": 0,
                "end": min(60, len(text)),
                "words": words,
                "score": score,
            }
        ],
        "warnings": [],
        "elapsed_ms": 1.0,
        "model": "fake-model",
    }


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(detector, "run_analysis", fake_analysis)
    with TestClient(app) as test_client:
        test_client.delete("/api/history")
        yield test_client
