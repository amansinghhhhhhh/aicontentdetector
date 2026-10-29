import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
HF_HOME = MODELS_DIR / "hf-home"
LOCAL_MODEL_DIR = MODELS_DIR / "detector"

MODELS_DIR.mkdir(exist_ok=True)
HF_HOME.mkdir(exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = os.environ.get("DETECTOR_MODEL", "wasitaigeneratedcom/ai-text-detector-small")

MAX_SEQ_LENGTH = 768
CHUNK_RESERVE_TOKENS = 2
SENTENCE_MAX_LENGTH = 512
BATCH_SIZE = int(os.environ.get("DETECTOR_BATCH_SIZE", "8"))
MAX_TOKENS_PER_BATCH = int(os.environ.get("DETECTOR_TOKEN_BUDGET", "2048"))

# 4-class order published in the model card (config.id2label wins if present).
DEFAULT_LABELS = ["human", "ai", "ai_edited", "humanized"]
HUMAN_LABEL = "human"

# ai_score is 1 - P(human); verdict thresholds on the CALIBRATED display percent.
AI_VERDICT_THRESHOLD = float(os.environ.get("AI_VERDICT_THRESHOLD", "60"))
MIXED_VERDICT_THRESHOLD = float(os.environ.get("MIXED_VERDICT_THRESHOLD", "25"))
MIN_RELIABLE_WORDS = 50

# Display calibration: raw model scores are heavily skewed (AI saturates near 1.0).
# Map raw -> display percent in logit space so the published operating point
# (threshold_at_0.5fpr in serving_head.json) lands at CALIBRATION_TARGET percent.
CALIBRATION_SCALE = float(os.environ.get("CALIBRATION_SCALE", "1.0"))
CALIBRATION_TARGET = float(os.environ.get("CALIBRATION_TARGET", "0.70"))

# Final percent = CONTEXT_WEIGHT * whole-document score + (1-w) * sentence score.
CONTEXT_WEIGHT = 0.6

# Per-sentence score blending: alpha = own score weight, (1-alpha) = neighbour context.
BLEND_ALPHA = {  # keyed by word-count bucket: min_words -> alpha
    0: 0.30,   # very short sentence: trust neighbours
    8: 0.50,
    15: 0.70,  # long enough to stand on its own
}


ONNX_DIR = MODELS_DIR / "detector-onnx"
ONNX_MODEL_ENV = os.environ.get("ONNX_MODEL_PATH", "")


def resolve_model_source() -> str:
    if (LOCAL_MODEL_DIR / "config.json").exists():
        return str(LOCAL_MODEL_DIR)
    return MODEL_ID


def onnx_model_path() -> Optional[Path]:
    """Path to the ONNX model, or None if not exported yet.

    DeBERTa-v3 does not survive int8 quantization (accuracy collapses), so the
    serving model is fp32 (model_fp32.onnx); a user-supplied model.onnx wins
    if present.
    """
    if ONNX_MODEL_ENV:
        p = Path(ONNX_MODEL_ENV)
        return p if p.exists() else None
    for name in ("model.onnx", "model_fp32.onnx"):
        p = ONNX_DIR / name
        if p.exists():
            return p
    return None


def _load_operating_threshold() -> float:
    import json

    default = 0.9756
    path = LOCAL_MODEL_DIR / "serving_head.json"
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("threshold_at_0.5fpr", default))
    except (ValueError, TypeError, OSError):
        return default


OPERATING_THRESHOLD = _load_operating_threshold()
