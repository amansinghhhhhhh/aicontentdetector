import os
import re
import threading
from typing import List, Optional, Sequence

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from ml import config as cfg


AI_TOKENS = {
    "ai",
    "gpt",
    "chatgpt",
    "llm",
    "machine",
    "generated",
    "synthetic",
    "fake",
    "bot",
    "robot",
    "edited",
    "humanized",
    "paraphrased",
    "artificial",
}
HUMAN_TOKENS = {
    "human",
    "real",
    "original",
    "organic",
    "genuine",
    "authentic",
    "person",
    "people",
    "natural",
    "written",
}


def _tokens(label: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", label.lower()) if t]


def _is_ai_label(label: str) -> bool:
    toks = _tokens(label)
    if "humanized" in label.lower():
        return True
    if any(t in AI_TOKENS for t in toks):
        return True
    return any(t.startswith("ai") and len(t) <= 9 for t in toks)


def _is_human_label(label: str) -> bool:
    if "humanized" in label.lower():
        return False
    return any(t in HUMAN_TOKENS for t in _tokens(label))


def _labels_from_config(config, fallback: Optional[List[str]]) -> List[str]:
    id2label = getattr(config, "id2label", None) or {}
    if id2label:
        try:
            labels = [id2label[i] for i in sorted(int(k) for k in id2label)]
        except (TypeError, ValueError):
            labels = [str(v) for _, v in sorted(id2label.items())]
        if labels and not all(re.fullmatch(r"LABEL_\d+", l or "") for l in labels):
            return [l or "" for l in labels]
    if fallback:
        return list(fallback)
    n = int(getattr(config, "detector_num_labels", None) or config.num_labels or 2)
    return [f"LABEL_{i}" for i in range(n)]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _select_backend() -> str:
    pref = os.environ.get("DETECTOR_BACKEND", "auto").lower()
    if pref == "torch":
        return "torch"
    if pref == "onnx":
        if cfg.onnx_model_path() is None:
            raise RuntimeError(
                "DETECTOR_BACKEND=onnx but no ONNX model found. "
                "Run: python ml/export_onnx.py"
            )
        return "onnx"
    if cfg.onnx_model_path() is not None and _onnx_available():
        return "onnx"
    return "torch"


class DetectorModel:
    def __init__(self, source: Optional[str] = None):
        self.source = source or cfg.resolve_model_source()
        self._lock = threading.Lock()
        self.tokenizer = None
        self.model = None  # torch module (torch backend only)
        self.session = None  # onnxruntime InferenceSession (onnx backend only)
        self.backend: Optional[str] = None
        self.labels: List[str] = []
        self.num_labels = 0
        self.is_tropa = False
        self.human_index: Optional[int] = None
        self.ai_indices: List[int] = []
        self.human_indices: List[int] = []

    def load(self) -> "DetectorModel":
        with self._lock:
            if self.model is not None or self.session is not None:
                return self
            tokenizer = AutoTokenizer.from_pretrained(self.source)
            config = AutoConfig.from_pretrained(self.source)
            self.backend = _select_backend()

            if self.backend == "onnx":
                self._load_onnx(config)
            else:
                self._load_torch(config)

            self.tokenizer = tokenizer
            self.num_labels = len(self.labels)

            if self.is_tropa:
                human = [i for i, l in enumerate(self.labels) if l.lower() == cfg.HUMAN_LABEL]
                self.human_index = human[0] if human else 0
            else:
                self.ai_indices = [i for i, l in enumerate(self.labels) if _is_ai_label(l)]
                self.human_indices = [
                    i for i, l in enumerate(self.labels) if _is_human_label(l)
                ]
                override = _env_label_override()
                if override:
                    lowered = [l.lower() for l in self.labels]
                    self.ai_indices = [i for i, l in enumerate(lowered) if l in override]
                    self.human_indices = []
        return self

    def _load_torch(self, config) -> None:
        from ml._torch_model import load_torch_model

        model, is_tropa = load_torch_model(self.source, config)
        model.eval()
        self.model = model
        self.is_tropa = is_tropa
        if is_tropa:
            self.labels = _labels_from_config(config, cfg.DEFAULT_LABELS)
        else:
            self.labels = _labels_from_config(config, None)

    def _load_onnx(self, config) -> None:
        import onnxruntime as ort

        path = cfg.onnx_model_path()
        if path is None:
            raise RuntimeError("ONNX model path not found")
        self.is_tropa = bool(getattr(config, "detector_num_labels", None))
        if self.is_tropa:
            self.labels = _labels_from_config(config, cfg.DEFAULT_LABELS)
        else:
            self.labels = _labels_from_config(config, None)

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        self.session = ort.InferenceSession(
            str(path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        out_meta = self.session.get_outputs()[0]
        out_dim = out_meta.shape[-1]
        if isinstance(out_dim, int) and out_dim != len(self.labels):
            raise RuntimeError(
                f"ONNX output dim {out_dim} != expected labels {len(self.labels)}"
            )
        # Warm the ORT threadpool/graph so the first request is not slow.
        dummy = np.zeros((1, 8), dtype=np.int64)
        self.session.run(["logits"], {"input_ids": dummy, "attention_mask": dummy})

    @property
    def loaded(self) -> bool:
        return self.model is not None or self.session is not None

    def predict_probs(self, texts: Sequence[str], max_length: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.num_labels), dtype=np.float32)
        if self.model is None and self.session is None:
            self.load()

        enc = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids, masks = enc["input_ids"], enc["attention_mask"]
        order = sorted(range(len(texts)), key=lambda i: len(input_ids[i]))

        # Padded-cost aware batching: max_len * count <= token budget, hard cap 16.
        batches: List[List[int]] = []
        current: List[int] = []
        current_max = 0
        for i in order:
            length = len(input_ids[i])
            if current and (
                current_max * (len(current) + 1) > cfg.MAX_TOKENS_PER_BATCH
                or len(current) >= 16
            ):
                batches.append(current)
                current, current_max = [], 0
            current.append(i)
            current_max = max(current_max, length)
        if current:
            batches.append(current)

        out = np.zeros((len(texts), self.num_labels), dtype=np.float32)

        with self._lock:
            for idxs in batches:
                padded = self.tokenizer.pad(
                    {
                        "input_ids": [input_ids[i] for i in idxs],
                        "attention_mask": [masks[i] for i in idxs],
                    },
                    padding="longest",
                    return_tensors="np",
                )
                ids_np = padded["input_ids"].astype(np.int64)
                mask_np = padded["attention_mask"].astype(np.int64)
                if self.session is not None:
                    logits = self.session.run(
                        ["logits"], {"input_ids": ids_np, "attention_mask": mask_np}
                    )[0]
                    if self.num_labels == 1:
                        probs = _sigmoid(logits)
                    else:
                        probs = _softmax(logits.astype(np.float32))
                else:
                    import torch

                    with torch.inference_mode():
                        result = self.model(
                            input_ids=torch.from_numpy(ids_np),
                            attention_mask=torch.from_numpy(mask_np),
                        )
                        if hasattr(result, "logits"):
                            result = result.logits
                        if self.num_labels == 1:
                            probs = torch.sigmoid(result)
                        else:
                            probs = torch.softmax(result.float(), dim=-1)
                    probs = probs.cpu().numpy()
                for row, i in enumerate(idxs):
                    out[i] = probs[row]
        return out

    def ai_scores_from_probs(self, probs: np.ndarray) -> np.ndarray:
        if probs.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self.num_labels == 1:
            scores = probs[:, 0]
            if self.human_indices:
                scores = 1.0 - scores
            return scores.astype(np.float32)
        if self.is_tropa and self.human_index is not None:
            return (1.0 - probs[:, self.human_index]).astype(np.float32)
        if self.ai_indices:
            return probs[:, self.ai_indices].sum(axis=1).astype(np.float32)
        if self.human_indices:
            return (1.0 - probs[:, self.human_indices].sum(axis=1)).astype(np.float32)
        if self.num_labels == 2:
            return probs[:, 1].astype(np.float32)
        return probs.max(axis=1).astype(np.float32)

    def ai_scores(self, texts: Sequence[str], max_length: int) -> List[float]:
        if not texts:
            return []
        probs = self.predict_probs(texts, max_length)
        return [float(s) for s in self.ai_scores_from_probs(probs)]

    def score_document(
        self, chunk_texts: Sequence[str], max_length: int = cfg.MAX_SEQ_LENGTH
    ) -> dict:
        """Length-weighted average class probabilities over sentence-aligned chunks."""
        if not chunk_texts:
            raise ValueError("No chunks to score")
        probs = self.predict_probs(chunk_texts, max_length)
        weights = np.array(
            [max(1, len(str(c).split())) for c in chunk_texts], dtype=np.float64
        )
        avg = (probs * weights[:, None]).sum(axis=0) / weights.sum()
        ai_score = float(self.ai_scores_from_probs(avg[None, :])[0])
        return {
            "ai_score": ai_score,
            "class_probs": {label: float(p) for label, p in zip(self.labels, avg)},
        }


def _env_label_override() -> set:
    raw = os.environ.get("DETECTOR_AI_LABELS", "")
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


_model: Optional[DetectorModel] = None
_model_lock = threading.Lock()


def get_model() -> DetectorModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = DetectorModel().load()
    return _model


def model_loaded() -> bool:
    return _model is not None


def model_source() -> Optional[str]:
    return _model.source if _model is not None else None


def model_backend() -> Optional[str]:
    return _model.backend if _model is not None else None
