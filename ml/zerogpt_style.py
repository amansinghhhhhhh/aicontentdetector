"""Local ZeroGPT-style score: distilgpt2 perplexity + burstiness features.

Fitted offline on the public ZeroGPT-labelled benchmark
(wasitaigeneratedcom/ai-detector-benchmark-2026-09); coefficients live in
models/zerogpt_style.json. Runtime inference is pure numpy.
"""

import json
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from ml.text_utils import clean_text, split_sentences, word_count

MODEL_ID = "distilgpt2"
MAX_TOKENS = 768
MAX_TOKENS_PER_BATCH = 1024
CALIBRATOR_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "zerogpt_style.json"
)

_BULLET_RE = re.compile(r"^\s*([-*\u2022]|\d+[.)])\s+")
_HEADING_RE = re.compile(r"^\s*#+\s+")

_lock = threading.Lock()
_state: dict = {"loaded": False, "calibrator": None, "tokenizer": None, "model": None}


def available() -> bool:
    return CALIBRATOR_PATH.exists()


def _load_calibrator() -> Optional[dict]:
    if _state["calibrator"] is not None:
        return _state["calibrator"]
    if not CALIBRATOR_PATH.exists():
        return None
    with CALIBRATOR_PATH.open(encoding="utf-8") as fh:
        _state["calibrator"] = json.load(fh)
    return _state["calibrator"]


def _load_ppl():
    if _state["model"] is not None:
        return _state["tokenizer"], _state["model"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
    model.config.pad_token_id = tokenizer.eos_token_id
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    return tokenizer, model


def token_nlls(tokenizer, model, texts):
    import torch

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_TOKENS,
        padding=False,
        return_offsets_mapping=True,
    )
    lengths = [len(ids) for ids in enc["input_ids"]]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])

    batches = []
    current: list = []
    current_max = 0
    for i in order:
        length = lengths[i]
        if current and (
            current_max * (len(current) + 1) > MAX_TOKENS_PER_BATCH
            or len(current) >= 8
        ):
            batches.append(current)
            current, current_max = [], 0
        current.append(i)
        current_max = max(current_max, length)
    if current:
        batches.append(current)

    results = [None] * len(texts)
    for idxs in batches:
        batch = tokenizer.pad(
            {
                "input_ids": [enc["input_ids"][i] for i in idxs],
                "attention_mask": [enc["attention_mask"][i] for i in idxs],
            },
            padding=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = batch["input_ids"][:, 1:]
        mask = batch["attention_mask"][:, 1:].bool()
        tok_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for row, i in enumerate(idxs):
            if lengths[i] < 2:
                results[i] = (np.zeros(0), [], enc["input_ids"][i])
                continue
            valid = mask[row].cpu().numpy()
            nll = tok_nll[row][mask[row]].cpu().numpy()
            offsets = [
                off
                for off, keep in zip(enc["offset_mapping"][i][1:], valid)
                if keep
            ]
            results[i] = (nll, offsets, enc["input_ids"][i])
    return results


def _sentence_means(nll, offsets, text):
    sents = split_sentences(clean_text(text))
    per_sent = []
    for s in sents:
        vals = [
            nll[j]
            for j, (a, b) in enumerate(offsets)
            if a >= s.start and a < s.end
        ]
        if len(vals) >= 2:
            per_sent.append(float(np.mean(vals)))
    return sents, per_sent


def extract_features(text: str) -> Optional[dict]:
    clean = clean_text(text)
    if not clean:
        return None
    with _lock:
        tokenizer, model = _load_ppl()
        nll, offsets, ids = token_nlls(tokenizer, model, [clean])[0]
    return _features_from(clean, nll, offsets, ids)


def extract_features_many(texts) -> list:
    cleaned = [clean_text(t) if t else "" for t in texts]
    prepared = [(i, c) for i, c in enumerate(cleaned) if c]
    out: list = [None] * len(texts)
    if not prepared:
        return out
    with _lock:
        tokenizer, model = _load_ppl()
        results = token_nlls(tokenizer, model, [c for _, c in prepared])
    for (i, clean), (nll, offsets, ids) in zip(prepared, results):
        out[i] = _features_from(clean, nll, offsets, ids)
    return out


def _features_from(clean: str, nll, offsets, ids) -> Optional[dict]:
    if len(nll) == 0:
        return None
    sents, per_sent = _sentence_means(nll, offsets, clean)
    words = word_count(clean) or 1
    word_list = re.findall(r"\b[\w'-]+\b", clean.lower())
    sent_words = [s.words for s in sents if s.words > 0] or [1]
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()] or [""]
    short_lines = sum(1 for ln in lines if word_count(ln) <= 8)
    bullets = sum(1 for ln in lines if _BULLET_RE.match(ln))
    headings = sum(1 for ln in lines if _HEADING_RE.match(ln))
    caps = sum(1 for ch in clean if ch.isupper())
    letters = sum(1 for ch in clean if ch.isalpha()) or 1
    punctuation = sum(1 for ch in clean if ch in "!?")
    sent_mean = np.array(per_sent) if per_sent else np.array([float(np.mean(nll))])
    word_counts = np.array(sent_words, dtype=float)
    return {
        "words": words,
        "tokens": len(ids),
        "sentences": len(sents) or 1,
        "doc_nll_mean": float(np.mean(nll)),
        "doc_nll_std": float(np.std(nll)),
        "doc_nll_p10": float(np.percentile(nll, 10)),
        "doc_nll_p50": float(np.percentile(nll, 50)),
        "doc_nll_p90": float(np.percentile(nll, 90)),
        "sent_nll_mean": float(sent_mean.mean()),
        "sent_nll_min": float(sent_mean.min()),
        "sent_nll_max": float(sent_mean.max()),
        "sent_nll_std": float(sent_mean.std()),
        "sent_nll_p25": float(np.percentile(sent_mean, 25)),
        "sent_nll_low_frac": float((sent_mean < 3.0).mean()),
        "burst_words_cv": float(word_counts.std() / max(word_counts.mean(), 1)),
        "burst_sent_words_std": float(word_counts.std()),
        "mean_sent_words": float(word_counts.mean()),
        "max_sent_words": float(word_counts.max()),
        "min_sent_words": float(word_counts.min()),
        "ttr": len(set(word_list)) / max(len(word_list), 1),
        "short_line_frac": short_lines / len(lines),
        "bullet_frac": bullets / len(lines),
        "heading_frac": headings / len(lines),
        "caps_frac": caps / letters,
        "punct_per_sent": punctuation / max(len(sents), 1),
        "commas_per_sent": clean.count(",") / max(len(sents), 1),
    }


def _linear(coef, intercept, mean, scale, values) -> float:
    z = (np.asarray(values, dtype=float) - np.asarray(mean)) / np.asarray(scale)
    return float(np.dot(coef, z) + intercept)


def _apply(calibrator, feats: Optional[dict]) -> Optional[dict]:
    if calibrator is None or feats is None:
        return None
    names = calibrator["features"]
    values = [feats.get(name, 0.0) for name in names]
    ridge = calibrator["ridge_pct"]
    logreg = calibrator["logreg"]
    pct = _linear(ridge["coef"], ridge["intercept"], ridge["mean"], ridge["scale"], values)
    pct = float(np.clip(pct, 0.0, 100.0))
    logit = _linear(
        logreg["coef"], logreg["intercept"], logreg["mean"], logreg["scale"], values
    )
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    return {"pct": round(pct, 1), "prob": round(prob, 4), "features": feats}


def score(text: str) -> Optional[dict]:
    """Return {"pct": 0-100 ZeroGPT-style estimate, "prob": 0-1 of pct>=50}."""
    return _apply(_load_calibrator(), extract_features(text))


def score_many(texts) -> list:
    calibrator = _load_calibrator()
    feats_list = extract_features_many(texts)
    return [_apply(calibrator, feats) for feats in feats_list]
