import math
import time
from typing import List, Optional

import numpy as np

from ml import config as cfg
from ml.model import get_model
from ml.text_utils import clean_text, pack_sentences, split_sentences, word_count


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


_BIAS = cfg.CALIBRATION_SCALE * _logit(cfg.CALIBRATION_TARGET)
_LOGIT_T0 = _logit(cfg.OPERATING_THRESHOLD)


def calibrate(raw: float) -> float:
    """Map raw model score to a display percent.

    Raw scores are heavily skewed (AI saturates near 1.0, formal human text sits
    high too). Logit-space mapping puts the published 0.5%-FPR operating point at
    CALIBRATION_TARGET% and keeps human text in the low double digits.
    """
    z = (_logit(raw) - _LOGIT_T0 + _BIAS) / cfg.CALIBRATION_SCALE
    return 100.0 / (1.0 + math.exp(-z))


def _blend_alpha(words: int) -> float:
    alpha = cfg.BLEND_ALPHA[0]
    for threshold in sorted(cfg.BLEND_ALPHA):
        if words >= threshold:
            alpha = cfg.BLEND_ALPHA[threshold]
    return alpha


def _blend(scores: List[float], word_counts: List[int]) -> List[float]:
    n = len(scores)
    if n <= 1:
        return list(scores)
    overall = sum(scores) / n
    blended = []
    for i, score in enumerate(scores):
        neighbours = []
        if i > 0:
            neighbours.append(scores[i - 1])
        if i < n - 1:
            neighbours.append(scores[i + 1])
        context = sum(neighbours) / len(neighbours) if neighbours else overall
        alpha = _blend_alpha(word_counts[i])
        blended.append(alpha * score + (1 - alpha) * context)
    return blended


def zerogpt_style_score(text: str) -> Optional[dict]:
    """Local ZeroGPT-style estimate (distilgpt2 perplexity + burstiness).

    Returns {"pct": 0-100, "prob": 0-1} or None when the calibrator or the
    language model is unavailable.
    """
    try:
        from ml import zerogpt_style
    except Exception:
        return None
    if not zerogpt_style.available():
        return None
    try:
        result = zerogpt_style.score(text)
    except Exception:
        return None
    if result is None:
        return None
    return {"pct": result["pct"], "prob": result["prob"]}


def zerogpt_style_scores(texts: List[str]) -> List[Optional[dict]]:
    """Batch variant of zerogpt_style_score (one language-model pass)."""
    try:
        from ml import zerogpt_style
    except Exception:
        return [None] * len(texts)
    if not zerogpt_style.available():
        return [None] * len(texts)
    try:
        results = zerogpt_style.score_many(texts)
    except Exception:
        return [None] * len(texts)
    return [
        {"pct": r["pct"], "prob": r["prob"]} if r else None for r in results
    ]


def zerogpt_style_features_scores(texts: List[str]) -> List[Optional[dict]]:
    """Batch ZeroGPT-style scores that also expose the LM features."""
    try:
        from ml import zerogpt_style
    except Exception:
        return [None] * len(texts)
    if not zerogpt_style.available():
        return [None] * len(texts)
    try:
        results = zerogpt_style.score_many(texts)
    except Exception:
        return [None] * len(texts)
    return [
        {"pct": r["pct"], "prob": r["prob"], "features": r["features"]}
        if r
        else None
        for r in results
    ]


def analyze_text(text: str) -> dict:
    started = time.perf_counter()
    model = get_model()

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text is empty after cleaning.")

    sentences = split_sentences(cleaned)
    if not sentences:
        raise ValueError("Could not detect any sentences in the text.")

    sent_scores_raw = model.ai_scores(
        [s.text for s in sentences], cfg.SENTENCE_MAX_LENGTH
    )
    sent_scores = _blend(
        [calibrate(s) for s in sent_scores_raw], [s.words for s in sentences]
    )

    chunk_texts = pack_sentences(
        sentences, model.tokenizer, cfg.MAX_SEQ_LENGTH, cfg.CHUNK_RESERVE_TOKENS
    )
    document = model.score_document(chunk_texts)
    doc_display = calibrate(document["ai_score"])

    total_words = max(1, sum(s.words for s in sentences))
    sentence_avg = sum(s.words * sc for s, sc in zip(sentences, sent_scores)) / total_words
    ai_percent = cfg.CONTEXT_WEIGHT * doc_display + (1 - cfg.CONTEXT_WEIGHT) * sentence_avg
    ai_percent = round(min(100.0, max(0.0, ai_percent)), 1)

    if ai_percent >= cfg.AI_VERDICT_THRESHOLD:
        verdict = "ai"
    elif ai_percent >= cfg.MIXED_VERDICT_THRESHOLD:
        verdict = "mixed"
    else:
        verdict = "human"

    words_total = word_count(cleaned)
    warnings: List[str] = []
    if words_total < cfg.MIN_RELIABLE_WORDS:
        warnings.append(
            f"Text is under {cfg.MIN_RELIABLE_WORDS} words — detection is less reliable on short inputs."
        )
    ascii_letters = sum(1 for ch in cleaned if ch.isascii() and ch.isalpha())
    letters = sum(1 for ch in cleaned if ch.isalpha()) or 1
    if ascii_letters / letters < 0.7:
        warnings.append("Model is English-first; non-English text may score poorly.")

    sentence_payload = [
        {
            "index": s.index,
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "words": s.words,
            "score": round(sc, 1),
        }
        for s, sc in zip(sentences, sent_scores)
    ]

    return {
        "text": cleaned,
        "ai_percent": ai_percent,
        "verdict": verdict,
        "zerogpt": zerogpt_style_score(cleaned),
        "confidence": round(min(100.0, abs(ai_percent - 50.0) * 2), 1),
        "class_percent": {
            label: round(p * 100, 1)
            for label, p in document["class_probs"].items()
        },
        "stats": {
            "characters": len(cleaned),
            "words": words_total,
            "sentences": len(sentences),
            "chunks": len(chunk_texts),
            "ai_sentences": sum(1 for sc in sent_scores if sc >= cfg.AI_VERDICT_THRESHOLD),
        },
        "sentences": sentence_payload,
        "warnings": warnings,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "model": model.source,
    }


def score_only(text: str) -> dict:
    """Fast document-level score (no per-sentence pass). Used to rank rewrites."""
    model = get_model()
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text is empty after cleaning.")
    sentences = split_sentences(cleaned)
    if not sentences:
        raise ValueError("Could not detect any sentences in the text.")
    chunk_texts = pack_sentences(
        sentences, model.tokenizer, cfg.MAX_SEQ_LENGTH, cfg.CHUNK_RESERVE_TOKENS
    )
    document = model.score_document(chunk_texts)
    ai_percent = round(min(100.0, max(0.0, calibrate(document["ai_score"]))), 1)
    if ai_percent >= cfg.AI_VERDICT_THRESHOLD:
        verdict = "ai"
    elif ai_percent >= cfg.MIXED_VERDICT_THRESHOLD:
        verdict = "mixed"
    else:
        verdict = "human"
    return {
        "ai_percent": ai_percent,
        "verdict": verdict,
        "class_percent": {
            label: round(p * 100, 1)
            for label, p in document["class_probs"].items()
        },
    }


def score_many(texts: List[str]) -> List[Optional[dict]]:
    """Batch document-level scores in one model pass (used to rank rewrites)."""
    if not texts:
        return []
    model = get_model()
    groups: List[Optional[tuple]] = []
    chunk_texts: List[str] = []
    for text in texts:
        cleaned = clean_text(text)
        if not cleaned:
            groups.append(None)
            continue
        sentences = split_sentences(cleaned)
        if not sentences:
            groups.append(None)
            continue
        chunks = pack_sentences(
            sentences, model.tokenizer, cfg.MAX_SEQ_LENGTH, cfg.CHUNK_RESERVE_TOKENS
        )
        groups.append((len(chunk_texts), len(chunks)))
        chunk_texts.extend(chunks)
    if not chunk_texts:
        return [None] * len(texts)
    probs = model.predict_probs(chunk_texts, cfg.MAX_SEQ_LENGTH)
    results: List[Optional[dict]] = []
    for group in groups:
        if group is None:
            results.append(None)
            continue
        start, count = group
        chunk_slice = chunk_texts[start : start + count]
        chunk_probs = probs[start : start + count]
        weights = np.array(
            [max(1, len(str(c).split())) for c in chunk_slice], dtype=np.float64
        )
        avg = (chunk_probs * weights[:, None]).sum(axis=0) / weights.sum()
        ai_percent = round(
            min(
                100.0,
                max(0.0, calibrate(float(model.ai_scores_from_probs(avg[None, :])[0]))),
            ),
            1,
        )
        if ai_percent >= cfg.AI_VERDICT_THRESHOLD:
            verdict = "ai"
        elif ai_percent >= cfg.MIXED_VERDICT_THRESHOLD:
            verdict = "mixed"
        else:
            verdict = "human"
        results.append(
            {
                "ai_percent": ai_percent,
                "verdict": verdict,
                "class_percent": {
                    label: round(p * 100, 1)
                    for label, p in zip(model.labels, avg)
                },
            }
        )
    return results


def warmup() -> Optional[str]:
    """Load the model at startup so the first request is fast."""
    try:
        return get_model().source
    except Exception:
        return None
