"""Local proxy scores for external AI detectors.

Style features are computed from the text; perplexity features come from
ml.zerogpt_style. Coefficients in models/stealth_proxies.json are fitted
offline on the public multi-detector benchmark
(wasitaigeneratedcom/ai-detector-benchmark-2026-09) for ZeroGPT, GPTZero,
Pangram and humanized-vs-human signals. Runtime inference is pure numpy.
"""

import json
import math
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from ml.text_utils import clean_text, split_sentences

PROXIES_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "stealth_proxies.json"
)

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_CONTRACTION_RE = re.compile(r"\b\w+'(?:t|re|ve|ll|d|m|s)\b", re.IGNORECASE)
_ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
_BULLET_RE = re.compile(r"^\s*([-*\u2022]|\d+[.)])\s+")
_HEADING_RE = re.compile(r"^\s*#+\s+")

_STOPWORDS = frozenset(
    "a an the and or but if while because as of to in on at for with from by about "
    "into over after before between during under above is are was were be been being "
    "am do does did doing have has had having i you he she it we they me him her us "
    "them my your his its our their this that these those there here what which who "
    "whom whose when where why how not no nor so than too very just can could will "
    "would shall should may might must".split()
)
_PRONOUNS = frozenset(
    "i me my mine we us our ours you your yours he him his she her hers it its they "
    "them their theirs".split()
)
_FIRST_PERSON = frozenset("i me my mine we us our ours".split())
_SECOND_PERSON = frozenset("you your yours".split())
_HEDGES = (
    "maybe", "perhaps", "probably", "kind of", "sort of", "seems", "apparently",
    "i think", "i guess", "honestly", "actually", "basically", "pretty much",
    "a bit", "a little", "really", "just", "quite", "somewhat",
)
_DISCOURSE = (
    "that said", "still", "plus", "even so", "however", "also", "look", "well",
    "you know", "anyway", "in the end", "overall", "basically", "honestly",
    "meanwhile", "besides", "on top of that",
)
_FILLERS = ("lol", "imo", "tbh", "idk", "btw", "fwiw", "anyway", "welp", "haha")
_MARKER_STARTS = (
    "honestly,", "look,", "so,", "that said,", "still,", "plus,", "even so,",
    "also,", "anyway,", "basically,", "well,",
)

STYLE_FEATURE_NAMES = [
    "avg_word_len",
    "long_word_frac",
    "stopword_frac",
    "first_person_frac",
    "second_person_frac",
    "pronoun_frac",
    "contraction_frac",
    "question_frac",
    "exclam_frac",
    "comma_per_100w",
    "period_per_100w",
    "semicolon_per_100w",
    "colon_per_100w",
    "dash_per_100w",
    "quote_per_100w",
    "paren_per_100w",
    "ellipsis_per_100w",
    "digit_frac",
    "currency_per_100w",
    "allcaps_frac",
    "lowercase_start_frac",
    "marker_start_frac",
    "lowercase_i_frac",
    "very_short_sent_frac",
    "long_sent_frac",
    "sent_len_p25",
    "sent_len_p75",
    "sent_len_cv",
    "sent_start_unique_frac",
    "paragraph_count",
    "mean_paragraph_words",
    "paragraph_words_cv",
    "line_count",
    "mean_line_words",
    "bigram_repeat_frac",
    "repeated_word_frac",
    "hedge_per_100w",
    "discourse_per_100w",
    "filler_per_100w",
    "readability",
    "style_heading_frac",
    "style_bullet_frac",
]

_lock = threading.Lock()
_state: dict = {"loaded": False, "data": None}


def available() -> bool:
    return PROXIES_PATH.exists()


def _load() -> Optional[dict]:
    if _state["data"] is not None:
        return _state["data"]
    if not PROXIES_PATH.exists():
        return None
    with _lock:
        if _state["data"] is None:
            with PROXIES_PATH.open(encoding="utf-8") as fh:
                _state["data"] = json.load(fh)
    return _state["data"]


def _count_phrases(lowered: str, phrases) -> int:
    return sum(lowered.count(phrase) for phrase in phrases)


def _syllables(word: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", word.lower())))


def style_features(text: str) -> dict:
    clean = clean_text(text) or ""
    words = _WORD_RE.findall(clean)
    lowered_words = [w.lower() for w in words]
    n_words = max(len(words), 1)
    sentences = split_sentences(clean)
    sent_lengths = [s.words for s in sentences if s.words > 0] or [0]
    n_sent = max(len(sent_lengths), 1)
    chars = max(len(clean), 1)
    letters = sum(1 for ch in clean if ch.isalpha()) or 1
    lowered = clean.lower()

    lengths = np.asarray(sent_lengths, dtype=float)
    sent_mean = float(lengths.mean()) or 1.0
    paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip()]
    para_words = [len(_WORD_RE.findall(p)) for p in paragraphs] or [0]
    lines = [ln.strip() for ln in clean.split("\n") if ln.strip()] or [""]

    bigrams = list(zip(lowered_words, lowered_words[1:]))
    repeated = 0
    seen = set()
    for word in lowered_words:
        if len(word) <= 2 or word in _STOPWORDS or not word.isalpha():
            continue
        if word in seen:
            repeated += 1
        else:
            seen.add(word)

    starts = [s.text.split()[0].lower() for s in sentences if s.text.split()]
    lowercase_starts = sum(1 for s in sentences if s.text[:1].islower())
    marker_starts = sum(
        1 for s in sentences if s.text.lower().startswith(_MARKER_STARTS)
    )

    return {
        "avg_word_len": sum(len(w) for w in words) / n_words,
        "long_word_frac": sum(1 for w in words if len(w) >= 8) / n_words,
        "stopword_frac": sum(1 for w in lowered_words if w in _STOPWORDS) / n_words,
        "first_person_frac": sum(1 for w in lowered_words if w in _FIRST_PERSON)
        / n_words,
        "second_person_frac": sum(1 for w in lowered_words if w in _SECOND_PERSON)
        / n_words,
        "pronoun_frac": sum(1 for w in lowered_words if w in _PRONOUNS) / n_words,
        "contraction_frac": len(_CONTRACTION_RE.findall(clean)) / n_words,
        "question_frac": clean.count("?") / n_sent,
        "exclam_frac": clean.count("!") / n_sent,
        "comma_per_100w": clean.count(",") / n_words * 100,
        "period_per_100w": clean.count(".") / n_words * 100,
        "semicolon_per_100w": clean.count(";") / n_words * 100,
        "colon_per_100w": clean.count(":") / n_words * 100,
        "dash_per_100w": (clean.count("\u2014") + clean.count("\u2013"))
        / n_words
        * 100,
        "quote_per_100w": sum(clean.count(ch) for ch in "\"\u201c\u201d")
        / n_words
        * 100,
        "paren_per_100w": (clean.count("(") + clean.count(")")) / n_words * 100,
        "ellipsis_per_100w": clean.count("...") / n_words * 100,
        "digit_frac": sum(1 for ch in clean if ch.isdigit()) / chars,
        "currency_per_100w": sum(
            clean.count(ch) for ch in "\u20b9$\u20ac\u00a3"
        )
        / n_words
        * 100,
        "allcaps_frac": len(_ALLCAPS_RE.findall(clean)) / n_words,
        "lowercase_start_frac": lowercase_starts / n_sent,
        "marker_start_frac": marker_starts / n_sent,
        "lowercase_i_frac": sum(1 for w in words if w == "i") / n_words,
        "very_short_sent_frac": float((lengths <= 5).mean()),
        "long_sent_frac": float((lengths >= 30).mean()),
        "sent_len_p25": float(np.percentile(lengths, 25)),
        "sent_len_p75": float(np.percentile(lengths, 75)),
        "sent_len_cv": float(lengths.std() / sent_mean),
        "sent_start_unique_frac": len(set(starts)) / n_sent,
        "paragraph_count": float(len(paragraphs)),
        "mean_paragraph_words": float(np.mean(para_words)),
        "paragraph_words_cv": float(
            np.std(para_words) / max(float(np.mean(para_words)), 1.0)
        ),
        "line_count": float(len(lines)),
        "mean_line_words": float(np.mean([len(_WORD_RE.findall(ln)) for ln in lines])),
        "bigram_repeat_frac": (
            1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        ),
        "repeated_word_frac": repeated / n_words,
        "hedge_per_100w": _count_phrases(lowered, _HEDGES) / n_words * 100,
        "discourse_per_100w": _count_phrases(lowered, _DISCOURSE) / n_words * 100,
        "filler_per_100w": _count_phrases(lowered, _FILLERS) / n_words * 100,
        "readability": 206.835
        - 1.015 * sent_mean
        - 84.6 * (sum(_syllables(w) for w in words) / n_words),
        "style_heading_frac": sum(1 for ln in lines if _HEADING_RE.match(ln))
        / len(lines),
        "style_bullet_frac": sum(1 for ln in lines if _BULLET_RE.match(ln))
        / len(lines),
    }


def merge_features(text: str, lm_features: Optional[dict]) -> Optional[dict]:
    if not lm_features:
        return None
    merged = dict(lm_features)
    merged.update(style_features(text))
    return merged


def _linear(proxy: dict, values) -> float:
    z = (np.asarray(values, dtype=float) - np.asarray(proxy["mean"])) / np.asarray(
        proxy["scale"]
    )
    return float(np.dot(proxy["coef"], z) + proxy["intercept"])


def score_from_features(features: Optional[dict]) -> Optional[dict]:
    data = _load()
    if data is None or not features:
        return None
    names = data["features"]
    values = [features.get(name, 0.0) for name in names]
    components = {}
    for key, proxy in data["proxies"].items():
        raw = _linear(proxy, values)
        if proxy.get("link") == "sigmoid":
            value = 1.0 / (1.0 + math.exp(-max(min(raw, 30.0), -30.0)))
        else:
            value = raw
        if proxy.get("output") == "pct":
            risk = float(np.clip(value, 0.0, 100.0))
        else:
            risk = float(np.clip(value, 0.0, 1.0) * 100.0)
        components[key] = round(risk, 1)

    weights = data.get("weights", {})
    total = sum(weights.get(key, 0.0) for key in components)
    if total <= 0:
        risk = sum(components.values()) / max(len(components), 1)
    else:
        risk = sum(
            components[key] * weights.get(key, 0.0) for key in components
        ) / total
    worst = max((components.get(key, 0.0) for key in ("zerogpt", "gptzero", "pangram")), default=0.0)
    return {"risk": round(risk, 1), "worst": round(worst, 1), "components": components}


def score_many_from_features(texts, lm_features_list) -> list:
    out = []
    for text, lm in zip(texts, lm_features_list):
        out.append(score_from_features(merge_features(text, lm)))
    return out
