"""Rule-based humanizer: rewrites AI-flavored text so it reads more human.

No LLM involved. Generates several candidate rewrites (phrase swaps,
contractions, sentence splitting, layout-preserving prose edits), scores each,
and returns the best one. Target "external" (default) ranks candidates with
local proxies for ZeroGPT, GPTZero and Pangram plus the local classifier;
target "local" optimizes this detector, including its ZeroGPT-style mode.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

from ml import config as cfg
from ml import stealth
from ml.detect import (
    score_many,
    score_only,
    zerogpt_style_features_scores,
    zerogpt_style_score,
    zerogpt_style_scores,
)
from ml.text_utils import clean_text, split_sentences, word_count

TONES = ("casual", "formal")
STRENGTHS = ("light", "medium", "strong")
TARGETS = ("external", "local")

# phrase -> (neutral/formal alternative, casual alternative)
_PHRASES: List[Tuple[str, Tuple[str, str]]] = [
    # openers / framing
    ("in today's fast-paced digital landscape", ("in the current digital landscape", "these days")),
    ("in today's fast-paced world", ("in the current environment", "these days")),
    ("in today's digital age", ("in the digital era", "these days")),
    ("in the modern workplace", ("in the workplace", "at work")),
    ("in the modern world", ("in the contemporary world", "today")),
    ("in the ever-evolving landscape of", ("across", "in")),
    ("in the realm of", ("in", "in")),
    ("at the forefront of", ("ahead in", "leading")),
    ("it is important to note that", ("", "remember that")),
    ("it is worth noting that", ("", "note that")),
    ("it should be noted that", ("", "note that")),
    ("it goes without saying that", ("", "clearly")),
    ("it is essential to", ("it is necessary to", "you need to")),
    ("one must", ("it is necessary to", "you have to")),
    ("in conclusion", ("overall", "so")),
    ("in summary", ("overall", "basically")),
    ("to summarize", ("in short", "basically")),
    ("all in all", ("overall", "basically")),
    # connectives
    ("on the other hand", ("conversely", "then again")),
    ("as a result", ("therefore", "so")),
    ("consequently", ("therefore", "so")),
    ("furthermore", ("also", "plus")),
    ("moreover", ("also", "plus")),
    ("additionally", ("also", "plus")),
    ("nevertheless", ("still", "even so")),
    ("nonetheless", ("still", "even so")),
    ("therefore", ("thus", "so")),
    ("thus", ("therefore", "so")),
    ("hence", ("therefore", "so")),
    ("however", ("yet", "but")),
    ("whilst", ("while", "while")),
    ("amongst", ("among", "among")),
    ("subsequently", ("later", "then")),
    ("notably", ("especially", "especially")),
    ("particularly", ("especially", "especially")),
    ("ultimately", ("in the end", "in the end")),
    ("prior to", ("before", "before")),
    ("in order to", ("to", "to")),
    ("due to the fact that", ("because", "because")),
    ("in the event that", ("if", "if")),
    ("with regard to", ("regarding", "about")),
    ("with respect to", ("regarding", "about")),
    ("in terms of", ("regarding", "for")),
    ("going forward", ("from now on", "from here")),
    # quantifiers
    ("a wide range of", ("many", "all kinds of")),
    ("a variety of", ("many", "different")),
    ("a plethora of", ("many", "a lot of")),
    ("a myriad of", ("many", "tons of")),
    ("numerous", ("many", "lots of")),
    ("various", ("several", "different")),
    ("myriad", ("many", "tons of")),
    ("plethora", ("plenty", "a lot")),
    # role / importance
    ("plays a crucial role", ("is essential", "matters a lot")),
    ("plays a vital role", ("is essential", "matters a lot")),
    ("plays an important role", ("is important", "counts")),
    ("pivotal role", ("key part", "big part")),
    ("is a testament to", ("shows", "proves")),
    ("game-changer", ("big deal", "big deal")),
    # adjectives
    ("cutting-edge", ("advanced", "new")),
    ("state-of-the-art", ("advanced", "top")),
    ("transformative", ("major", "big")),
    ("revolutionized", ("changed", "shook up")),
    ("revolutionize", ("change", "shake up")),
    ("rapidly evolving", ("fast-changing", "fast-changing")),
    ("ever-evolving", ("always changing", "always changing")),
    ("unprecedented", ("exceptional", "huge")),
    ("seamlessly", ("smoothly", "smoothly")),
    ("seamless", ("smooth", "smooth")),
    ("robust", ("solid", "solid")),
    ("optimal", ("best", "best")),
    ("comprehensive", ("complete", "full")),
    ("meticulously", ("carefully", "carefully")),
    ("meticulous", ("careful", "careful")),
    ("significantly", ("considerably", "a lot")),
    ("significant", ("major", "big")),
    ("substantially", ("considerably", "a lot")),
    ("substantial", ("large", "big")),
    ("beneficial", ("helpful", "helpful")),
    ("advantageous", ("helpful", "handy")),
    ("innovative", ("creative", "new")),
    ("holistic", ("complete", "whole")),
    ("paradigm", ("model", "model")),
    ("synergy", ("teamwork", "teamwork")),
    ("landscape", ("environment", "world")),
    ("best practices", ("proven methods", "what works")),
    # verbs
    ("leveraging", ("using", "using")),
    ("leverage", ("use", "put to work")),
    ("utilizing", ("using", "using")),
    ("utilized", ("used", "used")),
    ("utilize", ("use", "use")),
    ("utilization", ("use", "use")),
    ("facilitates", ("supports", "helps with")),
    ("facilitated", ("supported", "helped with")),
    ("facilitate", ("support", "help with")),
    ("demonstrates", ("shows", "shows")),
    ("demonstrated", ("showed", "showed")),
    ("demonstrate", ("show", "show")),
    ("optimizing", ("improving", "tuning")),
    ("optimize", ("improve", "tune")),
    ("enhances", ("improves", "boosts")),
    ("enhanced", ("improved", "improved")),
    ("enhance", ("improve", "boost")),
    ("streamlines", ("simplifies", "simplifies")),
    ("streamline", ("simplify", "speed up")),
    ("empowers", ("enables", "helps")),
    ("empower", ("enable", "help")),
    ("unlocking", ("opening up", "opening up")),
    ("unlock", ("open up", "open up")),
    ("fostering", ("building", "building")),
    ("foster", ("build", "encourage")),
    ("delving into", ("examining", "digging into")),
    ("delve into", ("examine", "dig into")),
    ("navigating", ("handling", "handling")),
    ("navigate", ("handle", "deal with")),
    ("harnessing", ("using", "using")),
    ("harness", ("use", "use")),
    ("implementing", ("setting up", "setting up")),
    ("implement", ("set up", "set up")),
    ("implementation requires", ("rollout requires", "getting this right takes")),
    ("successful implementation requires", ("a successful rollout requires", "getting this right takes")),
    ("ensuring", ("guaranteeing", "making sure of")),
    ("ensures", ("guarantees", "makes sure of")),
    ("ensure", ("guarantee", "make sure of")),
    ("providing", ("giving", "giving")),
    ("provides", ("gives", "gives")),
    ("provide", ("give", "give")),
    ("employing", ("using", "using")),
    ("employ", ("use", "use")),
    ("necessitates", ("requires", "needs")),
    ("necessitate", ("require", "need")),
    ("commence", ("start", "start")),
    ("initiate", ("start", "start")),
    ("obtain", ("get", "get")),
    ("purchase", ("buy", "buy")),
    ("ascertain", ("find out", "find out")),
    ("endeavor", ("effort", "try")),
    ("endeavors", ("efforts", "efforts")),
    ("assistance", ("help", "help")),
    ("assist", ("help", "help")),
    ("attempt", ("try", "try")),
    ("possess", ("have", "have")),
    ("reside", ("live", "live")),
    ("requires", ("needs", "needs")),
    ("requiring", ("needing", "needing")),
    ("require", ("need", "need")),
    ("requirement", ("need", "need")),
    ("aims to", ("tries to", "tries to")),
    ("seeks to", ("wants to", "wants to")),
    ("is designed to", ("is meant to", "is meant to")),
    ("commonly", ("often", "often")),
    ("effectively", ("well", "well")),
    ("efficiently", ("quickly", "quickly")),
    ("approximately", ("about", "about")),
    ("sufficient", ("enough", "enough")),
    ("additional", ("more", "more")),
    ("regarding", ("about", "about")),
    ("concerning", ("about", "about")),
    # AI-writing clichés (content words)
    ("artificial intelligence", ("AI", "AI")),
    ("has emerged as", ("has become", "is now")),
    ("have emerged as", ("have become", "are now")),
    ("emerged as", ("become", "is now")),
    ("transformative force", ("major force", "big deal")),
    ("across numerous industries", ("across many industries", "in pretty much every industry")),
    ("across various industries", ("across several industries", "in a lot of industries")),
    ("increasingly leveraging", ("increasingly using", "using more and more")),
    ("increasingly", ("more and more", "more and more")),
    ("AI-driven solutions", ("AI tools", "AI tools")),
    ("AI-powered solutions", ("AI tools", "AI tools")),
    ("streamline operations", ("simplify operations", "cut down busywork")),
    ("enhance decision-making processes", ("improve decision-making", "make better calls")),
    ("decision-making processes", ("decision-making", "decisions")),
    ("unlock valuable insights", ("find useful insights", "dig up useful insights")),
    ("valuable insights", ("useful insights", "useful insights")),
    ("from vast datasets", ("from large datasets", "in huge datasets")),
    ("vast datasets", ("large datasets", "huge datasets")),
    ("big datasets", ("large datasets", "huge datasets")),
    ("by embracing", ("by adopting", "by using")),
    ("embracing", ("adopting", "using")),
    ("embrace", ("adopt", "use")),
    ("these cutting-edge technologies", ("these advanced technologies", "these new tools")),
    ("maintain a competitive edge", ("stay competitive", "stay ahead")),
    ("competitive edge", ("advantage", "edge")),
    ("delivering superior value", ("delivering more value", "delivering more value")),
    ("superior value", ("more value", "more value")),
    ("to their customers", ("to customers", "to customers")),
    ("careful planning", ("careful planning", "real planning")),
    ("substantial investment in infrastructure", ("large infrastructure investment", "a lot of money on infrastructure")),
    ("investment in infrastructure", ("infrastructure investment", "money on infrastructure")),
    ("implementation", ("rollout", "getting this right")),
    ("adoption", ("use", "use")),
    ("adopting", ("using", "using")),
    ("adopt", ("use", "use")),
    ("organizations", ("organisations", "companies")),
    ("individuals", ("people", "people")),
    ("efficiency", ("speed", "speed")),
    ("methodology", ("method", "method")),
    ("pivotal", ("key", "central")),
    ("crucial", ("key", "important")),
    ("vital", ("essential", "needed")),
    ("essential", ("necessary", "needed")),
    # health / academic content
    ("regular physical exercise", ("regular exercise", "regular exercise")),
    ("physical exercise", ("exercise", "working out")),
    ("physical activity", ("exercise", "movement")),
    ("essential for maintaining", ("needed to maintain", "good for")),
    ("maintaining optimal health and well-being", ("maintaining good health and well-being", "staying healthy and feeling good")),
    ("enhance overall quality of life", ("improve overall quality of life", "make everyday life better")),
    ("overall quality of life", ("overall standard of living", "everyday life")),
    ("optimal health and well-being", ("good health and well-being", "feeling good")),
    ("optimal health", ("good health", "good health")),
    ("well-being", ("wellbeing", "feeling good")),
    ("chronic diseases", ("long-term diseases", "long-term health problems")),
    ("mental health", ("mental health", "how you feel")),
    ("quality of life", ("standard of living", "day-to-day life")),
    ("daily routine", ("daily life", "day")),
    ("incorporating", ("adding", "adding")),
    ("incorporate", ("add", "add")),
    ("boost energy levels", ("give you more energy", "give you more energy")),
    ("energy levels", ("energy", "energy")),
    ("healthy lifestyle", ("healthy life", "healthy life")),
    ("prioritize", ("prioritise", "make time for")),
    ("prioritise", ("prioritise", "make time for")),
    ("it is crucial to prioritize", ("it is key to prioritise", "you really need to make time for")),
    ("it is crucial to prioritise", ("it is key to prioritise", "you really need to make time for")),
    ("as part of a healthy lifestyle", ("as part of a healthy life", "if you want to stay healthy")),
    # email phrasing
    ("dear team", ("dear team", "hi team")),
    ("i hope this message finds you well", ("I hope you are well", "hope you're doing well")),
    ("i am writing to", ("I am writing to", "just writing to")),
    ("provide an update", ("give an update", "give you an update")),
    ("project timeline", ("project schedule", "project schedule")),
    ("due to unforeseen circumstances", ("because of unexpected circumstances", "because things came up")),
    ("has been adjusted", ("has been moved", "is now moved")),
    ("following week", ("next week", "next week")),
    ("please do not hesitate to reach out", ("please reach out", "just ping me")),
    ("if you have any questions or concerns", ("with any questions", "if anything's unclear")),
    ("best regards", ("kind regards", "thanks")),
    # generic
    ("research shows that", ("research indicates that", "studies show")),
    ("studies have shown", ("research indicates", "studies show")),
    ("it has been shown that", ("", "turns out")),
    ("in the field of", ("in", "in")),
    ("plays a significant role", ("is a major factor", "matters a lot")),
    ("a significant role", ("a major part", "a big part")),
]

_ALT_LOOKUP = {phrase.lower(): alts for phrase, alts in _PHRASES}
_COMBINED = re.compile(
    r"(?<![\w-])("
    + "|".join(
        re.escape(phrase)
        for phrase, _ in sorted(_PHRASES, key=lambda p: len(p[0]), reverse=True)
    )
    + r")(?![\w-])",
    re.IGNORECASE,
)

_CONTRACTIONS = [
    (re.compile(r"\bdo not\b", re.IGNORECASE), "don't"),
    (re.compile(r"\bdoes not\b", re.IGNORECASE), "doesn't"),
    (re.compile(r"\bdid not\b", re.IGNORECASE), "didn't"),
    (re.compile(r"\bis not\b", re.IGNORECASE), "isn't"),
    (re.compile(r"\bare not\b", re.IGNORECASE), "aren't"),
    (re.compile(r"\bwas not\b", re.IGNORECASE), "wasn't"),
    (re.compile(r"\bwere not\b", re.IGNORECASE), "weren't"),
    (re.compile(r"\bcannot\b", re.IGNORECASE), "can't"),
    (re.compile(r"\bcan not\b", re.IGNORECASE), "can't"),
    (re.compile(r"\bwill not\b", re.IGNORECASE), "won't"),
    (re.compile(r"\bwould not\b", re.IGNORECASE), "wouldn't"),
    (re.compile(r"\bshould not\b", re.IGNORECASE), "shouldn't"),
    (re.compile(r"\bcould not\b", re.IGNORECASE), "couldn't"),
    (re.compile(r"\bhave not\b", re.IGNORECASE), "haven't"),
    (re.compile(r"\bhas not\b", re.IGNORECASE), "hasn't"),
    (re.compile(r"\bhad not\b", re.IGNORECASE), "hadn't"),
    (re.compile(r"\bit is\b", re.IGNORECASE), "it's"),
    (re.compile(r"\bthat is\b", re.IGNORECASE), "that's"),
    (re.compile(r"\bthere is\b", re.IGNORECASE), "there's"),
    (re.compile(r"\bthey are\b", re.IGNORECASE), "they're"),
    (re.compile(r"\bwe are\b", re.IGNORECASE), "we're"),
    (re.compile(r"\byou are\b", re.IGNORECASE), "you're"),
    (re.compile(r"\bI am\b"), "I'm"),
]

_CASUAL_EXTRAS = [
    (re.compile(r"\bvery\b", re.IGNORECASE), "really"),
    (re.compile(r"\bin addition\b", re.IGNORECASE), "plus"),
]

_SPLIT_RE = re.compile(r",\s+(and|but|so|which|because)\s+", re.IGNORECASE)
_TRAILING_CONJ_RE = re.compile(r"[,\s]+(and|but|so|or)\s*$", re.IGNORECASE)
_SENTENCE_START_RE = re.compile(
    r"^(honestly|actually|look|so|and|but|also|plus|still|that said|even so)\b",
    re.IGNORECASE,
)
_AI_LIST_RE = re.compile(
    r"(\w[\w\s-]{2,40}),\s+(\w[\w\s-]{2,40}),\s+and\s+(\w[\w\s-]{2,40})"
)
_BULLET_LINE_RE = re.compile(r"^(\s*(?:[-*\u2022]|\d+[.)])\s+)(.*)$")
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s")

_MARKERS_CASUAL = ["That said, ", "Still, ", "Plus, ", "Even so, ", "Also, "]
_MARKERS_FORMAL = ["That said, ", "Still, ", "Even so, ", "However, "]
_OPENERS_CASUAL = ["Honestly, ", "Look, ", "So, "]

_PERSONAL_OPENERS = [
    "I have been thinking about this a lot lately. ",
    "I keep seeing this come up everywhere lately. ",
    "Honestly, I have been noticing this a lot. ",
]

_CASUAL_ENDINGS = [
    " dont @ me.",
    " imo.",
    " my two cents anyway.",
    " my two cents.",
    " but thats just me.",
    " anyway, thats my take.",
]


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.!?;:])", r"\1", text)
    text = re.sub(
        r"(^|[.!?]\s+|\n\s*)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        text,
    )
    return text.strip()


def _apply_phrases(
    text: str, tone: str, changes: Dict[str, list]
) -> str:
    def repl(match):
        original = match.group(0)
        alts = _ALT_LOOKUP[original.lower()]
        alt = alts[0] if tone == "formal" else alts[-1]
        before = text[: match.start()].rstrip()
        at_start = (not before) or before[-1] in ".!?\n"
        if alt:
            if at_start:
                alt = alt[0].upper() + alt[1:]
            entry = changes.setdefault(original.lower(), [alt, 0])
            entry[1] += 1
        else:
            entry = changes.setdefault(original.lower(), ["(removed)", 0])
            entry[1] += 1
        return alt

    return _tidy(_COMBINED.sub(repl, text))


def _apply_pairs(text: str, pairs) -> str:
    for pattern, replacement in pairs:
        def repl(match, replacement=replacement):
            if match.group(0)[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        text = pattern.sub(repl, text)
    return text


def _split_long(sentence: str, max_words: int, tone: str) -> List[str]:
    if word_count(sentence) <= max_words:
        return [sentence]

    parts = [sentence]
    rounds = 2 if max_words <= 22 else 1
    for _ in range(rounds):
        new_parts: List[str] = []
        for part in parts:
            if word_count(part) <= max_words:
                new_parts.append(part)
                continue
            matches = list(_SPLIT_RE.finditer(part))
            if not matches:
                new_parts.append(part)
                continue
            mid = len(part) / 2
            match = min(matches, key=lambda m: abs(m.start() - mid))
            left = part[: match.start()].rstrip()
            right = part[match.end():].lstrip()
            if word_count(left) < 4 or word_count(right) < 4:
                new_parts.append(part)
                continue
            conjunction = match.group(1).lower()
            if tone == "casual" and conjunction in ("and", "but", "so"):
                right = conjunction.capitalize() + " " + right
            else:
                right = right[0].upper() + right[1:]
            new_parts.extend([left + ".", right])
        parts = new_parts
    return parts


def _break_lists(sentence: str) -> List[str]:
    """Split triadic lists ("X, Y, and Z") into two sentences."""
    match = _AI_LIST_RE.search(sentence)
    if not match:
        return [sentence]
    left = _TRAILING_CONJ_RE.sub("", sentence[: match.start(3)].rstrip())
    last = sentence[match.start(3):].strip()
    if word_count(left) < 4 or word_count(last) < 2:
        return [sentence]
    return [left + ".", last[0].upper() + last[1:]]


def _is_structural(sentence: str) -> bool:
    """True if the chunk's first line is a heading, bullet, table or code fence.

    ``split_sentences`` can merge a structure line with the prose after it, so
    only the first line decides whether markers/openers may touch the chunk.
    """
    first_line = sentence.split("\n", 1)[0]
    stripped = first_line.lstrip()
    if not stripped:
        return True
    if _HEADING_LINE_RE.match(first_line) or _BULLET_LINE_RE.match(first_line):
        return True
    if stripped.startswith(("```", "|")):
        return True
    return False


def _add_markers(
    sentences: List[str], tone: str, every: int = 3, max_markers: int = 2
) -> List[str]:
    markers = _MARKERS_CASUAL if tone == "casual" else _MARKERS_FORMAL
    out: List[str] = []
    used = 0
    for i, sentence in enumerate(sentences):
        if (
            used < max_markers
            and i > 0
            and i % every == 1
            and word_count(sentence) >= 5
            and not _SENTENCE_START_RE.match(sentence)
            and not _is_structural(sentence)
        ):
            marker = markers[used % len(markers)]
            sentence = marker + sentence[0].lower() + sentence[1:]
            used += 1
        out.append(sentence)
    return out


def _add_opener(sentences: List[str], tone: str, which: int) -> List[str]:
    if tone != "casual" or not sentences:
        return sentences
    target = next(
        (i for i, s in enumerate(sentences) if not _is_structural(s)), None
    )
    if target is None:
        return sentences
    first = sentences[target]
    if _SENTENCE_START_RE.match(first) or word_count(first) < 3:
        return sentences
    opener = _OPENERS_CASUAL[which % len(_OPENERS_CASUAL)]
    sentences = list(sentences)
    sentences[target] = opener + first[0].lower() + first[1:]
    return sentences


def _prepend_personal(sentences: List[str], which: int) -> Tuple[List[str], int]:
    opener = _PERSONAL_OPENERS[which % len(_PERSONAL_OPENERS)]
    target = next(
        (i for i, s in enumerate(sentences) if not _is_structural(s)), 0
    )
    out = list(sentences)
    out.insert(target, opener.strip())
    return out, target


def _is_code_fence(text: str) -> bool:
    """True when the chunk is a pure ``` block (open + close, nothing else structural)."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return False
    body = stripped[3:]
    return body.rstrip().endswith("```") and "\n" in stripped


def _lower_style(text: str, style: int) -> str:
    if style >= 2:
        # Preserve headings, lists, tables and code fences verbatim.
        return "\n".join(
            line if _is_structural(line) else line.lower() for line in text.split("\n")
        )
    if style == 1:
        # Casual lowercase sentence starts, but never touch an all-caps word
        # (AI, NASA, TAIL_MARKER) or it would read as a typo like "aI".
        return re.sub(
            r"(^|[.!?]\s+)([A-Z])(?![A-Z])",
            lambda m: m.group(1) + m.group(2).lower(),
            text,
        )
    return text


# Long inputs are scored on a sample of at most this many characters (~one
# 768-token chunk) so large documents do not take minutes to humanize.
_SCORING_SAMPLE_CHARS = 2400


def _score_view(text: str) -> str:
    """Bounded sample of a long document used for scoring.

    The classifier is CPU-bound (DeBERTa-large costs ~6s per 768-token chunk)
    and the rewrite loop scores the text many times, so scoring a huge input in
    full takes minutes. This keeps a small sample drawn from across the whole
    document -- head, middle and the final sentence -- so long inputs score in
    about one chunk. Short inputs are returned untouched, and the rewrite
    itself always operates on the full text.
    """
    if len(text) <= _SCORING_SAMPLE_CHARS:
        return text
    cleaned = clean_text(text)
    if not cleaned or len(cleaned) <= _SCORING_SAMPLE_CHARS:
        return text
    sentences = split_sentences(cleaned)
    if not sentences:
        return text
    if sum(len(s.text) for s in sentences) <= _SCORING_SAMPLE_CHARS:
        return text

    count = len(sentences)
    total = sum(len(s.text) for s in sentences)
    last = sentences[-1].text
    reserve = min(len(last), _SCORING_SAMPLE_CHARS // 2)
    room = _SCORING_SAMPLE_CHARS - reserve
    # Size the slot count so that many evenly-spaced picks roughly fill the
    # budget: too many slots and the budget runs out before reaching the tail.
    avg = max(1, total // count)
    slots = max(4, min(count - 1, room // avg))
    picked: List[str] = []
    used = 0
    for i in range(slots):
        sample = sentences[round(i * (count - 1) / max(1, slots - 1))].text
        if sample in picked or used + len(sample) > room:
            continue
        picked.append(sample)
        used += len(sample)
    # The tail stays in view: the ending loop only changes the last sentence.
    if last not in picked:
        picked.append(last)
    return "\n".join(picked)


def _best_ending(text: str, after: dict) -> Tuple[str, dict, str]:
    base = text.rstrip()
    best_text, best_after, best_tag = text, after, ""
    for tag in _CASUAL_ENDINGS:
        candidate = base + tag
        try:
            scored = score_only(_score_view(candidate))
        except ValueError:
            continue
        if scored["ai_percent"] < best_after["ai_percent"]:
            best_text, best_after, best_tag = candidate, scored, tag.strip()
    return best_text, best_after, best_tag


def _rewrite(
    text: str,
    tone: str,
    *,
    contractions: bool = True,
    split_words: int = 0,
    markers: bool = False,
    opener: int = -1,
    break_lists: bool = False,
    personal: int = -1,
    lower_style: int = 0,
) -> Tuple[str, Dict[str, list]]:
    cleaned = clean_text(text)
    changes: Dict[str, list] = {}
    sentences = split_sentences(cleaned)

    # Original whitespace (incl. newlines) that followed each source sentence.
    # Restored on rejoin so paragraph and line breaks survive the rewrite.
    gaps: List[str] = []
    for i, sentence in enumerate(sentences):
        if i + 1 < len(sentences):
            gaps.append(cleaned[sentence.end : sentences[i + 1].start])
        else:
            gaps.append("")

    out: List[str] = []
    group: List[int] = []
    for idx, sentence in enumerate(sentences):
        if _is_code_fence(sentence.text):
            # Code fences stay verbatim; phrase/tidy transforms corrupt code.
            out.append(sentence.text)
            group.append(idx)
            continue
        piece = _apply_phrases(sentence.text, tone, changes)
        if contractions and tone == "casual":
            piece = _apply_pairs(piece, _CONTRACTIONS)
            piece = _apply_pairs(piece, _CASUAL_EXTRAS)
        parts = [piece]
        if break_lists:
            expanded: List[str] = []
            for part in parts:
                expanded.extend(_break_lists(part))
            parts = expanded
        if split_words:
            split_parts: List[str] = []
            for part in parts:
                split_parts.extend(_split_long(part, split_words, tone))
            parts = split_parts
        for part in parts:
            out.append(part)
            group.append(idx)

    if markers:
        marked = _add_markers(out, tone)
        if len(marked) == len(out):
            out = marked
    if opener >= 0:
        opened = _add_opener(out, tone, opener)
        if len(opened) == len(out):
            out = opened
    if personal >= 0:
        out, at = _prepend_personal(out, personal)
        # Sentinel -1 marks the opener so the rejoin can give it its own
        # paragraph before a heading/list, or a lead-in space before prose.
        if group:
            group.insert(at, -1)

    # Rejoin: pieces from one source sentence are split fragments -> space;
    # across source sentences -> the original gap (keeps \n / \n\n).
    pieces: List[str] = []
    for i, item in enumerate(out):
        pieces.append(item)
        if i + 1 >= len(out):
            break
        if group[i] == -1:
            # Leaving the personal opener: own line before structure, else lead-in.
            pieces.append("\n\n" if _is_structural(out[i + 1]) else " ")
        elif group[i + 1] == group[i]:
            pieces.append(" ")
        else:
            gap = gaps[group[i]] if group[i] < len(gaps) else ""
            pieces.append(gap if gap else " ")
    text = _lower_style("".join(pieces), lower_style)
    return text, changes


def _profiles(tone: str, strength: str) -> List[dict]:
    if tone == "formal":
        if strength == "light":
            return [dict(contractions=False)]
        if strength == "strong":
            return [
                dict(contractions=False, split_words=22, markers=True),
                dict(contractions=False, split_words=30, markers=True),
                dict(contractions=False, split_words=26, markers=True),
            ]
        return [
            dict(contractions=False, split_words=30, markers=True),
            dict(contractions=False, split_words=26, markers=True),
        ]

    if strength == "light":
        return [
            dict(split_words=0, markers=False),
            dict(split_words=0, markers=False, lower_style=1),
        ]
    if strength == "strong":
        return [
            dict(split_words=22, markers=True, opener=0, break_lists=True, lower_style=2),
            dict(split_words=22, markers=True, opener=1, break_lists=True, lower_style=2),
            dict(split_words=18, markers=True, opener=2, break_lists=True, lower_style=2),
            dict(split_words=26, markers=True, opener=0, break_lists=True, lower_style=1),
            dict(split_words=20, markers=True, opener=1, break_lists=True, lower_style=2),
            dict(split_words=24, markers=True, opener=0, break_lists=True, personal=0, lower_style=2),
        ]
    return [
        dict(split_words=26, markers=True, opener=0),
        dict(split_words=30, markers=True, opener=1, break_lists=True),
        dict(split_words=22, markers=True, opener=2),
        dict(split_words=22, markers=True, opener=0, break_lists=True, lower_style=1),
        dict(split_words=24, markers=True, opener=1, break_lists=True, lower_style=2),
    ]


def _external_profiles(tone: str, strength: str) -> List[dict]:
    if tone == "formal":
        if strength == "light":
            return [dict(contractions=False), dict(contractions=True)]
        if strength == "strong":
            return [
                dict(contractions=False, split_words=26),
                dict(contractions=False, split_words=22),
                dict(contractions=False, split_words=30, break_lists=True),
                dict(contractions=True, split_words=24),
                dict(contractions=True, split_words=20, break_lists=True),
            ]
        return [
            dict(contractions=False),
            dict(contractions=False, split_words=30),
            dict(contractions=False, split_words=26, break_lists=True),
            dict(contractions=True, split_words=28),
        ]

    if strength == "light":
        return [
            dict(contractions=True),
            dict(contractions=True, split_words=32),
        ]
    if strength == "strong":
        return [
            dict(contractions=True, split_words=22),
            dict(contractions=True, split_words=18, break_lists=True),
            dict(contractions=True, split_words=26),
            dict(contractions=True, split_words=20, break_lists=True),
        ]
    return [
        dict(contractions=True),
        dict(contractions=True, split_words=28),
        dict(contractions=True, split_words=24, break_lists=True),
        dict(contractions=True, split_words=32),
    ]


def _rewrite_line(
    line: str,
    tone: str,
    changes: Dict[str, list],
    *,
    contractions: bool,
    split_words: int,
    break_lists: bool,
) -> str:
    if not line.strip() or _HEADING_LINE_RE.match(line):
        return line

    bullet = _BULLET_LINE_RE.match(line)
    if bullet:
        prefix, body = bullet.groups()
        body = _apply_phrases(body, tone, changes)
        if contractions:
            body = _apply_pairs(body, _CONTRACTIONS)
        return prefix + body

    lead = line[: len(line) - len(line.lstrip())]
    piece = _apply_phrases(line.strip(), tone, changes)
    if contractions:
        piece = _apply_pairs(piece, _CONTRACTIONS)

    parts = [piece]
    if break_lists:
        expanded: List[str] = []
        for part in parts:
            expanded.extend(_break_lists(part))
        parts = expanded
    if split_words:
        split_parts: List[str] = []
        for part in parts:
            split_parts.extend(_split_long(part, split_words, tone))
        parts = split_parts
    return lead + " ".join(parts)


def _rewrite_layout(
    text: str,
    tone: str,
    *,
    contractions: bool = True,
    split_words: int = 0,
    break_lists: bool = False,
) -> Tuple[str, Dict[str, list]]:
    changes: Dict[str, list] = {}
    out: List[str] = []
    in_fence = False
    for line in clean_text(text).split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or stripped.startswith("|"):
            out.append(line)
            continue
        out.append(
            _rewrite_line(
                line,
                tone,
                changes,
                contractions=contractions,
                split_words=split_words,
                break_lists=break_lists,
            )
        )
    return "\n".join(out), changes


def _merge_changes(target: Dict[str, list], extra: Dict[str, list]) -> None:
    for key, (alt, count) in extra.items():
        entry = target.setdefault(key, [alt, 0])
        entry[1] += count


def _external_worst(after_stealth: dict) -> float:
    components = after_stealth.get("components") or {}
    return max(components.get(key, 0.0) for key in ("zerogpt", "gptzero", "pangram"))


def _humanize_external(original: str, tone: str, strength: str) -> Optional[dict]:
    if not stealth.available():
        return None

    started = time.perf_counter()
    before = score_only(_score_view(original))

    candidates: List[Tuple[str, Dict[str, list]]] = []
    seen = {original}
    for profile in _external_profiles(tone, strength):
        rewritten, changes = _rewrite_layout(original, tone, **profile)
        if rewritten.strip() and rewritten not in seen:
            seen.add(rewritten)
            candidates.append((rewritten, changes))
    if not candidates:
        return None

    texts = [original] + [text for text, _ in candidates]
    zg_items = zerogpt_style_features_scores(texts)
    if not zg_items or zg_items[0] is None:
        return None
    before_zg = zg_items[0]
    before_stealth = stealth.score_from_features(
        stealth.merge_features(original, before_zg.get("features"))
    )
    if before_stealth is None:
        return None

    batch = score_many([_score_view(text) for text, _ in candidates])
    ranked = []
    for (rewritten, changes), after, zg in zip(candidates, batch, zg_items[1:]):
        if after is None or not zg:
            continue
        after_stealth = stealth.score_from_features(
            stealth.merge_features(rewritten, zg.get("features"))
        )
        if after_stealth is None:
            continue
        ranked.append(
            (
                _external_worst(after_stealth),
                after_stealth["risk"],
                after["ai_percent"],
                rewritten,
                changes,
                after,
                zg,
                after_stealth,
            )
        )

    warnings: List[str] = []
    original_key = (_external_worst(before_stealth), before_stealth["risk"])
    if ranked:
        ranked.sort(key=lambda row: (row[0], row[1], row[2]))
        candidate = ranked[0]
        if (candidate[0], candidate[1]) < original_key:
            (
                _worst,
                risk,
                _classifier,
                best_text,
                best_changes,
                best_after,
                best_zg,
                best_stealth,
            ) = candidate
        else:
            risk = before_stealth["risk"]
            best_text = original
            best_changes = {}
            best_after = before
            best_zg = before_zg
            best_stealth = before_stealth
            warnings.append(
                "No rewrite lowered the external-risk estimate — the original is unchanged."
            )
    else:
        risk = before_stealth["risk"]
        best_text = original
        best_changes = {}
        best_after = before
        best_zg = before_zg
        best_stealth = before_stealth
        warnings.append("Could not score any rewrite — the original is unchanged.")

    if before_stealth["risk"] < cfg.MIXED_VERDICT_THRESHOLD:
        warnings.append(
            f"External proxies already rate this text low ({before_stealth['risk']}%) "
            "— the rewrite may be unnecessary."
        )
    else:
        warnings.append(
            f"Rewriting against external detectors (estimated risk "
            f"{before_stealth['risk']}%); headings, lists and layout are preserved."
        )
    if best_text != original:
        if risk < before_stealth["risk"]:
            warnings.append(
                f"Estimated external risk moved from {before_stealth['risk']}% to {risk}%."
            )
        else:
            warnings.append(
                f"External risk did not improve ({before_stealth['risk']}% → {risk}%) "
                "— try strong strength or edit a few sentences by hand."
            )

    components = best_stealth.get("components") or {}
    high = [
        label
        for key, label in (
            ("zerogpt", "ZeroGPT"),
            ("gptzero", "GPTZero"),
            ("pangram", "Pangram"),
        )
        if components.get(key, 0.0) >= 50.0
    ]
    if best_text != original and high:
        warnings.append(
            f"Still at or above 50% on {', '.join(high)} — try casual tone or "
            "strong strength, then verify in the real tools."
        )
    if (
        tone == "formal"
        and best_text != original
        and best_after["ai_percent"] >= cfg.MIXED_VERDICT_THRESHOLD
    ):
        warnings.append(
            "Formal tone keeps the original register, which detectors tend to flag. "
            "Casual tone usually drops the estimated risk further."
        )
    if best_after["ai_percent"] >= cfg.AI_VERDICT_THRESHOLD:
        warnings.append(
            "The local classifier still rates this rewrite as AI-like — check it in "
            "a second detector before publishing."
        )
    warnings.append(
        "External risk is estimated by local proxies trained on public ZeroGPT, "
        "GPTZero and Pangram labels; real tool results can differ."
    )

    return _result(
        original,
        best_text,
        tone,
        strength,
        before,
        best_after,
        best_changes,
        warnings,
        started,
        before_zg=before_zg,
        after_zg=best_zg,
        target="external",
        before_stealth=before_stealth,
        after_stealth=best_stealth,
    )


def humanize(
    text: str,
    tone: str = "casual",
    strength: str = "medium",
    target: str = "external",
) -> dict:
    original = clean_text(text)
    if not original:
        raise ValueError("Text is empty after cleaning.")
    if tone not in TONES:
        tone = "casual"
    if strength not in STRENGTHS:
        strength = "medium"
    if target not in TARGETS:
        target = "external"

    fallback: List[str] = []
    if target == "external":
        result = _humanize_external(original, tone, strength)
        if result is not None:
            return result
        fallback.append(
            "External proxy mode is unavailable — optimized for this detector instead."
        )
    return _local_humanize(original, tone, strength, extra_warnings=fallback)


def _local_humanize(
    text: str,
    tone: str = "casual",
    strength: str = "medium",
    extra_warnings: Optional[List[str]] = None,
) -> dict:
    started = time.perf_counter()
    original = clean_text(text)
    if not original:
        raise ValueError("Text is empty after cleaning.")
    if tone not in TONES:
        tone = "casual"
    if strength not in STRENGTHS:
        strength = "medium"

    before = score_only(_score_view(original))
    before_zg = zerogpt_style_score(original)
    warnings: List[str] = list(extra_warnings or [])
    if len(original) > _SCORING_SAMPLE_CHARS:
        warnings.append(
            "Long document: scores were estimated from a representative sample "
            "for speed, but the full text was rewritten."
        )

    already_human = before["ai_percent"] < cfg.MIXED_VERDICT_THRESHOLD
    zg_mode = already_human and before_zg is not None
    if already_human:
        if before_zg is not None and before_zg["pct"] >= 50:
            warnings.append(
                f"This detector rates the text as human ({before['ai_percent']}%), "
                f"but the ZeroGPT-style estimate is high ({before_zg['pct']}%) — "
                "rewriting for the stricter signal."
            )
        else:
            warnings.append(
                "This text already reads as human-written — the rewrite may be unnecessary."
            )
    if before_zg is None:
        warnings.append(
            "ZeroGPT-style estimate unavailable — the rewrite was optimized with "
            "the classifier only."
        )

    def combined(classifier_score: float, zg: Optional[dict]) -> float:
        if zg is None or before_zg is None:
            return classifier_score
        return 0.5 * classifier_score + 0.5 * zg["pct"]

    zg_cache: Dict[str, Optional[dict]] = {original: before_zg}

    candidates: List[Tuple[str, Dict[str, list]]] = []
    seen = {original}
    for profile in _profiles(tone, strength):
        rewritten, changes = _rewrite(original, tone, **profile)
        if rewritten.strip() and rewritten not in seen:
            seen.add(rewritten)
            candidates.append((rewritten, changes))

    scored: List[Tuple[float, str, Dict[str, list], dict]] = []
    if candidates:
        try:
            batch = score_many([_score_view(text) for text, _ in candidates])
        except Exception:
            batch = [None] * len(candidates)
        for (rewritten, changes), after in zip(candidates, batch):
            if after is None:
                continue
            scored.append((after["ai_percent"], rewritten, changes, after))

    best_text, best_changes, best_after = original, {}, before
    best_zg = before_zg
    best_combined = combined(before["ai_percent"], before_zg)
    best_classifier = before["ai_percent"]
    if scored:
        if zg_mode:
            zg_list = zerogpt_style_scores([rewritten for _, rewritten, _, _ in scored])
            ranked = []
            for (classifier_score, rewritten, changes, after), zg in zip(scored, zg_list):
                zg_cache[rewritten] = zg
                if zg is None:
                    continue
                ranked.append(
                    (zg["pct"], classifier_score, rewritten, changes, after, zg)
                )
            if ranked:
                ranked.sort(key=lambda row: (row[0], row[1]))
                (
                    _,
                    best_classifier,
                    best_text,
                    best_changes,
                    best_after,
                    best_zg,
                ) = ranked[0]
                best_combined = combined(best_classifier, best_zg)
            else:
                scored.sort(key=lambda row: row[0])
                best_classifier, best_text, best_changes, best_after = scored[0]
                best_combined = best_classifier
        elif before_zg is not None:
            scored.sort(key=lambda row: row[0])
            top = scored[:3]
            zg_list = zerogpt_style_scores([rewritten for _, rewritten, _, _ in top])
            reranked = []
            for (classifier_score, rewritten, changes, after), zg in zip(top, zg_list):
                zg_cache[rewritten] = zg
                reranked.append(
                    (
                        combined(classifier_score, zg),
                        classifier_score,
                        rewritten,
                        changes,
                        after,
                        zg,
                    )
                )
            reranked.sort(key=lambda row: row[0])
            best_combined, best_classifier, best_text, best_changes, best_after, best_zg = (
                reranked[0]
            )
        else:
            scored.sort(key=lambda row: row[0])
            best_classifier, best_text, best_changes, best_after = scored[0]
            best_combined = best_classifier

        needs_refinement = zg_mode or best_classifier >= cfg.MIXED_VERDICT_THRESHOLD
        if needs_refinement:
            refined, extra_changes = _rewrite(
                original, tone, split_words=20, markers=True, opener=0,
                break_lists=True, personal=0, lower_style=2,
            )
            if refined != best_text:
                try:
                    refined_after = score_only(_score_view(refined))
                    refined_zg = (
                        zerogpt_style_score(refined) if before_zg is not None else None
                    )
                    if before_zg is not None:
                        zg_cache[refined] = refined_zg
                    if zg_mode:
                        accept = refined_zg is not None and (
                            best_zg is None
                            or (refined_zg["pct"], refined_after["ai_percent"])
                            < (best_zg["pct"], best_classifier)
                        )
                    else:
                        accept = (
                            combined(refined_after["ai_percent"], refined_zg)
                            < best_combined
                        )
                    if accept:
                        _merge_changes(best_changes, extra_changes)
                        best_text = refined
                        best_after = refined_after
                        best_classifier = refined_after["ai_percent"]
                        if refined_zg is not None:
                            best_zg = refined_zg
                        best_combined = combined(best_classifier, best_zg)
                except ValueError:
                    pass

    needs_ending = tone == "casual" and (
        best_after["ai_percent"] >= cfg.MIXED_VERDICT_THRESHOLD
        or (zg_mode and best_zg is not None and best_zg["pct"] >= 50)
    )
    if needs_ending:
        tagged_text, tagged_after, tag = _best_ending(best_text, best_after)
        if tag:
            if zg_mode:
                tagged_zg = zerogpt_style_score(tagged_text)
                zg_cache[tagged_text] = tagged_zg
                if (
                    tagged_zg is not None
                    and best_zg is not None
                    and tagged_zg["pct"] <= best_zg["pct"]
                ):
                    best_text, best_after = tagged_text, tagged_after
                    best_zg = tagged_zg
                    best_changes.setdefault("(added ending)", [tag, 1])
            else:
                best_text, best_after = tagged_text, tagged_after
                best_changes.setdefault("(added ending)", [tag, 1])

    if before_zg is None:
        after_zg = None
    elif best_text in zg_cache:
        after_zg = zg_cache[best_text]
    else:
        after_zg = zerogpt_style_score(best_text)

    if best_text == original:
        warnings.append("No rewrite improved the score — the original is unchanged.")
    if zg_mode and after_zg is not None and best_text != original:
        if after_zg["pct"] < before_zg["pct"]:
            warnings.append(
                f"ZeroGPT-style estimate moved from {before_zg['pct']}% to "
                f"{after_zg['pct']}%."
            )
        else:
            warnings.append(
                "The ZeroGPT-style estimate did not improve — try casual tone with "
                "strong strength, or edit a few sentences by hand."
            )
    if best_after["ai_percent"] >= cfg.AI_VERDICT_THRESHOLD:
        warnings.append(
            "This text resisted rewriting. Try casual tone with strong strength, "
            "or rewrite a few sentences by hand."
        )
    if tone == "formal" and best_after["ai_percent"] >= cfg.MIXED_VERDICT_THRESHOLD:
        warnings.append(
            "Formal tone keeps the original register, which detectors tend to flag. "
            "Casual tone usually drops the score much further."
        )
    warnings.append(
        "Scores are tuned to this detector; other AI-detection tools may score differently."
    )
    return _result(
        original,
        best_text,
        tone,
        strength,
        before,
        best_after,
        best_changes,
        warnings,
        started,
        before_zg=before_zg,
        after_zg=after_zg,
        target="local",
    )


def _result(
    original: str,
    rewritten: str,
    tone: str,
    strength: str,
    before: dict,
    after: dict,
    changes: Dict[str, list],
    warnings: List[str],
    started: float,
    before_zg: Optional[dict] = None,
    after_zg: Optional[dict] = None,
    target: str = "local",
    before_stealth: Optional[dict] = None,
    after_stealth: Optional[dict] = None,
) -> dict:
    ordered = sorted(changes.items(), key=lambda item: item[1][1], reverse=True)
    return {
        "original": original,
        "rewritten": rewritten,
        "tone": tone,
        "strength": strength,
        "target": target,
        "before": {
            "ai_percent": before["ai_percent"],
            "verdict": before["verdict"],
        },
        "after": {
            "ai_percent": after["ai_percent"],
            "verdict": after["verdict"],
        },
        "zerogpt": {
            "before": before_zg["pct"] if before_zg else None,
            "after": after_zg["pct"] if after_zg else None,
        },
        "stealth": (
            {"before": before_stealth, "after": after_stealth}
            if before_stealth is not None or after_stealth is not None
            else None
        ),
        "changes": [
            {"source": key, "target": value[0], "count": value[1]}
            for key, value in ordered[:12]
        ],
        "warnings": warnings,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }
