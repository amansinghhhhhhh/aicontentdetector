import re
from dataclasses import dataclass
from typing import List, Sequence

_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+(?:[\"'”’)\]]*)(?=\s|$)|[^.!?]+$")
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_INLINE_GAP_RE = re.compile(r" *\n *")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

_ABBREV_END_RE = re.compile(
    r"(?:"
    r"\b[A-Za-z]\."
    r"|\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|approx|fig|no|inc|ltd|co|dept|vol|pp|cf|al|ca|est|vol|vs)\."
    r"|\bU\.S\.|\bU\.K\.|\bE\.U\."
    r")$",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


@dataclass
class Sentence:
    index: int
    text: str
    start: int
    end: int
    words: int


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _INLINE_GAP_RE.sub("\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _needs_merge(fragment: str) -> bool:
    return bool(_ABBREV_END_RE.search(fragment.strip()))


def _adjust(fragment: str, start: int) -> tuple:
    stripped = fragment.strip()
    if not stripped:
        return "", -1, -1
    offset = len(fragment) - len(fragment.lstrip())
    return stripped, start + offset, start + offset + len(stripped)


def split_sentences(text: str) -> List[Sentence]:
    if not text.strip():
        return []

    protected = _DECIMAL_RE.sub("\u0001", text)
    spans = []
    for match in _SENTENCE_RE.finditer(protected):
        frag, start, end = _adjust(match.group(), match.start())
        if frag and end > 0:
            spans.append([start, end])

    if not spans:
        return []

    merged: List[List[int]] = []
    for span in spans:
        if merged and _needs_merge(text[merged[-1][0]:span[1]]):
            merged[-1][1] = span[1]
        else:
            merged.append(span)

    sentences: List[Sentence] = []
    for idx, (start, end) in enumerate(merged):
        raw = text[start:end].replace("\u0001", ".")
        sentences.append(
            Sentence(index=idx, text=raw, start=start, end=end, words=word_count(raw))
        )
    return sentences


def pack_sentences(
    sentences: Sequence[Sentence],
    tokenizer,
    max_length: int,
    reserve: int = 2,
) -> List[str]:
    """Group consecutive sentences into <=max_length token chunks (sentence-aligned)."""
    if not sentences:
        return []
    budget = max_length - reserve
    token_counts = [
        len(ids)
        for ids in tokenizer(
            [s.text for s in sentences],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    ]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence, count in zip(sentences, token_counts):
        piece = sentence.text
        if count > budget:
            for piece in _hard_split(piece, tokenizer, budget):
                if current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                chunks.append(piece)
            continue
        if current and current_len + count > budget:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(piece)
        current_len += count

    if current:
        chunks.append(" ".join(current))
    return chunks


def _hard_split(text: str, tokenizer, budget: int) -> List[str]:
    """Split an over-long sentence on word boundaries within token budget."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    if len(offsets) <= budget:
        return [text]

    pieces: List[str] = []
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        if end < len(offsets):
            cut = offsets[end - 1][1]
            space = text.rfind(" ", offsets[start][0], cut)
            if space > offsets[start][0]:
                cut = space
            piece = text[offsets[start][0]:cut].strip()
            if piece:
                pieces.append(piece)
            while end < len(offsets) and offsets[end][0] < cut:
                end += 1
            start = end
        else:
            piece = text[offsets[start][0]:].strip()
            if piece:
                pieces.append(piece)
            break
    return pieces or [text]
