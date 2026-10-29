from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app import settings


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=settings.MAX_REQUEST_CHARS)


class HumanizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=settings.MAX_REQUEST_CHARS)
    tone: Literal["casual", "formal"] = "casual"
    strength: Literal["light", "medium", "strong"] = "medium"
    target: Literal["external", "local"] = "external"


class ScoreBrief(BaseModel):
    ai_percent: float
    verdict: Literal["ai", "mixed", "human"]


class ZerogptScore(BaseModel):
    pct: float
    prob: float


class ZerogptBrief(BaseModel):
    before: Optional[float] = None
    after: Optional[float] = None


class HumanizeChange(BaseModel):
    source: str
    target: str
    count: int


class StealthBrief(BaseModel):
    risk: float
    worst: Optional[float] = None
    components: dict[str, float]


class StealthResult(BaseModel):
    before: Optional[StealthBrief] = None
    after: Optional[StealthBrief] = None


class HumanizeResult(BaseModel):
    original: str
    rewritten: str
    tone: str
    strength: str
    target: Literal["external", "local"] = "local"
    before: ScoreBrief
    after: ScoreBrief
    zerogpt: Optional[ZerogptBrief] = None
    stealth: Optional[StealthResult] = None
    changes: list[HumanizeChange]
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: float


class SentenceScore(BaseModel):
    index: int
    text: str
    start: int
    end: int
    words: int
    score: float


class ScanStats(BaseModel):
    characters: int
    words: int
    sentences: int
    chunks: int
    ai_sentences: int


class ScanResult(BaseModel):
    id: Optional[int] = None
    source: Optional[str] = None
    filename: Optional[str] = None
    created_at: Optional[datetime] = None
    text: str
    ai_percent: float
    verdict: Literal["ai", "mixed", "human"]
    zerogpt: Optional[ZerogptScore] = None
    confidence: float
    class_percent: dict[str, float]
    stats: ScanStats
    sentences: list[SentenceScore]
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: float
    model: str


class HistoryItem(BaseModel):
    id: int
    source: str
    filename: Optional[str]
    preview: str
    words: int
    ai_percent: float
    verdict: str
    confidence: float
    created_at: datetime


class HistoryList(BaseModel):
    items: list[HistoryItem]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: Optional[str] = None
    model_backend: Optional[str] = None
    zerogpt_available: bool = False


class DeletedResponse(BaseModel):
    deleted: int
