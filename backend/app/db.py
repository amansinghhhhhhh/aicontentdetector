from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app import settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    source: Mapped[str] = mapped_column(String(16))
    filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    preview: Mapped[str] = mapped_column(String(300))
    words: Mapped[int] = mapped_column(Integer, default=0)
    ai_percent: Mapped[float] = mapped_column(Float, default=0.0)
    verdict: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    result_json: Mapped[str] = mapped_column(Text)


def _ensure_sqlite_dir(url: str) -> None:
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        path = Path(url[len("sqlite:///"):])
        path.parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_dir(settings.DATABASE_URL)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
    if settings.DATABASE_URL.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
