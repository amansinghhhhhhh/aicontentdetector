import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import schemas, settings
from app.db import init_db
from app.routers import detect as detect_router
from app.routers import history as history_router
from app.routers import humanize as humanize_router
from app.services import detector

logger = logging.getLogger("detector")


def _warmup() -> None:
    try:
        source = detector.warmup()
        logger.info("Model warmup complete: %s", source)
    except Exception:
        logger.exception("Model warmup failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if settings.WARMUP_MODEL:
        threading.Thread(target=_warmup, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="AI Content Detector API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "https://zerogptt.com",
        "https://www.zerogptt.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect_router.router)
app.include_router(history_router.router)
app.include_router(humanize_router.router)


@app.get("/api/health", response_model=schemas.HealthResponse)
def health():
    return {
        "status": "ok",
        "model_loaded": detector.is_model_loaded(),
        "model_source": detector.model_source(),
        "model_backend": detector.model_backend(),
        "zerogpt_available": detector.zerogpt_available(),
    }
