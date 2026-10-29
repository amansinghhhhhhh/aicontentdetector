import json
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app import schemas, settings
from app.db import Scan, get_session
from app.services import detector, extraction

router = APIRouter()


def _cut_at_sentence(text: str, limit: int) -> str:
    head = text[:limit]
    for sep in (". ", "! ", "? ", "\n"):
        pos = head.rfind(sep)
        if pos > limit // 2:
            return head[: pos + 1].rstrip()
    return head.rstrip()


def _truncate(text: str) -> tuple:
    if len(text) <= settings.MAX_ANALYSIS_CHARS:
        return text, []
    cut = _cut_at_sentence(text, settings.MAX_ANALYSIS_CHARS)
    warning = (
        f"Input is {len(text)} characters — analysis was limited to the first "
        f"{settings.MAX_ANALYSIS_CHARS} characters."
    )
    return cut, [warning]


def _analyze_and_store(
    text: str,
    source: str,
    filename: Optional[str],
    session: Session,
    extra_warnings: list,
) -> dict:
    try:
        result = detector.run_analysis(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if extra_warnings:
        result["warnings"] = list(result.get("warnings", [])) + extra_warnings

    scan = Scan(
        source=source,
        filename=filename,
        preview=result["text"][:300],
        words=result["stats"]["words"],
        ai_percent=result["ai_percent"],
        verdict=result["verdict"],
        confidence=result["confidence"],
        result_json=json.dumps(result, ensure_ascii=False),
    )
    session.add(scan)
    session.commit()

    result["id"] = scan.id
    result["source"] = source
    result["filename"] = filename
    result["created_at"] = scan.created_at
    return result


@router.post("/api/detect", response_model=schemas.ScanResult)
def detect(body: schemas.DetectRequest, session: Session = Depends(get_session)):
    text, warnings = _truncate(body.text)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    return _analyze_and_store(text, "text", None, session, warnings)


@router.post("/api/detect/file", response_model=schemas.ScanResult)
def detect_file(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    data = file.file.read()
    if len(data) > settings.MAX_UPLOAD_BYTES:
        limit_mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File exceeds {limit_mb} MB limit.")

    filename = file.filename or ""
    try:
        text = extraction.extract_text(filename, data)
    except extraction.UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except extraction.EmptyDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not read '{filename}': {exc}"
        ) from exc

    text, warnings = _truncate(text)
    return _analyze_and_store(text, "file", filename, session, warnings)
