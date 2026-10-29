import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import schemas
from app.db import Scan, get_session

router = APIRouter()


def _to_result(scan: Scan) -> dict:
    payload = json.loads(scan.result_json)
    payload["id"] = scan.id
    payload["source"] = scan.source
    payload["filename"] = scan.filename
    payload["created_at"] = scan.created_at
    return payload


@router.get("/api/history", response_model=schemas.HistoryList)
def list_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    total = session.scalar(select(func.count()).select_from(Scan)) or 0
    rows = session.scalars(
        select(Scan)
        .order_by(Scan.created_at.desc(), Scan.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    items = [
        schemas.HistoryItem(
            id=row.id,
            source=row.source,
            filename=row.filename,
            preview=row.preview,
            words=row.words,
            ai_percent=row.ai_percent,
            verdict=row.verdict,
            confidence=row.confidence,
            created_at=row.created_at,
        )
        for row in rows
    ]
    return {"items": items, "total": total}


@router.get("/api/history/{scan_id}", response_model=schemas.ScanResult)
def get_history_item(scan_id: int, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return _to_result(scan)


@router.delete("/api/history/{scan_id}", response_model=schemas.DeletedResponse)
def delete_history_item(scan_id: int, session: Session = Depends(get_session)):
    scan = session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found.")
    session.delete(scan)
    session.commit()
    return {"deleted": 1}


@router.delete("/api/history", response_model=schemas.DeletedResponse)
def clear_history(session: Session = Depends(get_session)):
    count = session.scalar(select(func.count()).select_from(Scan)) or 0
    session.execute(Scan.__table__.delete())
    session.commit()
    return {"deleted": count}
