from fastapi import APIRouter, HTTPException

from app import schemas
from app.services import humanizer

router = APIRouter()


@router.post("/api/humanize", response_model=schemas.HumanizeResult)
def humanize(body: schemas.HumanizeRequest):
    # Full text is rewritten; the schema caps input at MAX_REQUEST_CHARS.
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    try:
        result = humanizer.run(body.text, body.tone, body.strength, body.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result
