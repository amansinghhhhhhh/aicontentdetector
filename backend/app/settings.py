import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "scans.db"

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_ANALYSIS_CHARS = int(os.environ.get("MAX_ANALYSIS_CHARS", "20000"))
MAX_REQUEST_CHARS = 500_000
WARMUP_MODEL = os.environ.get("MODEL_WARMUP", "1") == "1"
