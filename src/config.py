"""Central configuration: paths and secrets (loaded from .env, never hard-coded)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_responses"
RAW_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Copy .env.example to .env and fill it in.")
    return value


RAG_BASE_URL = _require("RAG_BASE_URL").rstrip("/")
RAG_API_KEY = _require("RAG_API_KEY")

LEHMUS_BASE_URL = os.getenv("LEHMUS_BASE_URL", "").rstrip("/")
LEHMUS_API_KEY = os.getenv("LEHMUS_API_KEY", "")
LEHMUS_SMALL_MODEL = os.getenv("LEHMUS_SMALL_MODEL", "")
LEHMUS_LARGE_MODEL = os.getenv("LEHMUS_LARGE_MODEL", "")
LEHMUS_EMBED_MODEL = os.getenv("LEHMUS_EMBED_MODEL", "")
