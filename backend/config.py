import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", PROJECT_ROOT / "corpus"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
COLLECTION = os.getenv("COLLECTION", "am_support")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
