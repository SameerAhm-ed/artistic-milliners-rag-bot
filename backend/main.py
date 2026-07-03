from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import rag
from store import VectorStore
from session import SessionStore
from ingest import ingest_corpus

app = FastAPI(title="AM Support RAG Bot")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

store = VectorStore(config.CHROMA_DIR, config.COLLECTION)
sessions = SessionStore()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/health")
def health():
    return {"status": "ok", "chunks": store.count()}

@app.post("/ingest")
def ingest():
    n = ingest_corpus(config.CORPUS_DIR, store)
    return {"ingested": n}

@app.post("/chat")
def chat(req: ChatRequest):
    return rag.answer(req.message, req.session_id, store, sessions)
