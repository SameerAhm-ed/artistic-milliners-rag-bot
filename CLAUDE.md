# CLAUDE.md — RAG Support Bot

Project conventions for Claude Code. Read before editing.

## What this is
Local, zero-cost Retrieval-Augmented Generation chatbot. Portfolio piece for the
Artistic Milliners "Chatbot / AI Conversation Engineer" role. Answers questions grounded
in a self-written AM mock knowledge base, with cited sources and multi-turn memory.

Full design: [docs/design.md](docs/design.md). Follow it. If a change deviates, update the spec first.

## Hard constraints
- **$0 / fully local.** No paid APIs. Ever.
- **Machine: 7.7 GB RAM, Intel UHD (no dedicated GPU).** Keep memory small. CPU inference.
- LLM: Ollama `llama3.2:3b` (fallback `llama3.2:1b`). Embeddings: ChromaDB built-in `DefaultEmbeddingFunction` (ONNX all-MiniLM-L6-v2, CPU, no torch) — same lightweight MiniLM model, fewer deps than a separate fastembed install.

## Stack
- Backend: FastAPI (Python 3.12), Chroma (local persistent vector store).
- Frontend: React + Vite. **Mobile-first**, Stripe/Shopify-grade polish. PWA = stretch only.
- Model-agnostic: LLM/embeddings behind small interfaces so they can be swapped.

## Layout
```
backend/    FastAPI app — ingest.py, retriever.py, chat.py, session.py, main.py
frontend/   React + Vite app
corpus/     Self-written AM mock markdown docs (the knowledge base)
docs/        design.md (spec)
```

## Conventions
- Small, single-purpose modules. Each: clear input/output, testable in isolation.
- Anti-hallucination is the priority correctness lever: system prompt = "answer only from
  provided context; if not present, say you don't know." Never loosen this silently.
- Every answer must return its cited sources + the raw retrieved chunks.
- No secrets in code. Config via env / a `.env` (gitignored).

## Commands (fill in as built)
- Backend dev: `cd backend && uvicorn main:app --reload`
- Frontend dev: `cd frontend && npm run dev`
- Pull model: `ollama pull llama3.2:3b`
- Ingest corpus: (endpoint) `POST /ingest`

## Git
- Incremental commits, meaningful messages (portfolio signal: built incrementally, not dumped).
- Do NOT commit: venv, node_modules, chroma persistence dir, model files, `.env`.
