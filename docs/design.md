# Customer Support RAG Chatbot — Design Spec

**Date:** 2026-07-03
**Author:** Sameer (for Artistic Milliners — Chatbot / AI Conversation Engineer role, portfolio piece)
**Status:** Design — approved pending user review
**Related:** [PRD_Portfolio_Projects.md](../../../PRD_Portfolio_Projects.md) Project 1

---

## 1. Purpose

Build a Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded
in a fixed document set, with cited sources and multi-turn memory. Demonstrates core
AI Conversation Engineer skills: LLMs/NLP, Python, multi-turn conversation, retrieval.

Runs **fully local, zero paid API** — constrained by the build machine (7.7 GB RAM,
integrated Intel GPU, no dedicated VRAM).

## 2. Success Criteria

- Answers test questions correctly using **only** the provided documents.
- Follow-up questions correctly use prior turn context (e.g. "what about the second one?").
- Every answer shows its source (document + section/chunk).
- Retrieved chunks are surfaced in the UI (debug panel) — proves retrieval, not just LLM.
- Runs locally on the build machine without OOM.
- GitHub repo + README with architecture diagram. (Live deploy = stretch, post-MVP.)

## 3. Constraints (machine-driven)

| Constraint | Value | Design impact |
|---|---|---|
| RAM | 7.7 GB total | Small LLM only; light embeddings (no torch) |
| GPU | Intel UHD (integrated) | CPU inference; expect modest latency |
| Cost | $0 | Ollama local LLM + local ONNX embeddings |

## 4. Stack

| Layer | Choice | Reason |
|---|---|---|
| LLM | Ollama `llama3.2:3b` (fallback `llama3.2:1b`) | ~2GB, fits RAM; RAG LLM only synthesizes retrieved context |
| Embeddings | `fastembed` (all-MiniLM-L6-v2, ONNX, CPU) | ~90MB, no torch, low RAM; Chroma native support |
| Vector DB | Chroma (local, persistent) | Free, embedded, no server |
| Backend | FastAPI (Python 3.12) | Async, clean REST, matches role |
| Frontend | React + Vite, **mobile-first + polished** | Chat UI + retrieved-chunks debug panel; Stripe/Shopify-grade look |
| Corpus | Self-written AM mock policy/FAQ markdown | Defensible, personalized, no scraping/legal issue |

### Frontend design intent

- **Mobile-first**: design at ~375px width first, scale up to desktop.
- **Polished SaaS aesthetic** (Stripe / Shopify vibe): clean typography, generous spacing,
  restrained color, subtle motion/transitions, thoughtful empty/loading states.
- Chat interface + collapsible "retrieved sources / chunks" panel.
- Built using the `frontend-design` skill during implementation.
- PWA (installable, offline app-shell) = **stretch**, not v1 (see Out of Scope).

## 5. Architecture

```
frontend (React/Vite)  ──HTTP──►  backend (FastAPI)
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              fastembed         Chroma (vector    Ollama
              (embed query)      store, top-k)    (llama3.2:3b)
```

### Components (isolated, single-purpose)

1. **`ingest.py`** — load markdown → chunk → embed → upsert to Chroma.
   - In: doc folder path. Out: populated Chroma collection.
   - Depends on: fastembed, Chroma.
2. **`retriever.py`** — embed query → Chroma top-k similarity search.
   - In: query string, k. Out: list of chunks + metadata (source, section).
   - Depends on: fastembed, Chroma.
3. **`chat.py`** — build prompt (query + retrieved chunks + session history) → Ollama → answer.
   - In: query, session_id. Out: answer + cited sources + retrieved chunks.
   - Depends on: retriever, Ollama client, session store.
4. **`session.py`** — in-memory multi-turn history keyed by session_id.
   - In/Out: append + fetch recent turns.
5. **`main.py`** — FastAPI app wiring endpoints.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | (Re)build vector store from corpus |
| POST | `/chat` | `{session_id, message}` → `{answer, sources[], chunks[]}` |
| GET | `/health` | Liveness + model status |

## 6. Data Flow

1. User sends message + session_id from React UI.
2. Backend embeds query (fastembed).
3. Chroma returns top-k chunks with source metadata.
4. Build prompt: system instruction (answer only from context, cite, say "I don't know"
   if not covered) + retrieved chunks + recent session history + user query.
5. Ollama (llama3.2:3b) generates grounded answer.
6. Response = answer + cited sources + the raw retrieved chunks (for debug panel).
7. Append turn to session history.

## 7. Chunking / Retrieval Strategy

- Chunk by markdown section (headings) with a fallback char-size cap (~500–800 chars, small overlap).
- Store metadata: `source_file`, `section_title`, `chunk_index`.
- Retrieval: top-k = 3–4. Tune during testing against a fixed question set.
- Anti-hallucination: system prompt forces "answer only from provided context; if not
  present, say you don't know." This is the main correctness lever.

## 8. Corpus (self-written)

A small realistic Artistic Milliners knowledge base in markdown, e.g.:
- Company / about / sustainability blurb
- Product / fabric FAQ
- Support policy (returns, lead times, MOQ, contact)

Kept intentionally answerable-and-bounded so "I don't know" cases are testable too.

## 9. Out of Scope (v1)

- Authentication / user accounts
- Multi-language
- Voice interface
- PWA (installable + offline app-shell) → **stretch** (v1 is polished mobile-first web, not installable)
- WhatsApp/Telegram integration → **stretch** (job "Plus" skill)
- Cloud deployment → **stretch** after local MVP works
- Tool-calling / function calls → stretch

## 10. Testing

- Fixed question set (~10 Q): in-corpus answerable, out-of-corpus (expect "I don't know"),
  and one multi-turn follow-up ("what about the second one?").
- Assert: correct answer, correct cited source, graceful unknown handling.
- Manual latency check on the build machine (target: usable, not instant).

## 11. Repo & Process

- New standalone folder + own git repo. Incremental commits with meaningful messages.
- README written near end: problem, architecture diagram, stack, what's library vs custom,
  honest notes (local small model tradeoffs), screenshots/GIF.
- Framing: built specifically in response to the AM job posting.

## 12. Build Order (later, in implementation plan)

1. Env: venv, install deps, `ollama pull llama3.2:3b`, verify Ollama runs.
2. Corpus doc.
3. ingest + retriever (verify retrieval quality in isolation).
4. chat + session + Ollama wiring.
5. FastAPI endpoints.
6. React minimal UI + debug panel.
7. Test set + tuning.
8. README + diagram. Stretch: deploy / WhatsApp.
