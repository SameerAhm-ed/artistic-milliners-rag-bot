# RAG Support Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, zero-cost RAG chatbot that answers questions grounded in a self-written Artistic Milliners mock knowledge base, with cited sources and multi-turn memory.

**Architecture:** FastAPI backend. Markdown corpus → chunk → embed (ONNX MiniLM) → Chroma vector store. On query: embed → top-k retrieve → build prompt (system + chunks + session history) → Ollama `llama3.2:3b` → answer + sources + raw chunks. React+Vite mobile-first frontend talks to the backend over REST.

**Tech Stack:** Python 3.12, FastAPI, ChromaDB (local persistent), ChromaDB default ONNX all-MiniLM-L6-v2 embeddings, Ollama (`llama3.2:3b`), React 18 + Vite, plain CSS.

## Global Constraints

- **$0 / fully local.** No paid APIs anywhere.
- **Machine:** 7.7 GB RAM, Intel UHD (integrated, no CUDA). CPU inference only. Keep memory small.
- **LLM:** Ollama `llama3.2:3b` (fallback `llama3.2:1b`). Never a hosted API.
- **Embeddings:** ChromaDB built-in `DefaultEmbeddingFunction` = ONNX all-MiniLM-L6-v2, CPU, no torch. (This is the same lightweight MiniLM ONNX model the spec called "fastembed"; using Chroma's built-in avoids an extra dependency and extra RAM. Equivalent model, fewer moving parts.)
- **Anti-hallucination is mandatory:** system prompt must instruct "answer ONLY from provided context; if the answer is not in the context, say you don't know." Never loosen silently.
- **Every answer returns:** `answer`, `sources` (list), and `chunks` (raw retrieved text). No exceptions.
- **Python:** 3.12. **Node:** 24. Both confirmed present.

---

## File Structure

```
backend/
  config.py         # settings (paths, model names, k) from env with defaults
  chunker.py        # chunk_markdown() -> list[Chunk]
  store.py          # VectorStore: add/query/count/reset over Chroma
  ingest.py         # ingest_corpus(): corpus dir -> chunks -> store
  session.py        # SessionStore: multi-turn history
  llm.py            # generate(): Ollama wrapper (isolated for test mocking)
  rag.py            # build_prompt(), answer(): the RAG pipeline
  main.py           # FastAPI app + endpoints
  requirements.txt
  tests/
    test_chunker.py
    test_store.py
    test_ingest.py
    test_session.py
    test_rag.py
    test_api.py
  eval/
    questions.json  # fixed eval set
    run_eval.py     # runs questions through /chat, prints pass/fail
corpus/
  about.md          # AM mock: company/about/sustainability
  products.md       # AM mock: fabrics/products FAQ
  support.md        # AM mock: returns, lead times, MOQ, contact
frontend/
  (Vite React app — scaffolded in Task 11)
```

---

### Task 1: Backend scaffolding, deps, pytest

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/config.py`
- Create: `backend/tests/__init__.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config` module with attributes `CORPUS_DIR: Path`, `CHROMA_DIR: Path`, `COLLECTION: str`, `LLM_MODEL: str`, `TOP_K: int`, `MAX_HISTORY_TURNS: int`.

- [ ] **Step 1: Write requirements.txt**

Create `backend/requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
chromadb==0.5.5
ollama==0.3.3
pytest==8.3.2
httpx==0.27.2
```

- [ ] **Step 2: Create venv and install**

Run (from `backend/`):
```
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```
Expected: installs complete without error. (Chroma pulls onnxruntime + the MiniLM model on first embed, not now.)

- [ ] **Step 3: Write the failing test**

Create `backend/tests/__init__.py` (empty).
Create `backend/tests/test_config.py`:
```python
from pathlib import Path
import config

def test_config_defaults():
    assert config.COLLECTION == "am_support"
    assert config.LLM_MODEL == "llama3.2:3b"
    assert config.TOP_K == 4
    assert config.MAX_HISTORY_TURNS == 6
    assert isinstance(config.CORPUS_DIR, Path)
    assert isinstance(config.CHROMA_DIR, Path)
```

- [ ] **Step 4: Run test to verify it fails**

Run (from `backend/`): `.venv\Scripts\python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 5: Write config.py**

Create `backend/config.py`:
```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/requirements.txt backend/config.py backend/tests
git commit -m "chore: backend scaffolding, deps, config"
```

---

### Task 2: Author the AM mock corpus

**Files:**
- Create: `corpus/about.md`, `corpus/products.md`, `corpus/support.md`
- Test: `backend/tests/test_corpus.py`

**Interfaces:**
- Consumes: `config.CORPUS_DIR`.
- Produces: three non-empty markdown files with `##` sections. No code interface.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_corpus.py`:
```python
import config

def test_corpus_files_present_and_nonempty():
    files = list(config.CORPUS_DIR.glob("*.md"))
    assert len(files) >= 3
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert len(text) > 200
        assert "##" in text  # has sections
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_corpus.py -v`
Expected: FAIL (fewer than 3 files / files missing).

- [ ] **Step 3: Write corpus/about.md**

Create `corpus/about.md`:
```markdown
# Artistic Milliners — About

## Who We Are
Artistic Milliners is a vertically integrated denim manufacturer founded in 1949,
headquartered in Karachi, Pakistan. We produce denim fabric and finished garments
for global apparel brands, covering spinning, weaving, dyeing, and garment finishing
in-house.

## Sustainability
Our sustainability program focuses on water recycling, renewable energy, and reduced
chemical usage. We operate one of the region's largest solar power installations and
recycle a majority of process water through in-house treatment plants. We publish an
annual sustainability report covering water, energy, and waste metrics.

## Certifications
We hold certifications including ISO 9001 for quality management and OEKO-TEX Standard
100 for textile safety. Specific certification scope is available on request.
```

- [ ] **Step 4: Write corpus/products.md**

Create `corpus/products.md`:
```markdown
# Products & Fabrics — FAQ

## What products do you offer?
We offer denim fabric by the roll and finished denim garments including jeans, jackets,
and shirts. Fabric weights range from light 6 oz chambray to heavy 14 oz rigid denim.

## What is the minimum fabric weight available?
The lightest fabric we produce is approximately 6 oz per square yard (chambray). The
heaviest standard offering is 14 oz rigid denim.

## Do you offer stretch denim?
Yes. We produce stretch and super-stretch denim using elastane blends, with recovery
options suited to skinny and jegging fits.

## Can you match a custom shade?
Yes, custom shade matching is available for bulk orders. A physical or digital reference
is required, and a lab dip is produced for approval before bulk dyeing.
```

- [ ] **Step 5: Write corpus/support.md**

Create `corpus/support.md`:
```markdown
# Support & Ordering Policy

## What is the minimum order quantity (MOQ)?
The standard MOQ for custom fabric orders is 1,000 meters per shade. Sample yardage is
available in smaller quantities for development.

## What are typical lead times?
Standard bulk fabric lead time is 30 to 45 days after order confirmation and lab dip
approval. Garment orders typically add 15 to 20 days for cut-make-trim.

## What is the returns policy?
Bulk fabric orders are made to order and are non-returnable once lab dip is approved,
except in the case of a verified manufacturing defect. Defect claims must be raised
within 14 days of delivery with supporting photographs.

## How do I contact the sales team?
Email the trade and merchandising office to start an order or request samples. Include
your target fabric weight, shade reference, and required quantity.
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_corpus.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add corpus backend/tests/test_corpus.py
git commit -m "content: add Artistic Milliners mock knowledge base"
```

---

### Task 3: Markdown chunker

**Files:**
- Create: `backend/chunker.py`
- Test: `backend/tests/test_chunker.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Chunk` = `dict` with keys `id: str`, `text: str`, `source_file: str`, `section_title: str`, `chunk_index: int`.
  - `def chunk_markdown(text: str, source_file: str, max_chars: int = 700, overlap: int = 100) -> list[Chunk]`
  - Splits on `##`/`#` headings into sections; sections longer than `max_chars` are further split into overlapping windows. `id` is `f"{source_file}::{chunk_index}"`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_chunker.py`:
```python
from chunker import chunk_markdown

SAMPLE = """# Title

## Section A
Short section A body.

## Section B
""" + ("word " * 300)  # long body forces a window split

def test_splits_by_section():
    chunks = chunk_markdown(SAMPLE, "doc.md", max_chars=700, overlap=100)
    titles = {c["section_title"] for c in chunks}
    assert "Section A" in titles
    assert "Section B" in titles

def test_metadata_and_ids_unique():
    chunks = chunk_markdown(SAMPLE, "doc.md")
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for i, c in enumerate(chunks):
        assert c["source_file"] == "doc.md"
        assert c["chunk_index"] == i
        assert c["text"].strip() != ""

def test_long_section_is_windowed():
    chunks = chunk_markdown(SAMPLE, "doc.md", max_chars=700, overlap=100)
    b_chunks = [c for c in chunks if c["section_title"] == "Section B"]
    assert len(b_chunks) >= 2  # long body split into multiple windows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chunker'`.

- [ ] **Step 3: Write chunker.py**

Create `backend/chunker.py`:
```python
import re

def _split_sections(text: str):
    """Yield (title, body) pairs split on markdown headings (# or ##)."""
    lines = text.splitlines()
    title = "Untitled"
    buf = []
    for line in lines:
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if buf:
                yield title, "\n".join(buf).strip()
                buf = []
            title = m.group(1).strip()
        else:
            buf.append(line)
    if buf:
        yield title, "\n".join(buf).strip()

def _window(body: str, max_chars: int, overlap: int):
    if len(body) <= max_chars:
        return [body]
    windows = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(body):
        windows.append(body[start:start + max_chars])
        start += step
    return windows

def chunk_markdown(text: str, source_file: str, max_chars: int = 700, overlap: int = 100):
    chunks = []
    idx = 0
    for title, body in _split_sections(text):
        if not body:
            continue
        for window in _window(body, max_chars, overlap):
            window = window.strip()
            if not window:
                continue
            chunks.append({
                "id": f"{source_file}::{idx}",
                "text": window,
                "source_file": source_file,
                "section_title": title,
                "chunk_index": idx,
            })
            idx += 1
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_chunker.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/chunker.py backend/tests/test_chunker.py
git commit -m "feat: markdown section chunker with windowing"
```

---

### Task 4: Vector store (Chroma + ONNX MiniLM)

**Files:**
- Create: `backend/store.py`
- Test: `backend/tests/test_store.py`

**Interfaces:**
- Consumes: `config.CHROMA_DIR`, `config.COLLECTION`, `Chunk` shape from Task 3.
- Produces:
  - `class VectorStore(persist_dir: Path, collection: str)`
  - `.add(chunks: list[Chunk]) -> None`
  - `.query(text: str, k: int) -> list[dict]` — each result `{text, source_file, section_title, score}` (score = distance, lower = closer).
  - `.count() -> int`
  - `.reset() -> None`
  - Uses Chroma's `DefaultEmbeddingFunction` (ONNX MiniLM-L6-v2).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_store.py`:
```python
from store import VectorStore

CHUNKS = [
    {"id": "d::0", "text": "The minimum order quantity is 1000 meters.",
     "source_file": "support.md", "section_title": "MOQ", "chunk_index": 0},
    {"id": "d::1", "text": "We produce stretch denim with elastane blends.",
     "source_file": "products.md", "section_title": "Stretch", "chunk_index": 1},
]

def test_add_and_count(tmp_path):
    s = VectorStore(tmp_path / "chroma", "test_col")
    s.reset()
    s.add(CHUNKS)
    assert s.count() == 2

def test_query_returns_relevant_chunk(tmp_path):
    s = VectorStore(tmp_path / "chroma", "test_col")
    s.reset()
    s.add(CHUNKS)
    results = s.query("what is the minimum order quantity", k=1)
    assert len(results) == 1
    assert results[0]["source_file"] == "support.md"
    assert "text" in results[0] and "score" in results[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'store'`.

- [ ] **Step 3: Write store.py**

Create `backend/store.py`:
```python
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, persist_dir, collection: str):
        self._client = chromadb.PersistentClient(path=str(Path(persist_dir)))
        self._embed = embedding_functions.DefaultEmbeddingFunction()
        self._name = collection
        self._col = self._client.get_or_create_collection(
            name=collection, embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks):
        if not chunks:
            return
        self._col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {"source_file": c["source_file"], "section_title": c["section_title"]}
                for c in chunks
            ],
        )

    def query(self, text: str, k: int):
        res = self._col.query(query_texts=[text], n_results=k)
        out = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({
                "text": doc,
                "source_file": meta["source_file"],
                "section_title": meta["section_title"],
                "score": float(dist),
            })
        return out

    def count(self):
        return self._col.count()

    def reset(self):
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name, embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_store.py -v`
Expected: PASS. (First run downloads the MiniLM ONNX model — allow time.)

- [ ] **Step 5: Commit**

```bash
git add backend/store.py backend/tests/test_store.py
git commit -m "feat: Chroma vector store with ONNX MiniLM embeddings"
```

---

### Task 5: Ingest pipeline

**Files:**
- Create: `backend/ingest.py`
- Test: `backend/tests/test_ingest.py`

**Interfaces:**
- Consumes: `chunk_markdown` (Task 3), `VectorStore` (Task 4), `config.CORPUS_DIR`.
- Produces: `def ingest_corpus(corpus_dir, store) -> int` — reads all `*.md`, chunks, resets store, adds, returns total chunk count.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_ingest.py`:
```python
from ingest import ingest_corpus
from store import VectorStore

def test_ingest_populates_store(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\n## S1\nMinimum order quantity is 1000 meters.\n",
                                 encoding="utf-8")
    store = VectorStore(tmp_path / "chroma", "ingest_test")
    n = ingest_corpus(corpus, store)
    assert n >= 1
    assert store.count() == n
    hits = store.query("MOQ minimum order", k=1)
    assert hits[0]["source_file"] == "a.md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest'`.

- [ ] **Step 3: Write ingest.py**

Create `backend/ingest.py`:
```python
from pathlib import Path
from chunker import chunk_markdown

def ingest_corpus(corpus_dir, store) -> int:
    corpus_dir = Path(corpus_dir)
    store.reset()
    total = []
    for md in sorted(corpus_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        total.extend(chunk_markdown(text, md.name))
    store.add(total)
    return len(total)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_ingest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/ingest.py backend/tests/test_ingest.py
git commit -m "feat: corpus ingest pipeline"
```

---

### Task 6: Session store (multi-turn memory)

**Files:**
- Create: `backend/session.py`
- Test: `backend/tests/test_session.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class SessionStore()`
  - `.append(session_id: str, role: str, content: str) -> None`
  - `.history(session_id: str, max_turns: int) -> list[dict]` — returns last `max_turns` messages as `{"role", "content"}`, oldest first.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_session.py`:
```python
from session import SessionStore

def test_append_and_history_order():
    s = SessionStore()
    s.append("sess1", "user", "hi")
    s.append("sess1", "assistant", "hello")
    hist = s.history("sess1", max_turns=6)
    assert hist == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

def test_history_truncates_to_max_turns():
    s = SessionStore()
    for i in range(10):
        s.append("s", "user", f"m{i}")
    hist = s.history("s", max_turns=3)
    assert [m["content"] for m in hist] == ["m7", "m8", "m9"]

def test_sessions_isolated():
    s = SessionStore()
    s.append("a", "user", "x")
    assert s.history("b", max_turns=6) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'session'`.

- [ ] **Step 3: Write session.py**

Create `backend/session.py`:
```python
from collections import defaultdict

class SessionStore:
    def __init__(self):
        self._data = defaultdict(list)

    def append(self, session_id: str, role: str, content: str) -> None:
        self._data[session_id].append({"role": role, "content": content})

    def history(self, session_id: str, max_turns: int):
        return list(self._data.get(session_id, []))[-max_turns:]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_session.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/session.py backend/tests/test_session.py
git commit -m "feat: in-memory multi-turn session store"
```

---

### Task 7: LLM client (Ollama wrapper)

**Files:**
- Create: `backend/llm.py`
- Test: `backend/tests/test_llm.py`

**Interfaces:**
- Consumes: `config.LLM_MODEL`.
- Produces: `def generate(messages: list[dict], model: str | None = None) -> str` — calls `ollama.chat`, returns the assistant text. Isolated so tests monkeypatch `ollama.chat`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_llm.py`:
```python
import llm

def test_generate_returns_text(monkeypatch):
    def fake_chat(model, messages):
        assert model  # model passed through
        return {"message": {"content": "hello from llm"}}
    monkeypatch.setattr(llm.ollama, "chat", fake_chat)
    out = llm.generate([{"role": "user", "content": "hi"}], model="test-model")
    assert out == "hello from llm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm'`.

- [ ] **Step 3: Write llm.py**

Create `backend/llm.py`:
```python
import ollama
import config

def generate(messages, model: str | None = None) -> str:
    model = model or config.LLM_MODEL
    resp = ollama.chat(model=model, messages=messages)
    return resp["message"]["content"].strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Manual smoke (requires Ollama running)**

Run: `ollama pull llama3.2:3b` then start Ollama, then:
```
.venv\Scripts\python -c "import llm; print(llm.generate([{'role':'user','content':'say hi in 3 words'}]))"
```
Expected: a short reply. (If too slow/RAM-tight: `ollama pull llama3.2:1b` and set `LLM_MODEL=llama3.2:1b`.)

- [ ] **Step 6: Commit**

```bash
git add backend/llm.py backend/tests/test_llm.py
git commit -m "feat: Ollama LLM client wrapper"
```

---

### Task 8: RAG pipeline (prompt build + answer)

**Files:**
- Create: `backend/rag.py`
- Test: `backend/tests/test_rag.py`

**Interfaces:**
- Consumes: `VectorStore.query` (Task 4), `SessionStore` (Task 6), `llm.generate` (Task 7).
- Produces:
  - `SYSTEM_PROMPT: str` — the anti-hallucination instruction.
  - `def build_prompt(query, chunks, history) -> list[dict]` — returns Ollama `messages` list: system + history + a user message embedding the retrieved context and the query.
  - `def answer(query, session_id, store, sessions, generate_fn=llm.generate) -> dict` — returns `{"answer", "sources", "chunks"}`; appends user query and assistant answer to session. `generate_fn` is injectable for tests.
  - `sources` = deduped list of `{"source_file", "section_title"}` from retrieved chunks.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_rag.py`:
```python
import rag
from session import SessionStore

class FakeStore:
    def query(self, text, k):
        return [
            {"text": "MOQ is 1000 meters.", "source_file": "support.md",
             "section_title": "MOQ", "score": 0.1},
        ]

def fake_generate(messages, model=None):
    # echo that context was injected
    assert any("MOQ is 1000 meters." in m["content"] for m in messages)
    return "The minimum order quantity is 1000 meters."

def test_answer_returns_answer_sources_chunks():
    sessions = SessionStore()
    out = rag.answer("what is MOQ", "s1", FakeStore(), sessions,
                     generate_fn=fake_generate)
    assert "1000 meters" in out["answer"]
    assert out["sources"] == [{"source_file": "support.md", "section_title": "MOQ"}]
    assert len(out["chunks"]) == 1

def test_answer_records_history():
    sessions = SessionStore()
    rag.answer("what is MOQ", "s1", FakeStore(), sessions, generate_fn=fake_generate)
    hist = sessions.history("s1", max_turns=6)
    assert hist[0]["role"] == "user"
    assert hist[1]["role"] == "assistant"

def test_system_prompt_forbids_hallucination():
    assert "don't know" in rag.SYSTEM_PROMPT.lower() or \
           "do not know" in rag.SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag'`.

- [ ] **Step 3: Write rag.py**

Create `backend/rag.py`:
```python
import config
import llm

SYSTEM_PROMPT = (
    "You are the Artistic Milliners support assistant. "
    "Answer ONLY using the provided context. "
    "If the answer is not in the context, say you don't know and suggest contacting "
    "the sales team. Be concise. Do not invent facts, numbers, or policies."
)

def build_prompt(query, chunks, history):
    context = "\n\n".join(
        f"[{c['source_file']} - {c['section_title']}]\n{c['text']}" for c in chunks
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}",
    })
    return messages

def _dedupe_sources(chunks):
    seen = []
    for c in chunks:
        key = {"source_file": c["source_file"], "section_title": c["section_title"]}
        if key not in seen:
            seen.append(key)
    return seen

def answer(query, session_id, store, sessions, generate_fn=llm.generate):
    chunks = store.query(query, k=config.TOP_K)
    history = sessions.history(session_id, max_turns=config.MAX_HISTORY_TURNS)
    messages = build_prompt(query, chunks, history)
    text = generate_fn(messages)
    sessions.append(session_id, "user", query)
    sessions.append(session_id, "assistant", text)
    return {
        "answer": text,
        "sources": _dedupe_sources(chunks),
        "chunks": chunks,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_rag.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/rag.py backend/tests/test_rag.py
git commit -m "feat: RAG pipeline with anti-hallucination prompt and citations"
```

---

### Task 9: FastAPI endpoints

**Files:**
- Create: `backend/main.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `ingest_corpus` (Task 5), `VectorStore` (Task 4), `SessionStore` (Task 6), `rag.answer` (Task 8), `config`.
- Produces FastAPI app `app` with:
  - `GET /health` → `{"status": "ok", "chunks": int}`
  - `POST /ingest` → `{"ingested": int}`
  - `POST /chat` body `{"session_id": str, "message": str}` → `{"answer", "sources", "chunks"}`
  - CORS enabled for local frontend.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_api.py`:
```python
from fastapi.testclient import TestClient
import main

def test_health():
    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_chat_uses_rag(monkeypatch):
    def fake_answer(query, session_id, store, sessions, generate_fn=None):
        return {"answer": "stub", "sources": [], "chunks": []}
    monkeypatch.setattr(main.rag, "answer", fake_answer)
    client = TestClient(main.app)
    r = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert r.status_code == 200
    assert r.json()["answer"] == "stub"

def test_chat_requires_message():
    client = TestClient(main.app)
    r = client.post("/chat", json={"session_id": "s1"})
    assert r.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`.

- [ ] **Step 3: Write main.py**

Create `backend/main.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_api.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full backend smoke (Ollama running)**

Run (from `backend/`):
```
.venv\Scripts\uvicorn main:app --reload
```
Then in another shell:
```
curl -X POST http://127.0.0.1:8000/ingest
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"session_id\":\"s1\",\"message\":\"what is the MOQ?\"}"
```
Expected: ingest returns a count; chat returns a grounded answer citing `support.md`.

- [ ] **Step 6: Run full backend test suite**

Run: `.venv\Scripts\python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/main.py backend/tests/test_api.py
git commit -m "feat: FastAPI endpoints (health, ingest, chat) with CORS"
```

---

### Task 10: Evaluation set

**Files:**
- Create: `backend/eval/questions.json`
- Create: `backend/eval/run_eval.py`

**Interfaces:**
- Consumes: running backend at `http://127.0.0.1:8000`.
- Produces: `run_eval.py` posts each question to `/chat` and checks the answer contains an expected substring (or, for out-of-corpus, an "I don't know"-style phrase). Prints pass/fail summary.

- [ ] **Step 1: Write questions.json**

Create `backend/eval/questions.json`:
```json
[
  {"q": "What is the minimum order quantity?", "expect": "1000"},
  {"q": "What are your typical lead times?", "expect": "30"},
  {"q": "Do you offer stretch denim?", "expect": "stretch"},
  {"q": "What is the lightest fabric weight?", "expect": "6"},
  {"q": "Can I return a bulk order I changed my mind about?", "expect": "non-returnable"},
  {"q": "Who is the CEO of Artistic Milliners?", "expect": "__unknown__"}
]
```
(`__unknown__` marks a question the corpus cannot answer — expect a don't-know response.)

- [ ] **Step 2: Write run_eval.py**

Create `backend/eval/run_eval.py`:
```python
import json
import sys
from pathlib import Path
import httpx

API = "http://127.0.0.1:8000/chat"
UNKNOWN_MARKERS = ["don't know", "do not know", "not sure", "contact"]

def main():
    questions = json.loads((Path(__file__).parent / "questions.json").read_text())
    passed = 0
    with httpx.Client(timeout=120) as client:
        for i, item in enumerate(questions):
            r = client.post(API, json={"session_id": "eval", "message": item["q"]})
            ans = r.json()["answer"].lower()
            if item["expect"] == "__unknown__":
                ok = any(m in ans for m in UNKNOWN_MARKERS)
            else:
                ok = item["expect"].lower() in ans
            passed += ok
            print(f"[{'PASS' if ok else 'FAIL'}] {item['q']}\n    -> {ans[:120]}")
    print(f"\n{passed}/{len(questions)} passed")
    sys.exit(0 if passed == len(questions) else 1)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the eval (Ollama + backend running)**

Run (from `backend/`): `.venv\Scripts\python eval/run_eval.py`
Expected: prints per-question results. Tune `TOP_K`, chunk size, or model (`3b`→`1b`) if failures. Record the honest score for the README.

- [ ] **Step 4: Commit**

```bash
git add backend/eval
git commit -m "test: evaluation question set and runner"
```

---

### Task 11: Frontend scaffold + API client

**Files:**
- Create: Vite React app in `frontend/`
- Create: `frontend/src/api.js`
- Create: `frontend/.env` (gitignored) with `VITE_API_URL=http://127.0.0.1:8000`

**Interfaces:**
- Consumes: backend `/chat`.
- Produces: `sendMessage(sessionId, message) -> Promise<{answer, sources, chunks}>`.

- [ ] **Step 1: Scaffold Vite React**

Run (from `frontend/` parent, i.e. project root):
```
npm create vite@latest frontend -- --template react
cd frontend
npm install
```
Expected: Vite React app created, deps installed.

- [ ] **Step 2: Write the API client**

Create `frontend/src/api.js`:
```javascript
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function sendMessage(sessionId, message) {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}
```

- [ ] **Step 3: Create env file**

Create `frontend/.env`:
```
VITE_API_URL=http://127.0.0.1:8000
```

- [ ] **Step 4: Verify dev server boots**

Run (from `frontend/`): `npm run dev`
Expected: Vite serves on `http://localhost:5173` without error.

- [ ] **Step 5: Commit**

```bash
git add frontend package*.json frontend/src/api.js
git commit -m "feat: frontend scaffold and API client"
```

---

### Task 12: Chat UI — mobile-first, polished

**Files:**
- Modify: `frontend/src/App.jsx`
- Create: `frontend/src/App.css`
- Modify: `frontend/src/index.css`

**Interfaces:**
- Consumes: `sendMessage` (Task 11).
- Produces: a mobile-first chat screen: message list, input bar, per-answer collapsible "Sources" showing `sources` + retrieved `chunks`. Loading + error states.

> **Design note:** Before building, invoke the `frontend-design` skill for the visual direction (typography, spacing, restraint — Stripe/Shopify feel). The code below is the functional baseline; refine styling under that skill's guidance. Mobile-first: base styles target ~375px; add `min-width` media queries for larger screens.

- [ ] **Step 1: Write App.jsx**

Create/replace `frontend/src/App.jsx`:
```jsx
import { useState, useRef, useEffect } from "react";
import { sendMessage } from "./api";
import "./App.css";

const SESSION_ID = crypto.randomUUID();

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    setError(null);
    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    try {
      const res = await sendMessage(SESSION_ID, text);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.answer, sources: res.sources, chunks: res.chunks },
      ]);
    } catch (err) {
      setError("Could not reach the assistant. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Artistic Milliners</h1>
        <p>Support Assistant</p>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty">Ask about MOQ, lead times, fabrics, or returns.</div>
        )}
        {messages.map((m, i) => (
          <Message key={i} m={m} />
        ))}
        {loading && <div className="msg assistant"><TypingDots /></div>}
        {error && <div className="error">{error}</div>}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={handleSend}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question…"
          aria-label="Message"
        />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </div>
  );
}

function Message({ m }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`msg ${m.role}`}>
      <div className="bubble">{m.content}</div>
      {m.role === "assistant" && m.sources?.length > 0 && (
        <div className="sources">
          <button className="sources-toggle" onClick={() => setOpen((o) => !o)}>
            {open ? "Hide" : "Show"} sources ({m.sources.length})
          </button>
          {open && (
            <div className="sources-body">
              {m.sources.map((s, i) => (
                <span key={i} className="source-tag">
                  {s.source_file} · {s.section_title}
                </span>
              ))}
              <details className="chunks">
                <summary>Retrieved passages</summary>
                {m.chunks.map((c, i) => (
                  <p key={i} className="chunk">{c.text}</p>
                ))}
              </details>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TypingDots() {
  return <span className="dots"><span>.</span><span>.</span><span>.</span></span>;
}
```

- [ ] **Step 2: Write App.css (mobile-first baseline)**

Create/replace `frontend/src/App.css`:
```css
.app {
  display: flex; flex-direction: column;
  height: 100dvh; max-width: 720px; margin: 0 auto;
  background: var(--bg, #fafafa);
}
.header { padding: 16px 20px; border-bottom: 1px solid #ececf1; background: #fff; }
.header h1 { margin: 0; font-size: 18px; font-weight: 650; letter-spacing: -0.01em; }
.header p { margin: 2px 0 0; font-size: 13px; color: #6b7280; }

.chat { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.empty { margin: auto; color: #9ca3af; font-size: 14px; text-align: center; }

.msg { display: flex; flex-direction: column; max-width: 85%; }
.msg.user { align-self: flex-end; align-items: flex-end; }
.msg.assistant { align-self: flex-start; }
.bubble {
  padding: 10px 14px; border-radius: 16px; font-size: 15px; line-height: 1.45;
  box-shadow: 0 1px 2px rgba(16,24,40,0.05);
}
.msg.user .bubble { background: #635bff; color: #fff; border-bottom-right-radius: 4px; }
.msg.assistant .bubble { background: #fff; color: #111827; border: 1px solid #ececf1;
  border-bottom-left-radius: 4px; }

.sources { margin-top: 6px; }
.sources-toggle { background: none; border: none; color: #635bff; font-size: 12px;
  cursor: pointer; padding: 2px 0; }
.sources-body { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.source-tag { background: #f0f0ff; color: #4338ca; font-size: 11px; padding: 3px 8px;
  border-radius: 999px; }
.chunks { flex-basis: 100%; margin-top: 6px; font-size: 12px; color: #6b7280; }
.chunk { background: #f6f6f9; padding: 8px 10px; border-radius: 8px; margin: 4px 0; }

.composer { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid #ececf1;
  background: #fff; }
.composer input { flex: 1; padding: 12px 14px; border: 1px solid #e5e7eb;
  border-radius: 12px; font-size: 15px; outline: none; }
.composer input:focus { border-color: #635bff; box-shadow: 0 0 0 3px rgba(99,91,255,0.12); }
.composer button { padding: 0 18px; border: none; border-radius: 12px; background: #635bff;
  color: #fff; font-weight: 600; cursor: pointer; }
.composer button:disabled { opacity: 0.5; cursor: default; }

.error { color: #dc2626; font-size: 13px; }
.dots span { animation: blink 1.4s infinite both; }
.dots span:nth-child(2) { animation-delay: 0.2s; }
.dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }

@media (min-width: 640px) {
  .header h1 { font-size: 20px; }
  .bubble { font-size: 16px; }
}
```

- [ ] **Step 3: Reset base styles**

Replace `frontend/src/index.css`:
```css
* { box-sizing: border-box; }
html, body, #root { margin: 0; padding: 0; height: 100%; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #fafafa; color: #111827; -webkit-font-smoothing: antialiased; }
```

- [ ] **Step 4: Manual verify end-to-end**

Backend running (`uvicorn main:app`), Ollama running, then `npm run dev`. Open on mobile viewport (DevTools device toolbar). Send "What is the MOQ?" → grounded answer + working sources toggle. Check layout at 375px and desktop.

- [ ] **Step 5: Commit**

```bash
git add frontend/src
git commit -m "feat: mobile-first polished chat UI with sources panel"
```

---

### Task 13: README + architecture diagram

**Files:**
- Modify: `README.md` (project root)

**Interfaces:** none.

- [ ] **Step 1: Write README.md**

Create root `README.md` with: one-line pitch; "built for the Artistic Milliners job posting" note; architecture diagram (reuse the mermaid from `docs/design.md`); stack table; **what's library vs custom** (custom: chunker, RAG pipeline, prompt, UI; library: Chroma, MiniLM ONNX, Ollama, FastAPI); setup steps (venv, `pip install`, `ollama pull llama3.2:3b`, `/ingest`, `npm run dev`); honest eval score from Task 10; note on local small-model tradeoffs (CPU latency); screenshots/GIF placeholder; stretch goals (WhatsApp, deploy, PWA).

- [ ] **Step 2: Add screenshots**

Capture 2–3 screenshots (mobile chat, sources expanded) into `docs/screenshots/` and embed in README.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/screenshots
git commit -m "docs: README with architecture, honest metrics, screenshots"
```

---

## Self-Review

**Spec coverage:** All design.md sections mapped — corpus (T2), chunk/embed/Chroma (T3–T5), RAG + multi-turn + citations (T6–T8), endpoints (T9), retrieved-chunks panel (T12), mobile-first polished UI (T11–T12), README + diagram + honest metrics (T13), eval (T10). Stretch (WhatsApp/deploy/PWA) correctly deferred. ✓

**Placeholder scan:** No TBD/TODO in code steps; all steps carry real code or exact commands. README task (T13) is prose-descriptive by nature (content authored at build time) but lists exact required sections. ✓

**Type consistency:** `Chunk` dict shape (`id/text/source_file/section_title/chunk_index`) consistent T3→T5. `VectorStore.query` returns `{text, source_file, section_title, score}` consumed identically in T8. `answer()` returns `{answer, sources, chunks}` consumed in T9 and T12. `sources` = `{source_file, section_title}` consistent T8→T12. ✓

**Embedding note:** Plan uses Chroma's built-in ONNX MiniLM instead of the separately-installed `fastembed` package (same model, fewer deps, lower RAM). Flagged in Global Constraints; update spec/CLAUDE.md if desired.
