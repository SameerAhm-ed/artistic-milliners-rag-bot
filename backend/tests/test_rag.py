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
    prompt = rag.SYSTEM_PROMPT.lower()
    assert "don't know" in prompt or "do not know" in prompt
    # Locks in the context-only constraint: if someone silently loosens the
    # "answer ONLY using the provided context" clause, this must fail.
    assert "only" in prompt
    assert "context" in prompt
    assert "answer only using the provided context" in prompt


class DupeStore:
    def query(self, text, k):
        return [
            {"text": "MOQ is 1000 meters.", "source_file": "support.md",
             "section_title": "MOQ", "score": 0.1},
            {"text": "Minimum order is 1000m for cotton.", "source_file": "support.md",
             "section_title": "MOQ", "score": 0.2},
            {"text": "Lead time is 4 weeks.", "source_file": "support.md",
             "section_title": "Lead Time", "score": 0.3},
        ]

def test_dedupe_sources_collapses_duplicates():
    sessions = SessionStore()
    out = rag.answer("what is MOQ", "s1", DupeStore(), sessions,
                     generate_fn=fake_generate)
    sources = out["sources"]
    assert len(sources) == 2
    assert sources == [
        {"source_file": "support.md", "section_title": "MOQ"},
        {"source_file": "support.md", "section_title": "Lead Time"},
    ]
    # no duplicate entries
    assert len(sources) == len({(s["source_file"], s["section_title"]) for s in sources})
