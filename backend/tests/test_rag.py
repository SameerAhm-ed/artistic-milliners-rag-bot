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
