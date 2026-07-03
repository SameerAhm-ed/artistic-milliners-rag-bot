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
