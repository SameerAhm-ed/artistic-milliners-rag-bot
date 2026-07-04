import llm

def test_generate_returns_text(monkeypatch):
    def fake_chat(model, messages):
        assert model  # model passed through
        return {"message": {"content": "hello from llm"}}
    monkeypatch.setattr(llm.ollama, "chat", fake_chat)
    out = llm.generate([{"role": "user", "content": "hi"}], model="test-model")
    assert out == "hello from llm"
