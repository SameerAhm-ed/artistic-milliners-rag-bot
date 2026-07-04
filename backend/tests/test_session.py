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
