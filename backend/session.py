from collections import defaultdict

class SessionStore:
    def __init__(self):
        self._data = defaultdict(list)

    def append(self, session_id: str, role: str, content: str) -> None:
        self._data[session_id].append({"role": role, "content": content})

    def history(self, session_id: str, max_turns: int):
        return list(self._data.get(session_id, []))[-max_turns:]
