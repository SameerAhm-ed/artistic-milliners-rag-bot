import json
import sys
from pathlib import Path
import httpx

API = "http://127.0.0.1:8000/chat"
UNKNOWN_MARKERS = ["don't know", "do not know", "not sure"]

def normalize(text: str) -> str:
    # lowercase and strip digit-group separators so "1,000" matches "1000"
    return text.lower().replace(",", "")

def check_multiturn(client):
    """Prove context carry-over: a follow-up question in the SAME session must
    resolve a pronoun/reference against turn 1's answer. Uses a dedicated
    shared session_id distinct from the isolated per-question eval sessions."""
    session_id = "eval-multiturn"
    turn1_q = "Do you offer stretch denim?"
    turn2_q = "What fits is it suited for?"
    expected_tokens = ["skinny", "jegging"]

    client.post(API, json={"session_id": session_id, "message": turn1_q})
    r2 = client.post(API, json={"session_id": session_id, "message": turn2_q})
    ans2 = normalize(r2.json()["answer"])
    ok = any(normalize(t) in ans2 for t in expected_tokens)
    print(f"[{'PASS' if ok else 'FAIL'}] (multi-turn) {turn1_q!r} -> {turn2_q!r}\n    -> {ans2[:120]}")
    return ok

def main():
    questions = json.loads((Path(__file__).parent / "questions.json").read_text())
    passed = 0
    with httpx.Client(timeout=120) as client:
        for i, item in enumerate(questions):
            r = client.post(API, json={"session_id": f"eval-{i}", "message": item["q"]})
            ans = normalize(r.json()["answer"])
            expect = item["expect"]
            if expect == "__unknown__":
                ok = any(m in ans for m in UNKNOWN_MARKERS)
            else:
                # expect may be a single string or a list of acceptable phrasings
                # (any-of), tolerating LLM paraphrase while still asserting the fact
                accepted = expect if isinstance(expect, list) else [expect]
                ok = any(normalize(e) in ans for e in accepted)
            passed += ok
            print(f"[{'PASS' if ok else 'FAIL'}] {item['q']}\n    -> {ans[:120]}")

        multiturn_ok = check_multiturn(client)
        passed += multiturn_ok

    total = len(questions) + 1
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
