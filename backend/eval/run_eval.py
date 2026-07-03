import json
import sys
from pathlib import Path
import httpx

API = "http://127.0.0.1:8000/chat"
UNKNOWN_MARKERS = ["don't know", "do not know", "not sure", "contact"]

def normalize(text: str) -> str:
    # lowercase and strip digit-group separators so "1,000" matches "1000"
    return text.lower().replace(",", "")

def main():
    questions = json.loads((Path(__file__).parent / "questions.json").read_text())
    passed = 0
    with httpx.Client(timeout=120) as client:
        for i, item in enumerate(questions):
            r = client.post(API, json={"session_id": "eval", "message": item["q"]})
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
    print(f"\n{passed}/{len(questions)} passed")
    sys.exit(0 if passed == len(questions) else 1)

if __name__ == "__main__":
    main()
