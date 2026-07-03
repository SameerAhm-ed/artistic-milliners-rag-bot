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
