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
