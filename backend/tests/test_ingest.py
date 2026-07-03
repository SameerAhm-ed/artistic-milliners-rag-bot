from ingest import ingest_corpus
from store import VectorStore

def test_ingest_populates_store(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\n## S1\nMinimum order quantity is 1000 meters.\n",
                                 encoding="utf-8")
    store = VectorStore(tmp_path / "chroma", "ingest_test")
    n = ingest_corpus(corpus, store)
    assert n >= 1
    assert store.count() == n
    hits = store.query("MOQ minimum order", k=1)
    assert hits[0]["source_file"] == "a.md"
