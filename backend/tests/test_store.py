from store import VectorStore

CHUNKS = [
    {"id": "d::0", "text": "The minimum order quantity is 1000 meters.",
     "source_file": "support.md", "section_title": "MOQ", "chunk_index": 0},
    {"id": "d::1", "text": "We produce stretch denim with elastane blends.",
     "source_file": "products.md", "section_title": "Stretch", "chunk_index": 1},
]

def test_add_and_count(tmp_path):
    s = VectorStore(tmp_path / "chroma", "test_col")
    s.reset()
    s.add(CHUNKS)
    assert s.count() == 2

def test_query_returns_relevant_chunk(tmp_path):
    s = VectorStore(tmp_path / "chroma", "test_col")
    s.reset()
    s.add(CHUNKS)
    results = s.query("what is the minimum order quantity", k=1)
    assert len(results) == 1
    assert results[0]["source_file"] == "support.md"
    assert "text" in results[0] and "score" in results[0]
