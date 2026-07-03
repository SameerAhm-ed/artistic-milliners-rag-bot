from chunker import chunk_markdown

SAMPLE = """# Title

## Section A
Short section A body.

## Section B
""" + ("word " * 300)  # long body forces a window split

def test_splits_by_section():
    chunks = chunk_markdown(SAMPLE, "doc.md", max_chars=700, overlap=100)
    titles = {c["section_title"] for c in chunks}
    assert "Section A" in titles
    assert "Section B" in titles

def test_metadata_and_ids_unique():
    chunks = chunk_markdown(SAMPLE, "doc.md")
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for i, c in enumerate(chunks):
        assert c["source_file"] == "doc.md"
        assert c["chunk_index"] == i
        assert c["text"].strip() != ""

def test_long_section_is_windowed():
    chunks = chunk_markdown(SAMPLE, "doc.md", max_chars=700, overlap=100)
    b_chunks = [c for c in chunks if c["section_title"] == "Section B"]
    assert len(b_chunks) >= 2  # long body split into multiple windows
