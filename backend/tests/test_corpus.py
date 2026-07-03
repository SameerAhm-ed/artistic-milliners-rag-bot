import config

def test_corpus_files_present_and_nonempty():
    files = list(config.CORPUS_DIR.glob("*.md"))
    assert len(files) >= 3
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert len(text) > 200
        assert "##" in text  # has sections
