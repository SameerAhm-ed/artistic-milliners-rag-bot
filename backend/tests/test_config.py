from pathlib import Path
import config

def test_config_defaults():
    assert config.COLLECTION == "am_support"
    assert config.LLM_MODEL == "llama3.2:3b"
    assert config.TOP_K == 4
    assert config.MAX_HISTORY_TURNS == 6
    assert isinstance(config.CORPUS_DIR, Path)
    assert isinstance(config.CHROMA_DIR, Path)
