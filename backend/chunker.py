import re

def _split_sections(text: str):
    """Yield (title, body) pairs split on markdown headings (# or ##)."""
    lines = text.splitlines()
    title = "Untitled"
    buf = []
    for line in lines:
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if buf:
                yield title, "\n".join(buf).strip()
                buf = []
            title = m.group(1).strip()
        else:
            buf.append(line)
    if buf:
        yield title, "\n".join(buf).strip()

def _window(body: str, max_chars: int, overlap: int):
    if len(body) <= max_chars:
        return [body]
    windows = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(body):
        windows.append(body[start:start + max_chars])
        start += step
    return windows

def chunk_markdown(text: str, source_file: str, max_chars: int = 700, overlap: int = 100):
    chunks = []
    idx = 0
    for title, body in _split_sections(text):
        if not body:
            continue
        for window in _window(body, max_chars, overlap):
            window = window.strip()
            if not window:
                continue
            chunks.append({
                "id": f"{source_file}::{idx}",
                "text": window,
                "source_file": source_file,
                "section_title": title,
                "chunk_index": idx,
            })
            idx += 1
    return chunks
