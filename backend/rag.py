import config
import llm

SYSTEM_PROMPT = (
    "You are the Artistic Milliners support assistant. "
    "Answer ONLY using the provided context. "
    "If the answer is not in the context, say you don't know and suggest contacting "
    "the sales team. Be concise. Do not invent facts, numbers, or policies."
)

def build_prompt(query, chunks, history):
    context = "\n\n".join(
        f"[{c['source_file']} - {c['section_title']}]\n{c['text']}" for c in chunks
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}",
    })
    return messages

def _dedupe_sources(chunks):
    seen = []
    for c in chunks:
        key = {"source_file": c["source_file"], "section_title": c["section_title"]}
        if key not in seen:
            seen.append(key)
    return seen

def answer(query, session_id, store, sessions, generate_fn=llm.generate):
    chunks = store.query(query, k=config.TOP_K)
    history = sessions.history(session_id, max_turns=config.MAX_HISTORY_TURNS)
    messages = build_prompt(query, chunks, history)
    text = generate_fn(messages)
    sessions.append(session_id, "user", query)
    sessions.append(session_id, "assistant", text)
    return {
        "answer": text,
        "sources": _dedupe_sources(chunks),
        "chunks": chunks,
    }
