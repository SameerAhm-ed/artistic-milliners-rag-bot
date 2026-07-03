import ollama
import config

def generate(messages, model: str | None = None) -> str:
    model = model or config.LLM_MODEL
    resp = ollama.chat(model=model, messages=messages)
    return resp["message"]["content"].strip()
