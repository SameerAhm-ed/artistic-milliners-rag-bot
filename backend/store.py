from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, persist_dir, collection: str):
        self._client = chromadb.PersistentClient(path=str(Path(persist_dir)))
        self._embed = embedding_functions.DefaultEmbeddingFunction()
        self._name = collection
        self._col = self._client.get_or_create_collection(
            name=collection, embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks):
        if not chunks:
            return
        self._col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[
                {"source_file": c["source_file"], "section_title": c["section_title"]}
                for c in chunks
            ],
        )

    def query(self, text: str, k: int):
        res = self._col.query(query_texts=[text], n_results=k)
        out = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({
                "text": doc,
                "source_file": meta["source_file"],
                "section_title": meta["section_title"],
                "score": float(dist),
            })
        return out

    def count(self):
        return self._col.count()

    def reset(self):
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name, embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )
