import os, json, numpy as np
from typing import List, Tuple
from .bedrock_client import embed_texts_titan
from .deps import get_settings

settings = get_settings()

class Retriever:
    def __init__(self):
        self.backend = os.getenv("VECTOR_DB", "faiss")
        self.index_dir = settings["index_dir"]
        self._load()

    def _load(self):
        if self.backend == "faiss":
            import faiss
            import pathlib
            self.dim = 1536
            self.index_path = f"{self.index_dir}/faiss.index"
            self.meta_path  = f"{self.index_dir}/meta.jsonl"
            if not pathlib.Path(self.index_path).exists():
                # empty index
                self.index = faiss.IndexFlatIP(self.dim)
                self.meta = []
            else:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = [json.loads(line) for line in f]
        else:
            import chromadb
            self.chroma = chromadb.PersistentClient(path=self.index_dir)
            self.coll = self.chroma.get_or_create_collection("docs")

    def search(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        q_emb = embed_texts_titan([query])[0]
        if self.backend == "faiss":
            import faiss, numpy as np
            if self.index.ntotal == 0 or not self.meta:
                return []
            D, I = self.index.search(np.array([q_emb]).astype("float32"), k)
            out = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                m = self.meta[idx]
                out.append((m["text"], float(score)))
            return out
        else:
            res = self.coll.query(query_embeddings=[q_emb], n_results=k)
            docs = res.get("documents", [[]])[0]
            dists = res.get("distances", [[]])[0]
            return list(zip(docs, [1 - d for d in dists]))  # convert L2 to similarity-ish

def build_prompt(system_prompt: str, query: str, contexts: List[str]) -> list:
    ctx_block = "\n\n".join([f"- {c}" for c in contexts]) if contexts else "No context."
    user = (
        "Use the following context to answer.\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {query}\n"
        "If the answer is not in the context, say you don't know."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
