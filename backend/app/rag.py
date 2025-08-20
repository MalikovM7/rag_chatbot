import json
from typing import List, Tuple

import numpy as np

from .deps import get_settings

settings = get_settings()


class Retriever:
    def __init__(self):
        self.backend = settings["vector_db"]
        self.index_dir = settings["index_dir"]
        self._load()

    def _load(self):
        if self.backend == "faiss":
            import os
            import pickle

            import faiss

            idx_path = os.path.join(self.index_dir, "index.faiss")
            meta_path = os.path.join(self.index_dir, "meta.pkl")
            self.index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as fh:
                self.texts = pickle.load(fh)
        else:
            raise ValueError(f"Unsupported VECTOR_DB={self.backend}")

    def _embed(self, texts: List[str]) -> np.ndarray:

        import boto3


        bedrock = boto3.client("bedrock-runtime", region_name=settings["aws_region"])
        body = {"inputText": texts, "dimensions": 1536}
        resp = bedrock.invoke_model(
            modelId=settings["embedding_model_id"],
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read())
        vecs = [item["embedding"] for item in payload["embeddingResults"]]
        return np.array(vecs, dtype="float32")

    def search(self, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        q = self._embed([query])[0]
        scores, idxs = self.index.search(np.array([q], dtype="float32"), top_k)
        dists = scores[0]
        ids = idxs[0]
        results = []
        for i, d in zip(ids, dists):
            if i < 0:
                continue
            text = self.texts[i]

            sim = float(1.0 / (1.0 + d))
            results.append((text, sim))
        return results


def rag_tool_schema():
    return {
        "name": "get_knowledge_base_data",
        "description": "Retrieve top relevant passages for a user question from the enterprise knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "Natural-language question to search with",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many passages to return",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["user_query"],
        },
    }


def run_rag_tool(user_query: str, top_k: int = 4):
    retriever = Retriever()
    hits = retriever.search(user_query, top_k=top_k)
    sources = [
        {"id": str(i), "text": t, "score": float(s)} for i, (t, s) in enumerate(hits)
    ]
    return {"sources": sources}


def build_messages(system_prompt: str, user_query: str, sources: List[str]) -> list:
    ctx = "\n\n".join(f"- {c}" for c in sources) if sources else "No context available."
    user = (
        f"Use the context to answer.\n\nContext:\n{ctx}\n\nQuestion: {user_query}\n"
        "If the answer is not in the context, say you don't know."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]
