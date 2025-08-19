import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from .schemas import ChatRequest, ChatChunk, DocChunk
from .deps import get_settings
from .rag import Retriever, build_prompt
from .bedrock_client import stream_claude

app = FastAPI(title="RAG Chatbot Backend")
settings = get_settings()
retriever = Retriever()

@app.get("/health")
def health():
    return {"status": "ok", "model": settings["bedrock_model_id"]}

@app.get("/info")
def info():
    return {
        "rag": True,
        "vector_db": settings["vector_db"],
        "embedding_model": settings["embedding_model_id"]
    }

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    # 1) Retrieval → 2) Augmentation → 3) Generation (slides) :contentReference[oaicite:3]{index=3}
    sources = []
    contexts = []
    if req.use_rag:
        hits = retriever.search(req.query, k=req.top_k)
        for i, (text, score) in enumerate(hits):
            sources.append(DocChunk(id=str(i), text=text[:500], score=score).model_dump())
            contexts.append(text)

    messages = build_prompt(settings["system_prompt"], req.query, contexts)

    def gen():
        # first send sources (once)
        yield f"data:{json.dumps(ChatChunk(type='token', data='', sources=sources).model_dump())}\n\n"
        for token in stream_claude(messages, temperature=req.temperature):
            yield f"data:{json.dumps(ChatChunk(type='token', data=token).model_dump())}\n\n"
        yield f"data:{json.dumps(ChatChunk(type='done', data='').model_dump())}\n\n"

    return EventSourceResponse(gen(), media_type="text/event-stream")
