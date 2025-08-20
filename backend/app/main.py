import json
import os
import time
from typing import Optional

import boto3
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="RAG Chatbot Backend")

BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
USE_FAKE_STREAM = os.getenv("USE_FAKE_STREAM", "0") == "1"


bedrock = None
if not USE_FAKE_STREAM:
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def _get_bedrock():
    import boto3

    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


class ChatReq(BaseModel):
    message: str = Field(..., alias="query")
    temperature: float = 0.3
    max_tokens: int = 512
    use_rag: Optional[bool] = None
    top_k: Optional[int] = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": BEDROCK_MODEL_ID if not USE_FAKE_STREAM else "FAKE_STREAM",
        "region": AWS_REGION,
        "fake": USE_FAKE_STREAM,
    }


def _sse(obj: dict) -> str:
    return f"data:{json.dumps(obj, ensure_ascii=False)}\n\n"


def _fake_stream(prompt: str):
    text = f"(demo) You said: {prompt}"
    for ch in text:
        yield _sse({"type": "token", "data": ch})
        time.sleep(0.01)


def _bedrock_stream(prompt: str, temperature: float, max_tokens: int):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = _get_bedrock().invoke_model_with_response_stream(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    buffer = ""

    def flush(force: bool = False):
        nonlocal buffer
        text = buffer.strip()
        if text and (force or text.endswith((".", "!", "?", "…"))):
            yield _sse({"type": "token", "data": text})
            buffer = ""

    for event in resp.get("body"):
        chunk = event.get("chunk")
        if not chunk:
            continue

        data = json.loads(chunk.get("bytes").decode("utf-8"))
        t = data.get("type")

        if t == "content_block_delta":
            delta = data.get("delta", {}).get("text", "")
            if delta:
                buffer += delta

                if buffer.endswith((".", "!", "?", "…")):

                    for line in flush(force=False):
                        yield line

        elif t in ("message_stop", "content_block_stop", "response_stop"):

            for line in flush(force=True):
                yield line

    if buffer.strip():
        yield _sse({"type": "token", "data": buffer.strip()})


@app.post("/chat/stream")
def chat_stream(req: ChatReq):
    def gen():
        try:
            if USE_FAKE_STREAM or bedrock is None:
                for line in _fake_stream(req.message):
                    yield line
            else:
                for line in _bedrock_stream(
                    req.message, req.temperature, req.max_tokens
                ):
                    yield line
            yield _sse({"type": "done"})
        except Exception as e:
            yield _sse({"type": "error", "data": f"{type(e).__name__}: {e}"})
            yield _sse({"type": "done"})

    return EventSourceResponse(gen(), media_type="text/event-stream")
