import json, boto3
from typing import Iterator, Dict, Any
from .deps import get_settings

settings = get_settings()
bedrock = boto3.client("bedrock-runtime", region_name=settings["aws_region"])

def embed_texts_titan(texts: list[str]) -> list[list[float]]:
    body = {"inputText": texts, "dimensions": 1536}
    resp = bedrock.invoke_model(
        modelId=settings["embedding_model_id"],
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    # Titan v2 returns list of embeddings under "embeddingList" or similar key depending on version
    # Fall back to common patterns safely:
    vectors = payload.get("embedding", payload.get("embeddings", payload.get("embeddingList")))
    return [v.get("embedding", v) for v in vectors]

def stream_claude(messages: list[Dict[str, Any]], temperature: float = 0.3) -> Iterator[str]:
    """
    Streams tokens from Bedrock Anthropic Claude via response stream.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": temperature,
        "messages": messages,
    }
    resp = bedrock.invoke_model_with_response_stream(
        modelId=settings["bedrock_model_id"],
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    for event in resp.get("body"):
        chunk = event.get("chunk")
        if not chunk:
            continue
        data = json.loads(chunk.get("bytes").decode("utf-8"))
        if data.get("type") in ("message_start", "content_block_start", "content_block_stop", "message_stop"):
            continue
        if data.get("type") == "content_block_delta":
            delta = data["delta"].get("text", "")
            if delta:
                yield delta
