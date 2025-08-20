
import json
from typing import Iterator, Dict, Any, List, Optional
import boto3
from .deps import get_settings

settings = get_settings()
bedrock = boto3.client("bedrock-runtime", region_name=settings["aws_region"])

def _invoke(model_id: Optional[str]=None, *, body: Dict[str, Any], stream: bool=False):
    return bedrock.invoke_model_with_response_stream(modelId=model_id or settings["bedrock_model_id"],
                                                     body=json.dumps(body),
                                                     accept="application/json",
                                                     contentType="application/json") if stream else            bedrock.invoke_model(modelId=model_id or settings["bedrock_model_id"],
                                body=json.dumps(body),
                                accept="application/json",
                                contentType="application/json")

def invoke_once(messages: List[Dict[str, Any]], *, system: Optional[str]=None, tools: Optional[List[Dict[str, Any]]]=None,
                max_tokens: int=512, temperature: float=0.3) -> Dict[str, Any]:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    resp = _invoke(body=body, stream=False)
    payload = json.loads(resp["body"].read())
    return payload

def stream_completion(messages: List[Dict[str, Any]], *, system: Optional[str]=None, tools: Optional[List[Dict[str, Any]]]=None,
                      max_tokens: int=512, temperature: float=0.3) -> Iterator[str]:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    resp = _invoke(body=body, stream=True)
    for event in resp.get("body"):
        chunk = event.get("chunk")
        if not chunk:
            continue
        data = json.loads(chunk.get("bytes").decode("utf-8"))
        t = data.get("type")
        
        if t == "content_block_delta":
            delta = data.get("delta", {}).get("text", "")
            if delta:
                yield delta

def first_tool_use(message_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return first tool_use block, if any."""
    for block in message_payload.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return {"id": block.get("id"), "name": block.get("name"), "input": block.get("input")}
    return None
