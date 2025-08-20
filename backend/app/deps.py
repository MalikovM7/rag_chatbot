
import os

def get_settings():
    return {
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),
        "bedrock_model_id": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0"),
        "embedding_model_id": os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
        "vector_db": os.getenv("VECTOR_DB", "faiss"), 
        "index_dir": os.getenv("INDEX_DIR", "/data/index"),
        "rag_max_tokens_ctx": int(os.getenv("RAG_MAX_TOKENS_CTX", "512")),
        "system_prompt": os.getenv("SYSTEM_PROMPT", "You are a helpful RAG assistant."),
    }
