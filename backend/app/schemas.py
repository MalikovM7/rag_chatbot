
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    use_rag: bool = True
    top_k: int = 4
    temperature: float = 0.3

class DocChunk(BaseModel):
    id: str
    text: str
    score: float

class ChatChunk(BaseModel):
    type: str 
    data: Optional[str] = None
    sources: Optional[List[DocChunk]] = None
