from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000, example="What constitutes medical malpractice?")
    domain: Optional[Literal["legal", "medical", "auto"]] = Field(
        default="auto",
        description="Domain to search. 'auto' detects from the question."
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the elements of negligence in a personal injury case?",
                "domain": "legal",
                "top_k": 5
            }
        }


class SourceDocument(BaseModel):
    text: str
    source: str
    domain: str
    score: float
    chunk_id: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    domain_detected: str
    sources: List[SourceDocument]
    latency_ms: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the elements of negligence?",
                "answer": "The four elements of negligence are: duty of care, breach of duty, causation, and damages.",
                "confidence": 0.91,
                "domain_detected": "legal",
                "sources": [],
                "latency_ms": 320.5
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_size: int


class IndexStats(BaseModel):
    total_documents: int
    legal_documents: int
    medical_documents: int
    index_dimension: int
    model_name: str