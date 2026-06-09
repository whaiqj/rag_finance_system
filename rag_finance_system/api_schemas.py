"""Pydantic models for API request/response."""

from typing import Optional
from pydantic import BaseModel, Field


# ── Document Upload ──

class UploadResponse(BaseModel):
    filename: str
    file_path: str
    doc_type: str
    size_bytes: int


# ── Index ──

class IndexRequest(BaseModel):
    file_path: str
    doc_type: str = "law"


class IndexResponse(BaseModel):
    chunk_count: int
    doc_type: str
    file_path: str


# ── Search ──

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    doc_type_filter: Optional[str] = None
    law_name_filter: Optional[str] = None
    authority_filter: Optional[str] = None
    status_filter: Optional[str] = "有效"
    use_reranker: bool = True


class SearchResultItem(BaseModel):
    text: str
    source: str
    article_num: str
    score: float
    law_name: str
    doc_type: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


# ── Q&A ──

class QARequest(BaseModel):
    question: str
    doc_type_filter: Optional[str] = None
    use_reranker: bool = True
    use_query_rewrite: bool = True
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)
    include_historical: bool = False


class SourceItem(BaseModel):
    source: str
    article_num: str
    text: str
    score: float


class ConfidenceScores(BaseModel):
    total: float
    retrieval: float
    coverage: float


class QAResponse(BaseModel):
    question: str
    answer: str
    rewritten_query: Optional[str] = None
    sources: list[SourceItem]
    confidence: ConfidenceScores
