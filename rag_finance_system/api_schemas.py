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


# ── Article Relations ──

class ArticleRelationsRequest(BaseModel):
    law_name: str = Field(..., min_length=1, description="法规名称，如 '中华人民共和国公司法'")
    article_num: str = Field(..., min_length=1, description="条文编号，如 '16' 或 '第一百二十三条'")


class ArticleInfo(BaseModel):
    """单条条文展示信息。"""
    text: str
    chunk_id: str
    law_name: str
    article_num: str
    source: str


class ReferenceInfo(BaseModel):
    """一条 REFERENCES 边 + 引用方/被引用方的条文信息。"""
    text: str
    chunk_id: str
    law_name: str
    article_num: str
    source: str
    target_law: str = ""
    target_article: str = ""


class DocumentInfo(BaseModel):
    """Document 节点摘要。"""
    name: str
    doc_type: str
    source: str = ""


class RelatedDocumentInfo(BaseModel):
    """一条 RELATES_TO 边，含方向。"""
    name: str
    doc_type: str
    relation_type: str
    direction: str  # 'outgoing' | 'incoming'


class ArticleRelationsResponse(BaseModel):
    target: Optional[ArticleInfo] = None
    incoming_refs: list[ReferenceInfo] = Field(default_factory=list)
    outgoing_refs: list[ReferenceInfo] = Field(default_factory=list)
    parent_document: Optional[DocumentInfo] = None
    related_documents: list[RelatedDocumentInfo] = Field(default_factory=list)
    related_articles: list[ArticleInfo] = Field(default_factory=list)


# ── Category Management ──

class CategoriesResponse(BaseModel):
    categories: dict[str, list[str]] = Field(
        description="按类型分组的分类名: {term: [...], law: [...], authority: [...]}"
    )


class CategoryRenameRequest(BaseModel):
    old_name: str = Field(..., min_length=1)
    new_name: str = Field(..., min_length=1)


class CategoryAffectedCount(BaseModel):
    term: int = 0
    law: int = 0
    authority: int = 0


class CategoryDeleteResponse(BaseModel):
    deleted: str
    affected: CategoryAffectedCount


class CategoryRenameResponse(BaseModel):
    old_name: str
    new_name: str
    affected: CategoryAffectedCount


class DictionaryItem(BaseModel):
    name: str
    category: str = ""


class DictionaryItemList(BaseModel):
    items: list[DictionaryItem]


class SetCategoryRequest(BaseModel):
    item_type: str = Field(..., pattern=r"^(term|law|authority)$")
    item_name: str = Field(..., min_length=1)
    category: str
