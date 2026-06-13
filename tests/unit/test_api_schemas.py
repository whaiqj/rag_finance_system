"""api_schemas Pydantic 模型单元测试。"""

import pytest
from pydantic import ValidationError


class TestSearchRequest:
    def test_defaults(self):
        from rag_finance_system.api_schemas import SearchRequest

        req = SearchRequest(query="测试")
        assert req.top_k == 10
        assert req.status_filter == "有效"
        assert req.use_reranker is True

    def test_custom_values(self):
        from rag_finance_system.api_schemas import SearchRequest

        req = SearchRequest(query="测试", top_k=5, status_filter=None, use_reranker=False)
        assert req.top_k == 5
        assert req.status_filter is None
        assert req.use_reranker is False

    def test_top_k_too_low(self):
        from rag_finance_system.api_schemas import SearchRequest

        with pytest.raises(ValidationError):
            SearchRequest(query="测试", top_k=0)

    def test_top_k_too_high(self):
        from rag_finance_system.api_schemas import SearchRequest

        with pytest.raises(ValidationError):
            SearchRequest(query="测试", top_k=101)


class TestQARequest:
    def test_defaults(self):
        from rag_finance_system.api_schemas import QARequest

        req = QARequest(question="测试")
        assert req.max_new_tokens == 1024
        assert req.use_reranker is True
        assert req.use_query_rewrite is True
        assert req.include_historical is False

    def test_custom_values(self):
        from rag_finance_system.api_schemas import QARequest

        req = QARequest(question="测试", max_new_tokens=512, include_historical=True)
        assert req.max_new_tokens == 512
        assert req.include_historical is True

    def test_max_tokens_too_low(self):
        from rag_finance_system.api_schemas import QARequest

        with pytest.raises(ValidationError):
            QARequest(question="测试", max_new_tokens=32)

    def test_max_tokens_too_high(self):
        from rag_finance_system.api_schemas import QARequest

        with pytest.raises(ValidationError):
            QARequest(question="测试", max_new_tokens=5000)


class TestIndexRequest:
    def test_required_fields(self):
        from rag_finance_system.api_schemas import IndexRequest

        req = IndexRequest(file_path="/path/to/file.txt")
        assert req.file_path == "/path/to/file.txt"
        assert req.doc_type == "law"

    def test_custom_doc_type(self):
        from rag_finance_system.api_schemas import IndexRequest

        req = IndexRequest(file_path="/path/to/file.txt", doc_type="case")
        assert req.doc_type == "case"


class TestArticleRelationsRequest:
    def test_required_fields(self):
        from rag_finance_system.api_schemas import ArticleRelationsRequest

        req = ArticleRelationsRequest(law_name="中华人民共和国公司法", article_num="16")
        assert req.law_name == "中华人民共和国公司法"

    def test_empty_law_name_rejected(self):
        from rag_finance_system.api_schemas import ArticleRelationsRequest

        with pytest.raises(ValidationError):
            ArticleRelationsRequest(law_name="", article_num="16")

    def test_empty_article_num_rejected(self):
        from rag_finance_system.api_schemas import ArticleRelationsRequest

        with pytest.raises(ValidationError):
            ArticleRelationsRequest(law_name="中华人民共和国公司法", article_num="")


class TestSetCategoryRequest:
    def test_valid_item_types(self):
        from rag_finance_system.api_schemas import SetCategoryRequest

        for item_type in ["term", "law", "authority"]:
            req = SetCategoryRequest(item_type=item_type, item_name="测试", category="分类")
            assert req.item_type == item_type

    def test_invalid_item_type(self):
        from rag_finance_system.api_schemas import SetCategoryRequest

        with pytest.raises(ValidationError):
            SetCategoryRequest(item_type="invalid", item_name="测试", category="分类")


class TestCategoryRenameRequest:
    def test_valid(self):
        from rag_finance_system.api_schemas import CategoryRenameRequest

        req = CategoryRenameRequest(old_name="旧分类", new_name="新分类")
        assert req.old_name == "旧分类"

    def test_empty_old_name_rejected(self):
        from rag_finance_system.api_schemas import CategoryRenameRequest

        with pytest.raises(ValidationError):
            CategoryRenameRequest(old_name="", new_name="新分类")


class TestResponseModels:
    def test_search_response(self):
        from rag_finance_system.api_schemas import SearchResponse, SearchResultItem

        resp = SearchResponse(
            query="测试",
            results=[SearchResultItem(text="内容", source="法.txt", article_num="1", score=0.9, law_name="法", doc_type="law")],
        )
        assert resp.query == "测试"
        assert len(resp.results) == 1

    def test_qa_response(self):
        from rag_finance_system.api_schemas import QAResponse, ConfidenceScores, SourceItem

        resp = QAResponse(
            question="测试",
            answer="回答",
            sources=[SourceItem(source="法.txt", article_num="1", text="内容", score=0.9)],
            confidence=ConfidenceScores(total=0.9, retrieval=0.85, coverage=0.8),
        )
        assert resp.answer == "回答"
        assert resp.confidence.total == 0.9

    def test_upload_response(self):
        from rag_finance_system.api_schemas import UploadResponse

        resp = UploadResponse(filename="法.txt", file_path="/path", doc_type="law", size_bytes=1024)
        assert resp.size_bytes == 1024

    def test_index_response(self):
        from rag_finance_system.api_schemas import IndexResponse

        resp = IndexResponse(chunk_count=10, doc_type="law", file_path="/path")
        assert resp.chunk_count == 10

    def test_article_relations_response(self):
        from rag_finance_system.api_schemas import ArticleRelationsResponse

        resp = ArticleRelationsResponse()
        assert resp.incoming_refs == []
        assert resp.outgoing_refs == []
        assert resp.target is None

    def test_categories_response(self):
        from rag_finance_system.api_schemas import CategoriesResponse

        resp = CategoriesResponse(categories={"term": ["银行监管"], "law": [], "authority": []})
        assert "term" in resp.categories
