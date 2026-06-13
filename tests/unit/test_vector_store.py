"""vector_store 单元测试。"""

from unittest.mock import MagicMock, patch

import pytest

from rag_finance_system.src.vector_store import DEFAULT_TEXT_MAX_LENGTH, VectorStore
from tests.fixtures.sample_chunks import make_chunk, make_chunks


class TestVectorStoreStatics:
    def test_normalize_score_in_0_1(self):
        assert VectorStore._normalize_score(0.8) == 0.8

    def test_normalize_score_in_neg1_1(self):
        assert VectorStore._normalize_score(-1.0) == 0.0
        assert VectorStore._normalize_score(1.0) == 1.0

    def test_normalize_score_out_of_range(self):
        value = VectorStore._normalize_score(10.0)
        assert 0.0 <= value <= 1.0

    def test_clean_text_truncates(self):
        text = "a" * 5000
        assert len(VectorStore._clean_text(text, 100)) == 100

    def test_escape_expr_value(self):
        escaped = VectorStore._escape_expr_value('a"b\\c')
        assert '\\' in escaped
        assert '\"' in escaped


class TestVectorStoreExpressions:
    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_build_expr_no_filters(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        vs._schema_fields = {"source", "doc_type", "law_name", "authority", "status"}
        assert vs._build_expr() == ""

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_build_expr_multiple_filters(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        vs._schema_fields = {"source", "doc_type", "law_name", "authority", "status"}
        expr = vs._build_expr(source_filter="a", doc_type_filter="law", status_filter="有效")
        assert 'source == "a"' in expr
        assert 'doc_type == "law"' in expr
        assert 'status == "有效"' in expr

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_build_expr_skips_missing_schema_fields(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        vs._schema_fields = {"source"}
        expr = vs._build_expr(source_filter="a", status_filter="有效")
        assert 'source == "a"' in expr
        assert "status" not in expr


class TestVectorStoreOperations:
    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_insert_empty_chunks(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        assert vs.insert([], []) == 0

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_insert_calls_create_when_missing(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.side_effect = [False, False]
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        chunks = [make_chunk(text="资本充足率")]
        embeddings = [[0.1] * 512]
        inserted = vs.insert(chunks, embeddings)
        assert inserted == 1
        assert mock_client.create_collection.called
        assert mock_client.insert.called

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_insert_batches_data(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.side_effect = [False, False]
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        chunks = make_chunks(5)
        embeddings = [[0.1] * 512 for _ in range(5)]
        inserted = vs.insert(chunks, embeddings, batch_size=2)
        assert inserted == 5
        assert mock_client.insert.call_count == 3

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_insert_mismatch_raises(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        with pytest.raises(AssertionError):
            vs.insert([make_chunk()], [])

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_search_no_collection(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        assert vs.search([0.1, 0.2]) == []

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_search_returns_normalized(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.side_effect = [False, True]
        mock_client.search.return_value = [[{
            "id": "1",
            "distance": 0.85,
            "entity": {
                "text": "测试文本",
                "source": "法.txt",
                "chunk_id": "c1",
                "article_num": "1",
                "file_path": "/tmp/a.txt",
                "chunk_index": 0,
                "doc_type": "law",
                "law_name": "中华人民共和国公司法",
                "authority": "全国人大",
                "effective_date": "2024-01-01",
                "status": "有效",
            },
        }]]
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        vs._schema_fields = {"effective_date", "status"}
        results = vs.search([0.1, 0.2], top_k=1)
        assert len(results) == 1
        assert 0.0 <= results[0]["score"] <= 1.0
        assert results[0]["status"] == "有效"

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_search_with_filter(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.side_effect = [False, True]
        mock_client.search.return_value = [[]]
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        vs._schema_fields = {"source", "status"}
        vs.search([0.1, 0.2], source_filter="法.txt", status_filter="有效")
        kwargs = mock_client.search.call_args.kwargs
        assert 'source == "法.txt"' in kwargs["filter"]
        assert 'status == "有效"' in kwargs["filter"]

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_get_collection_stats_no_collection(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.return_value = False
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        stats = vs.get_collection_stats()
        assert stats == {"count": 0, "row_count": 0}

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_get_collection_stats_existing(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.side_effect = [False, True]
        mock_client.get_collection_stats.return_value = {"row_count": 12}
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        stats = vs.get_collection_stats()
        assert stats["count"] == 12

    @patch("rag_finance_system.src.vector_store.MilvusClient")
    def test_drop_collection(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.has_collection.side_effect = [False, True]
        mock_client_cls.return_value = mock_client
        vs = VectorStore()
        vs.drop_collection()
        mock_client.drop_collection.assert_called_once()
