"""api_app 单元测试。"""

from unittest.mock import MagicMock, patch

import pytest


class TestAPIAppFunctions:
    """直接测试 api_app 里的业务函数，跳过 FastAPI 路由层。"""

    def test_get_embedder(self):
        with patch("rag_finance_system.src.embedder.Embedder") as mock_cls:
            from rag_finance_system.api_app import _get_embedder
            import rag_finance_system.api_app as api_module
            api_module._embedder = None
            mock_cls.return_value = "embedder"
            assert _get_embedder() == "embedder"
            assert _get_embedder() == "embedder"
            assert mock_cls.call_count == 1

    def test_get_vector_store(self):
        with patch("rag_finance_system.src.vector_store.VectorStore") as mock_cls:
            from rag_finance_system.api_app import _get_vector_store
            import rag_finance_system.api_app as api_module
            api_module._vector_store = None
            mock_cls.return_value = "vs"
            assert _get_vector_store() == "vs"

    def test_get_reranker_failed(self):
        with patch("rag_finance_system.src.embedder.Reranker", side_effect=RuntimeError("fail")):
            from rag_finance_system.api_app import _get_reranker
            import rag_finance_system.api_app as api_module
            api_module._reranker = None
            api_module._reranker_failed = False
            assert _get_reranker() is None
            assert api_module._reranker_failed is True

    def test_get_bm25_load(self):
        with patch("rag_finance_system.src.bm25_index.BM25Index") as mock_cls:
            mock_cls.load.return_value = "bm25"
            from rag_finance_system.api_app import _get_bm25
            import rag_finance_system.api_app as api_module
            api_module._bm25_index = None
            assert _get_bm25() == "bm25"

    def test_get_bm25_create_empty(self):
        with patch("rag_finance_system.src.bm25_index.BM25Index") as mock_cls:
            mock_cls.load.return_value = None
            mock_cls.return_value = "empty_bm25"
            from rag_finance_system.api_app import _get_bm25
            import rag_finance_system.api_app as api_module
            api_module._bm25_index = None
            assert _get_bm25() == "empty_bm25"

    def test_get_es_connected(self):
        with patch("rag_finance_system.src.es_index.ESIndex") as mock_cls:
            es = MagicMock()
            es._connected = True
            es.doc_count = 10
            mock_cls.return_value = es
            from rag_finance_system.api_app import _get_es
            import rag_finance_system.api_app as api_module
            api_module._es_index = None
            api_module._es_failed = False
            assert _get_es() is es

    def test_get_es_not_connected(self):
        with patch("rag_finance_system.src.es_index.ESIndex") as mock_cls:
            es = MagicMock()
            es._connected = False
            mock_cls.return_value = es
            from rag_finance_system.api_app import _get_es
            import rag_finance_system.api_app as api_module
            api_module._es_index = None
            api_module._es_failed = False
            assert _get_es() is None
            assert api_module._es_failed is True

    def test_get_term_index_load(self):
        with patch("rag_finance_system.src.term_index.TermIndex") as mock_cls:
            term_idx = MagicMock()
            mock_cls.load.return_value = term_idx
            with patch("rag_finance_system.api_app._get_dictionary", return_value="dict"):
                from rag_finance_system.api_app import _get_term_index
                import rag_finance_system.api_app as api_module
                api_module._term_index = None
                result = _get_term_index()
                assert result is term_idx

    def test_get_processor(self):
        with patch("rag_finance_system.src.document_processor.DocumentProcessor") as mock_cls:
            mock_cls.return_value = "processor"
            from rag_finance_system.api_app import _get_processor
            import rag_finance_system.api_app as api_module
            api_module._processor = None
            assert _get_processor() == "processor"

    def test_resolve_single_file_status(self):
        from rag_finance_system.api_app import _resolve_single_file_status
        with patch("rag_finance_system.api_app._get_bm25") as mock_bm25:
            bm25 = MagicMock()
            bm25.corpus = [
                {"law_name": "中华人民共和国公司法", "effective_date": "20180101"},
            ]
            mock_bm25.return_value = bm25
            chunks = [
                {"law_name": "中华人民共和国公司法", "effective_date": "20050101", "status": "有效"},
            ]
            result = _resolve_single_file_status(chunks, {"中华人民共和国公司法"})
            assert result[0]["status"] == "已修订"

    def test_check_milvus_connected(self):
        with patch("rag_finance_system.api_app._get_vector_store") as mock_vs:
            vs = MagicMock()
            vs.client.has_collection.return_value = True
            mock_vs.return_value = vs
            from rag_finance_system.api_app import _check_milvus
            assert _check_milvus() is True

