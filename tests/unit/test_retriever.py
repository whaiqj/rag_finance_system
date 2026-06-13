"""retriever 单元测试。"""

from unittest.mock import MagicMock, patch

from rag_finance_system.src.retriever import Retriever, _rrf_fusion


class TestRRFFusion:
    def test_rrf_fusion_single_list(self):
        results = _rrf_fusion([
            {"chunk_id": "a", "score": 0.9},
            {"chunk_id": "b", "score": 0.8},
        ], top_k=10)
        assert len(results) == 2
        assert "rrf_score" in results[0]

    def test_rrf_fusion_two_lists(self):
        results = _rrf_fusion(
            [{"chunk_id": "a"}, {"chunk_id": "b"}],
            [{"chunk_id": "b"}, {"chunk_id": "c"}],
            top_k=10,
        )
        assert len(results) == 3
        assert results[0]["chunk_id"] == "b"

    def test_rrf_fusion_deduplicates(self):
        results = _rrf_fusion(
            [{"chunk_id": "a", "text": "x"}],
            [{"chunk_id": "a", "text": "y"}],
            top_k=10,
        )
        assert len(results) == 1
        assert results[0]["chunk_id"] == "a"

    def test_rrf_fusion_top_k(self):
        results = _rrf_fusion(
            [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}],
            top_k=2,
        )
        assert len(results) == 2


class TestRetriever:
    def make_retriever(self):
        embedder = MagicMock()
        embedder.encode_query.return_value = [0.1, 0.2]
        vector_store = MagicMock()
        reranker = MagicMock()
        reranker.rerank.return_value = [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.7}]
        bm25 = MagicMock()
        bm25.doc_count = 2
        term_index = MagicMock()
        term_index.doc_count = 2
        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            bm25_index=bm25,
            term_index=term_index,
            top_k=3,
            reranker_top_n=2,
        )
        return retriever, embedder, vector_store, reranker, bm25, term_index

    def test_retriever_flags(self):
        retriever, *_ = self.make_retriever()
        assert retriever._has_bm25 is True
        assert retriever._has_terms is True

    def test_vec_search(self):
        retriever, _, vector_store, _, _, _ = self.make_retriever()
        vector_store.search.side_effect = [
            [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.8}],
            [{"chunk_id": "b", "score": 0.7}, {"chunk_id": "c", "score": 0.6}],
        ]
        results = retriever._vec_search([0.1], [{"law_name": "法1"}, {"source": "s1"}], None, "有效", 3)
        assert [r["chunk_id"] for r in results] == ["a", "b", "c"]

    def test_ft_search_with_bm25(self):
        retriever, _, _, _, bm25, _ = self.make_retriever()
        retriever._has_es = False
        bm25.search.side_effect = [[{"chunk_id": "a", "bm25_score": 1.2}]]
        results = retriever._ft_search("资本充足率", [{}], None, "有效", 3)
        assert results[0]["chunk_id"] == "a"

    def test_term_search(self):
        retriever, *_rest, term_index = self.make_retriever()
        term_index.search.return_value = [{"chunk_id": "t1", "term_score": 2.0}]
        results = retriever._term_search("NPL", None, None, None, None, "有效", 3)
        assert results[0]["chunk_id"] == "t1"

    @patch("rag_finance_system.src.retriever._get_pool")
    def test_retrieve_vector_only(self, mock_get_pool):
        retriever, *_ = self.make_retriever()
        retriever._has_bm25 = False
        retriever._has_terms = False
        retriever._vec_search = MagicMock(return_value=[{"chunk_id": "a", "text": "doc a", "score": 0.9}])

        class FakeFuture:
            def __init__(self, value): self.value = value
            def result(self): return self.value
        class FakePool:
            def submit(self, fn, *args, **kwargs): return FakeFuture(fn(*args, **kwargs))
        mock_get_pool.return_value = FakePool()

        with patch("rag_finance_system.src.retriever.as_completed", side_effect=lambda fs: list(fs)):
            results = retriever.retrieve("问题", use_reranker=False)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "a"

    @patch("rag_finance_system.src.retriever._get_pool")
    def test_retrieve_with_rrf_and_reranker(self, mock_get_pool):
        retriever, *_ = self.make_retriever()
        retriever._vec_search = MagicMock(return_value=[{"chunk_id": "a", "text": "doc a", "score": 0.9}])
        retriever._ft_search = MagicMock(return_value=[{"chunk_id": "b", "text": "doc b", "bm25_score": 1.0}])
        retriever._term_search = MagicMock(return_value=[{"chunk_id": "c", "text": "doc c", "term_score": 2.0}])

        class FakeFuture:
            def __init__(self, value): self.value = value
            def result(self): return self.value
        class FakePool:
            def submit(self, fn, *args, **kwargs): return FakeFuture(fn(*args, **kwargs))
        mock_get_pool.return_value = FakePool()

        with patch("rag_finance_system.src.retriever.as_completed", side_effect=lambda fs: list(fs)):
            results = retriever.retrieve("问题", use_reranker=True)
        assert len(results) == 2
        assert all("reranker_score" in r for r in results)

    def test_compute_confidence_no_chunks(self):
        retriever, *_ = self.make_retriever()
        result = retriever.compute_confidence("q", "a", [])
        assert result == {"total": 0.0, "retrieval": 0.0, "coverage": 0.0}

    def test_compute_confidence_with_chunks(self):
        retriever, *_ = self.make_retriever()
        chunks = [
            {"text": "capital adequacy ratio", "score": 0.8},
            {"text": "other words", "score": 0.6},
        ]
        result = retriever.compute_confidence("q", "The capital adequacy ratio is important", chunks)
        assert result["total"] > 0
        assert result["retrieval"] > 0
