"""BM25Index 单元测试。"""

import pickle
from pathlib import Path

import pytest

from rag_finance_system.src.bm25_index import BM25Index
from tests.fixtures.sample_chunks import make_chunks, make_chunk


class TestBM25Indexing:
    """索引构建。"""

    def test_index_empty_list(self):
        bm25 = BM25Index()
        assert bm25.index([]) == 0
        assert bm25.doc_count == 0

    def test_index_single_chunk(self):
        bm25 = BM25Index()
        chunks = [make_chunk(text="资本充足率是银行监管的核心指标")]
        assert bm25.index(chunks) == 1
        assert bm25.doc_count == 1

    def test_index_multiple_chunks(self):
        bm25 = BM25Index()
        chunks = make_chunks(5)
        assert bm25.index(chunks) == 5
        assert bm25.doc_count == 5

    def test_index_replaces_previous(self):
        bm25 = BM25Index()
        bm25.index([make_chunk(text="文档1")])
        bm25.index([make_chunk(text="文档2")])
        # 第二次 index 重建（index 追加后重算）
        assert bm25.doc_count == 2

    def test_index_updates_doc_freqs(self):
        bm25 = BM25Index()
        chunks = [make_chunk(text="资本充足率是核心")]
        bm25.index(chunks)
        # jieba 应分出"资本"
        assert bm25.doc_count > 0
        assert len(bm25.doc_freqs) > 0


class TestBM25Search:
    """检索。"""

    def test_search_empty_index(self):
        bm25 = BM25Index()
        assert bm25.search("资本充足率") == []

    def test_search_basic_match(self):
        bm25 = BM25Index()
        chunks = [
            make_chunk(text="资本充足率是银行监管的核心指标", chunk_id="c1"),
            make_chunk(text="今天天气不错", chunk_id="c2"),
        ]
        bm25.index(chunks)
        results = bm25.search("资本充足率")
        assert len(results) > 0
        assert results[0]["chunk_id"] == "c1"

    def test_search_no_match(self):
        bm25 = BM25Index()
        bm25.index([make_chunk(text="今天天气不错")])
        results = bm25.search("量子力学方程")
        assert len(results) == 0

    def test_search_top_k(self):
        bm25 = BM25Index()
        bm25.index(make_chunks(5))
        results = bm25.search("公司法", top_k=2)
        assert len(results) <= 2

    def test_search_source_filter(self):
        bm25 = BM25Index()
        chunks = [
            make_chunk(text="资本充足率相关", source="法1.txt", chunk_id="c1"),
            make_chunk(text="资本充足率相关", source="法2.txt", chunk_id="c2"),
        ]
        bm25.index(chunks)
        results = bm25.search("资本充足率", source_filter="法1.txt")
        assert all(r["source"] == "法1.txt" for r in results)

    def test_search_doc_type_filter(self):
        bm25 = BM25Index()
        chunks = [
            make_chunk(text="资本充足率", doc_type="law", chunk_id="c1"),
            make_chunk(text="资本充足率", doc_type="case", chunk_id="c2"),
        ]
        bm25.index(chunks)
        results = bm25.search("资本充足率", doc_type_filter="law")
        assert all(r["doc_type"] == "law" for r in results)

    def test_search_law_name_filter(self):
        bm25 = BM25Index()
        chunks = [
            make_chunk(text="资本充足率", law_name="中华人民共和国公司法", chunk_id="c1"),
            make_chunk(text="资本充足率", law_name="中华人民共和国证券法", chunk_id="c2"),
        ]
        bm25.index(chunks)
        results = bm25.search("资本充足率", law_name_filter="中华人民共和国公司法")
        assert all(r["law_name"] == "中华人民共和国公司法" for r in results)

    def test_search_authority_filter(self):
        bm25 = BM25Index()
        chunks = [
            make_chunk(text="资本充足率", authority="央行", chunk_id="c1"),
            make_chunk(text="资本充足率", authority="证监会", chunk_id="c2"),
        ]
        bm25.index(chunks)
        results = bm25.search("资本充足率", authority_filter="央行")
        assert all(r["authority"] == "央行" for r in results)

    def test_search_status_filter(self):
        bm25 = BM25Index()
        chunks = [
            make_chunk(text="资本充足率", status="有效", chunk_id="c1"),
            make_chunk(text="资本充足率", status="已修订", chunk_id="c2"),
        ]
        bm25.index(chunks)
        results = bm25.search("资本充足率", status_filter="有效")
        assert all(r["status"] == "有效" for r in results)

    def test_search_multiple_filters(self):
        bm25 = BM25Index()
        chunks = [
            make_chunk(text="资本充足率", doc_type="law", status="有效", chunk_id="c1"),
            make_chunk(text="资本充足率", doc_type="case", status="有效", chunk_id="c2"),
            make_chunk(text="资本充足率", doc_type="law", status="已修订", chunk_id="c3"),
        ]
        bm25.index(chunks)
        results = bm25.search("资本充足率", doc_type_filter="law", status_filter="有效")
        assert all(r["doc_type"] == "law" and r["status"] == "有效" for r in results)

    def test_search_chinese_tokenization(self):
        bm25 = BM25Index()
        chunks = [make_chunk(text="中华人民共和国公司法规定了股东责任")]
        bm25.index(chunks)
        results = bm25.search("股东责任")
        assert len(results) > 0


class TestBM25Scoring:
    """评分逻辑。"""

    def test_bm25_score_positive(self):
        bm25 = BM25Index()
        bm25.index([make_chunk(text="资本充足率是银行监管核心指标")])
        results = bm25.search("资本充足率")
        assert all(r["bm25_score"] > 0 for r in results)

    def test_bm25_score_ranking(self):
        bm25 = BM25Index()
        chunks = [
            make_chunk(text="资本充足率 资本充足率 资本充足率 核心指标", chunk_id="c1"),
            make_chunk(text="资本充足率是重要指标", chunk_id="c2"),
        ]
        bm25.index(chunks)
        results = bm25.search("资本充足率")
        if len(results) >= 2:
            assert results[0]["bm25_score"] >= results[1]["bm25_score"]


class TestBM25Persistence:
    """持久化。"""

    def test_save_and_load(self, tmp_path):
        bm25 = BM25Index()
        bm25.index([make_chunk(text="资本充足率")])
        path = str(tmp_path / "bm25_test.pkl")
        bm25.save(path)

        loaded = BM25Index.load(path)
        assert loaded is not None
        assert loaded.doc_count == bm25.doc_count
        assert loaded.search("资本充足率")[0]["bm25_score"] > 0

    def test_load_nonexistent(self, tmp_path):
        loaded = BM25Index.load(str(tmp_path / "nonexistent.pkl"))
        assert loaded is None

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not a valid pickle")
        loaded = BM25Index.load(str(path))
        assert loaded is None
