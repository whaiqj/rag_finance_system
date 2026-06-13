"""TermIndex 单元测试。"""

import pytest

from rag_finance_system.src.term_index import TermIndex
from tests.fixtures.sample_chunks import make_chunks, make_chunk


class TestTermIndexing:
    """索引构建。"""

    def test_index_no_dictionary(self):
        ti = TermIndex(dictionary=None)
        assert ti.index([make_chunk()]) == 0

    def test_index_with_dictionary(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        chunks = [make_chunk(text="资本充足率是银行监管核心指标")]
        result = ti.index(chunks)
        assert result > 0
        assert ti.doc_count > 0

    def test_index_detects_terms_in_text(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        chunks = [make_chunk(text="资本充足率和不良贷款率都是重要指标")]
        ti.index(chunks)
        assert "资本充足率" in ti._inverted_index
        assert "不良贷款率" in ti._inverted_index

    def test_index_empty_chunks(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        assert ti.index([]) == 0

    def test_index_accumulates(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        ti.index([make_chunk(text="资本充足率指标一")])
        ti.index([make_chunk(text="资本充足率指标二")])
        assert ti.doc_count == 2


class TestTermSearch:
    """检索。"""

    def test_search_no_dictionary(self):
        ti = TermIndex(dictionary=None)
        assert ti.search("资本充足率") == []

    def test_search_matching_query(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        ti.index([make_chunk(text="资本充足率是银行监管核心指标")])
        results = ti.search("资本充足率要求")
        assert len(results) > 0
        assert results[0]["term_score"] > 0

    def test_search_no_matching_terms(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        ti.index([make_chunk(text="今天天气很好")])
        results = ti.search("量子力学方程")
        assert len(results) == 0

    def test_search_source_filter(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        chunks = [
            make_chunk(text="资本充足率", source="法1.txt", chunk_id="c1"),
            make_chunk(text="资本充足率", source="法2.txt", chunk_id="c2"),
        ]
        ti.index(chunks)
        results = ti.search("资本充足率", source_filter="法1.txt")
        assert all(r["source"] == "法1.txt" for r in results)

    def test_search_law_name_filter(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        chunks = [
            make_chunk(text="资本充足率", law_name="中华人民共和国公司法", chunk_id="c1"),
            make_chunk(text="资本充足率", law_name="中华人民共和国证券法", chunk_id="c2"),
        ]
        ti.index(chunks)
        results = ti.search("资本充足率", law_name_filter="中华人民共和国公司法")
        assert all(r["law_name"] == "中华人民共和国公司法" for r in results)

    def test_search_status_filter(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        chunks = [
            make_chunk(text="资本充足率", status="有效", chunk_id="c1"),
            make_chunk(text="资本充足率", status="已修订", chunk_id="c2"),
        ]
        ti.index(chunks)
        results = ti.search("资本充足率", status_filter="有效")
        assert all(r["status"] == "有效" for r in results)

    def test_search_top_k_limit(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        ti.index(make_chunks(5))
        results = ti.search("资本充足率", top_k=2)
        assert len(results) <= 2

    def test_search_term_score_injected(self, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        ti.index([make_chunk(text="资本充足率是核心指标")])
        results = ti.search("资本充足率要求")
        assert all("term_score" in r for r in results)

    def test_search_after_detect_entities(self, finance_dict):
        """验证 search 内部调用 detect_entities 的工作流。"""
        ti = TermIndex(dictionary=finance_dict)
        ti.index([make_chunk(text="NPL不良贷款率的计算方法")])
        # "NPL" 应触发 detect_entities → 识别 "不良贷款率"
        results = ti.search("NPL怎么算")
        assert len(results) > 0


class TestTermPersistence:
    """持久化。"""

    def test_save_and_load(self, tmp_path, finance_dict):
        ti = TermIndex(dictionary=finance_dict)
        ti.index([make_chunk(text="资本充足率核心指标")])
        path = str(tmp_path / "term_index_test.pkl")
        ti.save(path)

        loaded = TermIndex.load(path)
        assert loaded is not None
        assert loaded.doc_count == ti.doc_count
        # 加载后需重新设置 dictionary 才能搜索
        loaded._dictionary = finance_dict
        results = loaded.search("资本充足率")
        assert len(results) > 0

    def test_load_nonexistent(self, tmp_path):
        loaded = TermIndex.load(str(tmp_path / "nonexistent.pkl"))
        assert loaded is None
