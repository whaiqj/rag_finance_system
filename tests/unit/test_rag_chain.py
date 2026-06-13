"""rag_chain 单元测试。"""

from unittest.mock import MagicMock, patch

from rag_finance_system.src.rag_chain import (
    RAGChain,
    _build_authority_index,
    _build_law_name_index,
    _build_source_index,
    build_prompt,
)


class TestIndexBuilders:
    def test_build_law_name_index(self):
        index = _build_law_name_index()
        assert "公司法" in index
        assert index["公司法"] == "中华人民共和国公司法"

    def test_build_source_index(self):
        index = _build_source_index()
        assert isinstance(index, dict)

    def test_build_authority_index(self):
        index = _build_authority_index()
        assert isinstance(index, dict)


class TestBuildPrompt:
    def test_build_prompt_with_chunks(self):
        chunks = [
            {"text": "第一条 公司设立", "source": "公司法.txt", "article_num": "1"},
            {"text": "第二条 股东责任", "source": "公司法.txt", "article_num": "2"},
        ]
        messages = build_prompt("公司设立条件是什么", chunks)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "公司设立条件是什么" in messages[1]["content"]
        assert "第一条" in messages[1]["content"]

    def test_build_prompt_with_graph_facts(self):
        chunks = [{"text": "资本充足率", "source": "法.txt", "article_num": "1"}]
        facts = ["关系1", "关系2"]
        messages = build_prompt("资本充足率", chunks, graph_facts=facts)
        assert "关系1" in messages[1]["content"]
        assert "关系2" in messages[1]["content"]

    def test_build_prompt_no_chunks(self):
        messages = build_prompt("随意问题", [])
        assert len(messages) == 2
        assert "未检索到" in messages[1]["content"] or "未找到" in messages[1]["content"]


class TestRAGChain:
    def make_rag(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"chunk_id": "c1", "text": "资本充足率监管要求", "source": "法.txt", "article_num": "1", "score": 0.9}
        ]
        retriever.compute_confidence.return_value = {"total": 0.8, "retrieval": 0.85, "coverage": 0.75}
        llm = MagicMock()
        llm.generate.return_value = "资本充足率是指银行资本与风险加权资产的比例。"
        llm.generate_stream.return_value = iter(["资本", "充足率"])
        dictionary = MagicMock()
        dictionary.detect_entities.return_value = {"terms": ["资本充足率"], "law_names": [], "authorities": []}
        dictionary.expand_query.return_value = "资本充足率 银行监管"
        rewriter = MagicMock()
        rewriter.rewrite.return_value = "资本充足率 监管要求"
        kg = MagicMock()
        kg._connected = True
        kg.get_related_articles.return_value = []
        kg.get_graph_facts.return_value = []
        rag = RAGChain(
            retriever=retriever,
            llm=llm,
            dictionary=dictionary,
            rewriter=rewriter,
            knowledge_graph=kg,
        )
        return rag, retriever, llm, dictionary, rewriter, kg

    def test_rag_chain_init(self):
        rag, *_ = self.make_rag()
        assert rag.retriever is not None
        assert rag.llm is not None

    def test_detect_law_name(self):
        rag, *_ = self.make_rag()
        result = rag._detect_law_name("公司法第十六条")
        assert result == "中华人民共和国公司法"

    def test_detect_source(self):
        rag, *_ = self.make_rag()
        result = rag._detect_source("请问公司法规定")
        assert result is None or isinstance(result, str)

    def test_detect_authority(self):
        rag, *_ = self.make_rag()
        result = rag._detect_authority("上海银保监局")
        assert result is None or isinstance(result, str)

    def test_rewrite_query(self):
        rag, *_ = self.make_rag()
        result = rag.rewrite_query("资本充足率要求是什么")
        assert result == "资本充足率 监管要求"

    def test_query(self):
        rag, retriever, llm, dictionary, rewriter, kg = self.make_rag()
        result = rag.query("资本充足率要求是什么")
        assert result["question"] == "资本充足率要求是什么"
        assert result["answer"] == "资本充足率是指银行资本与风险加权资产的比例。"
        assert len(result["sources"]) == 1
        assert result["confidence"]["total"] == 0.8
        retriever.retrieve.assert_called_once()
        llm.generate.assert_called_once()

    def test_query_no_reranker(self):
        rag, retriever, *_ = self.make_rag()
        result = rag.query("资本充足率", use_reranker=False)
        assert retriever.retrieve.call_args.kwargs["use_reranker"] is False

    def test_query_no_rewrite(self):
        rag, retriever, *_ = self.make_rag()
        result = rag.query("资本充足率", use_query_rewrite=False)
        kwargs = retriever.retrieve.call_args.kwargs
        assert kwargs["query"] == "资本充足率 银行监管"

    def test_query_no_expansion(self):
        rag, retriever, llm, dictionary, *_ = self.make_rag()
        result = rag.query("资本充足率", use_query_expansion=False)
        dictionary.expand_query.assert_not_called()

    def test_query_with_doc_type_filter(self):
        rag, retriever, *_ = self.make_rag()
        with patch.object(rag, "_detect_source", return_value=None):
            with patch.object(rag, "_detect_law_name", return_value=None):
                with patch.object(rag, "_detect_authority", return_value=None):
                    result = rag.query("资本充足率", doc_type_filter="law")
                    kwargs = retriever.retrieve.call_args.kwargs
                    assert kwargs["doc_type_filter"] == "law"

    def test_query_include_historical(self):
        rag, retriever, *_ = self.make_rag()
        result = rag.query("资本充足率", include_historical=True)
        kwargs = retriever.retrieve.call_args.kwargs
        assert kwargs["status_filter"] is None

    def test_query_stream(self):
        rag, retriever, llm, *_ = self.make_rag()
        tokens = list(rag.query_stream("资本充足率"))
        assert len(tokens) > 0
        import json
        first = json.loads(tokens[0])
        assert first["type"] == "token"

    def test_query_with_kg_articles(self):
        rag, retriever, llm, dictionary, rewriter, kg = self.make_rag()
        kg.get_related_articles.return_value = [
            {"chunk_id": "g1", "text": "图谱条文", "source": "法.txt", "article_num": "2", "law_name": "中华人民共和国公司法", "relation": "references"}
        ]
        result = rag.query("资本充足率")
        assert len(result["sources"]) == 2

    def test_query_with_graph_facts(self):
        rag, retriever, llm, dictionary, rewriter, kg = self.make_rag()
        kg.get_graph_facts.return_value = ["关系1", "关系2"]
        result = rag.query("资本充足率")
        assert llm.generate.called

    def test_query_kg_failure_graceful(self):
        rag, retriever, llm, dictionary, rewriter, kg = self.make_rag()
        kg.get_related_articles.side_effect = RuntimeError("boom")
        result = rag.query("资本充足率")
        assert result["answer"] == "资本充足率是指银行资本与风险加权资产的比例。"
