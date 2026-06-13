"""graph_builder 单元测试。"""

from unittest.mock import MagicMock

from rag_finance_system.src.graph_builder import (
    GraphBuilder,
    _parse_cn_number,
    extract_authorities,
    extract_document_references,
    extract_references,
    extract_self_article_num,
    infer_document_relation,
)


class TestGraphBuilderFunctions:
    def test_parse_cn_number(self):
        assert _parse_cn_number("一百二十三") == "123"

    def test_parse_cn_number_digit(self):
        assert _parse_cn_number("16") == "16"

    def test_extract_references(self):
        refs = extract_references("根据《中华人民共和国公司法》第十六条和《证券法》第五条规定")
        assert {r["law"] for r in refs} == {"中华人民共和国公司法", "证券法"}

    def test_extract_references_multiple(self):
        refs = extract_references("见《公司法》第十条、《公司法》第十条、《证券法》第五条")
        assert len(refs) == 2

    def test_extract_self_article_num(self):
        assert extract_self_article_num("第一条 为了规范公司的组织和行为") == "1"

    def test_extract_document_references(self):
        refs = extract_document_references("根据《公司法》并参照《证券法》处理")
        assert refs == ["公司法", "证券法"]

    def test_infer_document_relation_basis(self):
        assert infer_document_relation("根据上位法要求执行") == "basis"

    def test_infer_document_relation_default(self):
        assert infer_document_relation("普通描述文本") == "related"

    def test_extract_authorities_with_dictionary(self, finance_dict):
        result = extract_authorities("银保监会要求加强监管", dictionary=finance_dict)
        assert "国家金融监督管理总局" in result

    def test_extract_authorities_without_dictionary(self):
        result = extract_authorities("中国人民银行发布通知")
        assert "中国人民银行" in result


class TestGraphBuilder:
    def test_build_from_chunks_disconnected(self):
        kg = MagicMock()
        kg._connected = False
        builder = GraphBuilder(kg)
        assert builder.build_from_chunks([]) == {}

    def test_build_from_chunks_calls_upserts(self, finance_dict):
        kg = MagicMock()
        kg._connected = True
        kg.upsert_documents_batch.return_value = 1
        kg.upsert_articles_batch.return_value = 2
        kg.link_articles_to_documents_batch.return_value = 2
        builder = GraphBuilder(kg, dictionary=finance_dict)
        builder._build_references = MagicMock(return_value=1)
        builder._build_document_relations = MagicMock(return_value=1)
        builder._build_document_authorities = MagicMock(return_value=1)
        builder._build_term_relations = MagicMock(return_value={"defines": 1, "mentions": 2})
        stats = builder.build_from_chunks([{"chunk_id": "c1", "text": "第一条", "law_name": "中华人民共和国公司法"}])
        assert stats["documents"] == 1
        assert stats["articles"] == 2
        assert stats["defines"] == 1

    def test_build_references(self, finance_dict):
        kg = MagicMock()
        kg._connected = True
        kg.link_article_reference.return_value = True
        builder = GraphBuilder(kg, dictionary=finance_dict)
        count = builder._build_references([{"chunk_id": "c1", "text": "根据《公司法》第十六条规定"}])
        assert count == 1

    def test_build_term_relations_no_dictionary(self):
        kg = MagicMock()
        kg._connected = True
        builder = GraphBuilder(kg, dictionary=None)
        assert builder._build_term_relations([{"chunk_id": "c1", "text": "内容"}]) == {"defines": 0, "mentions": 0}

    def test_sync_dictionary_to_graph(self, finance_dict):
        kg = MagicMock()
        kg._connected = True
        kg.upsert_term.return_value = True
        kg.upsert_authority.return_value = True
        builder = GraphBuilder(kg, dictionary=finance_dict)
        stats = builder.sync_dictionary_to_graph()
        assert stats["terms"] > 0
        assert stats["authorities"] > 0
