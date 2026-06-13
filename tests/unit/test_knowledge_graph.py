"""knowledge_graph 单元测试。"""

import types
from unittest.mock import MagicMock, patch

from rag_finance_system.src.knowledge_graph import KnowledgeGraph


class TestKnowledgeGraphInit:
    def test_kg_init_connected(self):
        driver = MagicMock()
        fake_graph_db = types.SimpleNamespace(driver=MagicMock(return_value=driver))
        fake_neo4j = types.SimpleNamespace(GraphDatabase=fake_graph_db)
        with patch.dict("sys.modules", {"neo4j": fake_neo4j}):
            kg = KnowledgeGraph()
            assert kg._connected is True
            driver.verify_connectivity.assert_called_once()

    def test_kg_init_connection_failure(self):
        fake_graph_db = types.SimpleNamespace(driver=MagicMock(side_effect=RuntimeError("fail")))
        fake_neo4j = types.SimpleNamespace(GraphDatabase=fake_graph_db)
        with patch.dict("sys.modules", {"neo4j": fake_neo4j}):
            kg = KnowledgeGraph()
            assert kg._connected is False
            assert kg._driver is None


class TestKnowledgeGraphOps:
    def make_kg(self):
        kg = KnowledgeGraph.__new__(KnowledgeGraph)
        kg.uri = "bolt://x"
        kg.user = "neo4j"
        kg.password = "neo4j"
        kg.database = "neo4j"
        kg._driver = MagicMock()
        kg._connected = True
        kg._run = MagicMock(return_value=[])
        return kg

    def test_upsert_document(self):
        kg = self.make_kg()
        assert kg.upsert_document("中华人民共和国公司法", "law", "公司法.txt") is True
        assert "MERGE (d:Document" in kg._run.call_args.args[0]

    def test_upsert_document_empty_name(self):
        kg = self.make_kg()
        assert kg.upsert_document("") is False

    def test_upsert_article(self):
        kg = self.make_kg()
        assert kg.upsert_article("c1", "文本", "1", "法", "源") is True
        assert "MERGE (a:Article" in kg._run.call_args.args[0]

    def test_upsert_term(self):
        kg = self.make_kg()
        assert kg.upsert_term("资本充足率", "定义", "银行监管") is True

    def test_upsert_authority(self):
        kg = self.make_kg()
        assert kg.upsert_authority("中国人民银行", "央行") is True

    def test_link_article_to_document(self):
        kg = self.make_kg()
        assert kg.link_article_to_document("c1", "中华人民共和国公司法") is True
        assert "BELONGS_TO" in kg._run.call_args.args[0]

    def test_link_article_reference(self):
        kg = self.make_kg()
        assert kg.link_article_reference("c1", "中华人民共和国公司法", "16") is True
        assert "REFERENCES" in kg._run.call_args.args[0]

    def test_link_term_definition(self):
        kg = self.make_kg()
        assert kg.link_term_definition("c1", "资本充足率") is True

    def test_link_document_authority(self):
        kg = self.make_kg()
        assert kg.link_document_authority("中华人民共和国公司法", "中国人民银行") is True

    def test_upsert_articles_batch(self):
        kg = self.make_kg()
        chunks = [{"chunk_id": "c1", "text": "t1"}, {"chunk_id": "c2", "text": "t2"}]
        assert kg.upsert_articles_batch(chunks) == 2

    def test_upsert_documents_batch_deduplicates(self):
        kg = self.make_kg()
        chunks = [
            {"law_name": "中华人民共和国公司法", "source": "a", "doc_type": "law"},
            {"law_name": "中华人民共和国公司法", "source": "a", "doc_type": "law"},
        ]
        assert kg.upsert_documents_batch(chunks) == 1

    def test_get_related_articles_disconnected(self):
        kg = self.make_kg()
        kg._connected = False
        kg._driver = None
        assert kg._run("MATCH (n) RETURN n") == []
