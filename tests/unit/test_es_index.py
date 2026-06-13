"""es_index 单元测试。"""

import types
from unittest.mock import MagicMock, patch

from rag_finance_system.src.es_index import ESIndex


class TestESIndex:
    def make_es_module(self, ping=True):
        client = MagicMock()
        client.ping.return_value = ping
        client.indices.exists.return_value = False
        client.indices.stats.return_value = {"indices": {"finance_regulations": {"total": {"docs": {"count": 3}}}}}
        es_module = types.SimpleNamespace(Elasticsearch=MagicMock(return_value=client))
        helpers_module = types.SimpleNamespace(bulk=MagicMock(return_value=(2, [])))
        return client, es_module, helpers_module

    def test_es_init_connected(self):
        client, es_module, helpers_module = self.make_es_module(True)
        with patch.dict("sys.modules", {"elasticsearch": es_module, "elasticsearch.helpers": helpers_module}):
            es = ESIndex()
            assert es._connected is True
            client.indices.create.assert_called_once()

    def test_es_init_connection_failure(self):
        client, es_module, helpers_module = self.make_es_module(False)
        with patch.dict("sys.modules", {"elasticsearch": es_module, "elasticsearch.helpers": helpers_module}):
            es = ESIndex()
            assert es._connected is False

    def test_index_not_connected(self):
        es = ESIndex.__new__(ESIndex)
        es.doc_count = 0
        es._connected = False
        es._client = None
        assert es.index([]) == 0

    def test_index_calls_bulk(self):
        client, es_module, helpers_module = self.make_es_module(True)
        with patch.dict("sys.modules", {"elasticsearch": es_module, "elasticsearch.helpers": helpers_module}):
            es = ESIndex()
            count = es.index([
                {"chunk_id": "c1", "text": "资本充足率", "source": "法1.txt"},
                {"chunk_id": "c2", "text": "不良贷款率", "source": "法2.txt"},
            ])
            assert count == 3
            helpers_module.bulk.assert_called_once()

    def test_index_skips_empty_chunk_id(self):
        client, es_module, helpers_module = self.make_es_module(True)
        with patch.dict("sys.modules", {"elasticsearch": es_module, "elasticsearch.helpers": helpers_module}):
            es = ESIndex()
            es.index([{"chunk_id": "", "text": "资本充足率"}])
            args = helpers_module.bulk.call_args.args
            assert args[1] == []

    def test_search_not_connected(self):
        es = ESIndex.__new__(ESIndex)
        es.doc_count = 0
        es._connected = False
        es._client = None
        assert es.search("资本充足率") == []

    def test_search_with_filters(self):
        client, es_module, helpers_module = self.make_es_module(True)
        client.search.return_value = {"hits": {"hits": [{"_source": {"chunk_id": "c1", "text": "资本充足率"}, "_score": 1.23}]}}
        with patch.dict("sys.modules", {"elasticsearch": es_module, "elasticsearch.helpers": helpers_module}):
            es = ESIndex()
            es.doc_count = 1
            results = es.search("资本充足率", source_filter="法1.txt", status_filter="有效")
            assert results[0]["bm25_score"] == 1.23
            body = client.search.call_args.kwargs["body"]
            assert {"term": {"source": "法1.txt"}} in body["query"]["bool"]["filter"]
            assert {"term": {"status": "有效"}} in body["query"]["bool"]["filter"]

    def test_save_writes_marker(self, tmp_path):
        es = ESIndex.__new__(ESIndex)
        es.index_name = "finance_regulations"
        path = tmp_path / "es.marker"
        es.save(str(path))
        assert path.read_text(encoding="utf-8").strip() == "finance_regulations"

    def test_load_connected(self):
        def fake_connect(self, hosts=None, username=None, password=None):
            self._connected = True
            self._client = MagicMock()

        with patch.object(ESIndex, "_connect", fake_connect):
            with patch.object(ESIndex, "_ensure_index", return_value=None):
                with patch.object(ESIndex, "_refresh_doc_count", return_value=None):
                    loaded = ESIndex.load()
                    assert loaded is not None
                    assert loaded._connected is True
