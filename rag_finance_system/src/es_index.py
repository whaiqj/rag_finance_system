"""
es_index.py
Elasticsearch 全文检索索引 — 替代/补充内存 BM25Index。
BM25 评分由 ES match 查询原生提供，支持 IK 中文分词。
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

ES_HOST = os.getenv("ES_HOST", "127.0.0.1")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_SCHEME = os.getenv("ES_SCHEME", "http")
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "finance_regulations")
ES_USERNAME = os.getenv("ES_USERNAME", "") or None
ES_PASSWORD = os.getenv("ES_PASSWORD", "") or None
ES_ANALYZER = os.getenv("ES_ANALYZER", "standard")


class ESIndex:
    """Elasticsearch 全文检索索引，接口兼容 BM25Index。"""

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        index_name: str = ES_INDEX_NAME,
        analyzer: str = ES_ANALYZER,
        username: Optional[str] = ES_USERNAME,
        password: Optional[str] = ES_PASSWORD,
    ):
        self.index_name = index_name
        self.analyzer = analyzer
        self.search_analyzer = "ik_smart" if analyzer == "ik_max_word" else "standard"
        self.doc_count: int = 0
        self._connected: bool = False
        self._client: Any = None

        self._connect(hosts, username, password)
        if self._connected:
            self._ensure_index()

    # ── 连接 ──

    def _connect(
        self,
        hosts: Optional[List[str]],
        username: Optional[str],
        password: Optional[str],
    ):
        try:
            from elasticsearch import Elasticsearch

            if hosts is None:
                hosts = [f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"]

            kwargs: dict = {
                "hosts": hosts,
                "request_timeout": 10,
                "max_retries": 2,
                "retry_on_timeout": True,
            }
            if username and password:
                kwargs["basic_auth"] = (username, password)

            self._client = Elasticsearch(**kwargs)

            if self._client.ping():
                self._connected = True
                logger.info(f"Elasticsearch 已连接: {hosts}")
            else:
                logger.warning("Elasticsearch ping 失败")
                self._client = None
        except Exception as e:
            logger.warning(f"Elasticsearch 连接失败，将回退到 BM25: {e}")
            self._client = None

    # ── 索引管理 ──

    def _build_mapping(self, analyzer: str, search_analyzer: str) -> dict:
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "1s",
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": analyzer,
                        "search_analyzer": search_analyzer,
                    },
                    "source": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "article_num": {"type": "keyword"},
                    "file_path": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "doc_type": {"type": "keyword"},
                    "law_name": {"type": "keyword"},
                    "authority": {"type": "keyword"},
                    "effective_date": {"type": "keyword"},
                    "status": {"type": "keyword"},
                }
            },
        }

    def _ensure_index(self):
        if not self._client:
            return
        if self._client.indices.exists(index=self.index_name):
            self._refresh_doc_count()
            return

        mapping = self._build_mapping(
            analyzer=self.analyzer, search_analyzer=self.search_analyzer
        )
        try:
            self._client.indices.create(index=self.index_name, body=mapping)
            logger.info(
                f"ES 索引已创建: {self.index_name} (analyzer={self.analyzer})"
            )
        except Exception as e:
            if self.analyzer != "standard" and "analyzer" in str(e).lower():
                logger.warning(f"IK 分词器不可用，回退 standard: {e}")
                self.analyzer = "standard"
                self.search_analyzer = "standard"
                mapping = self._build_mapping(analyzer="standard", search_analyzer="standard")
                self._client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"ES 索引已创建: {self.index_name} (analyzer=standard)")
            else:
                raise

    def _refresh_doc_count(self):
        if self._client:
            try:
                stats = self._client.indices.stats(index=self.index_name)
                self.doc_count = stats["indices"][self.index_name]["total"]["docs"]["count"]
            except Exception:
                self.doc_count = 0

    # ── 索引构建 ──

    def index(self, chunks: List[Dict[str, Any]]) -> int:
        """批量索引 chunks，以 chunk_id 为 _id 实现 upsert。"""
        if not self._connected or self._client is None:
            return self.doc_count
        if not chunks:
            return self.doc_count

        self._ensure_index()

        from elasticsearch.helpers import bulk

        actions = []
        skipped = 0
        for c in chunks:
            cid = c.get("chunk_id", "").strip()
            if not cid:
                skipped += 1
                continue
            actions.append(
                {
                    "_index": self.index_name,
                    "_id": cid,
                    "_source": {
                        "text": c.get("text", ""),
                        "source": c.get("source", ""),
                        "chunk_id": cid,
                        "article_num": c.get("article_num", ""),
                        "file_path": c.get("file_path", ""),
                        "chunk_index": int(c.get("chunk_index", 0)),
                        "doc_type": c.get("doc_type", "law"),
                        "law_name": c.get("law_name", ""),
                        "authority": c.get("authority", ""),
                        "effective_date": c.get("effective_date", ""),
                        "status": c.get("status", "有效"),
                    },
                }
            )

        if skipped:
            logger.warning(f"跳过 {skipped} 条缺少 chunk_id 的条目")

        try:
            success, errors = bulk(self._client, actions, raise_on_error=False)
            self._client.indices.refresh(index=self.index_name)
            self._refresh_doc_count()
            if errors:
                logger.warning(
                    f"ES 批量索引完成: {success} 成功, {len(errors)} 错误"
                )
            else:
                logger.info(
                    f"ES 批量索引完成: {success} 条, 总计 {self.doc_count} 篇"
                )
        except Exception as e:
            logger.warning(f"ES 批量索引失败: {e}")

        return self.doc_count

    # ── 检索 ──

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        law_name_filter: Optional[str] = None,
        authority_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """ES match 查询 + term 过滤，返回带 bm25_score 的 chunk 列表。"""
        if not self._connected or self._client is None or self.doc_count == 0:
            return []

        must = [{"match": {"text": query}}]
        filters = []
        if source_filter:
            filters.append({"term": {"source": source_filter}})
        if doc_type_filter:
            filters.append({"term": {"doc_type": doc_type_filter}})
        if law_name_filter:
            filters.append({"term": {"law_name": law_name_filter}})
        if authority_filter:
            filters.append({"term": {"authority": authority_filter}})
        if status_filter:
            filters.append({"term": {"status": status_filter}})

        body = {
            "query": {"bool": {"must": must, "filter": filters}},
            "size": top_k,
        }

        try:
            resp = self._client.search(index=self.index_name, body=body)
        except Exception as e:
            logger.warning(f"ES 查询失败: {e}")
            return []

        results: List[Dict[str, Any]] = []
        for hit in resp["hits"]["hits"]:
            src = dict(hit["_source"])
            src["bm25_score"] = round(hit.get("_score", 0.0) or 0.0, 6)
            results.append(src)
        return results

    # ── 持久化 (接口兼容桩) ──

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"{self.index_name}\n")
        logger.info(f"ES 索引标记已写入: {path}")

    @classmethod
    def load(cls, path: str = "") -> Optional["ESIndex"]:
        """尝试从 .env 配置连接 ES。忽略 path（接口兼容 BM25Index.load）。"""
        inst = cls()
        if inst._connected:
            inst._refresh_doc_count()
            return inst
        return None

    def __len__(self) -> int:
        return self.doc_count
