"""
bm25_index.py
BM25 关键词检索索引（内存实现 + jieba 中文分词）
配合向量检索做双路召回 + RRF 融合
"""

import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional

import jieba
from loguru import logger


class BM25Index:
    """BM25 倒排索引，纯内存实现，支持中文分词。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []          # 原始 chunk 列表
        self.tokenized: List[List[str]] = []            # 分词后的文档
        self.doc_freqs: Dict[str, int] = {}             # 词 → 出现文档数
        self.avgdl: float = 0.0
        self.doc_count: int = 0

    # ── 索引构建 ──

    def index(self, chunks: List[Dict[str, Any]]) -> int:
        """将 chunks 加入 BM25 索引。返回当前总文档数。"""
        if not chunks:
            return self.doc_count

        for c in chunks:
            tokens = list(jieba.lcut(c.get("text", "")))
            self.tokenized.append(tokens)
            self.corpus.append(c)

        self.doc_count = len(self.tokenized)
        total_len = sum(len(t) for t in self.tokenized)
        self.avgdl = total_len / max(self.doc_count, 1)

        # 重建词频表
        self.doc_freqs.clear()
        for tokens in self.tokenized:
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        logger.info(f"BM25 索引已构建: {self.doc_count} 篇文档, avgdl={self.avgdl:.1f}")
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
        """BM25 检索，返回带 bm25_score 的 chunk 列表。"""
        if self.doc_count == 0:
            return []

        query_tokens = list(jieba.lcut(query))

        scored: List[tuple[int, float]] = []
        for i, doc_tokens in enumerate(self.tokenized):
            # 标量过滤
            c = self.corpus[i]
            if source_filter and c.get("source", "") != source_filter:
                continue
            if doc_type_filter and c.get("doc_type", "") != doc_type_filter:
                continue
            if law_name_filter and c.get("law_name", "") != law_name_filter:
                continue
            if authority_filter and c.get("authority", "") != authority_filter:
                continue
            if status_filter and c.get("status", "") != status_filter:
                continue

            s = self._bm25_score(query_tokens, doc_tokens)
            if s > 0:
                scored.append((i, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        results: List[Dict[str, Any]] = []
        for idx, bm25_score in top:
            item = dict(self.corpus[idx])
            item["bm25_score"] = round(bm25_score, 6)
            results.append(item)
        return results

    def _bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        score = 0.0
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        for token in query_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            tf = doc_tokens.count(token)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        return score

    # ── 持久化 ──

    def save(self, path: str):
        """保存 BM25 索引到磁盘。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1, "b": self.b,
            "corpus": self.corpus,
            "tokenized": self.tokenized,
            "doc_freqs": self.doc_freqs,
            "avgdl": self.avgdl, "doc_count": self.doc_count,
        }
        with open(p, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 索引已保存: {path} ({self.doc_count} 篇)")

    @classmethod
    def load(cls, path: str) -> Optional["BM25Index"]:
        """从磁盘加载 BM25 索引。"""
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"BM25 索引加载失败: {e}")
            return None

        inst = cls(k1=data["k1"], b=data["b"])
        inst.corpus = data["corpus"]
        inst.tokenized = data["tokenized"]
        inst.doc_freqs = data["doc_freqs"]
        inst.avgdl = data["avgdl"]
        inst.doc_count = data["doc_count"]
        logger.info(f"BM25 索引已加载: {path} ({inst.doc_count} 篇)")
        return inst
