"""
term_index.py
术语精确倒排索引 — 将金融词典的 48 个规范术语与文档 chunk 建立倒排映射。

索引时：用词典别名匹配检测每个 chunk 包含哪些规范术语。
检索时：检测查询中的术语 → 查倒排表 → 返回匹配 chunk。

接口仿照 BM25Index：index() / search() / save() / load() / doc_count
"""

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger


class TermIndex:
    """术语级倒排索引：canonical_term → {corpus_index, ...}"""

    def __init__(self, dictionary=None):
        self._corpus: List[Dict[str, Any]] = []
        self._inverted_index: Dict[str, Set[int]] = {}
        self._term_df: Dict[str, int] = {}
        self.doc_count: int = 0
        self._dictionary = dictionary

    # ── 术语检测 ──

    def _detect_terms_in_text(self, text: str) -> Set[str]:
        """检测文本中包含的规范金融术语（复用词典别名→规范名映射）。"""
        if self._dictionary is None:
            return set()

        detected: Dict[str, int] = {}  # canonical → alias_len (保留最长匹配)
        text_lower = text.lower()

        sorted_aliases = sorted(
            self._dictionary._term_alias_to_canonical.items(),
            key=lambda x: -len(x[0]),
        )
        for alias, canonical in sorted_aliases:
            if alias in text_lower:
                if canonical not in detected or len(alias) > detected[canonical]:
                    detected[canonical] = len(alias)

        return set(detected.keys())

    # ── 索引构建 ──

    def index(self, chunks: List[Dict[str, Any]]) -> int:
        """对 chunk 列表构建术语倒排索引。返回 doc_count。"""
        if self._dictionary is None:
            logger.warning("TermIndex: 词典未加载，跳过索引构建")
            return 0

        start_idx = len(self._corpus)
        self._corpus.extend(chunks)

        for i, chunk in enumerate(chunks):
            corpus_idx = start_idx + i
            text = chunk.get("text", "")
            terms = self._detect_terms_in_text(text)
            for term in terms:
                if term not in self._inverted_index:
                    self._inverted_index[term] = set()
                self._inverted_index[term].add(corpus_idx)

        self._term_df = {t: len(idxs) for t, idxs in self._inverted_index.items()}
        self.doc_count = len(self._corpus)
        logger.info(
            f"TermIndex 已索引: {self.doc_count} chunks, "
            f"{len(self._inverted_index)} 个术语, "
            f"共 {sum(self._term_df.values())} 条映射"
        )
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
        """检索包含查询术语的 chunk。

        1. 从 query 检测金融术语
        2. 查倒排表，按 (match_count, idf_sum) 降序
        3. 应用标量过滤
        4. 返回 top_k，每条注入 term_score
        """
        if self._dictionary is None:
            return []

        entities = self._dictionary.detect_entities(query)
        query_terms = entities.get("terms", [])
        if not query_terms:
            return []

        # 累加评分：chunk_scores[corpus_idx] = [match_count, idf_sum]
        chunk_scores: Dict[int, list] = {}

        for term in query_terms:
            postings = self._inverted_index.get(term, set())
            df = self._term_df.get(term, 0)
            idf = math.log(1.0 + self.doc_count / max(df, 1))

            for idx in postings:
                if idx not in chunk_scores:
                    chunk_scores[idx] = [0, 0.0]
                chunk_scores[idx][0] += 1
                chunk_scores[idx][1] += idf

        # 排序 + 标量过滤
        scored = []
        for idx, (match_count, idf_sum) in chunk_scores.items():
            chunk = self._corpus[idx]

            # 标量过滤
            if source_filter and chunk.get("source", "") != source_filter:
                continue
            if doc_type_filter and chunk.get("doc_type", "") != doc_type_filter:
                continue
            if law_name_filter and chunk.get("law_name", "") != law_name_filter:
                continue
            if authority_filter and chunk.get("authority", "") != authority_filter:
                continue
            if status_filter and chunk.get("status", "") != status_filter:
                continue

            scored.append((match_count, idf_sum, chunk))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        results = []
        for match_count, idf_sum, chunk in scored[:top_k]:
            item = dict(chunk)
            item["term_score"] = round(match_count + 0.001 * idf_sum, 6)
            results.append(item)

        return results

    # ── 持久化 ──

    def save(self, path: str):
        """pickle 序列化到磁盘（不含 _dictionary 引用）。"""
        data = {
            "corpus": self._corpus,
            "inverted_index": {k: list(v) for k, v in self._inverted_index.items()},
            "term_df": self._term_df,
            "doc_count": self.doc_count,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"TermIndex 已保存到 {path}")

    @classmethod
    def load(cls, path: str) -> Optional["TermIndex"]:
        """从 pickle 文件加载。_dictionary 需调用方后续设置。"""
        if not Path(path).exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"TermIndex 加载失败: {e}")
            return None

        obj = cls(dictionary=None)
        obj._corpus = data.get("corpus", [])
        obj._inverted_index = {k: set(v) for k, v in data.get("inverted_index", {}).items()}
        obj._term_df = data.get("term_df", {})
        obj.doc_count = data.get("doc_count", 0)
        logger.info(f"TermIndex 已加载: {obj.doc_count} chunks, {len(obj._inverted_index)} 术语")
        return obj