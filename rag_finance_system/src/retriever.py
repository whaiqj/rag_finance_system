"""
retriever.py
检索模块：向量检索 + BM25关键词检索 → RRF融合 → Reranker精排
"""

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from loguru import logger

from .embedder import Embedder, Reranker
from .vector_store import VectorStore

load_dotenv()

TOP_K = int(os.getenv("RETRIEVER_TOP_K", 10))
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", 5))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 10))
RRF_K = int(os.getenv("RRF_K", 60))


def _rrf_fusion(
    *candidate_lists: List[Dict[str, Any]],
    key: str = "chunk_id",
    k: int = RRF_K,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """RRF (Reciprocal Rank Fusion) 融合多路召回结果。

    score(d) = sum_{r in rankers} 1 / (k + rank_r(d))
    支持 1~N 路召回列表，first-write-wins 元数据保留策略。
    """
    rrf_scores: Dict[str, float] = {}
    merged: Dict[str, Dict[str, Any]] = {}

    for candidates in candidate_lists:
        for rank, c in enumerate(candidates, start=1):
            cid = c.get(key, "")
            if not cid:
                continue
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in merged:
                merged[cid] = c

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results: List[Dict[str, Any]] = []
    for cid in sorted_ids[:top_k]:
        item = dict(merged[cid])
        item["rrf_score"] = round(rrf_scores[cid], 6)
        results.append(item)
    return results


class Retriever:
    """检索器：向量 + 全文检索(ES/BM25) + 术语倒排索引 多路召回 → RRF 融合 → Reranker 精排"""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        reranker: Optional[Reranker] = None,
        bm25_index: Optional[Any] = None,
        es_index: Optional[Any] = None,
        term_index: Optional[Any] = None,
        top_k: int = TOP_K,
        bm25_top_k: int = BM25_TOP_K,
        reranker_top_n: int = RERANKER_TOP_N,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.reranker = reranker
        self.bm25_index = bm25_index
        self.es_index = es_index
        self.term_index = term_index
        self.top_k = top_k
        self.bm25_top_k = bm25_top_k
        self.reranker_top_n = reranker_top_n

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
        source_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        law_name_filter: Optional[str] = None,
        authority_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """端到端检索：多路召回 → RRF融合 → Reranker精排"""
        k = top_k or self.top_k

        logger.info(f"检索: {query[:50]}...")

        # Step 1: 查询 Embedding
        query_vector = self.embedder.encode_query(query)

        # 内容过滤器拆为 OR 查询（多值字段分多次查询后合并去重）
        content_filters: list[dict] = []
        if law_name_filter:
            content_filters.append({"law_name": law_name_filter})
        if authority_filter:
            for _auth in authority_filter.split(","):
                _auth = _auth.strip()
                if _auth:
                    content_filters.append({"authority": _auth})
        if source_filter:
            content_filters.append({"source": source_filter})
        if not content_filters:
            content_filters.append({})

        # Step 2a: 向量召回
        all_vec: dict[str, dict] = {}
        for cf in content_filters:
            where = {**cf}
            if doc_type_filter:
                where["doc_type"] = doc_type_filter

            batch = self.vector_store.search(
                query_vector=query_vector,
                top_k=k,
                source_filter=where.get("source"),
                doc_type_filter=where.get("doc_type"),
                law_name_filter=where.get("law_name"),
                authority_filter=where.get("authority"),
            )
            for c in batch:
                cid = c.get("chunk_id", "")
                if cid and cid not in all_vec:
                    all_vec[cid] = c

        vec_candidates = sorted(
            all_vec.values(), key=lambda x: x.get("score", 0.0), reverse=True
        )[:k]
        logger.info(f"向量召回 {len(vec_candidates)} 条")

        # Step 2b: 全文检索召回（ES 优先 → BM25 回退 → 纯向量）
        if self.es_index and self.es_index.doc_count > 0:
            ft_backend: Any = self.es_index
            logger.info("使用 ES 全文检索")
        elif self.bm25_index and self.bm25_index.doc_count > 0:
            ft_backend = self.bm25_index
            logger.info("使用 BM25 关键词检索")
        else:
            ft_backend = None

        ft_candidates: List[Dict[str, Any]] = []
        if ft_backend is not None:
            all_ft: dict[str, dict] = {}
            for cf in content_filters:
                where = {**cf}
                if doc_type_filter:
                    where["doc_type"] = doc_type_filter

                try:
                    batch = ft_backend.search(
                        query=query,
                        top_k=self.bm25_top_k,
                        source_filter=where.get("source"),
                        doc_type_filter=where.get("doc_type"),
                        law_name_filter=where.get("law_name"),
                        authority_filter=where.get("authority"),
                    )
                except Exception as e:
                    logger.warning(f"全文检索后端查询失败: {e}")
                    batch = []
                for c in batch:
                    cid = c.get("chunk_id", "")
                    if cid and cid not in all_ft:
                        all_ft[cid] = c

            ft_candidates = sorted(
                all_ft.values(), key=lambda x: x.get("bm25_score", 0.0), reverse=True
            )[:k]
            logger.info(f"全文检索召回 {len(ft_candidates)} 条")

        # Step 2c: 术语倒排索引召回
        term_candidates: List[Dict[str, Any]] = []
        if self.term_index is not None and self.term_index.doc_count > 0:
            try:
                term_candidates_raw = self.term_index.search(
                    query=query,
                    top_k=self.bm25_top_k,
                    source_filter=source_filter,
                    doc_type_filter=doc_type_filter,
                    law_name_filter=law_name_filter,
                    authority_filter=authority_filter,
                )
                term_candidates = sorted(
                    term_candidates_raw,
                    key=lambda x: x.get("term_score", 0.0),
                    reverse=True,
                )[:k]
                logger.info(f"术语索引召回 {len(term_candidates)} 条")
            except Exception as e:
                logger.warning(f"术语索引查询失败: {e}")

        # Step 2d: RRF 多路融合
        recall_lists: list = [vec_candidates]
        if ft_candidates:
            recall_lists.append(ft_candidates)
        if term_candidates:
            recall_lists.append(term_candidates)

        if len(recall_lists) >= 2:
            candidates = _rrf_fusion(*recall_lists, top_k=k)
            logger.info(f"RRF 融合 ({len(recall_lists)} 路) 后 {len(candidates)} 条")
        else:
            candidates = recall_lists[0]

        if not candidates:
            return []

        # Step 3: Reranker 精排（可选）
        if use_reranker and self.reranker:
            texts = [c["text"] for c in candidates]
            reranked = self.reranker.rerank(
                query=query,
                documents=texts,
                top_n=self.reranker_top_n,
            )
            results: List[Dict[str, Any]] = []
            for r in reranked:
                orig = candidates[r["index"]]
                orig["reranker_score"] = r["score"]
                results.append(orig)
            logger.info(f"Reranker 精排后保留 {len(results)} 条")
            return results
        else:
            return candidates[:self.reranker_top_n]

    def compute_confidence(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """计算答案可信度"""
        if not retrieved_chunks:
            return {"total": 0.0, "retrieval": 0.0, "coverage": 0.0}

        # 取最高分：优先 reranker_score → rrf_score → score
        retrieval_score = max(
            c.get("reranker_score", c.get("rrf_score", c.get("score", 0.0)))
            for c in retrieved_chunks
        )

        answer_lower = answer.lower()
        matched = sum(
            1 for c in retrieved_chunks
            if any(word in answer_lower for word in c["text"][:100].lower().split()[:10])
        )
        coverage_score = min(matched / max(len(retrieved_chunks), 1), 1.0)

        total = 0.6 * retrieval_score + 0.4 * coverage_score

        return {
            "total": round(total, 3),
            "retrieval": round(retrieval_score, 3),
            "coverage": round(coverage_score, 3),
        }
