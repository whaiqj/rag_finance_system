"""
test_performance.py
检索链路性能诊断脚本 — 毫秒级计时，定位响应时间瓶颈
用法: python test_performance.py
目标: 全链路 ≤ 2秒
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_finance_system.src.bm25_index import BM25Index
from rag_finance_system.src.dictionary import FinanceDictionary
from rag_finance_system.src.embedder import Embedder, Reranker
from rag_finance_system.src.llm import get_llm
from rag_finance_system.src.rag_chain import RAGChain
from rag_finance_system.src.retriever import Retriever
from rag_finance_system.src.rewriter import QueryRewriter
from rag_finance_system.src.vector_store import VectorStore


# ---- 计时工具 ----
class StepTimer:
    def __init__(self):
        self.steps: list[dict] = []
        self._start = time.perf_counter()

    def measure(self, name: str) -> float:
        now = time.perf_counter()
        elapsed = (now - self._start) * 1000  # ms
        self.steps.append({"name": name, "elapsed_ms": round(elapsed, 1), "cumulative_ms": round(elapsed, 1)})
        self._start = now
        return elapsed

    def reset(self):
        self._start = time.perf_counter()
        self.steps.clear()

    def summary(self) -> list[dict]:
        cum = 0.0
        result = []
        for s in self.steps:
            cum += s["elapsed_ms"]
            result.append({"name": s["name"], "step_ms": s["elapsed_ms"], "cum_ms": round(cum, 1)})
        return result


# ---- 测试查询集 ----
TEST_QUERIES = [
    ("关键词检索-短", "资本充足率"),
    ("关键词检索-中", "不良贷款率的计算公式"),
    ("自然语言问答", "根据公司法，股东有哪些责任？"),
    ("自然语言问答-长", "商业银行在发放贷款时需要遵循哪些审慎经营规则？"),
    ("条文关联查询", "公司法第20条相关的规定有哪些？"),
    ("机构+法规组合", "上海银保监局关于防范非法集资的规定"),
]


def profile_single_query(
    rag: RAGChain,
    query: str,
    use_reranker: bool = True,
    use_query_rewrite: bool = True,
) -> dict:
    """对单条查询的每个环节逐一计时，返回耗时明细。"""
    timer = StepTimer()

    # 0. 词典实体检测
    entities = {"terms": [], "law_names": [], "authorities": []}
    if rag.dictionary:
        entities = rag.dictionary.detect_entities(query)
        timer.measure("词典实体检测")
        expanded_query = rag.dictionary.expand_query(query)
        timer.measure("词典查询扩展")
    else:
        timer.measure("词典实体检测(跳过)")
        expanded_query = query
        timer.measure("词典查询扩展(跳过)")

    # 0b. 文件名/法规名/机构名检测
    source_filter = rag._detect_source(query)
    law_name_filter = entities.get("law_names", [None])[0] if entities.get("law_names") else None
    authority_filter = ",".join(entities.get("authorities", [])) if entities.get("authorities") else None

    if source_filter:
        law_name_filter = None
        authority_filter = None
    else:
        if not law_name_filter:
            law_name_filter = rag._detect_law_name(query)
        if not authority_filter:
            authority_filter = rag._detect_authority(query)
    timer.measure("文件名/法规名/机构检测")

    # 1. 查询重写
    rewritten_query = expanded_query
    if use_query_rewrite:
        rewritten_query = rag.rewrite_query(expanded_query)
        timer.measure("查询重写")
    else:
        timer.measure("查询重写(跳过)")

    # 1b. 知识图谱
    graph_articles: list = []
    graph_facts: list = []
    if rag.kg:
        try:
            authority_list = entities.get("authorities") if rag.dictionary else None
            graph_articles = rag.kg.get_related_articles(
                law_names=[law_name_filter] if law_name_filter else None,
                terms=entities.get("terms") if rag.dictionary else None,
                authorities=authority_list,
                max_results=5,
            )
            graph_facts = rag.kg.get_graph_facts(
                law_names=[law_name_filter] if law_name_filter else None,
                terms=entities.get("terms") if rag.dictionary else None,
                authorities=authority_list,
                max_results=6,
            )
        except Exception:
            pass
    timer.measure("知识图谱扩展")

    # 2. 检索
    chunks = rag.retriever.retrieve(
        query=rewritten_query,
        top_k=None,
        use_reranker=use_reranker,
        source_filter=source_filter,
        doc_type_filter=None,
        law_name_filter=law_name_filter,
        authority_filter=authority_filter,
        status_filter="有效",
    )
    timer.measure("检索(向量+BM25+RRF+Reranker)")

    # 合并图谱结果
    seen_chunk_ids = {c.get("chunk_id", "") for c in chunks}
    for ga in graph_articles:
        gid = ga.get("chunk_id", "")
        if gid and gid not in seen_chunk_ids:
            seen_chunk_ids.add(gid)
            chunks.append({
                "text": ga.get("text", ""),
                "source": ga.get("source", ""),
                "article_num": ga.get("article_num", ""),
                "doc_type": "law",
                "law_name": ga.get("law_name", ""),
                "score": 0.0,
                "_graph_relation": ga.get("relation", ""),
            })

    # 3. Prompt 构建
    from rag_finance_system.src.rag_chain import build_prompt
    messages = build_prompt(query, chunks, graph_facts=graph_facts)
    timer.measure("Prompt构建")

    # 4. LLM 生成
    answer = rag.llm.generate(messages, max_new_tokens=1024)
    timer.measure("LLM生成")

    # 5. 溯源 + 可信度
    [
        {
            "source": c.get("source", ""),
            "article_num": c.get("article_num", ""),
            "text": c.get("text", "")[:300],
            "score": round(c.get("reranker_score", c.get("score", 0.0)), 4),
        }
        for c in chunks
    ]
    confidence = rag.retriever.compute_confidence(query, answer, chunks)
    timer.measure("溯源+可信度")

    steps = timer.summary()
    total_ms = steps[-1]["cum_ms"] if steps else 0

    return {
        "query": query,
        "total_ms": total_ms,
        "steps": steps,
        "answer_len": len(answer),
        "chunks_count": len(chunks),
        "confidence": confidence["total"],
    }


def profile_retriever_internals(
    rag: RAGChain,
    query: str,
    rounds: int = 3,
) -> dict:
    """深入 Retriever 内部，对向量/BM25/术语/RRF/Reranker 各自计时。"""
    from rag_finance_system.src.retriever import _rrf_fusion

    StepTimer()
    retriever = rag.retriever

    # 准备阶段
    rewritten_query = query
    if rag.rewriter:
        try:
            rewritten_query = rag.rewriter.rewrite(query).strip() or query
        except Exception:
            pass

    vec_times = []
    ft_times = []
    term_times = []
    rrf_times = []
    reranker_times = []

    for _ in range(rounds):
        # Embedding
        t0 = time.perf_counter()
        query_vector = retriever.embedder.encode_query(rewritten_query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # 向量召回
        t0 = time.perf_counter()
        vec_results = retriever.vector_store.search(
            query_vector=query_vector,
            top_k=retriever.top_k,
            status_filter="有效",
        )
        vec_ms = (time.perf_counter() - t0) * 1000
        vec_times.append(vec_ms)

        # 全文召回
        t0 = time.perf_counter()
        ft_results = []
        ft_backend = None
        if retriever.es_index and retriever.es_index.doc_count > 0:
            ft_backend = retriever.es_index
        elif retriever.bm25_index and retriever.bm25_index.doc_count > 0:
            ft_backend = retriever.bm25_index
        if ft_backend:
            ft_results = ft_backend.search(query=rewritten_query, top_k=retriever.bm25_top_k, status_filter="有效")
        ft_ms = (time.perf_counter() - t0) * 1000
        ft_times.append(ft_ms)

        # 术语索引
        t0 = time.perf_counter()
        if retriever.term_index and retriever.term_index.doc_count > 0:
            term_results = retriever.term_index.search(query=rewritten_query, top_k=retriever.bm25_top_k, status_filter="有效")
        else:
            term_results = []
        term_ms = (time.perf_counter() - t0) * 1000
        term_times.append(term_ms)

        # RRF 融合
        t0 = time.perf_counter()
        recall_lists = [vec_results]
        if ft_results:
            recall_lists.append(ft_results)
        if term_results:
            recall_lists.append(term_results)
        if len(recall_lists) >= 2:
            candidates = _rrf_fusion(*recall_lists, top_k=retriever.top_k)
        else:
            candidates = recall_lists[0]
        rrf_ms = (time.perf_counter() - t0) * 1000
        rrf_times.append(rrf_ms)

        # Reranker
        t0 = time.perf_counter()
        if retriever.reranker and candidates:
            texts = [c["text"] for c in candidates]
            retriever.reranker.rerank(query=rewritten_query, documents=texts, top_n=retriever.reranker_top_n)
        rerank_ms = (time.perf_counter() - t0) * 1000
        reranker_times.append(rerank_ms)

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    return {
        "embedding_ms": round(embed_ms, 1),
        "vector_avg_ms": avg(vec_times),
        "fulltext_avg_ms": avg(ft_times),
        "term_avg_ms": avg(term_times),
        "rrf_avg_ms": avg(rrf_times),
        "reranker_avg_ms": avg(reranker_times),
        "vector_details": [round(v, 1) for v in vec_times],
        "ft_details": [round(v, 1) for v in ft_times],
        "reranker_details": [round(v, 1) for v in reranker_times],
    }


def format_table(headers: list[str], rows: list[list[str]], col_widths: list[int]) -> str:
    """简单表格格式化。"""
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [sep]

    def pad_row(cells):
        return "| " + " | ".join(c.ljust(w) if i > 0 else c.rjust(w)
                                  for i, (c, w) in enumerate(zip(cells, col_widths, strict=False))) + " |"

    lines.append(pad_row(headers))
    lines.append(sep.replace("-", "="))
    for row in rows:
        lines.append(pad_row(row))
        lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RAG检索链路性能诊断")
    parser.add_argument("--no-reranker", action="store_true", help="跳过Reranker")
    parser.add_argument("--no-rewrite", action="store_true", help="跳过查询重写")
    parser.add_argument("--mock-llm", action="store_true", help="使用假的LLM（只测检索，不管LLM是否可用）")
    parser.add_argument("--rounds", type=int, default=2, help="每条查询重复轮数")
    parser.add_argument("--deep", action="store_true", help="深入Retriever内部计时")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    rag_data_dir = project_root / "rag_finance_system" / "data"

    # ===== 初始化 =====
    print("=" * 70)
    print("   RAG 检索链路性能诊断")
    print("=" * 70)

    print("\n[1/6] 初始化 Embedding 模型 (bge-small-zh-v1.5)...")
    t0 = time.perf_counter()
    embedder = Embedder()
    print(f"      耗时: {(time.perf_counter() - t0) * 1000:.0f}ms, 设备: {embedder.device}")

    print("[2/6] 初始化 Milvus 向量库...")
    vector_store = VectorStore()
    stats = vector_store.get_collection_stats()
    print(f"      记录数: {stats.get('count', 0)}")

    print("[3/6] 初始化 BM25 全文索引...")
    bm25_path = str(rag_data_dir / "bm25_index.pkl") if rag_data_dir.exists() else ""
    if bm25_path and Path(bm25_path).exists():
        bm25 = BM25Index.load(bm25_path)
        print(f"      已加载: {bm25.doc_count} 篇")
    else:
        bm25 = BM25Index()
        print("      空索引")

    print("[4/6] 初始化金融词典...")
    dict_path = str(data_dir / "finance_dictionary.json")
    dictionary = FinanceDictionary(dict_path=dict_path) if Path(dict_path).exists() else None
    if dictionary:
        print(f"      {dictionary.stats()}")

    print("[5/6] 初始化查询重写器 & Reranker...")
    rewriter = None
    try:
        rewriter = QueryRewriter()
        print("      重写器已加载")
    except Exception as e:
        print(f"      重写器跳过: {e}")

    reranker = None
    if not args.no_reranker:
        try:
            reranker = Reranker()
            print(f"      Reranker已加载, 设备: {reranker.device}")
        except Exception as e:
            print(f"      Reranker跳过: {e}")

    print("[6/6] 初始化 LLM...")
    if args.mock_llm:
        # 假的 LLM，用于仅测检索链路
        class MockLLM:
            def generate(self, messages, max_new_tokens=1024, temperature=0.1):
                return "（模拟LLM回答）根据相关法规条文，请参考检索结果中的具体条款。"
        llm = MockLLM()
        print("      使用 Mock LLM（仅测检索）")
    else:
        try:
            llm = get_llm(prefer_local=True)
            print("      LLM已就绪")
        except Exception as e:
            print(f"      LLM初始化失败: {e}")
            print("      自动切换到 Mock LLM（仅测检索链路）")
            class MockLLM:
                def generate(self, messages, max_new_tokens=1024, temperature=0.1):
                    return "（模拟回答）"
            llm = MockLLM()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25,
        reranker=reranker,
    )
    rag = RAGChain(
        retriever=retriever,
        llm=llm,
        rewriter=rewriter,
        dictionary=dictionary,
    )

    # ===== 全链路计时 =====
    print("\n" + "=" * 70)
    print("   全链路性能测试 (每条查询跑 {} 轮)".format(args.rounds))
    print("=" * 70)

    all_results = []
    for label, query in TEST_QUERIES:
        best_total = float("inf")
        best_result = None
        for _rnd in range(args.rounds):
            result = profile_single_query(
                rag, query,
                use_reranker=not args.no_reranker,
                use_query_rewrite=not args.no_rewrite,
            )
            if result["total_ms"] < best_total:
                best_total = result["total_ms"]
                best_result = result

        all_results.append((label, best_result))

        status = "OK" if best_total <= 2000 else "SLOW"
        print(f"\n  [{label}] \"{query[:40]}...\"  →  {best_total:.0f}ms [{status}]  "
              f"ans={best_result['answer_len']}chars  chunks={best_result['chunks_count']}  "
              f"conf={best_result['confidence']:.1%}")

        # 打印各环节明细
        step_rows = []
        for s in best_result["steps"]:
            pct = s["step_ms"] / best_total * 100 if best_total > 0 else 0
            bar = "█" * int(pct / 5) + ("▏" if pct % 5 >= 2.5 else "")
            step_rows.append([s["name"], f"{s['step_ms']:.0f}ms", f"{pct:.0f}%", bar])
        print(format_table(
            ["环节", "耗时", "占比", ""],
            step_rows,
            [22, 10, 6, 20],
        ))

    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print("   汇总对比")
    print("=" * 70)
    header = ["查询类型", "总耗时", "检索环节", "LLM生成", "答案", "状态"]
    rows = []
    for label, r in all_results:
        retrieval_ms = 0
        llm_ms = 0
        for s in r["steps"]:
            if "检索" in s["name"]:
                retrieval_ms = s["step_ms"]
            if "LLM" in s["name"]:
                llm_ms = s["step_ms"]

        total = r["total_ms"]
        status = "PASS(<=2s)" if total <= 2000 else f"FAIL(+{total - 2000:.0f}ms)"
        rows.append([label, f"{total:.0f}ms", f"{retrieval_ms:.0f}ms",
                     f"{llm_ms:.0f}ms", f"{r['answer_len']}chars", status])

    print(format_table(header, rows, [16, 10, 10, 10, 10, 16]))

    # ===== 可选：深入 Retriever 内部 =====
    if args.deep:
        print("\n" + "=" * 70)
        print("   深入 Retriever 内部计时 (每轮 {} 次取均值)".format(args.rounds))
        print("=" * 70)

        deep_results = []
        for label, query in TEST_QUERIES[:3]:  # 只测前3个
            d = profile_retriever_internals(rag, query, rounds=args.rounds)
            deep_results.append((label, d))

        header2 = ["查询", "Embed", "向量检索", "全文检索", "术语索引", "RRF融合", "Reranker"]
        rows2 = []
        for label, d in deep_results:
            d["vector_avg_ms"] + d["fulltext_avg_ms"] + d["term_avg_ms"] + d["rrf_avg_ms"] + d["reranker_avg_ms"]
            rows2.append([
                label[:12],
                f"{d['embedding_ms']:.0f}ms",
                f"{d['vector_avg_ms']:.0f}ms",
                f"{d['fulltext_avg_ms']:.0f}ms",
                f"{d['term_avg_ms']:.0f}ms",
                f"{d['rrf_avg_ms']:.0f}ms",
                f"{d['reranker_avg_ms']:.0f}ms",
            ])

        print(format_table(header2, rows2, [12, 8, 10, 10, 10, 10, 10]))

    # ===== 诊断建议 =====
    print("\n" + "=" * 70)
    print("   诊断建议")
    print("=" * 70)
    slow_count = sum(1 for _, r in all_results if r["total_ms"] > 2000)
    if slow_count == 0:
        print("  所有查询均在 2秒以内，无需优化。")
    else:
        # 找出最大瓶颈
        bottlenecks = {}
        for _, r in all_results:
            for s in r["steps"]:
                name = s["name"]
                bottlenecks[name] = max(bottlenecks.get(name, 0), s["step_ms"])
        sorted_b = sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)

        print(f"  {slow_count}/{len(all_results)} 查询超过 2秒限制\n")
        print("  Top 耗时环节:")
        for name, ms in sorted_b[:5]:
            bar = "█" * int(ms / 50)
            print(f"    {name:30s}  {ms:6.0f}ms  {bar}")

        print("\n  针对性优化方向:")
        if any("Reranker" in name and ms > 500 for name, ms in sorted_b):
            print("  - Reranker 耗时过高 → 考虑 GPU 推理 / 减小候选集 / 可选跳过")
        if any("LLM" in name and ms > 1000 for name, ms in sorted_b):
            print("  - LLM 生成过慢 → 换用更快 API / 减少 max_new_tokens")
        if any("重写" in name and ms > 300 for name, ms in sorted_b):
            print("  - 查询重写过慢 → 检查模型路径 / 跳过重写")
        if any("向量" in name or "BM25" in name and ms > 200 for name, ms in sorted_b):
            print("  - 检索过慢 → 检查 Milvus 索引 / 减少 OR-过滤器循环查询")


if __name__ == "__main__":
    main()
