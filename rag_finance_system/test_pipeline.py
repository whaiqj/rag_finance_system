"""
test_pipeline.py
全链路闭环验证脚本
功能：PDF/TXT解析 → Embedding → Milvus存储 → 检索(向量+ES/BM25+RRF) → LLM问答
用法：python test_pipeline.py --file data/raw/your_file.pdf --query "你的问题"
"""

import sys
import argparse
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_finance_system.src.document_processor import DocumentProcessor
from rag_finance_system.src.embedder import Embedder, Reranker
from rag_finance_system.src.vector_store import VectorStore
from rag_finance_system.src.retriever import Retriever
from rag_finance_system.src.llm import get_llm
from rag_finance_system.src.rag_chain import RAGChain
from rag_finance_system.src.bm25_index import BM25Index
from rag_finance_system.src.dictionary import FinanceDictionary
from loguru import logger


def build_index(file_path: str, embedder: Embedder, vector_store: VectorStore,
               bm25: BM25Index = None, es_index=None, graph_builder=None):
    """将PDF或TXT解析并写入 Milvus + BM25 + ES + 知识图谱"""
    logger.info("===== Step 1: 文档解析 =====")
    processor = DocumentProcessor()
    chunks = processor.process_file(file_path)
    logger.info(f"解析出 {len(chunks)} 个chunk")

    logger.info("===== Step 2: Embedding =====")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_documents(texts, batch_size=16)
    logger.info(f"生成 {len(embeddings)} 个向量，维度={len(embeddings[0])}")

    logger.info("===== Step 3: 写入Milvus =====")
    count = vector_store.insert(chunks, embeddings)
    stats = vector_store.get_collection_stats()
    total_rows = stats.get("count", stats.get("row_count", count))
    logger.info(f"Milvus中共 {total_rows} 条记录")

    if bm25:
        logger.info("===== Step 3b: 写入BM25 =====")
        bm25.index(chunks)
        bm25.save("rag_finance_system/data/bm25_index.pkl")

    if es_index and es_index._connected:
        logger.info("===== Step 3c: 写入ES =====")
        es_index.index(chunks)
        es_index.save("rag_finance_system/data/es_index.txt")

    if graph_builder and graph_builder.kg._connected:
        logger.info("===== Step 3d: 写入知识图谱 =====")
        graph_builder.sync_dictionary_to_graph()
        graph_builder.build_from_chunks(chunks)


def run_query(query: str, rag_chain: RAGChain):
    """执行问答并打印结果"""
    logger.info(f"===== Step 4: 问答检索 =====")
    result = rag_chain.query(query)

    print("\n" + "=" * 60)
    print(f"问题: {result['question']}")
    print("=" * 60)
    print(f"答案:\n{result['answer']}")
    print("=" * 60)
    print(f"可信度: {result['confidence']['total']:.1%}  "
          f"(检索相关性: {result['confidence']['retrieval']:.1%}, "
          f"覆盖度: {result['confidence']['coverage']:.1%})")
    print("=" * 60)
    print("溯源条文:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] 【{src['source']} {src['article_num']}】 "
              f"(相关度: {src['score']:.4f})")
        print(f"      {src['text'][:100]}...")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="RAG金融知识系统 - 全链路闭环验证")
    parser.add_argument("--file", "--pdf", dest="file", type=str, help="PDF或TXT文件路径（首次运行需要）")
    parser.add_argument("--query", type=str, default="什么是资本充足率？", help="查询问题")
    parser.add_argument("--no-reranker", action="store_true", help="跳过Reranker（节省显存）")
    parser.add_argument("--no-es", action="store_true", help="跳过ES（仅用BM25）")
    parser.add_argument("--no-kg", action="store_true", help="跳过知识图谱")
    parser.add_argument("--no-dict", action="store_true", help="跳过金融词典")
    parser.add_argument("--api", action="store_true", help="使用API模式（不加载本地LLM）")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # ===== 初始化组件 =====
    logger.info("初始化Embedding模型...")
    embedder = Embedder()

    logger.info("初始化向量数据库...")
    vector_store = VectorStore()

    # Reranker（可选）
    reranker = None
    if not args.no_reranker:
        try:
            logger.info("初始化Reranker...")
            reranker = Reranker()
        except Exception as e:
            logger.warning(f"Reranker加载失败: {e}，跳过精排")

    # 金融词典
    dictionary = None
    if not args.no_dict:
        try:
            dict_path = str(data_dir / "finance_dictionary.json")
            logger.info("初始化金融词典...")
            dictionary = FinanceDictionary(dict_path=dict_path)
            logger.info(f"金融词典已加载: {dictionary.stats()}")
        except Exception as e:
            logger.warning(f"金融词典加载失败: {e}")

    # BM25 索引
    bm25_path = str(project_root / "rag_finance_system" / "data" / "bm25_index.pkl")
    bm25 = BM25Index.load(bm25_path)
    if bm25:
        logger.info(f"BM25 索引已加载: {bm25.doc_count} 篇")
    else:
        bm25 = BM25Index()
        logger.info("BM25 索引未找到，已创建空索引")

    # ES 索引
    es_index = None
    if not args.no_es:
        try:
            from rag_finance_system.src.es_index import ESIndex
            es_index = ESIndex()
            if es_index._connected:
                logger.info(f"ES 全文索引已就绪: {es_index.doc_count} 篇")
            else:
                logger.info("ES 不可用，将回退 BM25")
                es_index = None
        except Exception as e:
            logger.warning(f"ES 初始化失败，将回退 BM25: {e}")

    # 知识图谱
    kg = None
    graph_builder = None
    if not args.no_kg:
        try:
            from rag_finance_system.src.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            if kg._connected:
                from rag_finance_system.src.graph_builder import GraphBuilder
                graph_builder = GraphBuilder(kg=kg, dictionary=dictionary)
                graph_builder.sync_dictionary_to_graph()
                logger.info(f"知识图谱已就绪: {kg.stats()}")
            else:
                logger.info("Neo4j 不可用，图谱功能将跳过")
                kg = None
        except Exception as e:
            logger.warning(f"知识图谱初始化失败: {e}")

    # ===== 建立索引（如果提供了PDF或TXT） =====
    if args.file:
        build_index(args.file, embedder, vector_store, bm25, es_index, graph_builder)
    else:
        stats = vector_store.get_collection_stats()
        count = stats.get("count", stats.get("row_count", 0))
        if count == 0:
            logger.error("Milvus中没有数据！请先用 --file 参数加载PDF或TXT文件")
            logger.error("示例: python test_pipeline.py --file data/raw/your_file.pdf")
            sys.exit(1)
        logger.info(f"使用已有索引，共 {count} 条记录")

    # ===== 初始化LLM =====
    logger.info("初始化LLM...")
    llm = get_llm(prefer_local=not args.api)

    # ===== 构建RAG链路 =====
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25_index=bm25,
        es_index=es_index,
        reranker=reranker,
    )
    rag = RAGChain(
        retriever=retriever,
        llm=llm,
        dictionary=dictionary,
        knowledge_graph=kg,
    )

    # ===== 执行问答 =====
    run_query(args.query, rag)


if __name__ == "__main__":
    main()
