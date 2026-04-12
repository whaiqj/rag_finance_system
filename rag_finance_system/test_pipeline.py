"""
test_pipeline.py
最小闭环验证脚本
功能：PDF/TXT解析 → Embedding → Milvus存储 → 检索 → LLM问答
用法：python test_pipeline.py --file data/raw/your_file.pdf --query "你的问题"
"""

import sys
import argparse
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent))

from rag_finance_system.src.document_processor import DocumentProcessor
from rag_finance_system.src.embedder import Embedder, Reranker
from rag_finance_system.src.vector_store import VectorStore
from rag_finance_system.src.retriever import Retriever
from rag_finance_system.src.llm import get_llm
from rag_finance_system.src.rag_chain import RAGChain
from loguru import logger


def build_index(file_path: str, embedder: Embedder, vector_store: VectorStore):
    """将PDF或TXT解析并写入Milvus"""
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
    logger.info(f"Milvus中共 {stats.get('row_count', count)} 条记录")


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
    parser = argparse.ArgumentParser(description="RAG金融知识系统 - 最小闭环验证")
    parser.add_argument("--file", "--pdf", dest="file", type=str, help="PDF或TXT文件路径（首次运行需要）")
    parser.add_argument("--query", type=str, default="什么是资本充足率？", help="查询问题")
    parser.add_argument("--no-reranker", action="store_true", help="跳过Reranker（节省显存）")
    parser.add_argument("--api", action="store_true", help="使用API模式（不加载本地LLM）")
    args = parser.parse_args()

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

    # ===== 建立索引（如果提供了PDF或TXT） =====
    if args.file:
        build_index(args.file, embedder, vector_store)
    else:
        stats = vector_store.get_collection_stats()
        count = stats.get("row_count", 0)
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
        reranker=reranker,
    )
    rag = RAGChain(retriever=retriever, llm=llm)

    # ===== 执行问答 =====
    run_query(args.query, rag)


if __name__ == "__main__":
    main()
