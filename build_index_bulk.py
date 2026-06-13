"""
build_index_bulk.py
批量建索引：将 src/txt_files/ 下全部 83 部金融法律写入 Milvus + BM25
用法: py -3 build_index_bulk.py
"""
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from rag_finance_system.src.bm25_index import BM25Index
from rag_finance_system.src.document_processor import DocumentProcessor
from rag_finance_system.src.embedder import Embedder
from rag_finance_system.src.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).parent
TXT_DIR = PROJECT_ROOT / "rag_finance_system" / "src" / "txt_files"
BM25_PATH = str(PROJECT_ROOT / "rag_finance_system" / "data" / "bm25_index.pkl")


def main():
    txt_files = sorted(glob.glob(str(TXT_DIR / "*.txt")))
    if not txt_files:
        logger.error(f"未找到 txt 文件: {TXT_DIR}")
        sys.exit(1)

    logger.info(f"找到 {len(txt_files)} 个 txt 文件")

    # 初始化
    logger.info("初始化 Embedding 模型...")
    embedder = Embedder()

    logger.info("初始化 Milvus...")
    vector_store = VectorStore()

    # 检查是否已有数据
    stats = vector_store.get_collection_stats()
    existing = stats.get("count", 0)
    if existing > 0:
        logger.warning(f"Milvus 中已有 {existing} 条记录，将追加新数据")

    logger.info("初始化 BM25...")
    bm25 = BM25Index()

    processor = DocumentProcessor()

    total_chunks = 0
    t0 = time.perf_counter()

    for i, fp in enumerate(txt_files, 1):
        fname = Path(fp).name
        logger.info(f"[{i}/{len(txt_files)}] 处理: {fname}")

        try:
            chunks = processor.process_file(fp, doc_type="law")
        except Exception as e:
            logger.error(f"  解析失败: {e}")
            continue

        if not chunks:
            logger.warning("  解析结果为空，跳过")
            continue

        # Embedding
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode_documents(texts, batch_size=16)

        # 写入 Milvus
        try:
            vector_store.insert(chunks, embeddings)
        except Exception as e:
            logger.error(f"  Milvus 写入失败: {e}")
            continue

        # 写入 BM25
        try:
            bm25.index(chunks)
        except Exception as e:
            logger.error(f"  BM25 写入失败: {e}")

        total_chunks += len(chunks)
        logger.info(f"  写入 {len(chunks)} chunks")

    # 保存 BM25
    Path(BM25_PATH).parent.mkdir(parents=True, exist_ok=True)
    bm25.save(BM25_PATH)
    logger.info(f"BM25 索引已保存: {BM25_PATH}")

    elapsed = time.perf_counter() - t0
    logger.info("===== 完成 =====")
    logger.info(f"处理 {len(txt_files)} 个文件, 共 {total_chunks} chunks")
    logger.info(f"总耗时: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    # 验证
    stats = vector_store.get_collection_stats()
    logger.info(f"Milvus 总记录数: {stats.get('count', 0)}")
    logger.info(f"BM25 文档数: {bm25.doc_count}")


if __name__ == "__main__":
    main()
