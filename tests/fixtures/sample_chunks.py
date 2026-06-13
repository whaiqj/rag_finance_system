"""测试用 chunk 工厂函数。"""

from typing import Any, Dict, List


def make_chunk(
    text: str = "中华人民共和国公司法 第一条 为了规范公司的组织和行为",
    source: str = "中华人民共和国公司法.txt",
    chunk_id: str = "test-chunk-001",
    article_num: str = "1",
    doc_type: str = "law",
    law_name: str = "中华人民共和国公司法",
    authority: str = "全国人民代表大会",
    status: str = "有效",
    effective_date: str = "2024-07-01",
    chunk_index: int = 0,
    **overrides: Any,
) -> Dict[str, Any]:
    """生成一个标准 chunk 字典。"""
    chunk = {
        "text": text,
        "source": source,
        "chunk_id": chunk_id,
        "article_num": article_num,
        "doc_type": doc_type,
        "law_name": law_name,
        "authority": authority,
        "status": status,
        "effective_date": effective_date,
        "chunk_index": chunk_index,
    }
    chunk.update(overrides)
    return chunk


def make_chunks(n: int = 5, **common_overrides: Any) -> List[Dict[str, Any]]:
    """生成 n 个 chunk，自动编号。"""
    return [
        make_chunk(
            chunk_id=f"test-chunk-{i:03d}",
            chunk_index=i,
            text=f"中华人民共和国公司法 第{['一','二','三','四','五'][i]}条 示例条文内容{i+1}",
            article_num=str(i + 1),
            **common_overrides,
        )
        for i in range(n)
    ]
