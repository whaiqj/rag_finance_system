"""
rag_chain.py
RAG主链路：检索 → Prompt组装 → LLM生成 → 答案+溯源
"""

from typing import List, Dict, Any, Optional
from loguru import logger


# System Prompt：约束LLM只基于上下文回答，防止幻觉
SYSTEM_PROMPT = """你是一个专业的金融法规知识助手。你的任务是根据提供的金融制度条文，准确回答用户的问题。

**重要规则**：
1. 只基于【参考条文】中的内容回答，不要凭空推测或引用参考条文之外的知识
2. 如果参考条文中没有相关内容，请明确告知"暂未找到相关条文，建议查阅完整法规原文"
3. 回答时请引用具体的条文来源（文件名、条文编号）
4. 保持回答专业、简洁、准确

回答格式：
- 先给出直接答案
- 再列出相关条文（格式：【来源：文件名 条文编号】）"""


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> List[dict]:
    """
    组装Prompt消息列表
    Args:
        query: 用户问题
        chunks: 检索到的相关chunks
    Returns:
        messages列表（适配OpenAI格式）
    """
    if not chunks:
        context = "（未检索到相关条文）"
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "未知文件")
            article = chunk.get("article_num", "")
            source_tag = f"【{source}{'  ' + article if article else ''}】"
            context_parts.append(f"{i}. {source_tag}\n{chunk['text']}")
        context = "\n\n".join(context_parts)

    user_message = f"""【参考条文】
{context}

【用户问题】
{query}

请根据以上参考条文回答问题："""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


class RAGChain:
    """
    RAG主链路
    整合：Retriever（检索） + LLM（生成）
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
        source_filter: Optional[str] = None,
        max_new_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        完整问答流程
        Returns:
            {
                "question": str,
                "answer": str,
                "sources": [{"source": str, "article_num": str, "text": str, "score": float}],
                "confidence": {"total": float, "retrieval": float, "coverage": float},
            }
        """
        logger.info(f"问答请求: {question[:60]}...")

        # 1. 检索
        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            use_reranker=use_reranker,
            source_filter=source_filter,
        )

        # 2. 构建Prompt
        messages = build_prompt(question, chunks)

        # 3. LLM生成
        answer = self.llm.generate(messages, max_new_tokens=max_new_tokens)
        logger.info(f"生成答案长度: {len(answer)} 字符")

        # 4. 整理溯源信息
        sources = [
            {
                "source": c.get("source", ""),
                "article_num": c.get("article_num", ""),
                "text": c.get("text", "")[:300],  # 前端展示截断
                "score": round(c.get("reranker_score", c.get("score", 0.0)), 4),
            }
            for c in chunks
        ]

        # 5. 计算可信度
        confidence = self.retriever.compute_confidence(question, answer, chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }
