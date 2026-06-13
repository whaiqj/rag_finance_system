# src包初始化

__all__ = [
    "DocumentProcessor",
    "Embedder",
    "Reranker",
    "VectorStore",
    "Retriever",
    "get_llm",
    "LocalLLM",
    "QwenAPILLM",
    "RAGChain",
    "TermIndex",
]


def __getattr__(name):
    if name == "DocumentProcessor":
        from .document_processor import DocumentProcessor

        return DocumentProcessor
    if name in {"Embedder", "Reranker"}:
        from .embedder import Embedder, Reranker

        return {"Embedder": Embedder, "Reranker": Reranker}[name]
    if name in {"LocalLLM", "QwenAPILLM", "get_llm"}:
        from .llm import LocalLLM, QwenAPILLM, get_llm

        return {
            "LocalLLM": LocalLLM,
            "QwenAPILLM": QwenAPILLM,
            "get_llm": get_llm,
        }[name]
    if name == "RAGChain":
        from .rag_chain import RAGChain

        return RAGChain
    if name == "Retriever":
        from .retriever import Retriever

        return Retriever
    if name == "TermIndex":
        from .term_index import TermIndex

        return TermIndex
    if name == "VectorStore":
        from .vector_store import VectorStore

        return VectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
