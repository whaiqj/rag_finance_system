"""
app.py
Streamlit前端 - 金融制度RAG问答系统
功能：文档上传、问答交互、溯源展示、可信度评分
"""

import sys
import os
import time
from pathlib import Path

import streamlit as st
import os

sys.path.insert(0, str(Path(__file__).parent))

# ===== 页面配置 =====
st.set_page_config(
    page_title="金融制度知识问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===== 缓存：全局组件只初始化一次 =====
@st.cache_resource(show_spinner="加载模型中，请稍候...")
def load_components(use_reranker: bool = True, use_api: bool = False):
    from rag_finance_system.src.document_processor import DocumentProcessor
    from rag_finance_system.src.embedder import Embedder, Reranker
    from rag_finance_system.src.vector_store import VectorStore
    from rag_finance_system.src.retriever import Retriever
    from rag_finance_system.src.llm import get_llm
    from rag_finance_system.src.rag_chain import RAGChain

    embedder = Embedder()
    vector_store = VectorStore()

    reranker = None
    if use_reranker:
        try:
            reranker = Reranker()
        except Exception as e:
            st.warning(f"Reranker加载失败: {e}")

    llm = get_llm(prefer_local=not use_api)
    retriever = Retriever(embedder=embedder, vector_store=vector_store, reranker=reranker)
    rag = RAGChain(retriever=retriever, llm=llm)
    processor = DocumentProcessor()

    return {
        "processor": processor,
        "embedder": embedder,
        "vector_store": vector_store,
        "rag": rag,
    }


# ===== 侧边栏 =====
with st.sidebar:
    st.title("⚙️ 系统设置")

    use_api = st.toggle("使用API模式（通义千问）", value=False,
                        help="显存不足时开启，需要在.env配置DASHSCOPE_API_KEY")
    use_reranker = st.toggle("启用Reranker精排", value=True,
                             help="提升检索精度，需额外~1.1GB显存")

    st.divider()
    st.subheader("文档管理")

    uploaded_file = st.file_uploader(
        "上传金融法规PDF或TXT",
        type=["pdf", "txt"],
        help="支持中文PDF或纯文本TXT，最大200MB"
    )

    if uploaded_file:
        if st.button("解析并建立索引", type="primary"):
            # 保存上传文件
            save_path = Path("data/raw") / uploaded_file.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # 建立索引
            with st.spinner(f"正在解析 {uploaded_file.name}..."):
                comps = load_components(use_reranker, use_api)
                chunks = comps["processor"].process_file(str(save_path))
                texts = [c["text"] for c in chunks]

                progress_bar = st.progress(0, text="生成向量...")
                batch_size = 16
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    embs = comps["embedder"].encode_documents(batch)
                    all_embeddings.extend(embs)
                    progress_bar.progress(
                        min((i + batch_size) / len(texts), 1.0),
                        text=f"生成向量... {min(i + batch_size, len(texts))}/{len(texts)}"
                    )

                comps["vector_store"].insert(chunks, all_embeddings)
                st.success(f"✅ 索引建立完成！共 {len(chunks)} 个知识片段")

    st.divider()
    st.caption("Phase 1 最小闭环版本\n向量检索 + Reranker + Qwen2.5")


# ===== 主界面 =====
st.title("📚 金融制度知识问答系统")
st.caption("基于RAG的金融法规智能问答 | bge-large-zh-v1.5 + Qwen2.5-7B")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 查看溯源条文 ({len(msg['sources'])} 条)"):
                for i, src in enumerate(msg["sources"], 1):
                    conf_color = "green" if src["score"] > 0.7 else "orange" if src["score"] > 0.4 else "red"
                    st.markdown(
                        f"**[{i}] 【{src['source']} {src['article_num']}】** "
                        f":{conf_color}[相关度 {src['score']:.2%}]"
                    )
                    st.text(src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"])
                    if i < len(msg["sources"]):
                        st.divider()

        if msg.get("confidence"):
            conf = msg["confidence"]
            total = conf["total"]
            col1, col2, col3 = st.columns(3)
            col1.metric("综合可信度", f"{total:.1%}")
            col2.metric("检索相关性", f"{conf['retrieval']:.1%}")
            col3.metric("答案覆盖度", f"{conf['coverage']:.1%}")


# 用户输入
if question := st.chat_input("请输入您的金融法规问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("检索相关条文并生成答案..."):
            try:
                comps = load_components(use_reranker, use_api)
                result = comps["rag"].query(question)

                answer = result["answer"]
                sources = result["sources"]
                confidence = result["confidence"]

                st.markdown(answer)

                # 溯源展示
                if sources:
                    with st.expander(f"📎 查看溯源条文 ({len(sources)} 条)"):
                        for i, src in enumerate(sources, 1):
                            conf_color = "green" if src["score"] > 0.7 else "orange" if src["score"] > 0.4 else "red"
                            st.markdown(
                                f"**[{i}] 【{src['source']} {src['article_num']}】** "
                                f":{conf_color}[相关度 {src['score']:.2%}]"
                            )
                            st.text(src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"])
                            if i < len(sources):
                                st.divider()

                # 可信度
                total = confidence["total"]
                col1, col2, col3 = st.columns(3)
                col1.metric("综合可信度", f"{total:.1%}")
                col2.metric("检索相关性", f"{confidence['retrieval']:.1%}")
                col3.metric("答案覆盖度", f"{confidence['coverage']:.1%}")

                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                })

            except Exception as e:
                err_msg = f"出错了: {str(e)}\n\n请检查：\n1. 是否已上传PDF并建立索引\n2. 模型路径是否正确\n3. 显存是否充足"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
