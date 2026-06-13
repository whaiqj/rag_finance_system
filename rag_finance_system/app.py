"""
app.py
Streamlit 前端 — 金融制度 RAG 问答系统
纯 HTTP 客户端，所有后端调用走 FastAPI。
"""

import json
import os
from pathlib import Path
from typing import Generator

import requests
import streamlit as st
from requests.exceptions import ConnectionError, RequestException, Timeout

DEFAULT_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

# ===== 页面配置 =====
st.set_page_config(
    page_title="金融制度知识问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Session State 初始化 =====
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_URL
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_ok" not in st.session_state:
    st.session_state.api_ok = None


# ===== HTTP Helper 函数 =====

def check_api_health(api_base: str) -> bool:
    try:
        r = requests.get(f"{api_base}/openapi.json", timeout=3)
        return r.status_code == 200
    except (ConnectionError, Timeout, RequestException):
        return False


def upload_file(api_base: str, file_bytes: bytes, filename: str, doc_type: str) -> dict:
    resp = requests.post(
        f"{api_base}/api/documents/upload",
        files={"file": (filename, file_bytes)},
        data={"doc_type": doc_type},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def index_file(api_base: str, file_path: str, doc_type: str) -> dict:
    resp = requests.post(
        f"{api_base}/api/documents/index",
        json={"file_path": file_path, "doc_type": doc_type},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def ask_question(
    api_base: str,
    question: str,
    doc_type_filter: str | None,
    use_reranker: bool,
    use_query_rewrite: bool,
) -> dict:
    payload: dict = {
        "question": question,
        "use_reranker": use_reranker,
        "use_query_rewrite": use_query_rewrite,
    }
    if doc_type_filter:
        payload["doc_type_filter"] = doc_type_filter
    resp = requests.post(
        f"{api_base}/api/qa",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def ask_question_stream(
    api_base: str,
    question: str,
    doc_type_filter: str | None,
    use_reranker: bool,
    use_query_rewrite: bool,
) -> tuple[Generator[str, None, None], dict]:
    """流式问答：返回 (token_generator, metadata_holder)。
    metadata_holder 会在流结束后被填充 sources/confidence/rewritten_query。
    """
    payload: dict = {
        "question": question,
        "use_reranker": use_reranker,
        "use_query_rewrite": use_query_rewrite,
    }
    if doc_type_filter:
        payload["doc_type_filter"] = doc_type_filter

    metadata: dict = {}

    def token_stream() -> Generator[str, None, None]:
        try:
            resp = requests.post(
                f"{api_base}/api/qa/stream",
                json=payload,
                timeout=180,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "token":
                        yield event["content"]
                    elif event.get("type") == "done":
                        metadata["sources"] = event.get("sources", [])
                        metadata["confidence"] = event.get("confidence", {})
                        metadata["rewritten_query"] = event.get("rewritten_query")
                    elif event.get("type") == "error":
                        yield f"\n\n**错误：{event['message']}**"
        except RequestException as e:
            yield f"\n\n**{_handle_api_error(e)}**"

    return token_stream(), metadata


def _handle_api_error(exc: RequestException) -> str:
    if hasattr(exc, "response") and exc.response is not None:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = exc.response.text or str(exc)
        return f"API错误 {exc.response.status_code}: {detail}"
    if isinstance(exc, ConnectionError):
        return "无法连接到API服务器，请确认FastAPI已启动"
    if isinstance(exc, Timeout):
        return "请求超时，服务器处理时间过长"
    return f"请求失败: {exc}"


# ── 分类管理 Helper ──

def fetch_categories(api_base: str) -> dict[str, list[str]]:
    resp = requests.get(f"{api_base}/api/categories", timeout=10)
    resp.raise_for_status()
    return resp.json().get("categories", {})


def rename_category_api(api_base: str, old_name: str, new_name: str) -> dict:
    resp = requests.put(
        f"{api_base}/api/categories/rename",
        json={"old_name": old_name, "new_name": new_name},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def delete_category_api(api_base: str, name: str) -> dict:
    resp = requests.delete(f"{api_base}/api/categories/{name}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_dictionary_items(api_base: str, item_type: str) -> list[dict]:
    resp = requests.get(f"{api_base}/api/dictionary/{item_type}", timeout=10)
    resp.raise_for_status()
    return resp.json().get("items", [])


def set_item_category_api(api_base: str, item_type: str, item_name: str, category: str) -> dict:
    resp = requests.put(
        f"{api_base}/api/dictionary/{item_name}/category",
        json={"item_type": item_type, "item_name": item_name, "category": category},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ===== 渲染 Helper =====

def render_sources(sources: list[dict]):
    if not sources:
        return
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


def render_confidence(conf: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("综合可信度", f"{conf['total']:.1%}")
    col2.metric("检索相关性", f"{conf['retrieval']:.1%}")
    col3.metric("答案覆盖度", f"{conf['coverage']:.1%}")


# ── 条文关联查询 Helper ──

@st.cache_data(ttl=300)
def get_law_names(api_base: str) -> list[str]:
    """从 API 获取已索引的法规名称列表。缓存 5 分钟。"""
    try:
        resp = requests.get(f"{api_base}/api/laws", timeout=10)
        resp.raise_for_status()
        return resp.json().get("law_names", [])
    except (ConnectionError, Timeout, RequestException):
        return []


def query_article_relations(api_base: str, law_name: str, article_num: str) -> dict:
    """调用 POST /api/articles/relations 查询条文关联关系。"""
    resp = requests.post(
        f"{api_base}/api/articles/relations",
        json={"law_name": law_name, "article_num": article_num},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ===== 侧边栏 =====
with st.sidebar:
    st.title("⚙️ 系统设置")

    # API URL 配置
    url_input = st.text_input(
        "API服务器URL",
        value=st.session_state.api_base_url,
        key="api_url_input",
        help="FastAPI 后端地址，默认 http://localhost:8000",
    )
    if url_input.strip() != st.session_state.api_base_url:
        st.session_state.api_base_url = url_input.strip()
        st.session_state.api_ok = None  # 重置健康状态

    # 健康检查
    api_base = st.session_state.api_base_url
    st.session_state.api_ok = check_api_health(api_base)
    if st.session_state.api_ok:
        st.success("API 服务正常", icon="✅")
    else:
        st.warning("API 服务未响应，请启动 FastAPI 后刷新", icon="⚠️")

    st.divider()

    # 检索设置
    use_reranker = st.toggle("启用 Reranker 精排", value=True,
                             help="提升检索精度，需要服务端加载 Reranker 模型")
    use_query_rewrite = st.toggle("启用查询重写", value=True,
                                  help="将用户问题改写为更适合向量检索的查询")
    mode = st.selectbox(
        "检索模式",
        ["全部", "仅法条", "仅案例", "仅其他"],
        index=0,
        help="限定本次问答的检索范围",
    )

    st.divider()
    st.subheader("文档管理")

    doc_type_label = st.selectbox(
        "文档类型",
        ["法规 (law)", "案例 (case)", "其他 (other)"],
        index=0,
        help="选择上传文件对应的类型",
    )
    doc_type_map = {"法规 (law)": "law", "案例 (case)": "case", "其他 (other)": "other"}
    doc_type = doc_type_map[doc_type_label]

    uploaded_file = st.file_uploader(
        "上传金融法规 PDF / TXT / 图片",
        type=["pdf", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        help="支持中文 PDF、纯文本 TXT、或图片(PNG/JPG/BMP/TIFF/WEBP)，最大 200MB",
    )

    if uploaded_file and Path(uploaded_file.name).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
        st.info("图片将通过 OCR 自动识别文字内容后建立索引，处理时间约 10-30 秒/张")

    if uploaded_file:
        if st.button("解析并建立索引", type="primary", disabled=not st.session_state.api_ok):
            try:
                with st.spinner("正在上传文件..."):
                    upload_resp = upload_file(
                        api_base,
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        doc_type,
                    )
                file_path = upload_resp["file_path"]
                with st.spinner("正在建立索引（大文件需数分钟）..."):
                    index_resp = index_file(api_base, file_path, doc_type)
                st.success(f"索引建立完成！共 {index_resp['chunk_count']} 个知识片段")
            except RequestException as e:
                st.error(_handle_api_error(e))

    st.divider()
    st.subheader("批量导入")
    if st.button(
        "一键导入 src/txt_files 中的 TXT 文件",
        type="secondary",
        key="batch_import",
        disabled=not st.session_state.api_ok,
    ):
        import glob as _glob
        from pathlib import Path as _Path
        _txt_dir = _Path(__file__).resolve().parent / "src" / "txt_files"
        _txt_files = sorted(_glob.glob(str(_txt_dir / "*.txt")))
        if not _txt_files:
            st.warning(f"未找到 .txt 文件: {_txt_dir}")
        else:
            progress = st.progress(0, text=f"0/{len(_txt_files)}")
            total_chunks = 0
            errors = 0
            for i, fp in enumerate(_txt_files):
                fname = _Path(fp).name
                try:
                    with open(fp, "rb") as fh:
                        up = upload_file(api_base, fh.read(), fname, "law")
                    ix = index_file(api_base, up["file_path"], "law")
                    total_chunks += ix["chunk_count"]
                except Exception as e:
                    st.warning(f"跳过 {fname}: {e}")
                    errors += 1
                progress.progress(
                    (i + 1) / len(_txt_files),
                    text=f"{i + 1}/{len(_txt_files)}  ({total_chunks} chunks)",
                )
            st.success(f"批量导入完成！{len(_txt_files) - errors} 个文件，共 {total_chunks} 个 chunk")

    st.divider()
    st.caption("金融制度 RAG 问答系统\n向量检索 + Reranker + LLM")


# ===== 主界面 =====
st.title("📚 金融制度知识问答系统")
st.caption("基于 RAG 的金融法规智能问答 | bge-small-zh-v1.5 + Qwen2.5")

tab_qa, tab_relations, tab_categories = st.tabs(["💬 智能问答", "🔗 条文关联查询", "🏷️ 标签分类管理"])

# ── Tab 1: 智能问答 ──
with tab_qa:
    # 检索模式 → doc_type_filter
    _mode_map = {"全部": None, "仅法条": "law", "仅案例": "case", "仅其他": "other"}
    doc_type_filter = _mode_map[mode]

    # 历史消息回放
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("rewritten_query") and msg["rewritten_query"] != msg.get("question"):
                st.caption(f"已改写查询：{msg['rewritten_query']}")
            if msg.get("sources"):
                render_sources(msg["sources"])
            if msg.get("confidence"):
                render_confidence(msg["confidence"])

    # 用户输入
    if question := st.chat_input(
        "请输入您的金融法规问题...",
        disabled=not st.session_state.api_ok,
    ):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            token_gen, metadata = ask_question_stream(
                api_base,
                question,
                doc_type_filter,
                use_reranker,
                use_query_rewrite,
            )
            answer = st.write_stream(token_gen)

            if answer is None or answer.strip() == "":
                answer = "（未获取到回答）"

            rewritten = metadata.get("rewritten_query")
            sources = metadata.get("sources", [])
            confidence = metadata.get("confidence", {})

            if rewritten and rewritten != question and rewritten != answer:
                st.caption(f"已改写查询：{rewritten}")
            render_sources(sources)
            if confidence:
                render_confidence(confidence)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "question": question,
                "rewritten_query": rewritten,
                "sources": sources,
                "confidence": confidence,
            })

# ── Tab 2: 条文关联查询 ──
with tab_relations:
    st.subheader("🔗 条文关联查询")
    st.caption("查询法规条文之间的引用关系、所属文档及关联法规")

    law_names = get_law_names(api_base)
    law_names_sorted = sorted(law_names) if law_names else []

    col1, col2 = st.columns([3, 1])
    with col1:
        if law_names_sorted:
            selected_law = st.selectbox(
                "选择法规",
                options=law_names_sorted,
                index=None,
                placeholder="输入或选择法规名称...",
                key="article_rel_law_name",
            )
        else:
            selected_law = st.text_input(
                "法规名称",
                placeholder="例如: 中华人民共和国公司法",
                key="article_rel_law_name",
                disabled=not st.session_state.api_ok,
            )
    with col2:
        article_num = st.text_input(
            "条文编号",
            placeholder="例如: 16",
            key="article_rel_num",
            disabled=not st.session_state.api_ok,
        )

    if st.button("查询关联关系", type="primary", disabled=not st.session_state.api_ok or not selected_law or not article_num):
        with st.spinner("正在查询知识图谱..."):
            try:
                data = query_article_relations(api_base, selected_law, article_num)
            except RequestException as e:
                st.error(_handle_api_error(e))
                data = None

        if data:
            target = data.get("target")
            if target:
                st.markdown("### 📄 目标条文")
                with st.container(border=True):
                    st.markdown(f"**{target['law_name']}** — 第{target['article_num']}条")
                    st.text(target["text"])

                incoming = data.get("incoming_refs", [])
                st.markdown(f"### ⬅️ 引用此条文的条文 ({len(incoming)})")
                if incoming:
                    for i, ref in enumerate(incoming, 1):
                        with st.container(border=True):
                            st.markdown(f"**[{i}] {ref['law_name']} 第{ref['article_num']}条**")
                            st.caption(f"来源: {ref['source']}")
                            st.text(ref["text"][:300] + ("..." if len(ref["text"]) > 300 else ""))
                else:
                    st.info("暂无其他条文引用此条文")

                outgoing = data.get("outgoing_refs", [])
                st.markdown(f"### ➡️ 此条文引用的条文 ({len(outgoing)})")
                if outgoing:
                    for i, ref in enumerate(outgoing, 1):
                        target_label = ""
                        if ref.get("target_law"):
                            target_label = f" → {ref['target_law']} 第{ref.get('target_article', '')}条"
                        with st.container(border=True):
                            st.markdown(f"**[{i}] {ref['law_name']} 第{ref['article_num']}条**{target_label}")
                            st.caption(f"来源: {ref['source']}")
                            st.text(ref["text"][:300] + ("..." if len(ref["text"]) > 300 else ""))
                else:
                    st.info("此条文未引用其他条文")

                parent = data.get("parent_document")
                if parent:
                    st.markdown("### 📁 所属文档")
                    with st.container(border=True):
                        st.markdown(f"**{parent['name']}**")
                        st.caption(f"类型: {parent.get('doc_type', '')} | 来源: {parent.get('source', '')}")

                related_docs = data.get("related_documents", [])
                st.markdown(f"### 📎 关联法规 ({len(related_docs)})")
                if related_docs:
                    for i, rd in enumerate(related_docs, 1):
                        direction_icon = "→" if rd["direction"] == "outgoing" else "←"
                        with st.container(border=True):
                            st.markdown(f"**[{i}] {direction_icon} {rd['name']}**")
                            st.caption(f"关系类型: {rd['relation_type']} | 方向: {rd['direction']}")
                else:
                    st.info("无关联法规")

                related_articles = data.get("related_articles", [])
                st.markdown(f"### 📋 关联法规示例条文 ({len(related_articles)})")
                if related_articles:
                    for i, ra in enumerate(related_articles, 1):
                        with st.container(border=True):
                            st.markdown(f"**[{i}] {ra['law_name']} 第{ra['article_num']}条**")
                            st.caption(f"来源: {ra['source']}")
                            st.text(ra["text"][:300] + ("..." if len(ra["text"]) > 300 else ""))
                else:
                    st.info("关联法规中暂无示例条文")
            else:
                st.warning(f"未找到 '{selected_law}' 第{article_num}条")
    else:
        st.info('请输入法规名称和条文编号，点击"查询关联关系"查看引用关系')
        if law_names_sorted:
            st.caption(f"知识图谱中现有 {len(law_names_sorted)} 部法规可供查询")

# ── Tab 3: 标签分类管理 ──
with tab_categories:
    st.subheader("🏷️ 标签分类管理")
    st.caption("管理金融词典中术语、法规和机构的分类标签")

    api_ready = st.session_state.api_ok

    if not api_ready:
        st.warning("API 服务未连接，请先启动 FastAPI 后端")
    else:
        # ── 加载分类数据 ──
        try:
            categories = fetch_categories(api_base)
        except RequestException as e:
            st.error(_handle_api_error(e))
            categories = {}

        # 合并所有分类名
        all_categories: list[str] = []
        for cats in categories.values():
            for c in cats:
                if c not in all_categories:
                    all_categories.append(c)
        all_categories.sort()

        # ── 子标签: 分类概览 / 条目管理 ──
        sub_cat, sub_item = st.tabs(["📊 分类概览", "📋 条目管理"])

        with sub_cat:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### 术语分类")
                term_cats = categories.get("term", [])
                if term_cats:
                    for c in term_cats:
                        count = sum(
                            1 for t in categories.get("term", []) if t == c
                        )
                        st.text(f"{c}")
                else:
                    st.caption("（无）")

            with col_b:
                st.markdown("#### 法规分类 / 机构分类")
                law_cats = categories.get("law", [])
                auth_cats = categories.get("authority", [])
                for c in law_cats + auth_cats:
                    st.text(c)
                if not law_cats and not auth_cats:
                    st.caption("（无）")

            st.divider()

            # ── 重命名分类 ──
            st.markdown("#### 🔄 重命名分类")
            if all_categories:
                col_old, col_new, col_btn = st.columns([2, 2, 1])
                with col_old:
                    rename_old = st.selectbox("原分类名", options=all_categories, key="rename_old")
                with col_new:
                    rename_new = st.text_input("新分类名", key="rename_new", placeholder="输入新名称...")
                with col_btn:
                    st.write("")
                    if st.button("重命名", key="btn_rename", disabled=not rename_new.strip()):
                        try:
                            result = rename_category_api(api_base, rename_old, rename_new.strip())
                            st.success(
                                f"已重命名: {rename_old} → {rename_new.strip()} "
                                f"(术语{result['affected']['term']}, 法规{result['affected']['law']}, 机构{result['affected']['authority']})"
                            )
                            st.rerun()
                        except RequestException as e:
                            st.error(_handle_api_error(e))
            else:
                st.info("暂无分类可重命名")

            # ── 删除分类 ──
            st.markdown("#### 🗑️ 删除分类")
            st.caption("删除后该分类下条目的 category 将被置空，条目本身不删除")
            if all_categories:
                col_del, col_btn2 = st.columns([3, 1])
                with col_del:
                    delete_target = st.selectbox("选择要删除的分类", options=all_categories, key="delete_cat")
                with col_btn2:
                    st.write("")
                    if st.button("删除", key="btn_delete", type="secondary"):
                        try:
                            result = delete_category_api(api_base, delete_target)
                            st.success(
                                f"已删除分类 '{delete_target}' "
                                f"(术语{result['affected']['term']}, 法规{result['affected']['law']}, 机构{result['affected']['authority']})"
                            )
                            st.rerun()
                        except RequestException as e:
                            st.error(_handle_api_error(e))
            else:
                st.info("暂无分类可删除")

        # ── 条目管理 ──
        with sub_item:
            item_type_label = st.radio(
                "条目类型",
                options=["术语 (term)", "法规 (law)", "机构 (authority)"],
                horizontal=True,
                key="item_type_radio",
            )
            item_type_map = {"术语 (term)": "term", "法规 (law)": "law", "机构 (authority)": "authority"}
            selected_item_type = item_type_map[item_type_label]

            try:
                items = fetch_dictionary_items(api_base, selected_item_type)
            except RequestException as e:
                st.error(_handle_api_error(e))
                items = []

            if items:
                st.caption(f"共 {len(items)} 条，点击修改分类")
                # 按分类分组显示
                items_by_cat: dict[str, list[dict]] = {}
                for it in items:
                    c = it.get("category", "") or "（未分类）"
                    if c not in items_by_cat:
                        items_by_cat[c] = []
                    items_by_cat[c].append(it)

                for cat_name, cat_items in items_by_cat.items():
                    with st.expander(f"{cat_name} ({len(cat_items)} 条)"):
                        for it in cat_items:
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.text(it["name"])
                            with c2:
                                new_cat = st.selectbox(
                                    "分类",
                                    options=["（未分类）"] + all_categories,
                                    index=(
                                        all_categories.index(it.get("category", "")) + 1
                                        if it.get("category", "") in all_categories
                                        else 0
                                    ),
                                    key=f"cat_{selected_item_type}_{it['name']}",
                                    label_visibility="collapsed",
                                )
                                target_cat = "" if new_cat == "（未分类）" else new_cat
                                if target_cat != (it.get("category", "") or ""):
                                    try:
                                        set_item_category_api(
                                            api_base, selected_item_type, it["name"], target_cat
                                    )
                                        st.rerun()
                                    except RequestException as e:
                                        st.error(_handle_api_error(e))
            else:
                st.info(f"暂无{selected_item_type}类型的条目")
