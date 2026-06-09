# 金融制度知识问答系统 (RAG Finance System)

基于 RAG 的金融法规智能问答系统，当前采用 `Streamlit + FastAPI + Milvus` 架构，已接入金融词典、Elasticsearch/BM25 全文检索、术语倒排、RRF 融合、Reranker 精排、知识图谱补充召回与法规时效性过滤。

## 当前状态

- 核心问答链路已跑通：上传/建索引/检索/问答可用
- 默认前端：`rag_finance_system/app.py`（Streamlit）
- 默认后端：`rag_finance_system/api_app.py`（FastAPI）
- 默认向量库：Milvus
- 可选全文检索：Elasticsearch，缺失时自动回退 BM25
- 可选知识图谱：Neo4j，缺失时自动跳过
- 时效性管理已接入：默认仅检索 `有效` 版本，支持展开历史版本
- MySQL 建表与基于 MySQL 的元数据管理尚未实现

## 架构概览

```text
app.py (Streamlit HTTP 客户端)
  -> api_app.py (FastAPI)
     -> DocumentProcessor
     -> Embedder / Reranker
     -> VectorStore (Milvus)
     -> ESIndex / BM25Index
     -> TermIndex
     -> FinanceDictionary
     -> KnowledgeGraph (optional)
     -> RAGChain.query()
        -> 实体检测
        -> 查询扩展
        -> 查询重写
        -> 混合检索 (向量 + ES/BM25 + 术语倒排)
        -> RRF 融合
        -> Reranker 精排
        -> LLM 生成
        -> 溯源 + 可信度评分
```

## 主要能力

- 金融法规 PDF/TXT 解析与分段
- 法规、案例、其他资料三类文档问答
- 金融词典实体检测、别名归一、缩写展开
- Milvus 向量检索
- Elasticsearch 全文检索，自动回退 BM25
- 术语精确倒排召回
- RRF 融合 + Reranker 精排
- Neo4j 图谱补充召回（可选）
- 法规 `effective_date` / `status` 时效性过滤
- Streamlit 前端与 FastAPI 后端分离

## 技术栈

| 组件 | 方案 |
|------|------|
| 前端 | Streamlit |
| 后端 | FastAPI |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| 查询重写 | `Qwen2.5-0.5B-Instruct + LoRA` |
| 向量数据库 | Milvus |
| 全文检索 | Elasticsearch 8.x / 内存 BM25 回退 |
| 知识图谱 | Neo4j 5.x（可选） |
| LLM | 本地 Qwen / DeepSeek API / 通义千问 API |

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── docker-compose.yml              # Milvus 本地部署
├── download_model.py               # 本地 Reranker 模型校验脚本
├── data/
│   ├── finance_dictionary.json
│   ├── dictionary_candidates.json
│   └── raw/                        # 上传文件存储（运行时生成）
├── models/                         # 本地模型目录（需自行准备，gitignored）
└── rag_finance_system/
    ├── .env
    ├── .env.example
    ├── app.py
    ├── api_app.py
    ├── api_schemas.py
    ├── test_pipeline.py
    ├── src/
    │   ├── document_processor.py
    │   ├── embedder.py
    │   ├── vector_store.py
    │   ├── bm25_index.py
    │   ├── es_index.py
    │   ├── term_index.py
    │   ├── dictionary.py
    │   ├── knowledge_graph.py
    │   ├── graph_builder.py
    │   ├── retriever.py
    │   ├── rag_chain.py
    │   ├── llm.py
    │   ├── rewriter.py
    │   └── txt_files/              # 83 部金融法律 txt 语料
    └── tools/
        ├── convert_testfiles.py
        ├── import_testfiles.py
        ├── extract_dictionary.py
        ├── merge_dictionary.py
        ├── generate_questions.py
        ├── generate_rewrite_data.py
        ├── rewrite_questions_for_rag.py
        └── train_rewriter.py
```

## 环境准备

### 1. Python

推荐 `Python 3.12.x`。

```bash
py -3 -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

### 2. 本地模型

至少准备：

- `models/bge-small-zh-v1.5`
- `models/bge-reranker-v2-m3`

可选：

- 本地 Qwen 主模型
- 本地查询重写模型与 LoRA 权重

`download_model.py` 会从 `RERANKER_MODEL_PATH` 或默认 `./models/bge-reranker-v2-m3` 读取模型做本地校验。

### 3. 环境变量

复制 `rag_finance_system/.env.example` 为 `rag_finance_system/.env` 后填写。

关键配置示例：

```env
EMBEDDING_MODEL_PATH=./models/bge-small-zh-v1.5
RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3
LLM_MODEL_PATH=./models/Qwen2.5-7B-Int4

DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=

MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=finance_regulations
MILVUS_EMBED_DIM=512

ES_HOST=127.0.0.1
ES_PORT=9200
ES_SCHEME=http
ES_INDEX_NAME=finance_regulations
ES_ANALYZER=standard

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j
NEO4J_DATABASE=neo4j

# 可选：LibreOffice 路径
# SOFFICE_PATH=C:/Program Files/LibreOffice/program/soffice.exe
```

## 启动方式

### 启动 Milvus

仓库根目录已提供 `docker-compose.yml`，当前只包含 Milvus 依赖服务：

```bash
docker compose up -d
```

默认暴露：

- Milvus: `19530`
- Milvus health/admin: `9091`
- MinIO Console: `9001`

### 启动 FastAPI

```bash
py -3 -m uvicorn rag_finance_system.api_app:app --host 0.0.0.0 --port 8000
```

### 启动 Streamlit

```bash
py -3 -m streamlit run rag_finance_system/app.py
```

## 使用说明

### 前端使用

1. 启动 FastAPI 与 Streamlit
2. 在侧边栏确认 API URL 可连通
3. 上传 PDF/TXT 文件并建立索引
4. 选择检索范围：全部 / 仅法条 / 仅案例 / 仅其他
5. 输入问题，查看答案、溯源与可信度

### 批量导入内置法规语料

前端侧边栏支持一键导入 `rag_finance_system/src/txt_files/*.txt`。

命令行也可直接执行：

```bash
py -3 rag_finance_system/tools/import_testfiles.py
```

该脚本当前会按 `law` 类型导入 `src/txt_files` 目录中的法规语料。

### 词典自动抽取

```bash
py -3 rag_finance_system/tools/extract_dictionary.py
py -3 rag_finance_system/tools/merge_dictionary.py
```

- `extract_dictionary.py`：从 `src/txt_files` 扫描候选法规名、机构名、术语
- `merge_dictionary.py`：做质量过滤并合并到 `data/finance_dictionary.json`

### LibreOffice 文档转换

```bash
py -3 rag_finance_system/tools/convert_testfiles.py
```

该脚本用于把 `data/testfiles` 下的 `.doc/.docx` 转成 `.txt`。优先读取 `SOFFICE_PATH`，否则尝试从 `PATH` 或常见安装目录查找 `soffice`。

## 检索与时效性说明

- 检索后端优先级：`Elasticsearch -> BM25 -> 纯向量`
- 术语倒排召回会与向量/全文检索一起参与 RRF 融合
- 默认 `status_filter="有效"`
- 问答接口可通过 `include_historical=true` 展开历史版本
- 旧 Milvus collection 若缺少 `effective_date/status` 字段，需要删除旧 collection 后全量重建索引

## 当前未完成项

- MySQL 建表与元数据管理
- FastAPI + Streamlit + 检索依赖的一体化 Docker Compose 部署
- `.env.example` 之外的完整部署模板与初始化脚本
- README 中未单独整理 Windows / Linux 部署差异

## 工程化清理说明

本轮已完成：

- 移除未使用依赖：`faiss-cpu`、`langchain*`、`langgraph*`
- 去除 `download_model.py` 中的本机绝对路径
- 去除 `convert_testfiles.py` 中固定 LibreOffice 路径依赖
- 修正 `embedder.py` 中 `bge-large` 的过期描述为当前 `bge-small`
- 修正批量导入 `src/txt_files` 时误用 `other` 类型的问题
