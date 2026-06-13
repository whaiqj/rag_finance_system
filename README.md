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

## API 端点

所有接口详见 `rag_finance_system/RAG_Finance_API.postman_collection.json`（Postman 导入即用）。

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/documents/upload` | 上传 PDF/TXT/图片文件 |
| `POST` | `/api/documents/index` | 解析 + 分段 + 写入 Milvus + BM25 |
| `GET` | `/api/laws` | 列出已索引法规及版本状态 |
| `POST` | `/api/search` | 混合检索（向量 + 全文 + 术语 + RRF + Reranker） |
| `POST` | `/api/qa` | 完整 RAG 问答（含溯源 + 可信度评分） |
| `POST` | `/api/qa/stream` | SSE 流式问答（token 级别实时推送） |
| `POST` | `/api/articles/relations` | Neo4j 图谱条文关联查询 |
| `GET` | `/api/categories` | 列出词典分类 |
| `PUT` | `/api/categories/rename` | 重命名词典分类 |
| `DELETE` | `/api/categories/{name}` | 删除词典分类 |
| `GET` | `/api/dictionary/{item_type}` | 列出词典条目（law/authority/term/abbreviation） |
| `PUT` | `/api/dictionary/{item_name}/category` | 设置词典条目分类 |

## 主要能力

- 金融法规 PDF/TXT/图片 解析与分段
- PDF 文本层不足时自动 OCR 回退（PaddleOCR + Docling 双后端）
- 独立图片文件 OCR 解析（支持 PNG/JPG/BMP/TIFF/WEBP），低分辨率图片基础预处理增强（灰度化+对比度+锐化）
- 法规、案例、其他资料三类文档问答
- 金融词典实体检测、别名归一、缩写展开
- Milvus 向量检索
- Elasticsearch 全文检索，自动回退 BM25
- 术语精确倒排召回
- RRF 融合 + Reranker 精排
- Neo4j 图谱补充召回（可选）
- SSE 流式问答输出（token 级别实时推送）
- 查询重写器 LoRA 微调训练管线
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
| OCR | PaddleOCR + Docling 双后端（PDF 文本层不足时自动回退）；独立图片文件 OCR；低分辨率图片基础预处理增强 |
| LLM | 本地 Qwen / DeepSeek API / 通义千问 API |
| 流式输出 | FastAPI SSE (Server-Sent Events)，token 级别实时推送 |

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── requirements-dev.txt              # pytest / coverage 等开发依赖
├── pyproject.toml                    # ruff + pytest 配置
├── docker-compose.yml              # Milvus 本地部署
├── download_model.py               # 本地 Reranker 模型校验脚本
├── build_index_bulk.py             # 批量索引构建（83 部法律一键入库）
├── test_performance.py             # 性能基准测试（毫秒级分步计时）
├── conftest.py                     # 根级测试配置（OCR 禁用）
├── data/
│   ├── finance_dictionary.json
│   ├── dictionary_candidates.json
│   └── raw/                        # 上传文件存储（运行时生成）
├── models/                         # 本地模型目录（需自行准备，gitignored）
├── tests/                          # 单元测试（15 个测试文件 + fixtures）
│   ├── conftest.py                 # 共享 mock fixtures
│   ├── fixtures/                   # 样本 chunks + 测试词典
│   ├── unit/                       # 各模块单元测试
│   └── api/                        # API 层测试（待补全）
└── rag_finance_system/
    ├── .env
    ├── .env.example
    ├── app.py
    ├── api_app.py
    ├── api_schemas.py
    ├── test_pipeline.py
    ├── test_retrieval_baseline.py
    ├── RAG_Finance_API.postman_collection.json
    ├── src/
    │   ├── document_processor.py
    │   ├── embedder.py
    │   ├── change.py
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

# OCR（PDF 文本层不足时自动回退）
# OCR_BACKEND=auto              # auto | docling | paddleocr | none
# OCR_LANG=ch
# OCR_MIN_TEXT_CHARS=80
# OCR_EMPTY_PAGE_RATIO=0.5
# OCR_RENDER_SCALE=2.0

# 图片预处理（低质量手机照片可增强OCR效果）
# IMAGE_PREPROCESS=auto         # auto | always | never
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
3. 上传 PDF/TXT/图片 文件并建立索引
4. 选择检索范围：全部 / 仅法条 / 仅案例 / 仅其他
5. 输入问题，查看答案、溯源与可信度

### 批量导入内置法规语料

前端侧边栏支持一键导入 `rag_finance_system/src/txt_files/*.txt`。

命令行也可直接执行：

```bash
py -3 rag_finance_system/tools/import_testfiles.py
```

该脚本当前会按 `law` 类型导入 `src/txt_files` 目录中的法规语料。

### 批量索引构建（全量重建）

```bash
py -3 build_index_bulk.py
```

一键完成 83 部金融法律的解析、分段、向量化和索引写入（Milvus + BM25），适合首次部署或全量重建场景。

### 图片 OCR 解析

项目支持独立图片文件的 OCR 文字提取，可用于扫描件、手机拍屏等场景：

- 支持格式：PNG / JPG / BMP / TIFF / WEBP
- 双后端自动回退：`PaddleOCR`（中文优先）→ `Docling`
- 低分辨率图片基础预处理（仅 PaddleOCR 后端）：灰度化 + 对比度增强 (1.5×) + 锐化 (1.2×)
- 通过 `IMAGE_PREPROCESS` 环境变量控制：`auto`（仅低分辨率触发）/ `always` / `never`
- 也可通过 `OCR_BACKEND=none` 完全禁用 OCR

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

### 查询重写器训练

基于 `Qwen2.5-0.5B-Instruct` 的 LoRA 微调管线，将自然语言问题改写为检索友好的查询：

```bash
py -3 rag_finance_system/tools/generate_rewrite_data.py   # 正向 + 反向改写数据生成
py -3 rag_finance_system/tools/rewrite_questions_for_rag.py # DeepSeek API 标注
py -3 rag_finance_system/tools/train_rewriter.py           # LoRA 微调训练
```

训练产物（`checkpoints/rewriter_lora/`）为可选组件，缺失时自动回退主 LLM 改写。注意：预训练 checkpoint 不在本仓库中（gitignored），需自行运行训练脚本生成。

### 运行测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -v                                          # 单元测试（15 个模块）
pytest rag_finance_system/test_pipeline.py                # 端到端管线验证
pytest rag_finance_system/test_retrieval_baseline.py      # 检索基线评估
py -3 test_performance.py                                 # 性能基准（毫秒级分步计时）
```

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
