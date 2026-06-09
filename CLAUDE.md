# 行为规则

- **每次回复前**必须先输出: `hyha！！！你好whaiqj！`

# rag_finance_system — 项目上下文

## 项目概述

基于 RAG 的金融法规智能问答系统。Streamlit 前端 + 自研检索链路，无外部框架依赖（LangChain 在 requirements 中但未实际使用）。

- Python 环境: **3.12.3** (`py -3`)
- 入口: `rag_finance_system/app.py` (Streamlit)
- 测试脚本: `rag_finance_system/test_pipeline.py`

## 核心调用链

```
app.py (Streamlit HTTP 客户端) → api_app.py (FastAPI)
  → load_components(): DocumentProcessor, Embedder, VectorStore(Milvus), BM25Index, Retriever, LLM, QueryRewriter, FinanceDictionary, RAGChain
  → RAGChain.query(question)
      → 词典实体检测 (FinanceDictionary.detect_entities: 术语/法规名/机构名)
      → 词典查询扩展 (FinanceDictionary.expand_query: 别名追加提升召回)
      → 查询重写 (Qwen2.5-0.5B + LoRA → 回退主 LLM)
      → Retriever.retrieve()
          → VectorStore.search() × N(OR合并)  ← 向量召回
          → BM25Index.search() × N(OR合并)    ← 关键词召回
          → RRF 融合 (k=60)                    ← 双路融合
          → Reranker 精排
      → build_prompt() → LLM.generate()
      → 溯源 + 可信度评分
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `rag_finance_system/src/dictionary.py` | **金融词典**: 术语归一/别名召回/法规名+机构名映射/缩写解析/实体检测/查询扩展 |
| `rag_finance_system/src/rag_chain.py` | 主链路编排，含实体识别索引、prompt 构建、RAGChain 类 |
| `rag_finance_system/src/retriever.py` | 检索器：向量+BM25双路召回 + RRF融合 + Reranker精排 |
| `rag_finance_system/src/bm25_index.py` | **BM25 关键词索引**：jieba分词 + 内存BM25 + 标量过滤 + pickle持久化 |
| `rag_finance_system/src/vector_store.py` | **Milvus 适配层**：insert/search/get_collection_stats/drop_collection |
| `rag_finance_system/src/document_processor.py` | PDF/TXT解析 + 三轨智能分段(法条/案例/其他) |
| `rag_finance_system/src/embedder.py` | bge-small-zh-v1.5 Embedding + bge-reranker-v2-m3 Reranker |
| `rag_finance_system/src/llm.py` | LLM工厂：本地Qwen → DeepSeek API → Qwen API 降级 |
| `rag_finance_system/src/rewriter.py` | 查询重写小模型 (Qwen2.5-0.5B + 可选 LoRA) |
| `rag_finance_system/src/change.py` | fdata/*.docx → txt_files/*.txt 转换 |
| `rag_finance_system/app.py` | Streamlit 前端 |
| `rag_finance_system/tools/` | 批量导入、问题生成、重写器训练等辅助脚本 |
| `rag_finance_system/src/txt_files/` | 83部中国金融法律txt |
| `rag_finance_system/src/fdata/` | 83部中国金融法律docx源文件 |
| `requirements.txt` | 依赖清单 |

## 已修复的 Bug (本轮会话)

### P0 阻断级
- **[已修复]** `document_processor.py:310`: `split_into_chunks()` 缺少 `return chunks`，导致所有文档处理返回 None
- **[已修复]** `requirements.txt`: 缺少 `chromadb`(后移除了)、`python-docx`、`peft`、`datasets`、`accelerate`

### P1 功能级
- **[已修复]** `app.py:150`: 案例上传 `c["source"] = "case"` 覆写原始文件名 → 改为 `c.get("source", case_file.name)`
- **[已修复]** `embedder.py:19`: Reranker 路径改为 `os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")`
- **[已修复]** `rag_chain.py:230`: 移除未使用的 `case_retriever` 参数

## 混合检索 (2026-06-07 完成)

- **新建** `rag_finance_system/src/bm25_index.py` — BM25 内存索引，jieba 中文分词，标量过滤，pickle 持久化
- **改造** `rag_finance_system/src/retriever.py` — `_rrf_fusion()` + `Retriever(bm25_index=)` 可选参数，BM25 缺失时自动降级纯向量模式
- **更新** `rag_finance_system/api_app.py` — `_get_bm25()` 单例 + 启动预加载 + 建索引同步写入 BM25 + 持久化到 `data/bm25_index.pkl`
- **依赖** `jieba==0.42.1`

## ChromaDB → Milvus 迁移 (已完成)

### 改动文件
- **重写**: `rag_finance_system/src/vector_store.py` — 从 Chroma/ORM 迁移到 `pymilvus.MilvusClient` (3.0 非废弃 API)
- **修复**: `rag_finance_system/test_pipeline.py` — `get_collection_stats()` 兼容 `count`/`row_count`
- **更新**: `requirements.txt` — 移除 `chromadb`，`pymilvus>=2.6.0`
- **更新**: `README.md` — ChromaDB → Milvus 所有引用

### Milvus 连接配置 (.env)
```
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
# 或: MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION_NAME=finance_regulations
MILVUS_EMBED_DIM=512
# 可选: MILVUS_DB_NAME=, MILVUS_USER=, MILVUS_PASSWORD=
```

### VectorStore 关键设计
- 使用 `MilvusClient` (非废弃 ORM API)
- 嵌入度量: `COSINE`，优先 `AUTOINDEX`，失败回退 `HNSW`
- 主键: UUID，文本截断 4096
- 标量过滤: `source/doc_type/law_name/authority` 精确匹配，AND 组合
- OR 逻辑在 `Retriever` 层通过多次搜索+合并实现
- 分数归一化: `_normalize_score()` 保证输出在 [0,1]

### VectorStore schema
```
id VARCHAR(64) PK, embedding FLOAT_VECTOR(dim), text VARCHAR(4096),
source VARCHAR(512), chunk_id VARCHAR(512), article_num VARCHAR(64),
file_path VARCHAR(2048), chunk_index INT64, doc_type VARCHAR(64),
law_name VARCHAR(512), authority VARCHAR(512)
```

## 已知问题 (2026-06-09 更新)

### P2 工程化 (已修复)
- **[已修复]** `requirements.txt` 含未使用依赖: `pymilvus`(已留用)、`faiss-cpu`、`langchain*`、`langgraph*`
- **[已修复]** `download_model.py` 硬编码 `C:\Users\wangx\...` 路径
- **[已修复]** `convert_testfiles.py` 硬编码 LibreOffice 路径
- **[已修复]** `app.py` 批量导入 `src/txt_files` 误用 doc_type="other" → "law"

### P3 文档 (已修复)
- **[已修复]** README Embedding 描述 bge-large vs bge-small 不一致(实际用 small)
- **[已修复]** 无 `.env.example` 模板 → 已补 SOFFICE_PATH 等

### 当前剩余
- `models/` 目录需手动创建并下载模型
- `rag_finance_system/rag金融知识技术路线.md` 中 Embedding 仍描述为 `bge-large-zh-v1.5`（历史路线文档）
- 无 pytest 单元测试覆盖
- MySQL 建表与元数据管理未实现
- 完整 Docker Compose 部署未实现（当前仅 Milvus）

## 验证状态

| 验证项 | 状态 |
|--------|------|
| Python 语法编译 (5个核心文件) | 通过 |
| 全包导入 (8个模块) | 通过 |
| VectorStore 方法完整性 | 通过 |
| BM25Index 索引/搜索/过滤/持久化 | 通过 |
| RRF 融合排序正确性 | 通过 |
| 空 BM25 回退纯向量模式 | 通过 |
| 无废弃 pymilvus API | 通过 |
| 分数归一化闭区间 | 通过 |
| 过滤表达式构建 | 通过 |
| 无 Milvus 服务时错误提示 | 通过 |
| Milvus 服务下 insert/search 冒烟 | **通过** (2026-06-07) |
| 混合检索端到端 (双路召回+RRF+Reranker) | **通过** (2026-06-07, BM25+向量+RRF+Reranker, 95.5%置信度) |
| Streamlit app.py 端到端 | **通过** (2026-06-08, Docker Milvus + FastAPI + Streamlit 全链路) |

## 端到端验证结果 (2026-06-07)

- Embedding (bge-small-zh-v1.5, CPU, 512维) → Milvus (finance_regulations, 303条) → Reranker → DeepSeek API → 溯源+可信度
- 查询"股东责任是什么？" → 可信度 96.0%，溯源 5 条全部来自《公司法》+ 条文号准确
- 查询"资本充足率是什么？" → 可信度 29.2%，系统诚实回答"未找到相关定义"（索引缺银行监管文档）
- test_pipeline.py sys.path 已修复 (parent → parent.parent)

## 金融词典 (2026-06-07 完成)

- **新建** `rag_finance_system/src/dictionary.py` — FinanceDictionary 类，JSON词典加载+反向索引+大小写不敏感实体检测
- **新建** `data/finance_dictionary.json` — 58个金融术语(296别名)、41个法规名(138别名)、15个监管机构(62别名)、25个英文缩写
- **更新** `rag_finance_system/src/rag_chain.py` — RAGChain 新增 `dictionary` 参数，词典优先→旧文件名索引回退
- **更新** `rag_finance_system/api_app.py` — `_get_dictionary()` 单例 + 预加载 + 传入 RAGChain
- **功能**:
  - `detect_entities(question)` — 检测术语/法规名/机构名
  - `expand_query(query)` — 追加术语别名提升向量+BM25召回
  - `resolve_law_name(name)` / `resolve_authority(name)` / `resolve_term(term)` — 别名→规范名
  - `resolve_abbreviation(abbr)` — 英文缩写→中文全称 (NPL→不良贷款率, AML→反洗钱)
  - `search_terms(query)` — 模糊匹配术语表

## 下一步

- pytest 单元测试补齐
- MySQL 建表与元数据管理
- 完整 Docker Compose 一键部署 (FastAPI + Streamlit + Milvus + 可选 ES/Neo4j/MySQL)
- Milvus/ES 重新全量建索引（schema 已变更为含 effective_date/status 的 12 字段）

## 时效性管理 (2026-06-08 完成)

- **更新** `rag_finance_system/src/document_processor.py` — `_extract_law_name_and_date()` 从文件名提取日期 + `resolve_version_status()` 多版本状态判定
- **更新** `rag_finance_system/src/vector_store.py` — schema 新增 `effective_date`(VARCHAR 16) + `status`(VARCHAR 8)，`_build_expr/search` 支持 `status_filter`
- **更新** `rag_finance_system/src/es_index.py` — mapping 新增 keyword 字段 `effective_date`/`status`，search 支持 `status_filter` term 过滤
- **更新** `rag_finance_system/src/bm25_index.py` — search 新增 `status_filter` 参数
- **更新** `rag_finance_system/src/term_index.py` — search 新增 `status_filter` 参数
- **更新** `rag_finance_system/src/retriever.py` — `retrieve()` 默认 `status_filter="有效"`，传 `None` 展开全部版本
- **更新** `rag_finance_system/src/rag_chain.py` — `query()` 新增 `include_historical` 参数
- **更新** `rag_finance_system/api_app.py` — 建索引时 `_resolve_single_file_status()` 单文件版本判定；`/api/search` 支持 `status_filter`；`/api/qa` 支持 `include_historical`
- **更新** `rag_finance_system/api_schemas.py` — `SearchRequest.status_filter` / `QARequest.include_historical`

### 设计要点
- v1 纯 Milvus/ES 标量字段，不依赖 MySQL
- `resolve_version_status()` 同名法规按 `effective_date` 降序，最新→"有效"，其余→"已修订"
- 检索默认 `status_filter="有效"`，用户传 `None` 可查历史版本
- 单文件 API 建索引时比对 BM25 已有数据自动判定新旧

## 环境信息

- OS: Windows 11
- Python: 3.12.3 (`py -3`)
- pymilvus: 3.0.0
- 依赖已全部安装 (`pip install -r requirements.txt` 成功)
- 非 git 仓库
