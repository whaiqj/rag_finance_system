# 金融制度知识问答系统 (RAG Finance System)

基于 RAG（检索增强生成）的金融法规智能问答系统，支持法条与案例的混合检索、语义问答、条文溯源与可信度评估。

## 架构概览

```
用户问题
  │
  ├── 查询重写 (LLM)
  ├── 向量检索 (bge-small-zh-v1.5 + ChromaDB)
  ├── Reranker 精排 (bge-reranker-v2-m3)
  ├── Prompt 组装 (System + Context + Question)
  ├── LLM 生成 (Qwen2.5-7B / DeepSeek / 通义千问)
  └── 可信度评分 + 溯源展示
```

## 功能

- **多模态文档解析**：支持 PDF / TXT 格式的法条文档和裁判文书
- **双轨切分策略**：法条按"第XX条"结构切分；案例按裁判文书段落（当事人信息、诉讼请求、经审理查明、本院认为、裁判结果）切分
- **语义检索 + 精排**：向量检索召回 Top-10，Cross-Encoder Reranker 精排保留 Top-5
- **查询重写**：将用户口语问题自动改写为适合向量检索的简洁查询
- **三种检索模式**：法条+案例 / 仅法条 / 仅案例
- **答案溯源**：每条回答附带来源条文、条文编号和相关度评分
- **可信度评分**：综合检索相关性（60%）与答案覆盖度（40%）
- **多 LLM 后端**：本地 Qwen2.5-7B-Int4 / DeepSeek API / 通义千问 API，自动降级

## 技术栈

| 组件 | 方案 |
|------|------|
| 前端 | Streamlit |
| Embedding | bge-small-zh-v1.5 (512d) |
| Reranker | bge-reranker-v2-m3 |
| 向量数据库 | ChromaDB (本地持久化) |
| LLM | Qwen2.5-7B-Instruct-GPTQ-Int4 / DeepSeek / 通义千问 |
| 文档解析 | pdfplumber (PDF) + 自研分段器 |

## 项目结构

```
rag_finance_system/
├── app.py                      # Streamlit 前端
├── src/
│   ├── document_processor.py   # 文档解析 + 智能分段（法条/案例双轨）
│   ├── embedder.py             # Embedding 模型 + Reranker
│   ├── vector_store.py         # ChromaDB 向量存储
│   ├── retriever.py            # 检索器（向量检索 + 精排 + 可信度）
│   ├── llm.py                  # LLM 推理（本地/API 三路可选）
│   ├── rag_chain.py            # RAG 主链路（重写 → 检索 → 生成 → 溯源）
│   └── change.py               # docx → txt 转换工具
├── models/
│   ├── bge-small-zh-v1.5/      # Embedding 模型
│   └── bge-reranker-v2-m3/     # Reranker 模型
├── data/raw/                   # 上传文档存储
├── db/chroma/                  # ChromaDB 持久化目录
├── requirements.txt
└── .env                        # 环境配置
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/intnerd/rag_finance_system.git
cd rag_finance_system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/Mac
# 或 venv\Scripts\activate  (Windows)

# 安装依赖
pip install -r requirements.txt

# 下载模型（如未包含在仓库中）
# Embedding: BAAI/bge-small-zh-v1.5
# Reranker:  BAAI/bge-reranker-v2-m3
# LLM:       Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
```

## 配置

复制 `.env` 文件并根据环境修改：

```bash
# 模型路径
EMBEDDING_MODEL_PATH=./models/bge-small-zh-v1.5
RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3
LLM_MODEL_PATH=./models/Qwen2.5-7B-Int4

# API 密钥（本地模型不可用时自动切换）
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=xxx

# 检索参数
RETRIEVER_TOP_K=10      # 向量检索召回数量
RERANKER_TOP_N=5         # Reranker 后保留数量
CHUNK_SIZE=512           # 分段字符数
CHUNK_OVERLAP=100        # 分段重叠字符数
```

## 使用

```bash
# 启动 Streamlit 前端
streamlit run rag_finance_system/app.py

# 或通过命令行测试
python rag_finance_system/test_pipeline.py \
  --file data/raw/your_document.pdf \
  --query "什么是资本充足率？"
```

### 前端操作流程

1. **上传文档**：在侧边栏上传法条 PDF/TXT，点击"解析并建立索引"
2. **上传案例**：在侧边栏上传案例 PDF/TXT，点击"解析案例并建立索引"
3. **选择模式**：法条+案例 / 仅法条 / 仅案例
4. **提问**：在对话框输入问题，获取带溯源和可信度评分的答案
5. **可选项**：侧边栏可切换 API 模式、关闭 Reranker、关闭查询重写

## 文档格式要求

### 法条文档

无特殊格式要求，系统会自动识别"第XX条"结构并按条切分。

### 案例文档（裁判文书）

需遵循中国裁判文书标准格式（最高人民法院法〔2016〕221 号规范），包含以下结构段落：

- `当事人信息：` / `原告诉称：` / `被告辩称：`
- `诉讼请求：`
- `事实与理由：`
- `经审理查明：`
- `本院认为：`
- `判决如下：` / `裁定如下：`

来自中国裁判文书网 (wenshu.court.gov.cn) 或北大法宝等数据库的文书天然符合此格式。

## 工作流程

```
文档上传 → PDF/TXT 解析 → 智能分段（法条按条/案例按段落）
  → Embedding → ChromaDB 存储

用户提问 → 查询重写 → 向量检索 (Top-10)
  → Reranker 精排 (Top-5) → Prompt 组装
  → LLM 生成 → 答案 + 溯源 + 可信度
```
