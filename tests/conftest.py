"""项目级共享 fixtures。"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── 路径常量 ──

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DICT_PATH = FIXTURES_DIR / "sample_dictionary.json"


# ── 词典 fixtures ──


@pytest.fixture(scope="session")
def sample_dict_path():
    """精简词典 JSON 文件路径。"""
    return str(SAMPLE_DICT_PATH)


@pytest.fixture(scope="session")
def sample_dict_data():
    """精简词典原始 JSON 数据。"""
    with open(SAMPLE_DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def finance_dict(sample_dict_path):
    """已加载的 FinanceDictionary 实例（使用精简词典）。"""
    os.environ["FINANCE_DICT_PATH"] = sample_dict_path
    from rag_finance_system.src.dictionary import FinanceDictionary

    return FinanceDictionary(dict_path=sample_dict_path)


# ── chunk fixtures ──


@pytest.fixture
def sample_chunks():
    """5 个标准法律 chunk。"""
    from tests.fixtures.sample_chunks import make_chunks

    return make_chunks(5)


@pytest.fixture
def single_chunk():
    """单个标准法律 chunk。"""
    from tests.fixtures.sample_chunks import make_chunk

    return make_chunk()


# ── Mock 服务 fixtures ──


@pytest.fixture
def mock_milvus_client():
    """Mock MilvusClient，覆盖常用方法。"""
    client = MagicMock()
    client.has_collection.return_value = False
    client.create_collection.return_value = None
    client.create_index.return_value = None
    client.load_collection.return_value = None
    client.insert.return_value = {"insert_count": 1}
    client.search.return_value = [[{"id": "1", "distance": 0.85, "entity": {"text": "测试"}}]]
    client.flush.return_value = None
    client.get_collection_stats.return_value = {"row_count": 100}
    client.drop_collection.return_value = None
    client.describe_collection.return_value = {"fields": []}
    client.list_indexes.return_value = []
    return client


@pytest.fixture
def mock_embedder():
    """Mock Embedder，encode 返回固定 512 维向量。"""
    import numpy as np

    embedder = MagicMock()
    embedder.dim = 512
    embedder.encode.return_value = np.random.rand(1, 512).astype(np.float32)
    embedder.encode_query.return_value = np.random.rand(1, 512).astype(np.float32)
    return embedder


@pytest.fixture
def mock_reranker():
    """Mock Reranker。"""
    reranker = MagicMock()
    reranker.rerank.return_value = [
        {"text": "相关文档1", "score": 0.95, "source": "法1.txt"},
        {"text": "相关文档2", "score": 0.80, "source": "法2.txt"},
    ]
    return reranker


@pytest.fixture
def mock_llm():
    """Mock LLM。"""
    llm = MagicMock()
    llm.generate.return_value = "根据公司法，股东应履行出资义务。"
    llm.generate_stream.return_value = iter(["根据", "公司法", "，", "股东", "应", "履行", "出资", "义务", "。"])
    return llm


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver。"""
    driver = MagicMock()
    driver.verify_connectivity.return_value = None
    session = MagicMock()
    result = MagicMock()
    result.__iter__ = MagicMock(return_value=iter([]))
    session.run.return_value = result
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = session
    return driver


@pytest.fixture
def tmp_dict_json(tmp_path):
    """可写的临时词典 JSON 文件路径。"""
    with open(SAMPLE_DICT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    p = tmp_path / "test_dictionary.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(p)
