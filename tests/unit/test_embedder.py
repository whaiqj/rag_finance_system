"""embedder 单元测试。"""

from unittest.mock import MagicMock, patch

import torch

from rag_finance_system.src.embedder import Embedder, Reranker


class TestEmbedder:
    @patch("sentence_transformers.SentenceTransformer")
    def test_embedder_init(self, mock_st):
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 512
        mock_st.return_value = model

        embedder = Embedder(model_path="mock-model")
        assert embedder.dimension == 512
        mock_st.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_single_text(self, mock_st):
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 512
        model.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_st.return_value = model

        embedder = Embedder(model_path="mock-model")
        result = embedder.encode("测试文本")
        assert result == [[0.1, 0.2, 0.3]]

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_batch(self, mock_st):
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 512
        model.encode.return_value.tolist.return_value = [[0.1], [0.2]]
        mock_st.return_value = model

        embedder = Embedder(model_path="mock-model")
        result = embedder.encode(["文本1", "文本2"])
        assert len(result) == 2

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_query_prefix(self, mock_st):
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 512
        model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
        mock_st.return_value = model

        embedder = Embedder(model_path="mock-model")
        result = embedder.encode_query("资本充足率")
        called_texts = model.encode.call_args[0][0]
        assert called_texts[0].startswith("为这个句子生成表示以用于检索相关文章：")
        assert result == [0.1, 0.2]

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_documents(self, mock_st):
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 512
        model.encode.return_value.tolist.return_value = [[0.1], [0.2]]
        mock_st.return_value = model

        embedder = Embedder(model_path="mock-model")
        result = embedder.encode_documents(["文档1", "文档2"])
        assert len(result) == 2
        assert model.encode.call_args.kwargs["show_progress_bar"] is True


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeLogits:
    def __init__(self, values):
        self.values = values

    def squeeze(self, dim):
        return self

    def squeeze_(self, dim):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _FakeModelOutput:
    def __init__(self, values):
        self.logits = torch.tensor(values, dtype=torch.float32)


class TestReranker:
    @patch("rag_finance_system.src.embedder.AutoModelForSequenceClassification.from_pretrained")
    @patch("rag_finance_system.src.embedder.AutoTokenizer.from_pretrained")
    def test_reranker_init(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        tokenizer = MagicMock()
        mock_tokenizer_from_pretrained.return_value = tokenizer
        model = MagicMock()
        model.to.return_value = model
        mock_model_from_pretrained.return_value = model

        reranker = Reranker(model_path="mock-reranker")
        assert reranker.batch_size > 0
        model.eval.assert_called_once()

    @patch("rag_finance_system.src.embedder.AutoModelForSequenceClassification.from_pretrained")
    @patch("rag_finance_system.src.embedder.AutoTokenizer.from_pretrained")
    def test_rerank_empty_documents(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        tokenizer = MagicMock()
        mock_tokenizer_from_pretrained.return_value = tokenizer
        model = MagicMock()
        model.to.return_value = model
        mock_model_from_pretrained.return_value = model

        reranker = Reranker(model_path="mock-reranker")
        assert reranker.rerank("query", []) == []

    @patch("torch.sigmoid")
    @patch("rag_finance_system.src.embedder.AutoModelForSequenceClassification.from_pretrained")
    @patch("rag_finance_system.src.embedder.AutoTokenizer.from_pretrained")
    def test_rerank_returns_sorted(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained, mock_sigmoid):
        tokenizer = MagicMock()
        tokenizer.return_value = _FakeInputs()
        mock_tokenizer_from_pretrained.return_value = tokenizer

        model = MagicMock()
        model.to.return_value = model
        model.return_value = _FakeModelOutput([[0.2], [0.8], [0.5]])
        mock_model_from_pretrained.return_value = model

        class _FakeSigmoidResult:
            def __init__(self, vals):
                self.vals = vals
            def cpu(self):
                return self
            def tolist(self):
                return self.vals

        mock_sigmoid.return_value = _FakeSigmoidResult([0.2, 0.8, 0.5])

        reranker = Reranker(model_path="mock-reranker", batch_size=10)
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"])
        assert results[0]["score"] >= results[1]["score"]
        assert results[0]["index"] == 1

    @patch("torch.sigmoid")
    @patch("rag_finance_system.src.embedder.AutoModelForSequenceClassification.from_pretrained")
    @patch("rag_finance_system.src.embedder.AutoTokenizer.from_pretrained")
    def test_rerank_top_n(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained, mock_sigmoid):
        tokenizer = MagicMock()
        tokenizer.return_value = _FakeInputs()
        mock_tokenizer_from_pretrained.return_value = tokenizer

        model = MagicMock()
        model.to.return_value = model
        model.return_value = _FakeModelOutput([[0.2], [0.8], [0.5]])
        mock_model_from_pretrained.return_value = model

        class _FakeSigmoidResult:
            def __init__(self, vals):
                self.vals = vals
            def cpu(self):
                return self
            def tolist(self):
                return self.vals

        mock_sigmoid.return_value = _FakeSigmoidResult([0.2, 0.8, 0.5])

        reranker = Reranker(model_path="mock-reranker", batch_size=10)
        results = reranker.rerank("query", ["doc1", "doc2", "doc3"], top_n=2)
        assert len(results) == 2
