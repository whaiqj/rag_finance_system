"""rewriter 单元测试。"""

import types
from unittest.mock import MagicMock, patch

from rag_finance_system.src.rewriter import QueryRewriter


class FakeInputs(dict):
    def __init__(self):
        super().__init__()
        self.input_ids = types.SimpleNamespace(shape=(1, 2))
    def to(self, device):
        return self


class TestQueryRewriter:
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_rewriter_init(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        tokenizer = MagicMock()
        model = MagicMock()
        model.to.return_value = model
        mock_tokenizer_from_pretrained.return_value = tokenizer
        mock_model_from_pretrained.return_value = model
        rewriter = QueryRewriter(model_path="mock-model", device="cpu")
        assert rewriter.device == "cpu"
        model.eval.assert_called_once()

    def test_rewrite(self):
        rewriter = QueryRewriter.__new__(QueryRewriter)
        rewriter.device = "cpu"
        rewriter.tokenizer = MagicMock()
        rewriter.model = MagicMock()
        rewriter.tokenizer.apply_chat_template.return_value = "prompt"
        rewriter.tokenizer.return_value = FakeInputs()
        rewriter.model.generate.return_value = [[101, 102, 201, 202]]
        rewriter.tokenizer.decode.return_value = "资本充足率 监管要求"
        result = rewriter.rewrite("资本充足率要求是什么")
        assert result == "资本充足率 监管要求"

    def test_rewrite_fallback_on_empty(self):
        rewriter = QueryRewriter.__new__(QueryRewriter)
        rewriter.device = "cpu"
        rewriter.tokenizer = MagicMock()
        rewriter.model = MagicMock()
        rewriter.tokenizer.apply_chat_template.return_value = "prompt"
        rewriter.tokenizer.return_value = FakeInputs()
        rewriter.model.generate.return_value = [[101, 102, 201, 202]]
        rewriter.tokenizer.decode.return_value = "  "
        result = rewriter.rewrite("原问题")
        assert result == "原问题"

    @patch("os.path.exists", return_value=True)
    @patch("peft.PeftModel.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_rewriter_lora_load(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained, mock_peft_from_pretrained, mock_exists):
        tokenizer = MagicMock()
        model = MagicMock()
        model.to.return_value = model
        mock_tokenizer_from_pretrained.return_value = tokenizer
        mock_model_from_pretrained.return_value = model
        mock_peft_from_pretrained.return_value = model
        QueryRewriter(model_path="mock-model", lora_path="mock-lora", device="cpu")
        mock_peft_from_pretrained.assert_called_once()

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_rewriter_4bit_quantization(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        tokenizer = MagicMock()
        model = MagicMock()
        mock_tokenizer_from_pretrained.return_value = tokenizer
        mock_model_from_pretrained.return_value = model
        QueryRewriter(model_path="mock-model", device="cuda", load_in_4bit=True)
        assert mock_model_from_pretrained.call_args.kwargs["load_in_4bit"] is True
