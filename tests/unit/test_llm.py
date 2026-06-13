"""llm 单元测试。"""

import types
from unittest.mock import MagicMock, patch

from rag_finance_system.src import llm as llm_module
from rag_finance_system.src.llm import DeepseekAPILLM, LocalLLM, QwenAPILLM, get_llm


class FakeInputs(dict):
    def __init__(self):
        super().__init__()
        self.input_ids = types.SimpleNamespace(shape=(1, 2))
    def to(self, device):
        return self


class TestLocalLLM:
    def test_generate(self):
        llm = LocalLLM.__new__(LocalLLM)
        llm.device = "cpu"
        llm.tokenizer = MagicMock()
        llm.model = MagicMock()
        llm.tokenizer.apply_chat_template.return_value = "prompt"
        llm.tokenizer.return_value = FakeInputs()
        llm.model.generate.return_value = [[101, 102, 201, 202]]
        llm.tokenizer.decode.return_value = "回答内容"
        result = llm.generate([{"role": "user", "content": "你好"}])
        assert result == "回答内容"

    @patch("threading.Thread")
    @patch("transformers.TextIteratorStreamer")
    def test_generate_stream(self, mock_streamer_cls, mock_thread_cls):
        llm = LocalLLM.__new__(LocalLLM)
        llm.device = "cpu"
        llm.tokenizer = MagicMock()
        llm.model = MagicMock()
        llm.tokenizer.apply_chat_template.return_value = "prompt"
        llm.tokenizer.return_value = FakeInputs()
        mock_streamer_cls.return_value = iter(["你", "好"])
        thread = MagicMock()
        mock_thread_cls.return_value = thread
        result = list(llm.generate_stream([{"role": "user", "content": "你好"}]))
        assert result == ["你", "好"]
        thread.start.assert_called_once()
        thread.join.assert_called_once()


class TestAPILLMs:
    def test_qwen_api_no_key_raises(self):
        try:
            QwenAPILLM(api_key="")
            assert False
        except ValueError:
            assert True

    def test_deepseek_api_no_key_raises(self):
        try:
            DeepseekAPILLM(api_key="")
            assert False
        except ValueError:
            assert True

    def test_qwen_api_generate(self):
        fake_response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="qwen答复"))])
        fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=MagicMock(return_value=fake_response))))
        fake_openai = types.SimpleNamespace(OpenAI=MagicMock(return_value=fake_client))
        with patch.dict("sys.modules", {"openai": fake_openai}):
            llm = QwenAPILLM(api_key="k")
            assert llm.generate([{"role": "user", "content": "hi"}]) == "qwen答复"

    def test_qwen_api_generate_stream(self):
        chunks = [types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="你"))]), types.SimpleNamespace(choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="好"))])]
        fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=MagicMock(return_value=chunks))))
        fake_openai = types.SimpleNamespace(OpenAI=MagicMock(return_value=fake_client))
        with patch.dict("sys.modules", {"openai": fake_openai}):
            llm = QwenAPILLM(api_key="k")
            assert list(llm.generate_stream([{"role": "user", "content": "hi"}])) == ["你", "好"]

    def test_deepseek_api_generate(self):
        fake_response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="deepseek答复"))])
        fake_client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=MagicMock(return_value=fake_response))))
        fake_openai = types.SimpleNamespace(OpenAI=MagicMock(return_value=fake_client))
        with patch.dict("sys.modules", {"openai": fake_openai}):
            llm = DeepseekAPILLM(api_key="k")
            assert llm.generate([{"role": "user", "content": "hi"}]) == "deepseek答复"


class TestGetLLM:
    def test_get_llm_local_fallback_to_api(self):
        with patch.object(llm_module, "DEEPSEEK_API_KEY", "abc"):
            with patch("rag_finance_system.src.llm.LocalLLM", side_effect=RuntimeError("boom")):
                with patch("rag_finance_system.src.llm.DeepseekAPILLM", return_value="deepseek"):
                    assert get_llm(prefer_local=True) == "deepseek"

    def test_get_llm_prefer_api(self):
        with patch.object(llm_module, "DEEPSEEK_API_KEY", "abc"):
            with patch("rag_finance_system.src.llm.DeepseekAPILLM", return_value="deepseek"):
                assert get_llm(prefer_local=False) == "deepseek"

    def test_get_llm_all_fail_raises(self):
        with patch.object(llm_module, "DEEPSEEK_API_KEY", ""):
            with patch.object(llm_module, "DASHSCOPE_API_KEY", ""):
                with patch("rag_finance_system.src.llm.LocalLLM", side_effect=RuntimeError("boom")):
                    try:
                        get_llm(prefer_local=True)
                        assert False
                    except RuntimeError:
                        assert True
