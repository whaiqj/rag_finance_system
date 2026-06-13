"""
llm.py
LLM推理模块
- 纯文本：本地Qwen2.5-7B-Int4（优先）→ DeepSeek API → 通义千问 API
- 多模态：本地Qwen2.5-VL（可选）→ Qwen-VL API (DashScope) → GPT-4V (OpenAI)
"""

import os
from typing import Iterator, List

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "./models/Qwen2.5-7B-Int4")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")
QWEN_VL_API_BASE_URL = os.getenv("QWEN_VL_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


# ========================
# 本地推理（GPTQ量化模型）
# ========================

class LocalLLM:
    """
    本地Qwen2.5-7B-Instruct-GPTQ-Int4推理
    显存需求：~4.5GB（4060 8GB够用）
    """

    def __init__(self, model_path: str = LLM_MODEL_PATH):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"加载本地LLM: {model_path}，设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",           # 自动分配GPU/CPU
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("本地LLM加载完成")

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """
        生成回答
        Args:
            messages: [{"role": "system/user/assistant", "content": str}]
            max_new_tokens: 最大生成token数
            temperature: 温度（金融场景用低温，减少幻觉）
        """
        import torch

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                repetition_penalty=1.1,
            )

        # 只取新生成的部分
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        """流式生成回答，逐 token yield。"""
        from threading import Thread

        from transformers import TextIteratorStreamer

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            repetition_penalty=1.1,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()


# ========================
# API调用（通义千问，备选）
# ========================

class QwenAPILLM:
    """
    通义千问API调用（当本地显存不足时使用）
    免费额度：qwen-plus 每分钟1M tokens
    """

    def __init__(self, api_key: str = DASHSCOPE_API_KEY, model: str = "qwen-plus"):
        if not api_key:
            raise ValueError("请在.env中设置DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.model = model
        logger.info(f"使用通义千问API: {model}")

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def generate_stream(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        """通义千问 API 流式生成。"""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


class DeepseekAPILLM:
    """
    Deepseek API调用（OpenAI兼容接口）
    """

    def __init__(self, api_key: str = DEEPSEEK_API_KEY, model: str = DEEPSEEK_MODEL, base_url: str = DEEPSEEK_API_BASE_URL):
        if not api_key:
            raise ValueError("请在.env中设置DEEPSEEK_API_KEY")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        logger.info(f"使用Deepseek API: {model}，base_url={base_url}")

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def generate_stream(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        """Deepseek API 流式生成。"""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# ========================
# 多模态 API（Qwen-VL，图片+文本理解）
# ========================

class QwenVLAPILLM:
    """
    通义千问多模态 API（Qwen-VL），支持图片+文本联合输入。

    用途：
    - 用户上传扫描件/手机拍屏照片直接提问
    - 含图表/印章的文档页面理解
    - 需要视觉能力的金融文档分析

    content 格式：纯文本 → str；含图片 → list[dict]
    [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
     {"type": "text", "text": "这张合同里有什么问题？"}]
    """

    def __init__(self, api_key: str = DASHSCOPE_API_KEY, model: str = QWEN_VL_MODEL,
                 base_url: str = QWEN_VL_API_BASE_URL):
        if not api_key:
            raise ValueError("请在.env中设置DASHSCOPE_API_KEY以使用Qwen-VL多模态API")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        logger.info(f"使用Qwen-VL多模态API: {model}")

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """
        生成回答。messages 中的 content 可以是：
        - str: 纯文本
        - list[dict]: 多模态内容 [{"type": "text", ...}, {"type": "image_url", ...}]
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def generate_stream(
        self,
        messages: list[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        """多模态流式生成。"""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    @staticmethod
    def encode_image_to_data_uri(image_path: str) -> str:
        """将本地图片编码为 base64 data URI，用于传入多模态 API。"""
        import base64
        import mimetypes

        mime, _ = mimetypes.guess_type(image_path)
        if not mime or not mime.startswith("image/"):
            mime = "image/png"

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def build_image_message(text: str, image_paths: list[str]) -> list[dict]:
        """构建包含图片的多模态消息。
        Returns: [{"role": "user", "content": [{"type": "image_url", ...}, {"type": "text", ...}]}]
        """
        content_parts = []
        for path in image_paths:
            data_uri = QwenVLAPILLM.encode_image_to_data_uri(path)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": data_uri},
            })
        content_parts.append({"type": "text", "text": text})
        return [{"role": "user", "content": content_parts}]


# ========================
# 工厂函数：自动选择LLM
# ========================

def get_llm(prefer_local: bool = True):
    """
    自动选择LLM：优先本地，失败则退回API
    """
    if prefer_local:
        try:
            return LocalLLM()
        except Exception as e:
            logger.warning(f"本地LLM加载失败: {e}，切换到API模式")

    if DEEPSEEK_API_KEY:
        return DeepseekAPILLM()
    if DASHSCOPE_API_KEY:
        return QwenAPILLM()

    raise RuntimeError(
        "无法初始化LLM：本地模型加载失败且未配置 Deepseek 或 通义千问 API Key。\n"
        "请在.env中设置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY，或确保本地模型路径正确。"
    )


def get_multimodal_llm():
    """
    获取多模态 LLM（适用于图片+文本联合输入场景）。
    优先级：本地 Qwen-VL 系列 → Qwen-VL API (DashScope)
    """
    # 本地多模态模型路径（可选，需手动下载 Qwen2.5-VL 系列）
    local_vl_path = os.getenv("QWEN_VL_LOCAL_PATH", "")
    if local_vl_path:
        try:
            logger.info(f"尝试加载本地多模态模型: {local_vl_path}")
            # 注意：Qwen2.5-VL 需要 transformers >= 4.49 + qwen-vl-utils
            # 如果环境不支持，会回退 API
            from transformers import AutoModelForVision2Seq, AutoProcessor

            processor = AutoProcessor.from_pretrained(local_vl_path, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                local_vl_path, device_map="auto", trust_remote_code=True
            )
            model.eval()
            logger.info("本地多模态模型加载完成")
            # 这里可以包装成类似接口的类，暂时先保持简洁：本地不可用时直接 API
            # TODO: 封装 LocalVLModel 类
        except Exception as e:
            logger.warning(f"本地多模态模型加载失败: {e}，回退 API")

    if DASHSCOPE_API_KEY:
        return QwenVLAPILLM()
    if os.getenv("OPENAI_API_KEY"):
        return QwenVLAPILLM(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    raise RuntimeError(
        "无法初始化多模态LLM：本地模型不可用且未配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY。\n"
        "请下载 Qwen2.5-VL 或设置 Qwen-VL API Key。"
    )
