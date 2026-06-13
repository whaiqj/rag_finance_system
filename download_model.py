import os
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_RERANKER_PATH = Path(__file__).resolve().parent / "models" / "bge-reranker-v2-m3"
MODEL_PATH = Path(os.getenv("RERANKER_MODEL_PATH", str(DEFAULT_RERANKER_PATH)))

print(f"开始加载 tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)

print("开始加载 model...")
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

print("加载成功！")
