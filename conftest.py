"""Root conftest — 环境隔离，防止测试时加载真实模型/连接真实服务。"""

import os

# 在 pytest 收集测试前设置环境变量，阻止真实模型/服务加载
os.environ.setdefault("OCR_BACKEND", "none")
os.environ.setdefault("FINANCE_DICT_PATH", "")  # 由 tests/conftest.py 覆盖
