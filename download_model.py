"""
模型下载脚本：Reranker + Qwen2.5-7B 主模型 + (可选) Embedding

用法:
    py -3 download_model.py              # 下载所有模型
    py -3 download_model.py --reranker   # 仅下载 Reranker
    py -3 download_model.py --llm        # 仅下载 Qwen-7B 主模型
    py -3 download_model.py --embedding  # 仅下载 Embedding 模型
    py -3 download_model.py --verify     # 仅校验已下载模型
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).resolve().parent / "models"

MODEL_REPOS = {
    "reranker": {
        "repo_id": "BAAI/bge-reranker-v2-m3",
        "local_dir": str(MODELS_DIR / "bge-reranker-v2-m3"),
        "description": "BGE Reranker v2-m3 (交叉编码器精排)",
    },
    "embedding": {
        "repo_id": "BAAI/bge-small-zh-v1.5",
        "local_dir": str(MODELS_DIR / "bge-small-zh-v1.5"),
        "description": "BGE Embedding small-zh-v1.5 (512维向量)",
    },
    "llm": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "local_dir": str(MODELS_DIR / "Qwen2.5-7B-Int4"),
        "description": "Qwen2.5-7B-Instruct GPTQ Int4 量化 (~4.5GB显存)",
    },
}


def download_model(name: str, force: bool = False) -> bool:
    """下载单个模型，已存在则跳过（除非 force=True）。"""
    info = MODEL_REPOS[name]
    local = Path(info["local_dir"])

    if local.exists() and any(local.iterdir()):
        if force:
            print(f"[!] 强制重下载: {info['description']}")
        else:
            print(f"[✓] 已存在，跳过: {info['description']}  ({local})")
            return True

    print(f"[↓] 下载中: {info['description']}")
    print(f"    repo: {info['repo_id']}")
    print(f"    目标: {local}")

    try:
        # 排除不需要的大文件以加速下载
        ignore_patterns = ["*.msgpack", "*.h5", "flax_model.*", "rust_model.*"]
        snapshot_download(
            repo_id=info["repo_id"],
            local_dir=str(local),
            ignore_patterns=ignore_patterns,
            resume_download=True,
        )
        print(f"[✓] 下载完成: {info['description']}")
        return True
    except Exception as e:
        print(f"[✗] 下载失败: {info['description']}  — {e}", file=sys.stderr)
        return False


def verify_model(name: str) -> bool:
    """校验模型目录存在且包含必要文件。"""
    info = MODEL_REPOS[name]
    local = Path(info["local_dir"])

    if not local.exists() or not any(local.iterdir()):
        print(f"[✗] 未找到: {info['description']}  ({local})")
        return False

    # 检查关键文件
    has_config = (local / "config.json").exists()
    has_weights = (
        (local / "model.safetensors").exists()
        or (local / "pytorch_model.bin").exists()
        or list(local.glob("*.safetensors"))
    )
    has_tokenizer = (
        (local / "tokenizer.json").exists()
        or (local / "tokenizer_config.json").exists()
    )

    status = "✓"
    if not has_config:
        status = "✗"
    print(f"[{status}] {info['description']}")
    print(f"    路径: {local}")
    print(f"    config.json: {'✓' if has_config else '✗ 缺失'}")
    print(f"    权重文件:   {'✓' if has_weights else '✗ 缺失'}")
    print(f"    tokenizer:   {'✓' if has_tokenizer else '✗ 缺失'}")

    return has_config and has_weights and has_tokenizer


def main():
    parser = argparse.ArgumentParser(description="RAG Finance 模型下载工具")
    parser.add_argument("--reranker", action="store_true", help="仅下载 Reranker")
    parser.add_argument("--llm", action="store_true", help="仅下载 Qwen-7B 主模型")
    parser.add_argument("--embedding", action="store_true", help="仅下载 Embedding 模型")
    parser.add_argument("--verify", action="store_true", help="仅校验不下载")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 默认下载所有模型
    if not args.reranker and not args.llm and not args.embedding:
        args.reranker = True
        args.llm = True
        args.embedding = True

    targets = []
    if args.reranker:
        targets.append("reranker")
    if args.llm:
        targets.append("llm")
    if args.embedding:
        targets.append("embedding")

    if args.verify:
        print("=" * 50)
        print("模型校验")
        print("=" * 50)
        all_ok = True
        for name in targets:
            if not verify_model(name):
                all_ok = False
            print()
        if all_ok:
            print("全部模型校验通过！")
        else:
            print("存在缺失模型，请运行 py -3 download_model.py 下载。")
            sys.exit(1)
        return

    print("=" * 50)
    print("RAG Finance 模型下载")
    print(f"目标目录: {MODELS_DIR}")
    print("=" * 50)
    print()

    # 下载前先显示所需磁盘空间（仅提示）
    print("预计磁盘占用:")
    if args.reranker:
        print("  bge-reranker-v2-m3:         ~2.2 GB")
    if args.llm:
        print("  Qwen2.5-7B-Int4 (GPTQ):     ~5.0 GB")
    if args.embedding:
        print("  bge-small-zh-v1.5:          ~0.1 GB")
    print()

    failed = []
    for name in targets:
        if not download_model(name, force=args.force):
            failed.append(name)
        print()

    if failed:
        print(f"[✗] {len(failed)} 个模型下载失败: {', '.join(failed)}")
        print("提示: 中国大陆用户可在 HuggingFace 设置镜像或使用 modelscope 下载。")
        sys.exit(1)
    else:
        print("=" * 50)
        print("全部模型下载完成！")
        print(f"请确认 .env 中 LLM_MODEL_PATH 指向: {MODEL_REPOS['llm']['local_dir']}")
        print("=" * 50)


if __name__ == "__main__":
    main()