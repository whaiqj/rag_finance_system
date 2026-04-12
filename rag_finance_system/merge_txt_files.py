import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent
source_dir = base_dir / "src" / "txt_files"
output_file = base_dir / "merged_txt_files.txt"

if not source_dir.exists():
    raise FileNotFoundError(f"源目录不存在: {source_dir}")

text_files = sorted(source_dir.glob("*.txt"))
if not text_files:
    raise FileNotFoundError(f"在目录 {source_dir} 中未找到任何 .txt 文件")

with output_file.open("w", encoding="utf-8") as out_f:
    for idx, txt_path in enumerate(text_files, start=1):
        out_f.write(f"# 文件 {idx}: {txt_path.name}\n")
        out_f.write(txt_path.read_text(encoding="utf-8"))
        out_f.write("\n\n")

print(f"已合并 {len(text_files)} 个文件到 {output_file}")
