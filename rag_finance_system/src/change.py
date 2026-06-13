import os

from docx import Document

base_dir = os.path.dirname(__file__)
input_dir = os.path.join(base_dir, "fdata")
output_dir = os.path.join(base_dir, "txt_files")

if not os.path.isdir(input_dir):
    raise FileNotFoundError(f"输入目录不存在: {input_dir}\n请将 .docx 文件放在此目录，或修改 input_dir 变量。")

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".docx"):
        doc_path = os.path.join(input_dir, filename)
        txt_path = os.path.join(output_dir, filename.replace(".docx", ".txt"))

        doc = Document(doc_path)
        full_text = []

        for para in doc.paragraphs:
            full_text.append(para.text)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(full_text))

print(f"转换完成，输出目录: {output_dir}")
