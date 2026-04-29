"""
document_processor.py
PDF解析 + 文本分段模块
支持：PDF文本层提取（pdfplumber）、章节层级保留、自定义文本分段
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))


class RecursiveCharacterTextSplitter:
    """本地实现的文本分段器（增强版：法律/中文友好）。"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        separators=None,
        keep_separator: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # ⚠️ 不再把 "第" 作为普通分隔符，避免破坏“第XX条”结构
        self.separators = separators or ["\n\n", "\n", "。", "；", "！", "？", "，", " "]
        self.keep_separator = keep_separator

    # ===================== 核心：对外接口 =====================
    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        # 1️⃣ 强制优先：按“第XX条”切分（命中则绝不跨条合并）
        articles = self._split_by_article(text)

        # 未命中条文结构 → 走原递归
        if len(articles) == 1:
            return self._split_recursive(text)

        # 2️⃣ 条内再切分，并且“每条独立合并”，避免跨条拼接
        final_chunks: list[str] = []
        for art in articles:
            art = art.strip()
            if not art:
                continue

            if len(art) <= self.chunk_size:
                # 单条较短，直接作为一个chunk（仍保留条头）
                final_chunks.append(art)
            else:
                # 条内递归切分
                sub_chunks = self._split_recursive(art)
                # ⚠️ 关键：只在“条内”做merge，不与其他条混合
                sub_chunks = self._merge_chunks(sub_chunks)
                final_chunks.extend(sub_chunks)

        return final_chunks

    # ===================== 新增：按条切分 =====================
    def _split_by_article(self, text: str) -> list[str]:
        """按“第XX条”切分，保留条头"""
        pattern = r"(第[零一二三四五六七八九十百千万\d]+条[^第]*)"
        matches = re.findall(pattern, text)
        if not matches:
            return [text]
        return [m.strip() for m in matches if m.strip()]

    # ===================== 原递归逻辑（轻微改造） =====================
    def _split_recursive(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        for separator in self.separators:
            if not separator:
                continue

            if separator in text:
                parts = text.split(separator)

                if len(parts) == 1:
                    continue

                chunks = []
                for i, part in enumerate(parts):
                    if not part.strip():
                        continue

                    piece = part
                    if self.keep_separator and i < len(parts) - 1:
                        piece += separator

                    piece = piece.strip()

                    # 防止递归不收敛
                    if len(piece) >= len(text):
                        return self._split_by_char(text)

                    chunks.extend(self._split_recursive(piece))

                if len(chunks) > 1:
                    return self._merge_chunks(chunks)

        return self._split_by_char(text)

    # ===================== 底层工具 =====================
    def _split_by_char(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        merged = []
        current = ""
        for chunk in chunks:
            if not current:
                current = chunk
                continue
            if len(current) + len(chunk) <= self.chunk_size:
                current += chunk
            else:
                merged.append(current)
                overlap = current[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current = overlap + chunk
        if current:
            merged.append(current)
        return merged


class DocumentProcessor:
    """PDF解析与文本分段"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " "],
            keep_separator=True,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        提取PDF全文，保留页码信息
        Returns:
            {
                "title": str,       # 文件名（作为文档标题）
                "pages": [          # 按页的文本列表
                    {"page_num": int, "text": str}
                ],
                "full_text": str    # 合并全文
            }
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        pages = []
        full_text_parts = []

        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"解析PDF: {path.name}，共{len(pdf.pages)}页")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text = self._clean_text(text)
                    pages.append({"page_num": i + 1, "text": text})
                    full_text_parts.append(text)

        return {
            "title": path.stem,
            "file_path": str(path.resolve()),
            "pages": pages,
            "full_text": "\n\n".join(full_text_parts),
        }

    def _clean_text(self, text: str) -> str:
        """清理文本：去除多余空白，修正常见OCR错误"""
        lines = [line.strip() for line in text.splitlines()]
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    cleaned_lines.append("")
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        return "\n".join(cleaned_lines).strip()

    def split_into_chunks(self, doc_info: Dict[str, Any], doc_type: str = "law") -> List[Dict[str, Any]]:
        """
        将文档全文分段，返回带元数据的chunk列表
        每个chunk包含：text, source, chunk_id, article_num(条文号，如有), doc_type
        """
        full_text = doc_info["full_text"]
        chunks_text = self.splitter.split_text(full_text)

        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            article_match = re.search(r"第[零一二三四五六七八九十百\d]+条", chunk_text)
            article_num = article_match.group(0) if article_match else ""

            chunks.append({
                "chunk_id": f"{doc_info['title']}_chunk_{idx:04d}",
                "text": chunk_text,
                "source": doc_info["title"],
                "file_path": doc_info["file_path"],
                "article_num": article_num,
                "chunk_index": idx,
                "doc_type": doc_type,
            })

        logger.info(f"文档 [{doc_info['title']}] 分段完成，共 {len(chunks)} 个chunk")
        return chunks

    def process_pdf(self, pdf_path: str, doc_type: str = "law") -> List[Dict[str, Any]]:
        doc_info = self.extract_text_from_pdf(pdf_path)
        return self.split_into_chunks(doc_info, doc_type=doc_type)

    def extract_text_from_txt(self, txt_path: str) -> Dict[str, Any]:
        path = Path(txt_path)
        if not path.exists():
            raise FileNotFoundError(f"文本文件不存在: {txt_path}")

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        text = self._clean_text(text)
        return {
            "title": path.stem,
            "file_path": str(path.resolve()),
            "pages": [{"page_num": 1, "text": text}],
            "full_text": text,
        }

    def process_txt(self, txt_path: str, doc_type: str = "law") -> List[Dict[str, Any]]:
        doc_info = self.extract_text_from_txt(txt_path)
        return self.split_into_chunks(doc_info, doc_type=doc_type)

    def process_file(self, file_path: str, doc_type: str = "law") -> List[Dict[str, Any]]:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self.process_pdf(file_path, doc_type=doc_type)
        if suffix == ".txt":
            return self.process_txt(txt_path=file_path, doc_type=doc_type)
        raise ValueError(f"不支持的文件类型: {suffix}，仅支持PDF和TXT")

    def process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        all_chunks = []
        files = list(Path(dir_path).glob("**/*"))
        supported_files = [f for f in files if f.suffix.lower() in {".pdf", ".txt"}]
        logger.info(f"发现 {len(supported_files)} 个支持的文件（PDF/TXT）")
        for file_path in supported_files:
            try:
                chunks = self.process_file(str(file_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"处理 {file_path.name} 失败: {e}")
        return all_chunks
      
    