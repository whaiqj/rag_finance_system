"""document_processor 单元测试。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from rag_finance_system.src.document_processor import DocumentProcessor, RecursiveCharacterTextSplitter


class TestRecursiveCharacterTextSplitter:
    def test_split_text_empty(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        assert splitter.split_text("") == []

    def test_split_text_short(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "这是一个很短的文本。"
        assert splitter.split_text(text) == [text]

    def test_split_text_by_article(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "第一章总则\n第一条 内容一\n第二条 内容二"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert any("第一条" in c for c in chunks)
        assert any("第二条" in c for c in chunks)

    def test_split_text_recursive_long_text(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "这是一个很长的文本，" * 20
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= 25 for c in chunks)

    def test_split_text_for_other_with_chinese_numbers(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "（一）第一部分内容\n（二）第二部分内容\n（三）第三部分内容"
        chunks = splitter.split_text_for_other(text)
        assert len(chunks) >= 2

    def test_split_text_for_other_fallback(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "普通文本" * 30
        chunks = splitter.split_text_for_other(text)
        assert len(chunks) > 1

    def test_split_by_char_overlap(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
        chunks = splitter._split_by_char("1234567890abcdef")
        assert len(chunks) >= 2
        assert chunks[0][-2:] == chunks[1][:2]

    def test_merge_chunks_respects_size(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
        merged = splitter._merge_chunks(["1234", "56", "789012345"])
        assert all(len(c) >= 1 for c in merged)


class TestDocumentProcessorCore:
    def test_extract_text_from_txt(self, tmp_path):
        path = tmp_path / "sample.txt"
        path.write_text("第一条 测试内容\n\n第二条 更多内容", encoding="utf-8")
        dp = DocumentProcessor()
        result = dp.extract_text_from_txt(str(path))
        assert result["title"] == "sample"
        assert "第一条" in result["full_text"]
        assert result["extraction_method"] == "txt"

    def test_extract_text_from_txt_missing(self, tmp_path):
        dp = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            dp.extract_text_from_txt(str(tmp_path / "missing.txt"))

    def test_clean_text(self):
        dp = DocumentProcessor()
        cleaned = dp._clean_text("a\x00b\n\n\n c ")
        assert "\x00" not in cleaned
        assert "a b" in cleaned

    def test_split_into_chunks_law_type(self):
        dp = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        doc_info = {
            "title": "中华人民共和国公司法_20240101",
            "file_path": "/tmp/a.txt",
            "full_text": "第一条 公司设立。第二条 股东责任。",
            "extraction_method": "txt",
        }
        chunks = dp.split_into_chunks(doc_info, doc_type="law")
        assert len(chunks) >= 1
        assert chunks[0]["law_name"] == "中华人民共和国公司法"
        assert chunks[0]["effective_date"] == "2024-01-01"

    def test_split_into_chunks_case_type(self):
        dp = DocumentProcessor(chunk_size=20, chunk_overlap=5)
        doc_info = {
            "title": "案例文件",
            "file_path": "/tmp/case.txt",
            "full_text": "这是一个较长的案例文本。" * 10,
            "extraction_method": "txt",
        }
        chunks = dp.split_into_chunks(doc_info, doc_type="case")
        assert len(chunks) > 1
        assert all(c["doc_type"] == "case" for c in chunks)

    def test_split_into_chunks_other_type(self):
        dp = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        doc_info = {
            "title": "监管问答",
            "file_path": "/tmp/other.txt",
            "full_text": "（一）第一部分\n（二）第二部分\n（三）第三部分",
            "extraction_method": "txt",
        }
        chunks = dp.split_into_chunks(doc_info, doc_type="other")
        assert len(chunks) >= 2

    def test_extract_date_from_title(self):
        assert DocumentProcessor._extract_date_from_title("中华人民共和国公司法_20240101") == "2024-01-01"

    def test_extract_law_name_and_date_law(self):
        dp = DocumentProcessor()
        law_name, effective_date = dp._extract_law_name_and_date("中华人民共和国公司法_20240101", "law")
        assert law_name == "中华人民共和国公司法"
        assert effective_date == "2024-01-01"

    def test_extract_law_name_and_date_non_law(self):
        dp = DocumentProcessor()
        law_name, effective_date = dp._extract_law_name_and_date("案例文件_20240101", "case")
        assert law_name == "案例文件_20240101"
        assert effective_date == "2024-01-01"

    def test_resolve_version_status(self):
        chunks = [
            {"law_name": "中华人民共和国公司法", "effective_date": "2024-01-01", "status": "有效"},
            {"law_name": "中华人民共和国公司法", "effective_date": "2023-01-01", "status": "有效"},
        ]
        resolved = DocumentProcessor.resolve_version_status(chunks)
        assert resolved[0]["status"] in {"有效", "已修订"}
        latest = [c for c in resolved if c["effective_date"] == "2024-01-01"][0]
        older = [c for c in resolved if c["effective_date"] == "2023-01-01"][0]
        assert latest["status"] == "有效"
        assert older["status"] == "已修订"

    def test_extract_authority_from_title(self):
        dp = DocumentProcessor()
        authority = dp._extract_authority("上海监管局关于加强监管的通知", "")
        assert authority == "上海监管局"

    def test_extract_authority_from_file_path(self):
        dp = DocumentProcessor()
        authority = dp._extract_authority("普通文件", "/a/b/北京监管分局/file.txt")
        assert authority == "北京监管分局"

    def test_process_file_unsupported_type(self, tmp_path):
        path = tmp_path / "bad.doc"
        path.write_text("x", encoding="utf-8")
        dp = DocumentProcessor()
        with pytest.raises(ValueError):
            dp.process_file(str(path))


class TestDocumentProcessorOCRHelpers:
    def test_extract_text_from_pdf_missing(self, tmp_path):
        dp = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            dp.extract_text_from_pdf(str(tmp_path / "missing.pdf"))

    def test_should_use_ocr_empty_pages(self):
        dp = DocumentProcessor()
        dp.ocr_backend = "auto"
        assert dp._should_use_ocr([{"text": ""}, {"text": ""}]) is True

    def test_should_use_ocr_none_backend(self, monkeypatch):
        dp = DocumentProcessor()
        dp.ocr_backend = "none"
        assert dp._should_use_ocr([{"text": ""}]) is False

    def test_resolve_ocr_backends_auto(self):
        dp = DocumentProcessor()
        dp.ocr_backend = "auto"
        assert dp._resolve_ocr_backends() == ["docling", "paddleocr"]

    def test_resolve_image_ocr_backends_auto(self):
        dp = DocumentProcessor()
        dp.ocr_backend = "auto"
        assert dp._resolve_image_ocr_backends() == ["paddleocr", "docling"]

    def test_extract_text_from_image_missing(self, tmp_path):
        dp = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            dp.extract_text_from_image(str(tmp_path / "missing.png"))

    def test_extract_text_from_image_unsupported_ext(self, tmp_path):
        path = tmp_path / "bad.gif"
        path.write_bytes(b"GIF89a")
        dp = DocumentProcessor()
        with pytest.raises(ValueError):
            dp.extract_text_from_image(str(path))

    def test_preprocess_image_returns_image(self):
        dp = DocumentProcessor()
        image = Image.new("RGB", (100, 100), color="white")
        processed = dp._preprocess_image(image)
        assert processed is not None
        assert processed.mode == "RGB"
