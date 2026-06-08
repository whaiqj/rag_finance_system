"""
graph_builder.py
知识图谱构建器 — 纯规则驱动，从条文文本中提取显式关系。
不依赖 LLM，仅用正则 + 词典匹配。
"""
import re
from typing import Dict, List, Any, Optional

from loguru import logger


# 匹配 "《XXX》第X条" 格式的引用
_REF_PATTERN = re.compile(
    r"《([^》]{2,60})》\s*第\s*([零一二三四五六七八九十百千\d]+)\s*条"
)

_DOC_REF_PATTERN = re.compile(r"《([^》]{2,80})》")
_AUTHORITY_TEXT_PATTERN = re.compile(
    r"(中国人民银行|国家金融监督管理总局|中国证券监督管理委员会|中国证监会|中国银保监会|中国银行保险监督管理委员会|金融监管总局|银保监会|证监会|人民银行)"
)
_RELATION_HINTS = {
    "根据": "basis",
    "依据": "basis",
    "依照": "basis",
    "按照": "basis",
    "遵循": "basis",
    "落实": "implements",
    "贯彻": "implements",
    "适用": "applies",
    "参照": "references",
    "配合": "supporting",
    "配套": "supporting",
    "上位法": "parent_law",
}

# 中文数字 → 阿拉伯数字映射
_CN_NUM_MAP: Dict[str, int] = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000,
}


def _parse_cn_number(text: str) -> str:
    """将"一百二十三"转为"123"。已是数字则直接返回。"""
    text = text.strip()
    if text.isdigit():
        return text
    result = 0
    current = 0
    for ch in text:
        val = _CN_NUM_MAP.get(ch)
        if val is None:
            continue
        if val >= 10:
            current = (current or 1) * val
            result += current
            current = 0
        else:
            current = val
    result += current
    return str(result)


def extract_references(text: str) -> List[Dict[str, str]]:
    """从条文文本中提取显式引用关系。

    Returns: [{"law": "公司法", "article": "16"}, ...]
    """
    refs: List[Dict[str, str]] = []
    seen: set = set()
    for match in _REF_PATTERN.finditer(text):
        law = match.group(1).strip()
        article_raw = match.group(2).strip()
        article = _parse_cn_number(article_raw)
        key = f"{law}|{article}"
        if key not in seen:
            seen.add(key)
            refs.append({"law": law, "article": article})
    return refs


# 匹配条文中的"第X条"以提取自身条文号
_ARTICLE_NUM_PATTERN = re.compile(r"第\s*([零一二三四五六七八九十百千\d]+)\s*条")


def extract_self_article_num(text: str) -> str:
    """从条文开头提取自身条文号。"""
    m = _ARTICLE_NUM_PATTERN.search(text[:200])
    if m:
        return _parse_cn_number(m.group(1))
    return ""


def extract_document_references(text: str) -> List[str]:
    """提取文档级法规引用，不要求带具体条文号。"""
    refs: List[str] = []
    seen: set[str] = set()
    for match in _DOC_REF_PATTERN.finditer(text):
        law = match.group(1).strip()
        if law and law not in seen:
            seen.add(law)
            refs.append(law)
    return refs


def infer_document_relation(text: str) -> str:
    """根据上下文关键词粗分类文档关系。"""
    preview = text[:400]
    for hint, relation_type in _RELATION_HINTS.items():
        if hint in preview:
            return relation_type
    return "related"


def extract_authorities(text: str, dictionary=None) -> List[str]:
    """从条文中提取监管机构名称。"""
    authorities: List[str] = []
    seen: set[str] = set()

    if dictionary:
        entities = dictionary.detect_entities(text)
        for authority in entities.get("authorities", []):
            if authority not in seen:
                seen.add(authority)
                authorities.append(authority)

    for match in _AUTHORITY_TEXT_PATTERN.finditer(text):
        raw = match.group(1).strip()
        authority = dictionary.resolve_authority(raw) if dictionary else raw
        authority = authority or raw
        if authority not in seen:
            seen.add(authority)
            authorities.append(authority)

    return authorities


class GraphBuilder:
    """图谱构建器 — 从 chunks + 词典实体构建 Neo4j 图。"""

    def __init__(self, kg, dictionary=None):
        """
        Args:
            kg: KnowledgeGraph 实例
            dictionary: FinanceDictionary 实例（可选，用于术语→条文定义匹配）
        """
        self.kg = kg
        self.dictionary = dictionary

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """从 chunk 列表构建完整的图谱子图。

        Steps:
            1. 写入 Document / Article 节点
            2. 建立 BELONGS_TO 边
            3. 正则提取 REFERENCES 边
            4. 词典匹配建立 DEFINES 边
        """
        if not self.kg or not self.kg._connected:
            return {}

        stats: Dict[str, int] = {}

        # Step 1: Document 节点
        doc_count = self.kg.upsert_documents_batch(chunks)
        stats["documents"] = doc_count
        logger.info(f"[图谱] Document 节点: {doc_count}")

        # Step 2: Article 节点
        art_count = self.kg.upsert_articles_batch(chunks)
        stats["articles"] = art_count
        logger.info(f"[图谱] Article 节点: {art_count}")

        # Step 3: BELONGS_TO 边
        belongs_count = self.kg.link_articles_to_documents_batch(chunks)
        stats["belongs_to"] = belongs_count
        logger.info(f"[图谱] BELONGS_TO 边: {belongs_count}")

        # Step 4: REFERENCES 边 (正则提取)
        ref_count = self._build_references(chunks)
        stats["references"] = ref_count
        logger.info(f"[图谱] REFERENCES 边: {ref_count}")

        # Step 5: 文档级法规关系
        doc_relation_count = self._build_document_relations(chunks)
        stats["document_relations"] = doc_relation_count
        logger.info(f"[图谱] RELATES_TO 边: {doc_relation_count}")

        # Step 6: 文档-机构关系
        issued_by_count = self._build_document_authorities(chunks)
        stats["issued_by"] = issued_by_count
        logger.info(f"[图谱] ISSUED_BY 边: {issued_by_count}")

        # Step 7: DEFINES / MENTIONS 边 (术语词典)
        term_stats = self._build_term_relations(chunks)
        stats.update(term_stats)
        logger.info(f"[图谱] DEFINES 边: {term_stats['defines']}")
        logger.info(f"[图谱] MENTIONS 边: {term_stats['mentions']}")

        return stats

    def _build_references(self, chunks: List[Dict[str, Any]]) -> int:
        """从每个 chunk 文本中提取《XXX》第X条引用，建立 REFERENCES 边。"""
        count = 0
        for c in chunks:
            chunk_id = c.get("chunk_id", "")
            if not chunk_id:
                continue
            text = c.get("text", "")
            refs = extract_references(text)
            for ref in refs:
                # 尝试用词典解析法规名（短名→全名）
                target_law = ref["law"]
                if self.dictionary:
                    resolved = self.dictionary.resolve_law_name(target_law)
                    if resolved:
                        target_law = resolved
                if self.kg.link_article_reference(chunk_id, target_law, ref["article"]):
                    count += 1
        return count

    def _build_document_relations(self, chunks: List[Dict[str, Any]]) -> int:
        """从条文文本提取文档级法规关系。"""
        count = 0
        seen: set[tuple[str, str, str]] = set()
        for c in chunks:
            doc_name = c.get("law_name", "") or c.get("source", "")
            if not doc_name:
                continue
            text = c.get("text", "")
            relation_type = infer_document_relation(text)
            for raw_ref in extract_document_references(text):
                target_doc = raw_ref
                if self.dictionary:
                    target_doc = self.dictionary.resolve_law_name(raw_ref) or raw_ref
                if not target_doc or target_doc == doc_name:
                    continue
                key = (doc_name, target_doc, relation_type)
                if key in seen:
                    continue
                seen.add(key)
                if self.kg.upsert_document(name=target_doc, doc_type="law", source=target_doc):
                    pass
                if self.kg.link_document_relation(doc_name, target_doc, relation_type):
                    count += 1
        return count

    def _build_document_authorities(self, chunks: List[Dict[str, Any]]) -> int:
        """从条文和来源信息提取文档-机构关系。"""
        count = 0
        seen: set[tuple[str, str]] = set()
        for c in chunks:
            doc_name = c.get("law_name", "") or c.get("source", "")
            if not doc_name:
                continue
            text = "\n".join([c.get("source", ""), c.get("text", "")])
            for authority in extract_authorities(text, dictionary=self.dictionary):
                key = (doc_name, authority)
                if key in seen:
                    continue
                seen.add(key)
                self.kg.upsert_authority(authority)
                if self.kg.link_document_authority(doc_name, authority):
                    count += 1
        return count

    def _build_term_relations(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """对每条 chunk 建立术语定义和提及关系。"""
        if not self.dictionary:
            return {"defines": 0, "mentions": 0}

        defines_count = 0
        mentions_count = 0
        for c in chunks:
            chunk_id = c.get("chunk_id", "")
            if not chunk_id:
                continue
            text = c.get("text", "")
            text_lower = text.lower()

            entities = self.dictionary.detect_entities(text)
            all_terms = set(entities.get("terms", []))
            for abbr, _full in self.dictionary._abbrev_to_full.items():
                if abbr in text_lower:
                    canonical = self.dictionary.resolve_term(abbr)
                    if canonical:
                        all_terms.add(canonical)

            defines_terms = set()
            if any(keyword in text for keyword in ["是指", "包括", "称为", "以下简称", "定义为"]):
                defines_terms = set(all_terms)

            for term in all_terms:
                if term in defines_terms:
                    if self.kg.link_term_definition(chunk_id, term):
                        defines_count += 1
                else:
                    if self.kg.link_term_mention(chunk_id, term):
                        mentions_count += 1

        return {"defines": defines_count, "mentions": mentions_count}

    def sync_dictionary_to_graph(self) -> Dict[str, int]:
        """将词典中的术语和机构写入图谱节点。"""
        if not self.kg or not self.kg._connected:
            return {}
        if not self.dictionary:
            return {}

        stats: Dict[str, int] = {}

        # Term 节点
        term_count = 0
        for name in self.dictionary._term_def:
            if self.kg.upsert_term(
                name=name,
                definition=self.dictionary._term_def.get(name, ""),
                category=self.dictionary._term_category.get(name, ""),
            ):
                term_count += 1
        stats["terms"] = term_count

        # Authority 节点
        auth_count = 0
        for full_name, info in self.dictionary._auth_full_info.items():
            if self.kg.upsert_authority(
                name=full_name,
                short=info.get("short", ""),
            ):
                auth_count += 1
        stats["authorities"] = auth_count

        logger.info(f"[图谱] 词典同步: {term_count} 术语, {auth_count} 机构")
        return stats
