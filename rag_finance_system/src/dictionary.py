"""
dictionary.py
金融法规词典 — 术语归一、别名召回、法规名命中、机构简称映射

用法:
    from .dictionary import FinanceDictionary
    fd = FinanceDictionary()
    fd.resolve_law_name("公司法")  # → "中华人民共和国公司法"
    fd.expand_query("NPL怎么算")   # → "NPL 不良贷款率 怎么算"
    fd.detect_entities("根据公司法，股东责任是什么")
        # → {"terms": ["股东责任"], "law_names": ["中华人民共和国公司法"], "authorities": []}
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


class FinanceDictionary:
    """金融法规词典 — 术语、法规名、机构名、缩写的统一查询入口。"""

    def __init__(self, dict_path: Optional[str] = None):
        if dict_path is None:
            dict_path = os.getenv(
                "FINANCE_DICT_PATH",
                str(Path(__file__).resolve().parent.parent.parent / "data" / "finance_dictionary.json"),
            )
        self._dict_path = dict_path

        with open(dict_path, "r", encoding="utf-8") as f:
            self._raw = json.load(f)

        # ── 反向索引 ──
        self._term_alias_to_canonical: Dict[str, str] = {}    # alias → canonical term
        self._term_aliases: Dict[str, List[str]] = {}          # canonical → all aliases
        self._term_def: Dict[str, str] = {}                    # canonical → definition
        self._term_category: Dict[str, str] = {}               # canonical → category

        for term_name, info in self._raw.get("terms", {}).items():
            self._term_def[term_name] = info.get("definition", "")
            self._term_category[term_name] = info.get("category", "")
            aliases = info.get("aliases", [term_name])
            self._term_aliases[term_name] = aliases
            for alias in aliases:
                alias_lower = alias.strip().lower()
                existing = self._term_alias_to_canonical.get(alias_lower)
                if existing and existing != term_name:
                    # 同一别名命中多个术语 → 保留较长的（更精确）
                    if len(term_name) > len(existing):
                        self._term_alias_to_canonical[alias_lower] = term_name
                else:
                    self._term_alias_to_canonical[alias_lower] = term_name

        self._law_alias_to_full: Dict[str, str] = {}           # alias → full law name
        self._law_full_info: Dict[str, dict] = {}              # full → {short, category, ...}
        for full_name, info in self._raw.get("law_names", {}).items():
            self._law_full_info[full_name] = info
            self._law_alias_to_full[full_name.lower()] = full_name
            short = info.get("short", "")
            if short:
                self._law_alias_to_full[short.lower()] = full_name
            for alias in info.get("aliases", []):
                alias_lower = alias.strip().lower()
                if alias_lower not in self._law_alias_to_full:
                    self._law_alias_to_full[alias_lower] = full_name

        self._auth_alias_to_full: Dict[str, str] = {}          # alias → full authority
        self._auth_full_info: Dict[str, dict] = {}             # full → {short, historical, ...}
        for full_name, info in self._raw.get("authorities", {}).items():
            self._auth_full_info[full_name] = info
            self._auth_alias_to_full[full_name.lower()] = full_name
            short = info.get("short", "")
            if short:
                self._auth_alias_to_full[short.lower()] = full_name
            for alias in info.get("aliases", []):
                alias_lower = alias.strip().lower()
                if alias_lower not in self._auth_alias_to_full:
                    self._auth_alias_to_full[alias_lower] = full_name
            # 历史名称也纳入映射
            for hist in info.get("historical", []):
                hist_lower = hist.strip().lower()
                if hist_lower not in self._auth_alias_to_full:
                    self._auth_alias_to_full[hist_lower] = full_name

        self._abbrev_to_full: Dict[str, str] = {
            k.strip().lower(): v for k, v in self._raw.get("abbreviations", {}).items()
        }

        logger.info(
            f"金融词典已加载: {len(self._term_alias_to_canonical)} 术语别名, "
            f"{len(self._law_alias_to_full)} 法规名, "
            f"{len(self._auth_alias_to_full)} 机构名, "
            f"{len(self._abbrev_to_full)} 缩写"
        )

    # ── 术语查询 ──

    def resolve_term(self, text: str) -> Optional[str]:
        """将别名/缩写解析为规范术语名。

        "NPL" → "不良贷款率", "拨备" → "拨备覆盖率"
        """
        canonical = self._term_alias_to_canonical.get(text.strip().lower())
        if canonical:
            return canonical
        full = self._abbrev_to_full.get(text.strip().lower())
        if full:
            return self._term_alias_to_canonical.get(full.lower(), full)
        return None

    def get_term_aliases(self, term: str) -> List[str]:
        """获取规范术语的所有别名（用于查询扩展）。"""
        canonical = self.resolve_term(term) or term
        return self._term_aliases.get(canonical, [term])

    def get_term_definition(self, term: str) -> Optional[str]:
        canonical = self.resolve_term(term) or term
        return self._term_def.get(canonical)

    def get_term_category(self, term: str) -> Optional[str]:
        canonical = self.resolve_term(term) or term
        return self._term_category.get(canonical)

    def search_terms(self, query: str, top_k: int = 10) -> List[Dict]:
        """模糊匹配术语表。返回 [{term, definition, category, score}]。"""
        results = []
        query_lower = query.lower()
        for canonical, aliases in self._term_aliases.items():
            score = 0
            for alias in aliases:
                if alias.lower() in query_lower:
                    score = max(score, len(alias))  # 匹配越长分越高
            if score > 0:
                results.append({
                    "term": canonical,
                    "definition": self._term_def.get(canonical, ""),
                    "category": self._term_category.get(canonical, ""),
                    "score": score,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ── 法规名查询 ──

    def resolve_law_name(self, name: str) -> Optional[str]:
        """短名/别名 → 全称。

        "公司法" → "中华人民共和国公司法", "PIPL" → "中华人民共和国个人信息保护法"
        """
        full = self._law_alias_to_full.get(name.strip().lower())
        if full:
            return full
        # 回退: 缩写解析
        full_from_abbr = self._abbrev_to_full.get(name.strip().lower())
        if full_from_abbr:
            return self._law_alias_to_full.get(full_from_abbr.lower(), full_from_abbr)
        return None

    def get_law_names(self, category: Optional[str] = None) -> List[Dict]:
        """列出所有法规名，可按分类筛选。"""
        result = []
        for full, info in self._law_full_info.items():
            if category and info.get("category") != category:
                continue
            result.append({"full_name": full, **info})
        return result

    # ── 机构名查询 ──

    def resolve_authority(self, name: str) -> Optional[str]:
        """短名/别名 → 全称。

        "银保监会" → "国家金融监督管理总局", "央行" → "中国人民银行"
        """
        full = self._auth_alias_to_full.get(name.strip().lower())
        if full:
            return full
        full_from_abbr = self._abbrev_to_full.get(name.strip().lower())
        if full_from_abbr:
            return self._auth_alias_to_full.get(full_from_abbr.lower(), full_from_abbr)
        return None

    def get_historical_names(self, full_name: str) -> List[str]:
        """获取机构的历史名称。"""
        info = self._auth_full_info.get(full_name, {})
        return info.get("historical", [])

    # ── 缩写查询 ──

    def resolve_abbreviation(self, abbr: str) -> Optional[str]:
        """英文缩写 → 中文全称。

        "AML" → "反洗钱", "NPL" → "不良贷款率"
        """
        return self._abbrev_to_full.get(abbr.strip().lower())

    # ── 实体检测 ──

    def detect_entities(self, question: str) -> Dict[str, List[str]]:
        """从问题中检测金融实体：术语、法规名、机构名。

        Returns:
            {"terms": [...], "law_names": [...], "authorities": [...]}
        """
        detected_terms: Dict[str, Tuple[str, int]] = {}     # canonical → (alias, len)
        detected_laws: Dict[str, Tuple[str, int]] = {}
        detected_auths: Dict[str, Tuple[str, int]] = {}

        q_lower = question.lower()

        # 术语检测：按别名长度降序（优先长匹配）
        sorted_terms = sorted(self._term_alias_to_canonical.items(), key=lambda x: -len(x[0]))
        for alias, canonical in sorted_terms:
            if alias in q_lower:
                if canonical not in detected_terms or len(alias) > detected_terms[canonical][1]:
                    detected_terms[canonical] = (alias, len(alias))

        # 法规名检测
        sorted_laws = sorted(self._law_alias_to_full.items(), key=lambda x: -len(x[0]))
        for alias, full in sorted_laws:
            if len(alias) >= 3 and alias in q_lower:
                if full not in detected_laws or len(alias) > detected_laws[full][1]:
                    detected_laws[full] = (alias, len(alias))

        # 机构名检测
        sorted_auths = sorted(self._auth_alias_to_full.items(), key=lambda x: -len(x[0]))
        for alias, full in sorted_auths:
            if len(alias) >= 2 and alias in q_lower:
                if full not in detected_auths or len(alias) > detected_auths[full][1]:
                    detected_auths[full] = (alias, len(alias))

        result = {
            "terms": list(detected_terms.keys()),
            "law_names": list(detected_laws.keys()),
            "authorities": list(detected_auths.keys()),
        }
        if any(result.values()):
            logger.info(f"词典实体检测: {result}")
        return result

    # ── 查询扩展 ──

    def expand_query(self, query: str, max_aliases_per_term: int = 2) -> str:
        """检测查询中的金融术语，追加别名以提升向量/BM25召回率。

        "NPL怎么算" → "NPL 不良贷款率 怎么算"
        "CAR要求"  → "CAR 资本充足率 要求"
        """
        entities = self.detect_entities(query)
        if not entities["terms"]:
            return query

        parts = [query]
        for term in entities["terms"]:
            aliases = self._term_aliases.get(term, [term])
            # 跳过查询中已有的别名
            new_aliases = [a for a in aliases if a.lower() not in query.lower()]
            # 最多追加 max_aliases_per_term 个新别名
            for alias in new_aliases[:max_aliases_per_term]:
                if alias not in parts:
                    parts.append(alias)

        if len(parts) == 1:
            return query
        expanded = " ".join(parts)
        logger.info(f"查询扩展: {query[:60]} → {expanded[:120]}")
        return expanded

    # ── 过滤值枚举 ──

    def enumerate_filter_values(
        self,
        law_name: Optional[str] = None,
        authority: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """给定短名，枚举所有可能的中文全称（用于标量过滤 OR 查询）。

        例如 "上海" → {"authority": ["上海银保监局", ...]}
        """
        result: Dict[str, List[str]] = {}
        if law_name:
            full = self.resolve_law_name(law_name)
            if full:
                result["law_name"] = [full]
        if authority:
            full = self.resolve_authority(authority)
            if full:
                result["authority"] = full.split(",")
        return result

    # ── 分类管理 ──

    def _save(self):
        """将当前词典数据写回 JSON 文件。"""
        with open(self._dict_path, "w", encoding="utf-8") as f:
            json.dump(self._raw, f, ensure_ascii=False, indent=2)
        logger.info(f"词典已保存: {self._dict_path}")

    def list_categories(self) -> Dict[str, List[str]]:
        """列出所有唯一分类名，按类型分组。"""
        cats: Dict[str, set] = {"term": set(), "law": set(), "authority": set()}
        for info in self._raw.get("terms", {}).values():
            c = info.get("category", "")
            if c:
                cats["term"].add(c)
        for info in self._raw.get("law_names", {}).values():
            c = info.get("category", "")
            if c:
                cats["law"].add(c)
        for info in self._raw.get("authorities", {}).values():
            c = info.get("category", "")
            if c:
                cats["authority"].add(c)
        return {k: sorted(v) for k, v in cats.items()}

    def rename_category(self, old_name: str, new_name: str) -> Dict[str, int]:
        """重命名分类，返回各类型受影响的条目数。"""
        counts: Dict[str, int] = {"term": 0, "law": 0, "authority": 0}
        for info in self._raw.get("terms", {}).values():
            if info.get("category") == old_name:
                info["category"] = new_name
                counts["term"] += 1
        for info in self._raw.get("law_names", {}).values():
            if info.get("category") == old_name:
                info["category"] = new_name
                counts["law"] += 1
        for info in self._raw.get("authorities", {}).values():
            if info.get("category") == old_name:
                info["category"] = new_name
                counts["authority"] += 1
        if any(counts.values()):
            self._save()
            self._reload_indexes()
        return counts

    def delete_category(self, name: str) -> Dict[str, int]:
        """删除分类名（将受影响的条目 category 置空）。"""
        counts: Dict[str, int] = {"term": 0, "law": 0, "authority": 0}
        for info in self._raw.get("terms", {}).values():
            if info.get("category") == name:
                info["category"] = ""
                counts["term"] += 1
        for info in self._raw.get("law_names", {}).values():
            if info.get("category") == name:
                info["category"] = ""
                counts["law"] += 1
        for info in self._raw.get("authorities", {}).values():
            if info.get("category") == name:
                info["category"] = ""
                counts["authority"] += 1
        if any(counts.values()):
            self._save()
            self._reload_indexes()
        return counts

    def _reload_indexes(self):
        """重载反向索引以同步 category 变更。"""
        self._term_category.clear()
        for term_name, info in self._raw.get("terms", {}).items():
            self._term_category[term_name] = info.get("category", "")
        for full_name, info in self._raw.get("law_names", {}).items():
            self._law_full_info[full_name] = info

    def list_items_by_category(self, item_type: str, category: str) -> List[Dict]:
        """列出指定分类下的所有条目。item_type: term/law/authority。"""
        results = []
        key_map = {"term": "terms", "law": "law_names", "authority": "authorities"}
        key = key_map.get(item_type)
        if key is None:
            return results
        for name, info in self._raw.get(key, {}).items():
            if info.get("category") == category:
                results.append({"name": name, **info})
        return results

    def list_all_items(self, item_type: str) -> List[Dict]:
        """列出指定类型的所有条目及其分类。item_type: term/law/authority。"""
        results = []
        key_map = {"term": "terms", "law": "law_names", "authority": "authorities"}
        key = key_map.get(item_type)
        if key is None:
            return results
        for name, info in self._raw.get(key, {}).items():
            results.append({"name": name, **info})
        results.sort(key=lambda x: x.get("category", ""))
        return results

    def set_item_category(self, item_type: str, item_name: str, category: str) -> bool:
        """设置单个条目的分类。返回是否成功。"""
        key_map = {"term": "terms", "law": "law_names", "authority": "authorities"}
        key = key_map.get(item_type)
        if key is None:
            return False
        if item_name not in self._raw.get(key, {}):
            return False
        self._raw[key][item_name]["category"] = category
        self._save()
        self._reload_indexes()
        return True

    # ── 统计 ──

    def stats(self) -> Dict[str, int]:
        return {
            "terms": len(self._term_aliases),
            "term_aliases": len(self._term_alias_to_canonical),
            "law_names": len(self._law_full_info),
            "law_aliases": len(self._law_alias_to_full),
            "authorities": len(self._auth_full_info),
            "authority_aliases": len(self._auth_alias_to_full),
            "abbreviations": len(self._abbrev_to_full),
        }
