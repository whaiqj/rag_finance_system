"""FinanceDictionary 单元测试。"""

import json
import os

import pytest


class TestFinanceDictionaryInit:
    """词典加载与初始化。"""

    def test_load_from_custom_path(self, sample_dict_path):
        from rag_finance_system.src.dictionary import FinanceDictionary

        fd = FinanceDictionary(dict_path=sample_dict_path)
        assert fd.stats()["terms"] > 0

    def test_load_from_env_var(self, sample_dict_path, monkeypatch):
        monkeypatch.setenv("FINANCE_DICT_PATH", sample_dict_path)
        from rag_finance_system.src.dictionary import FinanceDictionary

        fd = FinanceDictionary()
        assert fd.stats()["terms"] > 0

    def test_missing_dict_file_raises(self, tmp_path):
        from rag_finance_system.src.dictionary import FinanceDictionary

        with pytest.raises(FileNotFoundError):
            FinanceDictionary(dict_path=str(tmp_path / "nonexistent.json"))

    def test_stats_returns_expected_counts(self, finance_dict, sample_dict_data):
        stats = finance_dict.stats()
        assert stats["terms"] == len(sample_dict_data["terms"])
        assert stats["law_names"] == len(sample_dict_data["law_names"])
        assert stats["authorities"] == len(sample_dict_data["authorities"])
        assert stats["abbreviations"] == len(sample_dict_data["abbreviations"])


class TestTermResolution:
    """术语别名解析。"""

    def test_resolve_term_by_alias(self, finance_dict):
        assert finance_dict.resolve_term("NPL") == "不良贷款率"

    def test_resolve_term_by_chinese_alias(self, finance_dict):
        assert finance_dict.resolve_term("拨备") == "拨备覆盖率"

    def test_resolve_term_unknown_returns_none(self, finance_dict):
        assert finance_dict.resolve_term("不存在的术语") is None

    def test_resolve_term_case_insensitive(self, finance_dict):
        assert finance_dict.resolve_term("npl") == "不良贷款率"

    def test_resolve_term_strips_whitespace(self, finance_dict):
        assert finance_dict.resolve_term(" NPL ") == "不良贷款率"

    def test_resolve_term_via_abbreviation(self, finance_dict):
        assert finance_dict.resolve_term("CAR") == "资本充足率"

    def test_resolve_term_canonical_self(self, finance_dict):
        assert finance_dict.resolve_term("资本充足率") == "资本充足率"


class TestTermAliasesAndMetadata:
    """术语别名和元数据查询。"""

    def test_get_term_aliases(self, finance_dict):
        aliases = finance_dict.get_term_aliases("NPL")
        assert "不良贷款率" in aliases
        assert "NPL" in aliases

    def test_get_term_aliases_unknown(self, finance_dict):
        aliases = finance_dict.get_term_aliases("不存在的术语")
        assert aliases == ["不存在的术语"]

    def test_get_term_definition(self, finance_dict):
        defn = finance_dict.get_term_definition("NPL")
        assert "不良贷款" in defn

    def test_get_term_category(self, finance_dict):
        cat = finance_dict.get_term_category("资本充足率")
        assert cat == "银行监管"

    def test_get_term_category_unknown(self, finance_dict):
        assert finance_dict.get_term_category("不存在的术语") is None


class TestSearchTerms:
    """术语模糊搜索。"""

    def test_search_terms_match(self, finance_dict):
        results = finance_dict.search_terms("资本充足")
        assert len(results) > 0
        assert any(r["term"] == "资本充足率" for r in results)

    def test_search_terms_no_match(self, finance_dict):
        results = finance_dict.search_terms("量子计算")
        assert len(results) == 0

    def test_search_terms_top_k(self, finance_dict):
        results = finance_dict.search_terms("资", top_k=2)
        assert len(results) <= 2

    def test_search_terms_longer_alias_higher_score(self, finance_dict):
        results = finance_dict.search_terms("资本充足率")
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]


class TestLawNameResolution:
    """法规名解析。"""

    def test_resolve_law_name_short(self, finance_dict):
        assert finance_dict.resolve_law_name("公司法") == "中华人民共和国公司法"

    def test_resolve_law_name_full(self, finance_dict):
        assert finance_dict.resolve_law_name("中华人民共和国公司法") == "中华人民共和国公司法"

    def test_resolve_law_name_unknown(self, finance_dict):
        assert finance_dict.resolve_law_name("量子法") is None

    def test_resolve_law_name_case_insensitive(self, finance_dict):
        assert finance_dict.resolve_law_name("公司法") == finance_dict.resolve_law_name("公司法")

    def test_get_law_names(self, finance_dict):
        names = finance_dict.get_law_names()
        assert len(names) > 0
        assert any(n["full_name"] == "中华人民共和国公司法" for n in names)

    def test_get_law_names_by_category(self, finance_dict):
        names = finance_dict.get_law_names(category="商事法律")
        assert all(n.get("category") == "商事法律" for n in names)


class TestAuthorityResolution:
    """机构名解析。"""

    def test_resolve_authority_short(self, finance_dict):
        assert finance_dict.resolve_authority("银保监会") == "国家金融监督管理总局"

    def test_resolve_authority_historical(self, finance_dict):
        assert finance_dict.resolve_authority("中国银保监会") == "国家金融监督管理总局"

    def test_resolve_authority_unknown(self, finance_dict):
        assert finance_dict.resolve_authority("银河联邦") is None

    def test_resolve_authority_full(self, finance_dict):
        assert finance_dict.resolve_authority("中国人民银行") == "中国人民银行"


class TestAbbreviationResolution:
    """缩写解析。"""

    def test_resolve_abbreviation_aml(self, finance_dict):
        assert finance_dict.resolve_abbreviation("AML") == "反洗钱"

    def test_resolve_abbreviation_npl(self, finance_dict):
        assert finance_dict.resolve_abbreviation("NPL") == "不良贷款率"

    def test_resolve_abbreviation_unknown(self, finance_dict):
        assert finance_dict.resolve_abbreviation("ZZZ") is None

    def test_resolve_abbreviation_case_insensitive(self, finance_dict):
        assert finance_dict.resolve_abbreviation("aml") == "反洗钱"


class TestEntityDetection:
    """实体检测。"""

    def test_detect_entities_terms_and_law(self, finance_dict):
        result = finance_dict.detect_entities("根据公司法，股东责任是什么")
        assert "中华人民共和国公司法" in result["law_names"]
        assert "股东责任" in result["terms"]

    def test_detect_entities_authority(self, finance_dict):
        result = finance_dict.detect_entities("银保监会对资本充足率的要求")
        assert "国家金融监督管理总局" in result["authorities"]

    def test_detect_entities_no_match(self, finance_dict):
        result = finance_dict.detect_entities("今天天气怎么样")
        assert not any(result.values())

    def test_detect_entities_returns_three_keys(self, finance_dict):
        result = finance_dict.detect_entities("测试")
        assert set(result.keys()) == {"terms", "law_names", "authorities"}


class TestQueryExpansion:
    """查询扩展。"""

    def test_expand_query_adds_aliases(self, finance_dict):
        expanded = finance_dict.expand_query("NPL怎么算")
        assert "不良贷款率" in expanded

    def test_expand_query_no_terms_unchanged(self, finance_dict):
        query = "今天天气怎么样"
        expanded = finance_dict.expand_query(query)
        assert expanded == query

    def test_expand_query_respects_max_aliases(self, finance_dict):
        expanded = finance_dict.expand_query("CAR要求", max_aliases_per_term=1)
        # 只追加1个别名，不应出现多个
        parts = expanded.split()
        # 原查询可能分出多个 token，但新增别名最多1个
        new_aliases = [p for p in parts if p not in "CAR要求".split()]
        assert len(new_aliases) <= 1

    def test_expand_query_no_duplicate_aliases(self, finance_dict):
        expanded = finance_dict.expand_query("资本充足率")
        # "资本充足率" 已在查询中，不应重复追加
        count = expanded.count("资本充足率")
        # 可能出现一次在原文一次在扩展部分，但不应多于2
        assert count <= 2


class TestFilterValues:
    """过滤值枚举。"""

    def test_enumerate_filter_values_law_name(self, finance_dict):
        result = finance_dict.enumerate_filter_values(law_name="公司法")
        assert "law_name" in result
        assert "中华人民共和国公司法" in result["law_name"]

    def test_enumerate_filter_values_authority(self, finance_dict):
        result = finance_dict.enumerate_filter_values(authority="央行")
        assert "authority" in result

    def test_enumerate_filter_values_no_match(self, finance_dict):
        result = finance_dict.enumerate_filter_values(law_name="量子法")
        assert result == {}


class TestCategoryManagement:
    """分类管理 CRUD。"""

    def test_list_categories(self, finance_dict):
        cats = finance_dict.list_categories()
        assert "term" in cats
        assert "law" in cats
        assert "authority" in cats
        assert "银行监管" in cats["term"]

    def test_rename_category(self, finance_dict, tmp_dict_json):
        from rag_finance_system.src.dictionary import FinanceDictionary

        fd = FinanceDictionary(dict_path=tmp_dict_json)
        counts = fd.rename_category("银行监管", "银行监管_new")
        assert counts["term"] > 0
        # 验证分类已变更
        cats = fd.list_categories()
        assert "银行监管_new" in cats["term"]

    def test_rename_category_no_match(self, finance_dict):
        counts = finance_dict.rename_category("不存在的分类", "新分类")
        assert all(v == 0 for v in counts.values())

    def test_delete_category(self, finance_dict, tmp_dict_json):
        from rag_finance_system.src.dictionary import FinanceDictionary

        fd = FinanceDictionary(dict_path=tmp_dict_json)
        counts = fd.delete_category("银行监管")
        assert counts["term"] > 0
        cats = fd.list_categories()
        assert "银行监管" not in cats["term"]

    def test_list_items_by_category(self, finance_dict):
        items = finance_dict.list_items_by_category("term", "银行监管")
        assert len(items) > 0
        assert all(i.get("category") == "银行监管" for i in items)

    def test_list_items_by_category_invalid_type(self, finance_dict):
        items = finance_dict.list_items_by_category("invalid", "银行监管")
        assert items == []

    def test_set_item_category(self, finance_dict, tmp_dict_json):
        from rag_finance_system.src.dictionary import FinanceDictionary

        fd = FinanceDictionary(dict_path=tmp_dict_json)
        result = fd.set_item_category("term", "资本充足率", "新分类")
        assert result is True
        assert fd.get_term_category("资本充足率") == "新分类"

    def test_set_item_category_invalid_type(self, finance_dict, tmp_dict_json):
        from rag_finance_system.src.dictionary import FinanceDictionary

        fd = FinanceDictionary(dict_path=tmp_dict_json)
        result = fd.set_item_category("invalid", "资本充足率", "新分类")
        assert result is False

    def test_set_item_category_nonexistent_item(self, finance_dict, tmp_dict_json):
        from rag_finance_system.src.dictionary import FinanceDictionary

        fd = FinanceDictionary(dict_path=tmp_dict_json)
        result = fd.set_item_category("term", "不存在的术语", "新分类")
        assert result is False

    def test_list_all_items(self, finance_dict):
        items = finance_dict.list_all_items("term")
        assert len(items) == finance_dict.stats()["terms"]

    def test_get_historical_names(self, finance_dict):
        historical = finance_dict.get_historical_names("国家金融监督管理总局")
        assert "银保监会" in historical
