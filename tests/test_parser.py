"""Tests for query_parser — including the new filter: feature."""

from __future__ import annotations


from qdrant_advanced_search.parser import (
    ComplexQuery,
    Literal,
    SemExpr,
    SimpleQuery,
    parse_query,
)


# ---------------------------------------------------------------------------
# Simple queries
# ---------------------------------------------------------------------------


def test_simple_query_plain():
    q = parse_query("Victoire au superbowl")
    assert isinstance(q, SimpleQuery)
    assert q.text == "Victoire au superbowl"
    assert q.filters == []


def test_simple_query_with_filter_in():
    q = parse_query('Recettes filter: category IN ["sport", "cuisine"]')
    assert isinstance(q, SimpleQuery)
    assert q.text == "Recettes"
    assert len(q.filters) == 1
    f = q.filters[0]
    assert f.field == "category"
    assert f.values == ["sport", "cuisine"]
    assert f.exclude is False


def test_simple_query_with_filter_not_in():
    q = parse_query("Recettes filter: document_id NOT IN [12, 45]")
    assert isinstance(q, SimpleQuery)
    f = q.filters[0]
    assert f.field == "document_id"
    assert f.values == [12, 45]
    assert f.exclude is True


def test_simple_query_multiple_filters():
    q = parse_query('Alcène filter: category IN ["science"] filter: year NOT IN [2020]')
    assert isinstance(q, SimpleQuery)
    assert len(q.filters) == 2
    assert q.filters[0].field == "category"
    assert q.filters[0].exclude is False
    assert q.filters[1].field == "year"
    assert q.filters[1].exclude is True


# ---------------------------------------------------------------------------
# Complex queries — req filter
# ---------------------------------------------------------------------------


def test_complex_req_sem_with_filter_in():
    q = parse_query('c: req: sem: "football" filter: category IN ["sport"]')
    assert isinstance(q, ComplexQuery)
    assert q.prefetch is None
    assert q.req.kind == "sem"
    assert len(q.req.filters) == 1
    f = q.req.filters[0]
    assert f.field == "category"
    assert f.values == ["sport"]
    assert f.exclude is False


def test_complex_req_sem_with_filter_not_in():
    q = parse_query('c: req: sem: "football" filter: document_id NOT IN [1, 2, 3]')
    assert isinstance(q, ComplexQuery)
    f = q.req.filters[0]
    assert f.field == "document_id"
    assert f.values == [1, 2, 3]
    assert f.exclude is True


def test_complex_req_sem_with_lim_and_filter():
    q = parse_query('c: req: sem: "plage" lim: 20 filter: year IN [2023]')
    assert isinstance(q, ComplexQuery)
    assert q.req.lim == 20
    assert q.req.filters[0].field == "year"


def test_complex_req_keywords_with_filter():
    q = parse_query('c: req: keywords: "alcène" filter: document_id IN [14, 23, 98]')
    assert isinstance(q, ComplexQuery)
    assert q.req.kind == "keywords"
    assert q.req.filters[0].values == [14, 23, 98]


def test_complex_req_multiple_filters():
    q = parse_query(
        'c: req: sem: "IA" filter: category IN ["science"] filter: year NOT IN [2020] lim: 25'
    )
    assert isinstance(q, ComplexQuery)
    assert len(q.req.filters) == 2
    assert q.req.lim == 25


# ---------------------------------------------------------------------------
# Complex queries — pre filter
# ---------------------------------------------------------------------------


def test_complex_pre_sem_with_filter():
    q = parse_query(
        'c: pre: sem: "nutrition" lim: 100 filter: year IN [2022, 2023] req: sem: "protéines"'
    )
    assert isinstance(q, ComplexQuery)
    assert q.prefetch is not None
    assert q.prefetch.lim == 100
    assert len(q.prefetch.filters) == 1
    f = q.prefetch.filters[0]
    assert f.field == "year"
    assert f.values == [2022, 2023]


def test_complex_pre_keywords_with_filter():
    q = parse_query(
        'c: pre: keywords: "musculation" filter: category IN ["sport"] req: sem: "récupération"'
    )
    assert isinstance(q, ComplexQuery)
    assert q.prefetch is not None
    assert q.prefetch.filters[0].field == "category"


def test_complex_pre_and_req_different_filters():
    q = parse_query(
        'c: pre: sem: "nutrition" lim: 100 filter: year IN [2022, 2023]'
        ' req: sem: "protéines" filter: category NOT IN ["publicité"] lim: 30'
    )
    assert isinstance(q, ComplexQuery)
    assert q.prefetch is not None
    pre_f = q.prefetch.filters[0]
    assert pre_f.field == "year"
    assert pre_f.values == [2022, 2023]
    assert pre_f.exclude is False

    req_f = q.req.filters[0]
    assert req_f.field == "category"
    assert req_f.values == ["publicité"]
    assert req_f.exclude is True
    assert q.req.lim == 30


# ---------------------------------------------------------------------------
# Filter value types
# ---------------------------------------------------------------------------


def test_filter_integer_values():
    q = parse_query("test filter: document_id IN [1, 2, 3]")
    assert isinstance(q, SimpleQuery)
    assert q.filters[0].values == [1, 2, 3]
    assert all(isinstance(v, int) for v in q.filters[0].values)


def test_filter_string_values():
    q = parse_query('test filter: category IN ["sport", "culture"]')
    assert isinstance(q, SimpleQuery)
    assert q.filters[0].values == ["sport", "culture"]
    assert all(isinstance(v, str) for v in q.filters[0].values)


def test_filter_single_value():
    q = parse_query('test filter: author IN ["Dupont"]')
    assert isinstance(q, SimpleQuery)
    assert q.filters[0].values == ["Dupont"]


# ---------------------------------------------------------------------------
# Coexistence with tags:
# ---------------------------------------------------------------------------


def test_req_tags_and_filter_together():
    q = parse_query('c: req: sem: "football" tags: #SPORT filter: year IN [2023]')
    assert isinstance(q, ComplexQuery)
    assert q.req.tags is not None
    assert isinstance(q.req.tags, Literal)
    assert q.req.tags.value == "#SPORT"
    assert q.req.filters[0].field == "year"


# ---------------------------------------------------------------------------
# Existing queries still parse correctly (non-regression)
# ---------------------------------------------------------------------------


def test_existing_simple_query():
    q = parse_query("Victoire au superbowl")
    assert isinstance(q, SimpleQuery)
    assert q.text == "Victoire au superbowl"


def test_existing_complex_sem_sem():
    q = parse_query('c: pre: sem: "Il fait beau" lim: 50 req: sem: "Il fait chaud"')
    assert isinstance(q, ComplexQuery)
    assert q.prefetch is not None
    assert q.prefetch.sem == SemExpr(positive="Il fait beau")
    assert q.prefetch.lim == 50
    assert q.req.sem == SemExpr(positive="Il fait chaud")
    assert q.req.filters == []
    assert q.prefetch.filters == []


def test_existing_complex_keywords_sem_negative():
    q = parse_query(
        'c: pre: keywords: ("plage" OR "vacances") AND NOT "sport"'
        ' req: sem: "Crème de bronzage" NOT "Après soleil" lim: 50'
    )
    assert isinstance(q, ComplexQuery)
    assert q.req.sem == SemExpr(positive="Crème de bronzage", negative="Après soleil")
    assert q.req.lim == 50
    assert q.req.filters == []
