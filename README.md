# qdrant-advanced-search

Advanced query language for Qdrant: compose semantic search, keyword boolean logic, prefetch pipelines, and tag filters — all from a single query string.

---

## Installation

```bash
# with pip
pip install qdrant-advanced-search

# with uv
uv add qdrant-advanced-search

# include the Dash example app
uv add "qdrant-advanced-search[app]"
```

---

## Quick start

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_advanced_search import QueryExecutor

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

executor = QueryExecutor(
    client=client,
    model=model,
    collection_name="documents",
    parquet_path="documents.parquet",
)

results = executor.execute('c: pre: sem: "plage" lim: 50 req: sem: "crème solaire"')
for r in results:
    print(r.document_id, r.tags)
    print(r.paragraph_text[:120])
    print(r.document_text[:300])
```

---

## Query syntax

A **simple query** is any string that does not start with `c:` — it is executed as a plain semantic search.

A **complex query** must start with `c:` and supports the following clauses:

| Clause | Description | Example |
|---|---|---|
| `c:` | Required prefix for complex queries | `c: ...` |
| `pre: sem: "..."` | Semantic prefetch | `pre: sem: "plage"` |
| `pre: sem: "..." lim: N` | Semantic prefetch with limit | `pre: sem: "plage" lim: 100` |
| `pre: keywords: EXPR` | Keyword boolean prefetch | `pre: keywords: "plage" OR "vacances"` |
| `req: "..."` | Bare semantic request (shorthand) | `req: "crème solaire"` |
| `req: sem: "..." NOT "..."` | Semantic request with negative example | `req: sem: "bronzage" NOT "piscine"` |
| `req: keywords: EXPR` | Keyword boolean request | `req: keywords: "sport" AND NOT "foot"` |
| `lim: N` | Result limit (on `req:` or `pre: sem:`) | `lim: 50` |
| `tags: EXPR` | Tag filter (boolean expression on tag string) | `tags: "#SPORT" OR "#NATURE"` |

### Boolean expression syntax

Inside `keywords:` and `tags:` clauses:

- Terms are quoted strings `"word"` or bare words
- Operators (case-insensitive): `AND`, `OR`, `NOT`
- Parentheses for grouping: `("plage" OR "vacances") AND NOT "sport"`

### Full examples

```
# Simple semantic search
Victoire au superbowl

# Semantic prefetch → semantic request
c: pre: sem: "Il fait beau" lim: 50 req: sem: "Il fait chaud"

# Keyword prefetch → semantic request
c: pre: keywords: ("plage" OR "vacances") AND "sport" req: "Crème de bronzage" lim: 50

# Keyword prefetch with NOT → semantic request with negative example
c: pre: keywords: ("plage" OR "vacances") AND NOT "sport" req: sem: "Crème de bronzage" NOT "Après soleil" lim: 50

# Tag filtering
c: req: sem: "football" tags: "#SPORT"
```

---

## Configuration

All `QueryExecutor.__init__` parameters are keyword-only with sensible defaults:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `qdrant_url` | `str` | `"http://localhost:6333"` | URL of the Qdrant instance (ignored if `client` is passed) |
| `client` | `QdrantClient \| None` | `None` | An already-instantiated `QdrantClient` |
| `collection_name` | `str` | `"documents"` | Qdrant collection to search |
| `model` | `str \| SentenceTransformer` | `"paraphrase-multilingual-MiniLM-L12-v2"` | Model name or loaded instance |
| `text_field` | `str` | `"text"` | Payload field containing paragraph text |
| `tags_field` | `str` | `"tags"` | Payload field containing tag string |
| `document_id_field` | `str` | `"document_id"` | Payload field for the document ID |
| `paragraph_id_field` | `str` | `"paragraph_id"` | Payload field for the paragraph ID |
| `parquet_path` | `str \| Path \| None` | `None` | Path to a parquet file with full document texts. When provided, `SearchResult.document_text` is populated |
| `parquet_document_id_col` | `str` | `"document_id"` | Column name for document IDs in the parquet file |
| `parquet_text_col` | `str` | `"text"` | Column name for full text in the parquet file |
| `default_limit` | `int` | `50` | Default result limit for main queries |
| `default_pre_limit` | `int` | `50` | Default result limit for prefetch queries |

Passing an already-loaded `SentenceTransformer` instance avoids loading the model twice:

```python
from sentence_transformers import SentenceTransformer
from qdrant_advanced_search import QueryExecutor

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
executor = QueryExecutor(model=model, collection_name="my_collection")
```

Passing a parquet file enriches each `SearchResult` with the full document text:

```python
executor = QueryExecutor(
    collection_name="my_collection",
    parquet_path="documents.parquet",
    parquet_document_id_col="document_id",  # default
    parquet_text_col="text",                # default
)

results = executor.execute("alcène")
for r in results:
    print(r.document_id, r.tags)
    print(r.paragraph_text)   # matched paragraph (from Qdrant payload)
    print(r.document_text)    # full document text (from parquet)
```

---

## Dash app

An example Dash application is provided in `examples/dash_app.py`. Install the `app` extra then run:

```bash
uv run python main.py
```

The app starts a local development server at `http://127.0.0.1:8050` and provides a single text input that accepts both simple and complex query strings.

To seed the Qdrant collection with French Wikipedia articles before running the app:

```bash
uv run python seed_data.py
```

This requires a running Qdrant instance (e.g. via Docker: `docker run -p 6333:6333 qdrant/qdrant`).

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install dev dependencies: `uv sync --extra dev`
3. Run the linter: `uv run ruff check src/ examples/ main.py`
4. Submit a pull request with a clear description of your changes.
