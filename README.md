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

`execute()` returns a deduplicated list of `document_id` integers, ranked by relevance. Fetching tags, text, or any other payload is the caller's responsibility.

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
)

doc_ids = executor.execute('c: pre: sem: "plage" lim: 50 req: sem: "crème solaire"')
print(doc_ids)  # [42, 7, 91, ...]
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
| `req: sem: "..."` | Semantic request | `req: sem: "crème solaire"` |
| `req: sem: "..." NOT "..."` | Semantic request with negative example | `req: sem: "bronzage" NOT "piscine"` |
| `req: keywords: EXPR` | Keyword boolean request | `req: keywords: "sport" AND NOT "foot"` |
| `lim: N` | Result limit (on `req:` or `pre: sem:`) | `lim: 50` |
| `filter: field IN [v, ...]` | Payload field inclusion filter | `filter: category IN ["sport"]` |
| `filter: field NOT IN [v, ...]` | Payload field exclusion filter | `filter: year NOT IN [2020]` |

### Boolean expression syntax

Inside `keywords:` clauses:

- Terms are quoted strings `"word"` or bare words (e.g. `#TAG`)
- Operators (case-insensitive): `AND`, `OR`, `NOT`
- Parentheses for grouping: `("plage" OR "vacances") AND NOT "sport"`

### Payload filter syntax

`filter:` matches on any Qdrant payload field. String values use full-text (substring) matching; integer values use exact matching. It can be placed on `pre:`, `req:`, or a simple query. Multiple `filter:` clauses are combined with AND.

- `field` — exact payload field name
- Values — quoted strings `"val"` or integers `42`, comma-separated
- `IN` — keep only matching documents
- `NOT IN` — exclude matching documents

### Full examples

```
# Simple semantic search
Victoire au superbowl

# Simple search restricted to a category
Recettes de cuisine filter: category IN ["gastronomie", "végétarien"]

# Semantic prefetch → semantic request
c: pre: sem: "Il fait beau" lim: 50 req: sem: "Il fait chaud"

# Keyword prefetch → semantic request
c: pre: keywords: ("plage" OR "vacances") AND "sport" req: sem: "Crème de bronzage" lim: 50

# Keyword prefetch with NOT → semantic request with negative example
c: pre: keywords: ("plage" OR "vacances") AND NOT "sport" req: sem: "Crème de bronzage" NOT "Après soleil" lim: 50

# Filter by tag value (substring match on the tags payload field)
c: req: sem: "football" filter: tags IN ["#SPORT"]

# Payload filter on req — exclude specific documents
c: req: sem: "alcène" filter: document_id NOT IN [12, 45] lim: 20

# Payload filter on pre and req — different fields
c: pre: sem: "nutrition" lim: 100 filter: year IN [2022, 2023] req: sem: "protéines" filter: category NOT IN ["publicité"] lim: 30

# Multiple filters combined (AND)
c: req: sem: "intelligence artificielle" filter: category IN ["science"] filter: year NOT IN [2020] lim: 25
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
| `text_field` | `str` | `"text"` | Payload field containing paragraph text (full-text indexed) |
| `document_id_field` | `str` | `"document_id"` | Payload field for the document ID |
| `paragraph_id_field` | `str` | `"paragraph_id"` | Payload field for the paragraph ID |
| `default_limit` | `int` | `50` | Default result limit for main queries |
| `default_pre_limit` | `int` | `50` | Default result limit for prefetch queries |

Passing an already-loaded `SentenceTransformer` instance avoids loading the model twice:

```python
from sentence_transformers import SentenceTransformer
from qdrant_advanced_search import QueryExecutor

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
executor = QueryExecutor(model=model, collection_name="my_collection")
```

---

## Dash app

An example Dash application is provided in `examples/dash_app.py`. Install the `app` extra then run:

```bash
uv run python main.py
```

The app starts a local development server at `http://127.0.0.1:8050` and provides a single text input that accepts both simple and complex query strings. It fetches payload data (tags, paragraph text) directly from Qdrant for display.

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
