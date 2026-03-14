"""Translate parsed queries into Qdrant API calls and return document IDs."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchText,
    PayloadSchemaType,
    Prefetch,
    RecommendQuery,
)
from sentence_transformers import SentenceTransformer

from qdrant_advanced_search.parser import (
    BoolNode,
    ComplexQuery,
    Literal,
    SimpleQuery,
    parse_query,
)


@dataclass
class SearchResult:
    """A single search result with document metadata.

    Attributes:
        document_id: The parent document identifier.
        paragraph_id: The matched paragraph identifier.
        tags: Tag string for the document (e.g. "#SPORT, #NATURE").
        paragraph_text: The text of the matched paragraph.
    """

    document_id: int
    paragraph_id: int
    tags: str
    paragraph_text: str


# ---------------------------------------------------------------------------
# QueryExecutor
# ---------------------------------------------------------------------------


class QueryExecutor:
    """Execute structured search queries against a Qdrant collection.

    Loads a SentenceTransformer model and connects to Qdrant on instantiation.
    Ensures full-text payload indexes exist on the configured text and tags fields.
    """

    def __init__(
        self,
        *,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "documents",
        model: str | SentenceTransformer = "paraphrase-multilingual-MiniLM-L12-v2",
        text_field: str = "text",
        tags_field: str = "tags",
        document_id_field: str = "document_id",
        paragraph_id_field: str = "paragraph_id",
        default_limit: int = 10,
        default_pre_limit: int = 50,
    ) -> None:
        """Initialise the model, Qdrant client, and payload indexes.

        Args:
            qdrant_url: URL of the Qdrant instance.
            collection_name: Name of the Qdrant collection to search.
            model: Either a model name string or an already-loaded
                SentenceTransformer instance.
            text_field: Payload field containing paragraph text.
            tags_field: Payload field containing the tag string.
            document_id_field: Payload field containing the document identifier.
            paragraph_id_field: Payload field containing the paragraph identifier.
            default_limit: Default result limit for main queries.
            default_pre_limit: Default result limit for prefetch queries.
        """
        if isinstance(model, str):
            self._model = SentenceTransformer(model)
        else:
            self._model = model

        self._qdrant_url = qdrant_url
        self._collection_name = collection_name
        self._text_field = text_field
        self._tags_field = tags_field
        self._document_id_field = document_id_field
        self._paragraph_id_field = paragraph_id_field
        self._default_limit = default_limit
        self._default_pre_limit = default_pre_limit

        self._client = QdrantClient(url=qdrant_url)
        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create full-text payload indexes if they do not already exist."""
        for field_name in (self._text_field, self._tags_field):
            try:
                self._client.create_payload_index(
                    self._collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.TEXT,
                )
            except Exception:  # noqa: BLE001
                pass  # index already exists

    def _encode(self, text: str) -> list[float]:
        """Encode a string to a vector using the SentenceTransformer model.

        Args:
            text: The text to encode.

        Returns:
            A flat list of floats representing the embedding.
        """
        return self._model.encode(text).tolist()

    @staticmethod
    def _bool_to_filter(node: BoolNode, field_key: str) -> Filter:
        """Recursively convert a BoolNode to a Qdrant Filter.

        Args:
            node: The boolean expression node to convert.
            field_key: The payload field to match against (e.g. "text" or "tags").

        Returns:
            A Qdrant Filter object.
        """
        if isinstance(node, Literal):
            condition = FieldCondition(
                key=field_key,
                match=MatchText(text=node.value),
            )
            if node.negated:
                return Filter(must_not=[condition])
            return Filter(must=[condition])
        # BoolOp
        left_filter = QueryExecutor._bool_to_filter(node.left, field_key)
        right_filter = QueryExecutor._bool_to_filter(node.right, field_key)
        if node.op == "AND":
            return Filter(must=[left_filter, right_filter])
        # OR
        return Filter(should=[left_filter, right_filter])

    @staticmethod
    def _merge_filters(f1: Filter | None, f2: Filter | None) -> Filter | None:
        """Combine two filters with an AND relationship.

        Args:
            f1: First filter, may be None.
            f2: Second filter, may be None.

        Returns:
            A combined Filter, or None if both inputs are None.
        """
        if f1 is None:
            return f2
        if f2 is None:
            return f1
        return Filter(must=[f1, f2])

    def _extract_results(self, payload_list: list[dict]) -> list[SearchResult]:
        """Build deduplicated SearchResult objects from a list of payloads.

        One result per document_id, keeping the first (highest-ranked) paragraph.

        Args:
            payload_list: List of payload dicts from Qdrant points.

        Returns:
            Deduplicated list of SearchResult preserving result order.
        """
        seen: set[int] = set()
        results: list[SearchResult] = []
        for p in payload_list:
            if not p or self._document_id_field not in p:
                continue
            doc_id = int(p[self._document_id_field])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(
                SearchResult(
                    document_id=doc_id,
                    paragraph_id=int(p.get(self._paragraph_id_field, 0)),
                    tags=p.get(self._tags_field, ""),
                    paragraph_text=p.get(self._text_field, ""),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, raw_query: str) -> list[SearchResult]:
        """Parse and execute a raw query string.

        Args:
            raw_query: The user's raw query string.

        Returns:
            A deduplicated list of SearchResult matching the query, preserving
            result order.
        """
        parsed = parse_query(raw_query)
        if isinstance(parsed, SimpleQuery):
            return self._run_simple(parsed)
        return self._run_complex(parsed)

    # ------------------------------------------------------------------
    # Simple query
    # ------------------------------------------------------------------

    def _run_simple(self, q: SimpleQuery) -> list[SearchResult]:
        """Execute a plain semantic search query.

        Args:
            q: The SimpleQuery to execute.

        Returns:
            List of SearchResult objects.
        """
        vec = self._encode(q.text)
        result = self._client.query_points(
            collection_name=self._collection_name,
            query=vec,
            limit=self._default_limit,
        )
        payloads = [p.payload or {} for p in result.points]
        return self._extract_results(payloads)

    # ------------------------------------------------------------------
    # Complex query
    # ------------------------------------------------------------------

    def _run_complex(self, q: ComplexQuery) -> list[SearchResult]:
        """Execute a complex structured query with optional prefetch.

        Args:
            q: The ComplexQuery to execute.

        Returns:
            Deduplicated list of SearchResult preserving result order.
        """
        pre = q.prefetch
        req = q.req

        # Build req filters
        req_tags_filter: Filter | None = None
        if req.tags is not None:
            req_tags_filter = self._bool_to_filter(req.tags, self._tags_field)

        req_kw_filter: Filter | None = None
        if req.keywords is not None:
            req_kw_filter = self._bool_to_filter(req.keywords, self._text_field)

        req_lim = req.lim or self._default_limit

        # ----------------------------------------------------------------
        # No prefetch
        # ----------------------------------------------------------------
        if pre is None:
            if req.kind == "sem":
                vec = self._encode(req.sem.positive)  # type: ignore[union-attr]
                query: list[float] | RecommendQuery = vec
                if req.sem.negative:  # type: ignore[union-attr]
                    neg_vec = self._encode(req.sem.negative)  # type: ignore[union-attr]
                    query = RecommendQuery(positive=[vec], negative=[neg_vec])
                result = self._client.query_points(
                    collection_name=self._collection_name,
                    query=query,
                    query_filter=req_tags_filter,
                    limit=req_lim,
                )
                payloads = [p.payload or {} for p in result.points]
                return self._extract_results(payloads)

            # req kind == "keywords"
            combined = self._merge_filters(req_kw_filter, req_tags_filter)
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=combined,
                limit=req_lim,
                with_payload=True,
            )
            payloads = [r.payload or {} for r in records]
            return self._extract_results(payloads)

        # ----------------------------------------------------------------
        # Build prefetch filters
        # ----------------------------------------------------------------
        pre_tags_filter: Filter | None = None
        if pre.tags is not None:
            pre_tags_filter = self._bool_to_filter(pre.tags, self._tags_field)

        pre_kw_filter: Filter | None = None
        if pre.keywords is not None:
            pre_kw_filter = self._bool_to_filter(pre.keywords, self._text_field)

        pre_lim = pre.lim or self._default_pre_limit

        # ----------------------------------------------------------------
        # pre: sem + req: sem
        # ----------------------------------------------------------------
        if pre.kind == "sem" and req.kind == "sem":
            pre_vec = self._encode(pre.sem.positive)  # type: ignore[union-attr]
            pre_query: list[float] | RecommendQuery = pre_vec
            if pre.sem.negative:  # type: ignore[union-attr]
                pre_neg_vec = self._encode(pre.sem.negative)  # type: ignore[union-attr]
                pre_query = RecommendQuery(positive=[pre_vec], negative=[pre_neg_vec])

            req_vec = self._encode(req.sem.positive)  # type: ignore[union-attr]
            req_query: list[float] | RecommendQuery = req_vec
            if req.sem.negative:  # type: ignore[union-attr]
                req_neg_vec = self._encode(req.sem.negative)  # type: ignore[union-attr]
                req_query = RecommendQuery(positive=[req_vec], negative=[req_neg_vec])

            result = self._client.query_points(
                collection_name=self._collection_name,
                prefetch=Prefetch(
                    query=pre_query,
                    query_filter=pre_tags_filter,
                    limit=pre_lim,
                ),
                query=req_query,
                query_filter=req_tags_filter,
                limit=req_lim,
            )
            payloads = [p.payload or {} for p in result.points]
            return self._extract_results(payloads)

        # ----------------------------------------------------------------
        # pre: sem + req: keywords
        # ----------------------------------------------------------------
        if pre.kind == "sem" and req.kind == "keywords":
            pre_vec = self._encode(pre.sem.positive)  # type: ignore[union-attr]
            pre_q: list[float] | RecommendQuery = pre_vec
            if pre.sem.negative:  # type: ignore[union-attr]
                pre_neg_vec = self._encode(pre.sem.negative)  # type: ignore[union-attr]
                pre_q = RecommendQuery(positive=[pre_vec], negative=[pre_neg_vec])

            pre_result = self._client.query_points(
                collection_name=self._collection_name,
                query=pre_q,
                query_filter=pre_tags_filter,
                limit=pre_lim,
            )
            pre_ids: list[str] = [p.id for p in pre_result.points]  # type: ignore[misc]
            if not pre_ids:
                return []
            id_filter = Filter(must=[HasIdCondition(has_id=pre_ids)])
            combined = self._merge_filters(
                id_filter, self._merge_filters(req_kw_filter, req_tags_filter)
            )
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=combined,
                limit=req_lim,
                with_payload=True,
            )
            payloads = [r.payload or {} for r in records]
            return self._extract_results(payloads)

        # ----------------------------------------------------------------
        # pre: keywords + req: sem
        # ----------------------------------------------------------------
        if pre.kind == "keywords" and req.kind == "sem":
            pre_combined = self._merge_filters(pre_kw_filter, pre_tags_filter)
            pre_records, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=pre_combined,
                limit=1000,
                with_payload=False,
            )
            pre_ids_kw: list[str] = [r.id for r in pre_records]  # type: ignore[misc]
            if not pre_ids_kw:
                return []
            id_filter = Filter(must=[HasIdCondition(has_id=pre_ids_kw)])
            combined_filter = self._merge_filters(id_filter, req_tags_filter)

            req_vec = self._encode(req.sem.positive)  # type: ignore[union-attr]
            req_q: list[float] | RecommendQuery = req_vec
            if req.sem.negative:  # type: ignore[union-attr]
                req_neg_vec = self._encode(req.sem.negative)  # type: ignore[union-attr]
                req_q = RecommendQuery(positive=[req_vec], negative=[req_neg_vec])

            result = self._client.query_points(
                collection_name=self._collection_name,
                query=req_q,
                query_filter=combined_filter,
                limit=req_lim,
            )
            payloads = [p.payload or {} for p in result.points]
            return self._extract_results(payloads)

        # ----------------------------------------------------------------
        # pre: keywords + req: keywords
        # ----------------------------------------------------------------
        combined = self._merge_filters(
            self._merge_filters(pre_kw_filter, pre_tags_filter),
            self._merge_filters(req_kw_filter, req_tags_filter),
        )
        records, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=combined,
            limit=req_lim,
            with_payload=True,
        )
        payloads = [r.payload or {} for r in records]
        return self._extract_results(payloads)
