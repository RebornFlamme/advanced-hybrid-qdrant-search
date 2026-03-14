"""Translate parsed queries into Qdrant API calls and return document IDs."""

from __future__ import annotations

from query_parser import (
    BoolNode,
    ComplexQuery,
    Literal,
    SimpleQuery,
    parse_query,
)

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LIMIT = 10
DEFAULT_PRE_LIMIT = 50


# ---------------------------------------------------------------------------
# QueryExecutor
# ---------------------------------------------------------------------------


class QueryExecutor:
    """Execute structured search queries against a Qdrant collection.

    Loads a SentenceTransformer model and connects to Qdrant on instantiation.
    Ensures full-text payload indexes exist on ``text`` and ``tags`` fields.
    """

    def __init__(self) -> None:
        """Initialise the model, Qdrant client, and payload indexes."""
        self._model = SentenceTransformer(MODEL_NAME)
        self._client = QdrantClient(url=QDRANT_URL)
        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create full-text payload indexes if they do not already exist."""
        for field_name in ("text", "tags"):
            try:
                self._client.create_payload_index(
                    COLLECTION_NAME,
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

    @staticmethod
    def _extract_ids(payload_list: list[dict]) -> list[int]:
        """Extract document_id values from a list of payloads.

        Args:
            payload_list: List of payload dicts, each containing "document_id".

        Returns:
            Deduplicated list of document IDs preserving order.
        """
        ids = [p["document_id"] for p in payload_list if p and "document_id" in p]
        return list(dict.fromkeys(ids))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, raw_query: str) -> list[int]:
        """Parse and execute a raw query string.

        Args:
            raw_query: The user's raw query string.

        Returns:
            A deduplicated list of document IDs matching the query, preserving
            result order.
        """
        parsed = parse_query(raw_query)
        if isinstance(parsed, SimpleQuery):
            return self._run_simple(parsed)
        return self._run_complex(parsed)

    # ------------------------------------------------------------------
    # Simple query
    # ------------------------------------------------------------------

    def _run_simple(self, q: SimpleQuery) -> list[int]:
        """Execute a plain semantic search query.

        Args:
            q: The SimpleQuery to execute.

        Returns:
            List of document IDs.
        """
        vec = self._encode(q.text)
        result = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=DEFAULT_LIMIT,
        )
        payloads = [p.payload or {} for p in result.points]
        return self._extract_ids(payloads)

    # ------------------------------------------------------------------
    # Complex query
    # ------------------------------------------------------------------

    def _run_complex(self, q: ComplexQuery) -> list[int]:
        """Execute a complex structured query with optional prefetch.

        Args:
            q: The ComplexQuery to execute.

        Returns:
            Deduplicated list of document IDs preserving result order.
        """
        pre = q.prefetch
        req = q.req

        # Build req filters
        req_tags_filter: Filter | None = None
        if req.tags is not None:
            req_tags_filter = self._bool_to_filter(req.tags, "tags")

        req_kw_filter: Filter | None = None
        if req.keywords is not None:
            req_kw_filter = self._bool_to_filter(req.keywords, "text")

        req_lim = req.lim or DEFAULT_LIMIT

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
                    collection_name=COLLECTION_NAME,
                    query=query,
                    query_filter=req_tags_filter,
                    limit=req_lim,
                )
                payloads = [p.payload or {} for p in result.points]
                return self._extract_ids(payloads)

            # req kind == "keywords"
            combined = self._merge_filters(req_kw_filter, req_tags_filter)
            records, _ = self._client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=combined,
                limit=req_lim,
                with_payload=True,
            )
            payloads = [r.payload or {} for r in records]
            return self._extract_ids(payloads)

        # ----------------------------------------------------------------
        # Build prefetch filters
        # ----------------------------------------------------------------
        pre_tags_filter: Filter | None = None
        if pre.tags is not None:
            pre_tags_filter = self._bool_to_filter(pre.tags, "tags")

        pre_kw_filter: Filter | None = None
        if pre.keywords is not None:
            pre_kw_filter = self._bool_to_filter(pre.keywords, "text")

        pre_lim = pre.lim or DEFAULT_PRE_LIMIT

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
                collection_name=COLLECTION_NAME,
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
            return self._extract_ids(payloads)

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
                collection_name=COLLECTION_NAME,
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
                collection_name=COLLECTION_NAME,
                scroll_filter=combined,
                limit=req_lim,
                with_payload=True,
            )
            payloads = [r.payload or {} for r in records]
            return self._extract_ids(payloads)

        # ----------------------------------------------------------------
        # pre: keywords + req: sem
        # ----------------------------------------------------------------
        if pre.kind == "keywords" and req.kind == "sem":
            pre_combined = self._merge_filters(pre_kw_filter, pre_tags_filter)
            pre_records, _ = self._client.scroll(
                collection_name=COLLECTION_NAME,
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
                collection_name=COLLECTION_NAME,
                query=req_q,
                query_filter=combined_filter,
                limit=req_lim,
            )
            payloads = [p.payload or {} for p in result.points]
            return self._extract_ids(payloads)

        # ----------------------------------------------------------------
        # pre: keywords + req: keywords
        # ----------------------------------------------------------------
        combined = self._merge_filters(
            self._merge_filters(pre_kw_filter, pre_tags_filter),
            self._merge_filters(req_kw_filter, req_tags_filter),
        )
        records, _ = self._client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=combined,
            limit=req_lim,
            with_payload=True,
        )
        payloads = [r.payload or {} for r in records]
        return self._extract_ids(payloads)
