"""Translate parsed queries into Qdrant API calls and return document IDs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchText,
    PayloadSchemaType,
    Prefetch,
    RecommendQuery,
)

from sentence_transformers import SentenceTransformer

from qdrant_advanced_search.parser import (
    BoolNode,
    BoolOp,
    ComplexQuery,
    FilterClause,
    Literal,
    SimpleQuery,
    parse_query,
)


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
        client: QdrantClient | None = None,
        collection_name: str = "documents",
        model: str | SentenceTransformer = "paraphrase-multilingual-MiniLM-L12-v2",
        text_field: str = "text",
        document_id_field: str = "document_id",
        point_id_field: str = "paragraph_id",
        parquet_path: str | Path | None = None,
        parquet_text_column: str = "text",
        parquet_id_column: str = "document_id",
        default_limit: int = 50,
        default_pre_limit: int = 50,
    ) -> None:
        """Initialise the model, Qdrant client, and payload indexes.

        Args:
            qdrant_url: URL of the Qdrant instance. Ignored if ``client`` is provided.
            client: An already-instantiated QdrantClient. If provided, ``qdrant_url``
                is ignored.
            collection_name: Name of the Qdrant collection to search.
            model: Either a model name string or an already-loaded
                SentenceTransformer instance.
            text_field: Payload field containing paragraph text (used for tags index).
            document_id_field: Payload field in Qdrant containing the document identifier.
            point_id_field: Payload field in Qdrant containing the point/paragraph identifier
                (note: Qdrant point ``.id`` IS the paragraph ID, this names its payload field).
            parquet_path: Path to the parquet file used for keyword searches.
            parquet_text_column: Column name for document text in the parquet file.
            parquet_id_column: Column name for document IDs in the parquet file.
            default_limit: Default result limit for main queries.
            default_pre_limit: Default result limit for prefetch queries.
        """
        if isinstance(model, str):
            self._model = SentenceTransformer(model)
        else:
            self._model = model

        self._client = client if client is not None else QdrantClient(url=qdrant_url)
        self._collection_name = collection_name
        self._text_field = text_field
        self._document_id_field = document_id_field
        self._point_id_field = point_id_field
        self._default_limit = default_limit
        self._default_pre_limit = default_pre_limit

        self._parquet_df: pd.DataFrame | None = None
        self._parquet_text_col = parquet_text_column
        self._parquet_id_col = parquet_id_column
        if parquet_path is not None:
            self._parquet_df = pd.read_parquet(parquet_path, columns=[parquet_id_column, parquet_text_column])

        self._ensure_indexes()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        """The underlying QdrantClient instance."""
        return self._client

    @property
    def collection_name(self) -> str:
        """The Qdrant collection being searched."""
        return self._collection_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create a full-text payload index on text_field if it does not already exist."""
        try:
            self._client.create_payload_index(
                self._collection_name,
                field_name=self._text_field,
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

    def _eval_keywords_mask(self, node: BoolNode, series: pd.Series) -> pd.Series:
        """Recursively evaluate a BoolNode as a boolean mask over a text Series.

        Args:
            node: The boolean expression node.
            series: A pandas Series of text strings to match against.

        Returns:
            A boolean Series.
        """
        if isinstance(node, Literal):
            mask = series.str.contains(node.value, case=False, na=False, regex=False)
            return ~mask if node.negated else mask
        assert isinstance(node, BoolOp)
        left = self._eval_keywords_mask(node.left, series)
        right = self._eval_keywords_mask(node.right, series)
        if node.op == "AND":
            return left & right
        return left | right

    def _keywords_to_doc_ids(self, node: BoolNode) -> list[int]:
        """Search the parquet file for documents matching a keyword BoolNode.

        Args:
            node: The boolean keyword expression to evaluate.

        Returns:
            List of matching document IDs in parquet order.

        Raises:
            RuntimeError: If no parquet file was configured.
        """
        if self._parquet_df is None:
            raise RuntimeError("parquet_path must be provided to use keyword searches")
        mask = self._eval_keywords_mask(node, self._parquet_df[self._parquet_text_col])  # type: ignore[arg-type]
        return self._parquet_df.loc[mask, self._parquet_id_col].tolist()

    def _doc_ids_to_filter(self, doc_ids: list[int]) -> Filter:
        """Build a Qdrant Filter restricting results to the given document IDs.

        Args:
            doc_ids: List of document IDs to include.

        Returns:
            A Qdrant Filter using MatchAny on the document_id payload field.
        """
        return Filter(
            must=[FieldCondition(key=self._document_id_field, match=MatchAny(any=doc_ids))]
        )

    @staticmethod
    def _filter_clauses_to_filter(clauses: list[FilterClause]) -> Filter | None:
        """Convert a list of FilterClause objects into a Qdrant Filter (AND of all).

        String values use MatchText (full-text substring match, requires a text index).
        Integer values use MatchAny (exact match).

        Args:
            clauses: List of payload field filter clauses.

        Returns:
            A combined Filter, or None if the list is empty.
        """
        if not clauses:
            return None
        conditions: list[Filter] = []
        for clause in clauses:
            str_values = [v for v in clause.values if isinstance(v, str)]
            int_values = [v for v in clause.values if isinstance(v, int)]

            sub_conditions: list[FieldCondition] = []
            for sv in str_values:
                sub_conditions.append(FieldCondition(key=clause.field, match=MatchText(text=sv)))
            if int_values:
                sub_conditions.append(
                    FieldCondition(key=clause.field, match=MatchAny(any=int_values))
                )

            if not sub_conditions:
                continue

            if len(sub_conditions) == 1:
                inner: Filter | FieldCondition = sub_conditions[0]
            else:
                inner = Filter(should=sub_conditions)

            if clause.exclude:
                conditions.append(Filter(must_not=[inner]))
            else:
                conditions.append(Filter(must=[inner]))

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return Filter(must=conditions)

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

    def _extract_ids(self, payload_list: list[dict]) -> list[int]:
        """Extract deduplicated document IDs from a list of payloads.

        Args:
            payload_list: List of payload dicts from Qdrant points.

        Returns:
            Deduplicated list of document IDs preserving result order.
        """
        seen: set[int] = set()
        ids: list[int] = []
        for p in payload_list:
            if not p or self._document_id_field not in p:
                continue
            doc_id = int(p[self._document_id_field])
            if doc_id not in seen:
                seen.add(doc_id)
                ids.append(doc_id)
        return ids

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
        query_filter = self._filter_clauses_to_filter(q.filters)
        result = self._client.query_points(
            collection_name=self._collection_name,
            query=vec,
            query_filter=query_filter,
            limit=self._default_limit,
        )
        return self._extract_ids([p.payload or {} for p in result.points])

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

        req_base_filter = self._filter_clauses_to_filter(req.filters)
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
                    query_filter=req_base_filter,
                    limit=req_lim,
                )
                return self._extract_ids([p.payload or {} for p in result.points])

            # req: keywords → parquet search
            doc_ids = self._keywords_to_doc_ids(req.keywords)  # type: ignore[arg-type]
            if not doc_ids:
                return []
            if req_base_filter is not None:
                combined_filter = self._merge_filters(self._doc_ids_to_filter(doc_ids), req_base_filter)
                records, _ = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=combined_filter,
                    limit=req_lim,
                    with_payload=True,
                )
                return self._extract_ids([r.payload or {} for r in records])
            return doc_ids[:req_lim]

        # ----------------------------------------------------------------
        # Build prefetch info
        # ----------------------------------------------------------------
        pre_base_filter = self._filter_clauses_to_filter(pre.filters)
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
                    query_filter=pre_base_filter,
                    limit=pre_lim,
                ),
                query=req_query,
                query_filter=req_base_filter,
                limit=req_lim,
            )
            return self._extract_ids([p.payload or {} for p in result.points])

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
                query_filter=pre_base_filter,
                limit=pre_lim,
            )
            sem_doc_ids = self._extract_ids([p.payload or {} for p in pre_result.points])
            kw_doc_ids = set(self._keywords_to_doc_ids(req.keywords))  # type: ignore[arg-type]
            intersected = [d for d in sem_doc_ids if d in kw_doc_ids]
            if not intersected:
                return []
            if req_base_filter is not None:
                combined_filter = self._merge_filters(self._doc_ids_to_filter(intersected), req_base_filter)
                records, _ = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=combined_filter,
                    limit=req_lim,
                    with_payload=True,
                )
                return self._extract_ids([r.payload or {} for r in records])
            return intersected[:req_lim]

        # ----------------------------------------------------------------
        # pre: keywords + req: sem
        # ----------------------------------------------------------------
        if pre.kind == "keywords" and req.kind == "sem":
            pre_doc_ids = self._keywords_to_doc_ids(pre.keywords)  # type: ignore[arg-type]
            if not pre_doc_ids:
                return []
            pre_doc_filter = self._merge_filters(self._doc_ids_to_filter(pre_doc_ids), pre_base_filter)
            combined_filter = self._merge_filters(pre_doc_filter, req_base_filter)

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
            return self._extract_ids([p.payload or {} for p in result.points])

        # ----------------------------------------------------------------
        # pre: keywords + req: keywords
        # ----------------------------------------------------------------
        pre_doc_ids_list = self._keywords_to_doc_ids(pre.keywords)  # type: ignore[arg-type]
        req_doc_ids_set = set(self._keywords_to_doc_ids(req.keywords))  # type: ignore[arg-type]
        doc_ids = [d for d in pre_doc_ids_list if d in req_doc_ids_set]
        if not doc_ids:
            return []
        combined_tags = self._merge_filters(pre_base_filter, req_base_filter)
        if combined_tags is not None:
            final_filter = self._merge_filters(self._doc_ids_to_filter(doc_ids), combined_tags)
            records, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=final_filter,
                limit=req_lim,
                with_payload=True,
            )
            return self._extract_ids([r.payload or {} for r in records])
        return doc_ids[:req_lim]
