"""Seed script to populate a Qdrant collection with French Wikipedia articles.

This script loads ~100 French Wikipedia articles from HuggingFace, splits them
into paragraphs, generates embeddings, and upserts everything into a local
Qdrant instance. It also saves a parquet file with full article texts.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pandas as pd
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
VECTOR_SIZE = 384
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
NUM_DOCS = 100
MAX_PARAGRAPHS_PER_DOC = 10
MIN_PARAGRAPH_LEN = 50
PARQUET_PATH = "documents.parquet"

TAGS_POOL = [
    "#HISTOIRE",
    "#SCIENCE",
    "#GEOGRAPHIE",
    "#SPORT",
    "#CULTURE",
    "#POLITIQUE",
    "#ECONOMIE",
    "#NATURE",
    "#TECHNOLOGIE",
    "#ART",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def iter_wikipedia_articles(n: int) -> Iterator[dict]:
    """Yield the first *n* articles from the French Wikipedia HuggingFace dataset.

    Args:
        n: Maximum number of articles to yield.

    Yields:
        Raw article dictionaries containing at least a ``text`` field.
    """
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.fr",
        split="train",
        streaming=True,
    )
    count = 0
    for article in dataset:
        if count >= n:
            break
        yield article
        count += 1


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------


def split_paragraphs(text: str) -> list[str]:
    """Split article text into paragraphs and apply filtering rules.

    Paragraphs are obtained by splitting on double newlines. Only paragraphs
    longer than ``MIN_PARAGRAPH_LEN`` characters are kept, and at most
    ``MAX_PARAGRAPHS_PER_DOC`` paragraphs are returned.

    Args:
        text: Full article text.

    Returns:
        A list of paragraph strings (possibly empty if none qualify).
    """
    raw = text.split("\n\n")
    filtered = [p.strip() for p in raw if len(p.strip()) > MIN_PARAGRAPH_LEN]
    return filtered[:MAX_PARAGRAPHS_PER_DOC]


# ---------------------------------------------------------------------------
# Tag assignment
# ---------------------------------------------------------------------------


def assign_tags(document_id: int) -> str:
    """Assign 1–3 tags to a document based on its ``document_id``.

    The selection is deterministic and pseudo-random, derived from
    ``document_id`` modulo arithmetic over the tags pool.

    Args:
        document_id: Integer document identifier (1-based).

    Returns:
        A comma-separated string of tag names, e.g. ``"#HISTOIRE, #SPORT"``.
    """
    pool_size = len(TAGS_POOL)
    num_tags = (document_id % 3) + 1  # 1, 2, or 3 tags
    tags: list[str] = []
    for i in range(num_tags):
        idx = (document_id + i * 3) % pool_size
        tag = TAGS_POOL[idx]
        if tag not in tags:
            tags.append(tag)
    return ", ".join(tags)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def load_model() -> SentenceTransformer:
    """Load and return the sentence-transformer model.

    Returns:
        A loaded ``SentenceTransformer`` instance.
    """
    print(f"Loading model '{MODEL_NAME}'...")
    return SentenceTransformer(MODEL_NAME)


def encode_paragraphs(
    model: SentenceTransformer, texts: list[str]
) -> list[list[float]]:
    """Batch-encode a list of texts into embedding vectors.

    Args:
        model: A loaded ``SentenceTransformer`` instance.
        texts: List of paragraph strings to encode.

    Returns:
        A list of embedding vectors (each a list of floats).
    """
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client connected to localhost.

    Returns:
        A connected ``QdrantClient`` instance.
    """
    return QdrantClient(url=QDRANT_URL)


def recreate_collection(client: QdrantClient) -> None:
    """Delete (if existing) and recreate the Qdrant collection.

    Args:
        client: An active ``QdrantClient`` instance.
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def build_points(
    document_id: int,
    paragraphs: list[str],
    embeddings: list[list[float]],
    tags: str,
) -> list[PointStruct]:
    """Build Qdrant ``PointStruct`` objects for all paragraphs of one document.

    Args:
        document_id: Integer document identifier.
        paragraphs: List of paragraph text strings.
        embeddings: Corresponding embedding vectors.
        tags: Tag string assigned to this document.

    Returns:
        A list of ``PointStruct`` ready for upsert.
    """
    points: list[PointStruct] = []
    for para_idx, (text, vector) in enumerate(zip(paragraphs, embeddings)):
        paragraph_id = document_id * 100 + para_idx
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(paragraph_id))),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "paragraph_id": paragraph_id,
                    "tags": tags,
                    "text": text,
                },
            )
        )
    return points


def upsert_points(client: QdrantClient, points: list[PointStruct]) -> None:
    """Upsert a batch of points into the Qdrant collection.

    Args:
        client: An active ``QdrantClient`` instance.
        points: List of ``PointStruct`` objects to upsert.
    """
    client.upsert(collection_name=COLLECTION_NAME, points=points)


# ---------------------------------------------------------------------------
# Parquet export
# ---------------------------------------------------------------------------


def save_parquet(records: list[dict]) -> None:
    """Persist document-level records to a parquet file.

    Args:
        records: List of dicts with keys ``document_id`` (int) and ``text`` (str).
    """
    df = pd.DataFrame(records, columns=["document_id", "text"])
    df["document_id"] = df["document_id"].astype(int)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved {len(df)} documents to '{PARQUET_PATH}'.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def seed() -> None:
    """Main entry point: load data, embed, and populate Qdrant + parquet."""
    model = load_model()
    client = get_qdrant_client()
    recreate_collection(client)

    parquet_records: list[dict] = []
    total_points = 0

    for doc_index, article in enumerate(iter_wikipedia_articles(NUM_DOCS), start=1):
        text: str = article.get("text", "")
        paragraphs = split_paragraphs(text)

        if not paragraphs:
            print(f"  Doc {doc_index}: no valid paragraphs, skipping.")
            continue

        tags = assign_tags(doc_index)
        embeddings = encode_paragraphs(model, paragraphs)
        points = build_points(doc_index, paragraphs, embeddings, tags)

        upsert_points(client, points)

        parquet_records.append({"document_id": doc_index, "text": text})
        total_points += len(points)

        print(
            f"Doc {doc_index:>3}/{NUM_DOCS} processed — "
            f"{len(paragraphs)} paragraphs, tags: {tags}"
        )

    save_parquet(parquet_records)
    print(f"\nDone. Total points uploaded: {total_points}")


if __name__ == "__main__":
    seed()
