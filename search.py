"""Semantic search functionality for the document collection."""

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_RESULTS,
    EMBEDDING_MODEL,
)


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

def get_collection():
    """
    Connect to the persistent ChromaDB collection.

    Returns:
        The Sentinel ChromaDB collection.
    """

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load and cache the embedding model.

    Returns:
        The SentenceTransformer embedding model.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve(
    query: str,
    n_results: int = DEFAULT_RESULTS,
    sources: list[str] | None = None,
):
    """
    Retrieve the nearest document chunks.

    This function performs semantic retrieval but does not
    apply a distance threshold.

    Args:
        query: User's search question.
        n_results: Maximum number of candidates to retrieve.
        sources: Optional list of source filenames to search.

    Returns:
        A list of matching document chunks.

    Raises:
        ValueError: If the query is empty or n_results is invalid.
    """

    if not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    if n_results <= 0:
        raise ValueError(
            "n_results must be greater than zero."
        )

    collection = get_collection()
    model = get_embedding_model()

    query_embedding = model.encode(
        query
    ).tolist()

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }

    if sources:
        query_kwargs["where"] = {
            "source": {
                "$in": sources
            }
        }

    results = collection.query(
        **query_kwargs
    )

    matches = []

    for index, document in enumerate(
        results["documents"][0]
    ):
        matches.append(
            {
                "text": document,
                "source": results["metadatas"][0][index]["source"],
                "chunk_index": results["metadatas"][0][index]["chunk_index"],
                "distance": results["distances"][0][index],
            }
        )

    return matches


# ---------------------------------------------------------
# Threshold filtering
# ---------------------------------------------------------

def filter_results(
    results: list[dict],
    threshold: float | None = None,
):
    """
    Filter retrieved results by cosine distance.

    Lower distance means greater semantic similarity.

    Args:
        results: Results returned by retrieve().
        threshold: Maximum allowed distance.
            None disables threshold filtering.

    Returns:
        Results that meet the threshold.
    """

    if threshold is None:
        return results

    if threshold < 0:
        raise ValueError(
            "threshold cannot be negative."
        )

    return [
        result
        for result in results
        if result["distance"] <= threshold
    ]


# ---------------------------------------------------------
# Combined search
# ---------------------------------------------------------

def search(
    query: str,
    n_results: int = DEFAULT_RESULTS,
    threshold: float | None = None,
    sources: list[str] | None = None,
):
    """
    Retrieve and optionally filter semantic search results.

    Args:
        query: User's search question.
        n_results: Maximum number of candidates to retrieve.
        threshold: Maximum allowed distance.
            None disables threshold filtering.
        sources: Optional list of source filenames.

    Returns:
        A list of matching results.
    """

    results = retrieve(
        query=query,
        n_results=n_results,
        sources=sources,
    )

    return filter_results(
        results,
        threshold=threshold,
    )


# ---------------------------------------------------------
# Command-line testing
# ---------------------------------------------------------

if __name__ == "__main__":
    query = input("Search: ")

    results = search(
        query=query,
        n_results=5,
        
    )

    print(f"\nFound {len(results)} results.")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 60)
        print(f"Result {index}")
        print("=" * 60)

        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_index']}")
        print(f"Distance: {result['distance']:.4f}")

        print("\n" + result["text"][:1000])