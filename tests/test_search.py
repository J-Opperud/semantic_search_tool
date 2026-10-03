
import pytest

from search import filter_results, retrieve, search


class FakeCollection:
    """Minimal ChromaDB collection used for search tests."""

    def __init__(self, results):
        self.results = results
        self.last_query = None

    def query(self, **kwargs):
        self.last_query = kwargs
        return self.results


class FakeModel:
    """Minimal embedding model used for search tests."""

    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def sample_chroma_results():
    """Return a representative ChromaDB query response."""

    return {
        "documents": [[
            "Federal Acquisition Regulation information.",
            "Market research information.",
            "Contractor responsibility requirements.",
        ]],
        "metadatas": [[
            {
                "source": "far.txt",
                "chunk_index": 0,
            },
            {
                "source": "market_research.txt",
                "chunk_index": 1,
            },
            {
                "source": "responsibility.txt",
                "chunk_index": 2,
            },
        ]],
        "distances": [[
            0.10,
            0.30,
            0.70,
        ]],
    }


def test_retrieve_rejects_empty_query():
    """Reject blank search queries."""

    with pytest.raises(ValueError, match="cannot be empty"):
        retrieve("")


def test_retrieve_rejects_whitespace_query():
    """Reject queries containing only whitespace."""

    with pytest.raises(ValueError, match="cannot be empty"):
        retrieve("   ")


def test_retrieve_rejects_invalid_result_count():
    """Reject zero or negative result counts."""

    with pytest.raises(ValueError, match="greater than zero"):
        retrieve(
            "What is the FAR?",
            n_results=0,
        )


def test_filter_results_rejects_negative_threshold():
    """Reject invalid negative distance thresholds."""

    with pytest.raises(ValueError, match="cannot be negative"):
        filter_results(
            [],
            threshold=-0.1,
        )


def test_filter_results_keeps_results_within_threshold():
    """Keep results whose distance is within the threshold."""

    results = [
        {
            "text": "Close result",
            "source": "a.txt",
            "chunk_index": 0,
            "distance": 0.20,
        },
        {
            "text": "Far result",
            "source": "b.txt",
            "chunk_index": 1,
            "distance": 0.80,
        },
    ]

    filtered = filter_results(
        results,
        threshold=0.50,
    )

    assert len(filtered) == 1
    assert filtered[0]["text"] == "Close result"


def test_filter_results_without_threshold():
    """Return all results when no threshold is provided."""

    results = [
        {
            "text": "Result one",
            "source": "a.txt",
            "chunk_index": 0,
            "distance": 0.20,
        },
        {
            "text": "Result two",
            "source": "b.txt",
            "chunk_index": 1,
            "distance": 0.80,
        },
    ]

    assert filter_results(results) == results


def test_retrieve_returns_expected_result_structure(
    monkeypatch,
):
    """Convert ChromaDB results into the application's result format."""

    collection = FakeCollection(
        sample_chroma_results()
    )
    model = FakeModel()

    monkeypatch.setattr(
        "search.get_collection",
        lambda collection_name="docs": collection,
    )

    monkeypatch.setattr(
        "search.get_embedding_model",
        lambda: model,
    )

    results = retrieve(
        "What is the FAR?",
        n_results=3,
    )

    assert len(results) == 3

    assert results[0] == {
        "text": "Federal Acquisition Regulation information.",
        "source": "far.txt",
        "chunk_index": 0,
        "distance": 0.10,
    }


def test_retrieve_passes_source_filter_to_chroma(
    monkeypatch,
):
    """Apply source filtering through the ChromaDB query."""

    collection = FakeCollection(
        sample_chroma_results()
    )
    model = FakeModel()

    monkeypatch.setattr(
        "search.get_collection",
        lambda collection_name="docs": collection,
    )

    monkeypatch.setattr(
        "search.get_embedding_model",
        lambda: model,
    )

    retrieve(
        "What is the FAR?",
        n_results=3,
        sources=["far.txt"],
    )

    assert collection.last_query["where"] == {
        "source": {
            "$in": ["far.txt"],
        }
    }


def test_search_combines_retrieval_and_threshold(
    monkeypatch,
):
    """Search should retrieve results and then apply threshold filtering."""

    collection = FakeCollection(
        sample_chroma_results()
    )
    model = FakeModel()

    monkeypatch.setattr(
        "search.get_collection",
        lambda collection_name="docs": collection,
    )

    monkeypatch.setattr(
        "search.get_embedding_model",
        lambda: model,
    )

    results = search(
        "What is the FAR?",
        n_results=3,
        threshold=0.50,
    )

    assert len(results) == 2

    assert results[0]["distance"] == 0.10
    assert results[1]["distance"] == 0.30


def test_search_allows_custom_collection(
    monkeypatch,
):
    """Search should pass a custom collection name through retrieval."""

    collection = FakeCollection(
        sample_chroma_results()
    )
    model = FakeModel()

    requested_collection = {}

    def fake_get_collection(collection_name="docs"):
        requested_collection["name"] = collection_name
        return collection

    monkeypatch.setattr(
        "search.get_collection",
        fake_get_collection,
    )

    monkeypatch.setattr(
        "search.get_embedding_model",
        lambda: model,
    )

    search(
        "What is the FAR?",
        collection_name="test_collection",
    )

    assert requested_collection["name"] == "test_collection"

