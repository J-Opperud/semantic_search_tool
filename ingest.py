from pathlib import Path
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

from chunking import chunk_text
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
)

# ---------------------------------------------------------
# Document loading
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(path: Path) -> str:
    """
    Read a supported document.

    Args:
        path: Path to the document.

    Returns:
        The extracted document text.

    Raises:
        ValueError: If the file type is unsupported.
    """

    extension = path.suffix.lower()

    if extension in {".txt", ".md"}:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    if extension == ".pdf":
        reader = PdfReader(path)

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def find_documents(directory: Path = DOCS_DIR) -> list[Path]:
    """
    Find supported documents in the document directory.

    Args:
        directory: Directory containing source documents.

    Returns:
        A sorted list of document paths.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {directory}"
        )

    documents = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(documents)


# ---------------------------------------------------------
# Chunk creation
# ---------------------------------------------------------

def create_chunks(
    documents: list[Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Load documents and convert them into searchable chunks.

    Args:
        documents: Source document paths.
        chunk_size: Maximum chunk size.
        overlap: Number of overlapping characters.

    Returns:
        A list of dictionaries containing text and metadata.
    """

    records = []

    for document_path in documents:
        text = load_document(document_path)

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{document_path.name}::chunk-{index}",
                    "text": chunk,
                    "metadata": {
                        "source": document_path.name,
                        "chunk_index": index,
                    },
                }
            )

    return records


# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------

def create_collection(
    reset: bool = False,
    collection_name: str = COLLECTION_NAME,
):
    """
    Create or retrieve the persistent ChromaDB collection.

    Args:
        reset: Delete the existing collection before creating it.

    Returns:
        A ChromaDB collection.
    """

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    if reset:
        try:
            client.delete_collection(
                name=collection_name
            )
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    return collection


# ---------------------------------------------------------
# Indexing
# ---------------------------------------------------------

def index_documents(
    records: list[dict],
    collection,
    model: SentenceTransformer,
) -> int:
    """
    Generate embeddings and store records in ChromaDB.

    Args:
        records: Document chunk records.
        collection: ChromaDB collection.
        model: SentenceTransformer embedding model.

    Returns:
        Number of records indexed.
    """

    if not records:
        return 0

    texts = [
        record["text"]
        for record in records
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
    )

    collection.upsert(
        ids=[
            record["id"]
            for record in records
        ],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            record["metadata"]
            for record in records
        ],
    )

    return len(records)


# ---------------------------------------------------------
# Main indexing workflow
# ---------------------------------------------------------

def ingest(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset: bool = False,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Run the complete document ingestion pipeline.

    Args:
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between chunks.
        reset: Rebuild the ChromaDB collection.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Number of indexed chunks.
    """

    documents = find_documents()

    print(f"Found {len(documents)} documents.")

    records = create_chunks(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print(f"Created {len(records)} chunks.")

    collection = create_collection(
    reset=reset,
    collection_name=collection_name,
)

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    count = index_documents(
        records,
        collection,
        model,
    )

    print(
    f"Indexed {count} chunks into "
    f"'{collection_name}'."
)

    return count


if __name__ == "__main__":
    ingest(reset=True)