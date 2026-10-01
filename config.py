"""Application configuration for the semantic search system."""

from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs_wip"
CHROMA_DIR = BASE_DIR / "chroma_data"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENT_RESULTS_PATH = Path("experiment_results.json")

# ---------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------
UPSERT_BATCH_SIZE = 5000
COLLECTION_NAME = "docs"


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50


# ---------------------------------------------------------
# Search
# ---------------------------------------------------------

DEFAULT_RESULTS = 5
DEFAULT_THRESHOLD = 0.50