"""Streamlit dashboard for Sentinel."""

import streamlit as st

from evaluate import (
    load_experiment_results,
    run_chunking_experiment,
)
from ui.overview import render_overview
from ui.search_tab import render_search
from ui.evaluation import render_evaluation
from ui.indexing import render_indexing
from ui.experiments import render_experiments

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RESULTS,
    DEFAULT_THRESHOLD,
    EMBEDDING_MODEL,
    EXPERIMENT_RESULTS_PATH,
)
from ingest import ingest
from search import get_collection


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Sentinel",
    page_icon="🔎",
    layout="wide",
)


# =========================================================
# Session state
# =========================================================

def initialize_session_state():
    """Initialize application state used across reruns."""

    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    if "index_message" not in st.session_state:
        st.session_state.index_message = None


initialize_session_state()


# =========================================================
# Helper functions
# =========================================================

def get_corpus_sources(collection):
    """Return the unique source documents in the collection."""

    data = collection.get(
        include=["metadatas"]
    )

    metadatas = data["metadatas"]

    return sorted(
        {
            metadata["source"]
            for metadata in metadatas
            if metadata and "source" in metadata
        }
    )


def get_index_information(collection):
    """Return basic information about the current index."""

    sources = get_corpus_sources(collection)

    return {
        "documents": len(sources),
        "chunks": collection.count(),
        "sources": sources,
        "embedding_model": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
        "chunking_method": "Fixed Character",
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_CHUNK_OVERLAP,
    }


# =========================================================
# Application header
# =========================================================

st.title("🔎 Sentinel")

st.write(
    "Federal Acquisition Intelligence / Semantic Search"
)

st.divider()


# =========================================================
# Connect to ChromaDB
# =========================================================

try:
    collection = get_collection()

except Exception as exc:
    st.error(
        "Unable to connect to the Sentinel index."
    )

    st.exception(exc)

    st.stop()


index_info = get_index_information(
    collection
)


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("🔎 Sentinel")

st.sidebar.caption(
    "Retrieval Engineering Workbench"
)

st.sidebar.divider()

st.sidebar.subheader("Search Settings")

n_results = st.sidebar.slider(
    "Number of results",
    min_value=1,
    max_value=20,
    value=DEFAULT_RESULTS,
)

threshold = st.sidebar.slider(
    "Distance threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_THRESHOLD,
    step=0.01,
)

selected_sources = st.sidebar.multiselect(
    "Source filter",
    options=index_info["sources"],
)

st.sidebar.divider()

st.sidebar.subheader("Current Index")

st.sidebar.write(
    f"**Documents:** {index_info['documents']}"
)

st.sidebar.write(
    f"**Chunks:** {index_info['chunks']:,}"
)

st.sidebar.write(
    f"**Embedding:** {index_info['embedding_model']}"
)


# =========================================================
# Main navigation
# =========================================================

overview_tab, search_tab, evaluation_tab, indexing_tab, experiments_tab = st.tabs(
    [
        "📊 Overview",
        "🔎 Search",
        "🧪 Evaluation",
        "⚙️ Index & Chunking",
        "📈 Experiments",
    ]
)



# OVERVIEW
# =========================================
with overview_tab:
        render_overview(index_info)




# SEARCH
# =======================================

with search_tab:
        render_search(
        n_results=n_results,
        threshold=threshold,
        selected_sources=selected_sources,
    )




# EVALUATION
# ========================================

with evaluation_tab:

    render_evaluation()

    


# INDEX & CHUNKING
# =======================================

with indexing_tab:

    render_indexing(index_info)

    



# EXPERIMENTS
# ======================================

with experiments_tab:

        render_experiments()
