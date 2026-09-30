"""Streamlit dashboard for Sentinel."""

import streamlit as st

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RESULTS,
    DEFAULT_THRESHOLD,
    EMBEDDING_MODEL,
)
from ingest import ingest
from search import get_collection, search


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


# =========================================================
# OVERVIEW
# =========================================================

with overview_tab:

    st.header("📊 Overview")

    st.write(
        "Current state of the Sentinel semantic search index."
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Documents",
            index_info["documents"],
        )

    with col2:
        st.metric(
            "Indexed Chunks",
            f"{index_info['chunks']:,}",
        )

    with col3:
        st.metric(
            "Sources",
            len(index_info["sources"]),
        )

    with col4:
        st.metric(
            "Index Status",
            "Ready",
        )

    st.divider()

    # -----------------------------------------------------
    # Current index
    # -----------------------------------------------------

    st.subheader("Current Index")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write(
            f"**Chunking Method:** "
            f"{index_info['chunking_method']}"
        )

        st.write(
            f"**Chunk Size:** "
            f"{index_info['chunk_size']}"
        )

        st.write(
            f"**Overlap:** "
            f"{index_info['overlap']}"
        )

    with info_col2:

        st.write(
            f"**Embedding Model:** "
            f"{index_info['embedding_model']}"
        )

        st.write(
            f"**Vector Store:** "
            f"ChromaDB"
        )

        st.write(
            f"**Collection:** "
            f"{index_info['collection']}"
        )

    st.divider()

    # -----------------------------------------------------
    # Corpus
    # -----------------------------------------------------

    st.subheader("Corpus")

    if index_info["sources"]:

        corpus_rows = []

        for source in index_info["sources"]:
            corpus_rows.append(
                {
                    "Source": source,
                }
            )

        st.dataframe(
            corpus_rows,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No documents are currently indexed."
        )


# =========================================================
# SEARCH
# =========================================================

with search_tab:

    st.header("🔎 Search")

    st.write(
        "Search the Sentinel corpus using semantic similarity."
    )

    query = st.text_input(
        "Search the corpus",
        value=st.session_state.last_query,
        placeholder=(
            "Example: What is the purpose of "
            "the Federal Acquisition Regulation?"
        ),
    )

    search_button = st.button(
        "🔎 Search",
        type="primary",
    )

    if search_button:

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        else:

            st.session_state.last_query = query

            with st.spinner(
                "Searching the Sentinel corpus..."
            ):

                results = search(
                    query=query,
                    n_results=n_results,
                    threshold=threshold,
                    sources=selected_sources or None,
                )

            if not results:

                st.warning(
                    "No results matched the current "
                    "distance threshold."
                )

            else:

                st.success(
                    f"Found {len(results)} matching results."
                )

                for index, result in enumerate(
                    results,
                    start=1,
                ):

                    st.subheader(
                        f"Result {index}"
                    )

                    st.caption(
                        f"Source: {result['source']} | "
                        f"Chunk: {result['chunk_index']} | "
                        f"Distance: "
                        f"{result['distance']:.4f}"
                    )

                    st.write(
                        result["text"]
                    )

                    st.divider()


# =========================================================
# EVALUATION
# =========================================================

with evaluation_tab:

    st.header("🧪 Evaluation")

    st.write(
        "Controlled retrieval evaluation using Sentinel's "
        "five fixed evaluation questions."
    )

    st.info(
        "Evaluation always retrieves the top 3 results. "
        "The Search tab's distance threshold does not "
        "affect evaluation."
    )

    # -----------------------------------------------------
    # Initialize evaluation state
    # -----------------------------------------------------

    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = None

    if "relevance_scores" not in st.session_state:
        st.session_state.relevance_scores = {}

    # -----------------------------------------------------
    # Evaluation controls
    # -----------------------------------------------------

    st.subheader("Evaluation Controls")

    st.write(
        "Each question will retrieve the three closest "
        "chunks from the current index."
    )

    run_evaluation_button = st.button(
        "▶ Run Evaluation",
        type="primary",
        use_container_width=True,
    )

    if run_evaluation_button:

        from evaluate import run_evaluation

        with st.spinner(
            "Running evaluation questions..."
        ):

            st.session_state.evaluation_results = (
                run_evaluation(
                    n_results=3
                )
            )

        st.session_state.relevance_scores = {}

        st.success(
            "Evaluation complete. "
            "Review and score the retrieved results below."
        )

    # -----------------------------------------------------
    # Display evaluation results
    # -----------------------------------------------------

    if st.session_state.evaluation_results:

        st.divider()

        st.subheader("Retrieved Results")

        st.caption(
            "Rate each result based on how useful it is "
            "for answering the evaluation question."
        )

        for question_number, evaluation in enumerate(
            st.session_state.evaluation_results,
            start=1,
        ):

            question = evaluation["question"]
            results = evaluation["results"]

            st.markdown(
                f"### Question {question_number}"
            )

            st.write(
                f"**{question}**"
            )

            if not results:

                st.warning(
                    "No results were returned."
                )

                continue

            for result_number, result in enumerate(
                results,
                start=1,
            ):

                st.markdown(
                    f"#### Result {result_number}"
                )

                result_col1, result_col2 = st.columns(
                    [4, 1]
                )

                with result_col1:

                    st.caption(
                        f"Source: {result['source']} | "
                        f"Chunk: {result['chunk_index']} | "
                        f"Distance: "
                        f"{result['distance']:.4f}"
                    )

                with result_col2:

                    score_key = (
                        f"q{question_number}_"
                        f"r{result_number}"
                    )

                    current_score = (
                        st.session_state.relevance_scores
                        .get(
                            score_key,
                            "Not Rated",
                        )
                    )

                    score = st.selectbox(
                        "Relevance",
                        options=[
                            "Not Rated",
                            "Not Relevant",
                            "Partially Relevant",
                            "Relevant",
                        ],
                        index=[
                            "Not Rated",
                            "Not Relevant",
                            "Partially Relevant",
                            "Relevant",
                        ].index(current_score),
                        key=f"rating_{score_key}",
                    )

                    st.session_state.relevance_scores[
                        score_key
                    ] = score

                st.write(
                    result["text"]
                )

                st.divider()

        # -------------------------------------------------
        # Evaluation summary
        # -------------------------------------------------

        st.subheader("Evaluation Summary")

        total_results = sum(
            len(item["results"])
            for item in st.session_state.evaluation_results
        )

        rated_results = sum(
            1
            for score in (
                st.session_state.relevance_scores.values()
            )
            if score != "Not Rated"
        )

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.metric(
                "Results",
                total_results,
            )

        with summary_col2:

            st.metric(
                "Results Rated",
                rated_results,
            )

        if rated_results < total_results:

            st.warning(
                f"{total_results - rated_results} "
                "results still need a relevance rating."
            )

        else:

            st.success(
                "All retrieved results have been rated."
            )

    else:

        st.info(
            "Run the evaluation to retrieve the top three "
            "results for each fixed question."
        )

# =========================================================
# INDEX & CHUNKING
# =========================================================

with indexing_tab:

    st.header("⚙️ Index & Chunking")

    st.write(
        "Configure how Sentinel divides documents before "
        "embedding and storing them in ChromaDB."
    )

    st.info(
        "Changing chunk size or overlap requires the corpus "
        "to be re-indexed."
    )

    # -----------------------------------------------------
    # Current configuration
    # -----------------------------------------------------

    st.subheader("Current Configuration")

    current_col1, current_col2, current_col3 = st.columns(3)

    with current_col1:

        st.metric(
            "Chunk Size",
            index_info["chunk_size"],
        )

    with current_col2:

        st.metric(
            "Overlap",
            index_info["overlap"],
        )

    with current_col3:

        st.metric(
            "Indexed Chunks",
            f"{index_info['chunks']:,}",
        )

    st.divider()

    # -----------------------------------------------------
    # Chunking method
    # -----------------------------------------------------

    st.subheader("Chunking Method")

    chunking_method = st.selectbox(
        "Method",
        options=[
            "Fixed Character",
        ],
        help=(
            "Additional chunking methods will be added "
            "once they are implemented in chunking.py."
        ),
    )

    # -----------------------------------------------------
    # Presets
    # -----------------------------------------------------

    st.subheader("Chunking Configuration")

    preset = st.selectbox(
        "Experiment Preset",
        options=[
            "Custom",
            "Small — 150 / 25",
            "Baseline — 300 / 50",
            "Large — 600 / 100",
        ],
        index=2,
    )

    if preset == "Small — 150 / 25":

        default_chunk_size = 150
        default_overlap = 25

    elif preset == "Baseline — 300 / 50":

        default_chunk_size = 300
        default_overlap = 50

    elif preset == "Large — 600 / 100":

        default_chunk_size = 600
        default_overlap = 100

    else:

        default_chunk_size = DEFAULT_CHUNK_SIZE
        default_overlap = DEFAULT_CHUNK_OVERLAP

    chunk_size = st.number_input(
        "Chunk Size",
        min_value=50,
        max_value=2000,
        value=default_chunk_size,
        step=50,
    )

    overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=max(
            0,
            int(chunk_size) - 1,
        ),
        value=min(
            default_overlap,
            max(
                0,
                int(chunk_size) - 1,
            ),
        ),
        step=10,
    )

    # -----------------------------------------------------
    # Preview
    # -----------------------------------------------------

    st.subheader("Configuration Preview")

    preview_col1, preview_col2, preview_col3 = st.columns(3)

    with preview_col1:

        st.metric(
            "Method",
            chunking_method,
        )

    with preview_col2:

        st.metric(
            "Chunk Size",
            int(chunk_size),
        )

    with preview_col3:

        st.metric(
            "Overlap",
            int(overlap),
        )

    if overlap >= chunk_size:

        st.error(
            "Overlap must be smaller than chunk size."
        )

    else:

        st.success(
            f"Ready to index using "
            f"{chunking_method.lower()} chunking."
        )

    st.divider()

    # -----------------------------------------------------
    # Re-index
    # -----------------------------------------------------

    st.subheader("Re-index Corpus")

    st.write(
        "This will rebuild the ChromaDB collection using "
        "the selected configuration."
    )

    reindex = st.button(
        "🔄 Re-index Corpus",
        type="primary",
        disabled=overlap >= chunk_size,
        use_container_width=True,
    )

    if reindex:

        with st.status(
            "Re-indexing Sentinel corpus...",
            expanded=True,
        ) as status:

            st.write(
                f"Chunk size: {int(chunk_size)}"
            )

            st.write(
                f"Overlap: {int(overlap)}"
            )

            st.write(
                "Loading documents..."
            )

            count = ingest(
                chunk_size=int(chunk_size),
                overlap=int(overlap),
                reset=True,
            )

            status.update(
                label=(
                    f"Re-index complete — "
                    f"{count:,} chunks indexed."
                ),
                state="complete",
            )

        st.success(
            f"Successfully indexed {count:,} chunks."
        )

        st.rerun()


# =========================================================
# EXPERIMENTS
# =========================================================

with experiments_tab:

    st.header("📈 Experiments")

    st.write(
        "Compare different retrieval configurations "
        "using the same evaluation questions."
    )

    st.subheader("Planned Chunking Configurations")

    experiment_rows = [
        {
            "Configuration": "Small",
            "Chunk Size": 150,
            "Overlap": 25,
            "Status": "Planned",
        },
        {
            "Configuration": "Baseline",
            "Chunk Size": 300,
            "Overlap": 50,
            "Status": "Current",
        },
        {
            "Configuration": "Large",
            "Chunk Size": 600,
            "Overlap": 100,
            "Status": "Planned",
        },
    ]

    st.dataframe(
        experiment_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Experiment Results")

    st.info(
        "Experiment results will appear here after the "
        "evaluation runner is implemented."
    )