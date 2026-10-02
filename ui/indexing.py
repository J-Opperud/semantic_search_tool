import streamlit as st

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
from ingest import ingest


def render_indexing(index_info):

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

