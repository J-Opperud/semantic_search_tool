import streamlit as st

def render_overview(index_info):

    

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


# 