import streamlit as st
from search import search


def render_search(
    n_results,
    threshold,
    selected_sources,
):
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