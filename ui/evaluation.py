import streamlit as st

from evaluate import run_evaluation

def render_evaluation():
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