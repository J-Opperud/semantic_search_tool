import streamlit as st

from config import EXPERIMENT_RESULTS_PATH
from evaluate import (
    load_experiment_results,
    run_chunking_experiment,
    update_experiment_ratings,
    summarize_experiment,
)


def render_experiments():
    ratings_by_result = {}

    st.header("📈 Experiments")

    st.write(
        "Compare different retrieval configurations "
        "using the same evaluation questions."
    )

    st.subheader("Chunking Configurations")

    experiment_rows = [
        {
            "Configuration": "Small",
            "Chunk Size": 150,
            "Overlap": 25,
            "Status": "Ready",
        },
        {
            "Configuration": "Baseline",
            "Chunk Size": 300,
            "Overlap": 50,
            "Status": "Ready",
        },
        {
            "Configuration": "Large",
            "Chunk Size": 600,
            "Overlap": 100,
            "Status": "Ready",
        },
    ]

    st.dataframe(
        experiment_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Experiment Results")

    if EXPERIMENT_RESULTS_PATH.exists():

        results = load_experiment_results(
            EXPERIMENT_RESULTS_PATH
        )

        st.success(
            f"Loaded {len(results)} experiment results."
        )

        # -----------------------------------------------------------
        # Experiment metrics
        # -----------------------------------------------------------

        experiment_summary = summarize_experiment(
            results
        )

        st.subheader("Experiment Metrics")

        metric_rows = []

        for configuration in [
            "Small",
            "Baseline",
            "Large",
        ]:
            metrics = experiment_summary.get(
                configuration,
                {},
            )

            metric_rows.append(
                {
                    "Configuration": configuration,
                    "Precision@3": round(
                        metrics.get(
                            "precision_at_3",
                            0.0,
                        ),
                        3,
                    ),
                    "Mean Relevance": round(
                        metrics.get(
                            "mean_relevance",
                            0.0,
                        ),
                        3,
                    ),
                }
            )

        st.dataframe(
            metric_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # -----------------------------------------------------------
        # Individual experiment results and human ratings
        # -----------------------------------------------------------

        for configuration in [
            "Small",
            "Baseline",
            "Large",
        ]:

            st.subheader(
                f"{configuration} Configuration"
            )

            config_results = [
                result
                for result in results
                if result["configuration"] == configuration
            ]

            for result in config_results:

                st.markdown(
                    f"**{result['question']}**"
                )

                for result_number, chunk in enumerate(
                    result["results"],
                    start=1,
                ):

                    st.caption(
                        f"Result {result_number} | "
                        f"{chunk['source']} | "
                        f"Chunk {chunk['chunk_index']} | "
                        f"Distance: "
                        f"{chunk['distance']:.4f}"
                    )

                    st.write(
                        chunk["text"]
                    )

                    rating_options = [
                        "Not Relevant",
                        "Partially Relevant",
                        "Relevant",
                    ]

                    current_rating = None

                    if result.get("ratings"):
                        current_rating = result["ratings"][
                            result_number - 1
                        ]

                    rating = st.selectbox(
                        "Relevance",
                        options=rating_options,
                        index=(
                            rating_options.index(
                                current_rating
                            )
                            if current_rating in rating_options
                            else 0
                        ),
                        key=(
                            f"{result['configuration']}_"
                            f"{result['question']}_"
                            f"{result_number}"
                        ),
                    )

                    key = (
                        result["configuration"],
                        result["question"],
                    )

                    if key not in ratings_by_result:
                        ratings_by_result[key] = []

                    ratings_by_result[key].append(
                        rating
                    )

                    st.divider()

        # -----------------------------------------------------------
        # Save human relevance ratings
        # -----------------------------------------------------------

        if st.button(
            "💾 Save Relevance Ratings",
            type="primary",
        ):

            update_experiment_ratings(
                ratings_by_result,
                EXPERIMENT_RESULTS_PATH,
            )

            st.success(
                "Relevance ratings saved successfully."
            )

            st.rerun()

    else:

        st.info(
            "No experiment results yet. "
            "Run the experiment to generate them."
        )

        if st.button(
            "▶ Run Experiment",
            type="primary",
        ):

            with st.spinner(
                "Running chunking experiment..."
            ):

                run_chunking_experiment(
                    n_results=3
                )

            st.success(
                "Experiment complete."
            )

            st.rerun()
