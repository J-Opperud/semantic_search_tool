import streamlit as st

from config import EXPERIMENT_RESULTS_PATH
from evaluate import (
    load_experiment_results,
    run_chunking_experiment,
)

def render_experiments():
    st.header("📈 Experiments")
    
    st.write(
        "Compare different retrieval configurations "
        "using the same evaluation questions."
    )

    st.subheader(" Chunking Configurations")

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
        for configuration in ["Small", "Baseline", "Large"]:

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

                    st.write(chunk["text"])

                st.divider()

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
