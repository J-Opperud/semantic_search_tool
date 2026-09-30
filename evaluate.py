"""Evaluation utilities for Sentinel semantic retrieval."""

from search import retrieve


# ---------------------------------------------------------
# Fixed evaluation questions
# ---------------------------------------------------------

EVALUATION_QUESTIONS = [
    "What is the purpose of the Federal Acquisition Regulation?",

    "What does the FAR mean by market research?",

    "When should an agency conduct market research "
    "during the acquisition process?",

    "What factors are considered when determining "
    "whether a prospective contractor is responsible?",

    "What requirements or considerations apply "
    "to small businesses in federal contracting?",
]


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def run_evaluation(
    n_results: int = 3,
):
    """
    Run the fixed evaluation questions against
    the current Sentinel index.

    Args:
        n_results: Number of results to retrieve per question.

    Returns:
        A list containing evaluation questions and
        their retrieved results.
    """

    evaluation_results = []

    for question in EVALUATION_QUESTIONS:

        results = retrieve(
            query=question,
            n_results=n_results,
        )

        evaluation_results.append(
            {
                "question": question,
                "results": results,
            }
        )

    return evaluation_results