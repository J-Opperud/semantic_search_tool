
"""Evaluation and chunking experiment utilities for Sentinel."""

from dataclasses import dataclass, asdict
from typing import Any
from config import COLLECTION_NAME, EXPERIMENT_RESULTS_PATH
from search import retrieve
import json
from pathlib import Path


# Evaluation questions
# -------------------------------------------

EVALUATION_QUESTIONS = [
    "What is the purpose of the Federal Acquisition Regulation?",
    "What does the FAR mean by market research?",
    "When should an agency conduct market research during the acquisition process?",
    "What factors are considered when determining whether a prospective contractor is responsible?",
    "What requirements or considerations apply to small businesses in federal contracting?",
]


# Experiment configuration
# ----------------------------------------

@dataclass(frozen=True)
class ChunkConfig:
    """Describe one chunking configuration used in an experiment."""

    name: str
    chunk_size: int
    overlap: int


EXPERIMENT_CONFIGS = [
    ChunkConfig(
        name="Small",
        chunk_size=150,
        overlap=25,
    ),
    ChunkConfig(
        name="Baseline",
        chunk_size=300,
        overlap=50,
    ),
    ChunkConfig(
        name="Large",
        chunk_size=600,
        overlap=100,
    ),
]


# Relevance
# -----------------------------------------------

RELEVANCE_SCORES = {
    "Not Relevant": 0,
    "Partially Relevant": 1,
    "Relevant": 2,
}


def relevance_to_score(rating: str | None) -> int | None:
    """Convert a human relevance rating into a numeric score."""

    if rating is None or rating == "Not Rated":
        return None

    return RELEVANCE_SCORES.get(rating)



# Evaluation
# -----------------------------------------

def run_evaluation(
    n_results: int = 3,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    """Run the standard evaluation questions against the current index.
    Args:
        n_results: Number of results to retrieve for each question.
        collection_name: Name of the ChromaDB collection to evaluate.

    Returns:
        A list containing each question and its retrieved results.
    """

    if n_results <= 0:
        raise ValueError("n_results must be greater than zero.")

    evaluation_results = []

    for question in EVALUATION_QUESTIONS:
        results = retrieve(
            query=question,
            n_results=n_results,
            collection_name=collection_name,
            )

        evaluation_results.append(
            {
                "question": question,
                "results": results,
            }
        )

    return evaluation_results



# Experiment result structures
# --------------------------------------

@dataclass
class ExperimentResult:
    """Store the results of one query under one chunking configuration."""

    configuration: str
    chunk_size: int
    overlap: int
    question: str
    results: list[dict[str, Any]]
    ratings: list[str | None] | None = None




def create_experiment_result(
    config: ChunkConfig,
    question: str,
    results: list[dict[str, Any]],
) -> ExperimentResult:
    """Create a structured experiment result."""

    return ExperimentResult(
        configuration=config.name,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        question=question,
        results=results,
        ratings=[None] * len(results),
    )




# Relevance metrics
# -----------------------------------------------

def calculate_precision_at_k(
    ratings: list[str | None],
    k: int | None = None,
) -> float:
    """Calculate precision@k using 'Relevant' results as positive matches.

    'Partially Relevant' results are not counted as fully relevant.

    Unrated results are ignored rather than treated as irrelevant.
    """

    if k is not None:
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        ratings = ratings[:k]

    rated = [
        rating
        for rating in ratings
        if rating is not None and rating != "Not Rated"
    ]

    if not rated:
        return 0.0

    relevant = sum(
        rating == "Relevant"
        for rating in rated
    )

    return relevant / len(rated)


def calculate_mean_relevance(
    ratings: list[str | None],
    k: int | None = None,
) -> float:
    """Calculate the average graded relevance score.

    Scores:
        Not Relevant = 0
        Partially Relevant = 1
        Relevant = 2

    Unrated results are ignored.
    """

    if k is not None:
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        ratings = ratings[:k]

    scores = [
        score
        for rating in ratings
        if (score := relevance_to_score(rating)) is not None
    ]

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def summarize_experiment(
    ratings_by_configuration: dict[str, list[str | None]],
) -> dict[str, dict[str, float]]:
    """Create comparison metrics for each experiment configuration."""

    summary = {}

    for configuration, ratings in ratings_by_configuration.items():
        summary[configuration] = {
            "precision_at_3": calculate_precision_at_k(
                ratings,
                k=3,
            ),
            "mean_relevance": calculate_mean_relevance(
                ratings,
                k=3,
            ),
        }

    return summary


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------

def experiment_result_to_dict(result: ExperimentResult) -> dict[str, Any]:
    """Convert an experiment result into a JSON-friendly dictionary."""

    return asdict(result)


def load_experiment_results(
    path: Path,
) -> list[dict[str, Any]]:
    """Load saved experiment results from JSON."""
    
    if not path.exists():
        return []

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

def save_experiment_results(
    results: list[ExperimentResult],
    path: Path,
) -> None:
    """Save experiment results as JSON."""

    data = [
        experiment_result_to_dict(result)
        for result in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

def run_chunking_experiment(
    n_results: int = 3,
) -> list[ExperimentResult]:
    """Run the evaluation questions against each chunking configuration.

    Each configuration receives its own ChromaDB collection so the
    experiment does not modify Sentinel's normal search index.

    Args:
        n_results: Number of results retrieved for each question.

    Returns:
        A list of ExperimentResult objects.
    """

    if n_results <= 0:
        raise ValueError("n_results must be greater than zero.")

    experiment_results = []

    for config in EXPERIMENT_CONFIGS:
        collection_name = (
            f"s_experiment_{config.chunk_size}_{config.overlap}"
        )

        # Import here to avoid creating a circular import at module load time.
        from ingest import ingest

        ingest(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            reset=True,
            collection_name=collection_name,
        )

        evaluation_results = run_evaluation(
            n_results=n_results,
            collection_name=collection_name,
        )

        for evaluation in evaluation_results:
            experiment_results.append(
                create_experiment_result(
                    config=config,
                    question=evaluation["question"],
                    results=evaluation["results"],
                )
            )
    save_experiment_results(
        experiment_results,
        EXPERIMENT_RESULTS_PATH,
    )

    return experiment_results

def update_experiment_ratings(
    ratings_by_result: dict[tuple[str, str], list[str | None]],
    path: Path,
) -> None:
    """Update human relevance ratings for saved experiment results."""

    results = load_experiment_results(path)

    for result in results:
        key = (
            result["configuration"],
            result["question"],
        )

        if key in ratings_by_result:
            result["ratings"] = ratings_by_result[key]

    path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    

