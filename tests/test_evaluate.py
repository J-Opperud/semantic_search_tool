from evaluate import load_experiment_results
import json

from evaluate import (
    ExperimentResult,
    save_experiment_results,
    load_experiment_results,
)

def test_load_experiment_results_missing_file(tmp_path):
    """Return an empty list when the results file does not exist."""

    path = tmp_path / "experiment_results.json"

    results = load_experiment_results(path)

    assert results == []




def test_load_experiment_results_existing_file(tmp_path):
    """Load experiment results from an existing JSON file."""

    path = tmp_path / "experiment_results.json"

    expected = [
        {
            "configuration": "Small",
            "chunk_size": 150,
            "overlap": 25,
        }
    ]

    path.write_text(
        json.dumps(expected),
        encoding="utf-8",
    )

    results = load_experiment_results(path)

    assert results == expected

def test_experiment_result_to_dict():
    """Convert an ExperimentResult into a JSON-friendly dictionary."""

    from evaluate import ExperimentResult, experiment_result_to_dict

    result = ExperimentResult(
        configuration="Small",
        chunk_size=150,
        overlap=25,
        question="What is the FAR?",
        results=[
            {
                "text": "Federal Acquisition Regulation",
                "source": "far.txt",
                "chunk_index": 1,
                "distance": 0.25,
            }
        ],
    )

    data = experiment_result_to_dict(result)

    assert data["configuration"] == "Small"
    assert data["chunk_size"] == 150
    assert data["overlap"] == 25
    assert len(data["results"]) == 1

from evaluate import (
    calculate_mean_relevance,
    calculate_precision_at_k,
)


def test_precision_at_3():
    """Calculate precision using the first three ratings."""

    ratings = [
        "Relevant",
        "Not Relevant",
        "Relevant",
    ]

    assert calculate_precision_at_k(ratings, k=3) == 2 / 3


def test_mean_relevance():
    """Calculate the average graded relevance."""

    ratings = [
        "Relevant",          # 2
        "Partially Relevant", # 1
        "Not Relevant",      # 0
    ]

    assert calculate_mean_relevance(ratings, k=3) == 1.0


def test_experiment_results_persist(tmp_path):
    results = [
        ExperimentResult(
            configuration="baseline",
            chunk_size=300,
            overlap=50,
            question="What is market research?",
            results=[
                {
                    "source": "far_part_10_market_research.txt",
                    "chunk_index": 21,
                    "distance": 0.352,
                    "text": "Market research information...",
                }
            ],
            ratings=["Relevant"],
        )
    ]

    path = tmp_path / "experiment_results.json"

    save_experiment_results(results, path)

    assert path.exists()

    loaded_results = load_experiment_results(path)

    assert loaded_results == [
        {
            "configuration": "baseline",
            "chunk_size": 300,
            "overlap": 50,
            "question": "What is market research?",
            "results": [
                {
                    "source": "far_part_10_market_research.txt",
                    "chunk_index": 21,
                    "distance": 0.352,
                    "text": "Market research information...",
                }
            ],
            "ratings": ["Relevant"],
        }
    ]