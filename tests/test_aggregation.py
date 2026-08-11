"""Tests for server-side weighted metrics aggregation."""


def weighted_average(metrics):
    """Same logic as used in experiment server main.py files."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "federated_accuracy": sum(accuracies) / sum(examples),
        "federated_f1_score": sum(f1_scores) / sum(examples),
    }


def test_weighted_average_equal_clients():
    metrics = [
        (100, {"accuracy": 0.8, "f1_score": 0.7}),
        (100, {"accuracy": 0.6, "f1_score": 0.5}),
    ]
    out = weighted_average(metrics)
    assert abs(out["federated_accuracy"] - 0.7) < 1e-9
    assert abs(out["federated_f1_score"] - 0.6) < 1e-9


def test_weighted_average_unequal_clients():
    # Larger client should dominate
    metrics = [
        (10, {"accuracy": 1.0, "f1_score": 1.0}),
        (90, {"accuracy": 0.0, "f1_score": 0.0}),
    ]
    out = weighted_average(metrics)
    assert abs(out["federated_accuracy"] - 0.1) < 1e-9
    assert abs(out["federated_f1_score"] - 0.1) < 1e-9
