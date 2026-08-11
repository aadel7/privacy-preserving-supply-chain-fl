import json
import os
from datetime import datetime, timezone

import flwr as fl


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "federated_accuracy": sum(accuracies) / sum(examples),
        "federated_f1_score": sum(f1_scores) / sum(examples),
    }


def export_metrics(history, output_path="metrics.json", extra=None):
    """Export Flower History to metrics.json (dynamic, not hardcoded)."""
    losses = getattr(history, "losses_distributed", []) or []
    metrics_dist = getattr(history, "metrics_distributed", {}) or {}

    loss_per_round = [float(v) for _, v in losses]
    acc_pairs = metrics_dist.get("federated_accuracy", [])
    f1_pairs = metrics_dist.get("federated_f1_score", [])
    acc_per_round = [float(v) for _, v in acc_pairs]
    f1_per_round = [float(v) for _, v in f1_pairs]

    payload = {
        "experiment_id": os.environ.get("EXPERIMENT_ID", "01_logistic_regression_geographic"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "flower_history",
        "num_rounds": len(loss_per_round) or len(acc_per_round),
        "results": {
            "federated": {
                "accuracy": acc_per_round[-1] if acc_per_round else None,
                "f1_score": f1_per_round[-1] if f1_per_round else None,
                "loss": loss_per_round[-1] if loss_per_round else None,
                "per_round": {
                    "accuracy": acc_per_round,
                    "f1_score": f1_per_round,
                    "loss": loss_per_round,
                },
            }
        },
    }
    if extra:
        payload.update(extra)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[metrics] Wrote {output_path}")
    return payload


def main():
    print("Starting Federated Learning Central Server...")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    export_metrics(
        history,
        output_path=os.environ.get("METRICS_PATH", "metrics.json"),
        extra={
            "model": "Logistic Regression",
            "partitioning": "Geographic (Europe vs LATAM)",
            "non_iid_level": "mild",
            "local_epochs": 1,
        },
    )


if __name__ == "__main__":
    main()
