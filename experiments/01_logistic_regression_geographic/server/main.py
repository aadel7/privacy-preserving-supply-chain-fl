import json
import os
from datetime import datetime, timezone

import flwr as fl

METRIC_KEYS = ("accuracy", "f1_score", "precision", "recall", "roc_auc")


def weighted_average(metrics):
    totals = {k: 0.0 for k in METRIC_KEYS}
    examples = 0
    for num_examples, m in metrics:
        examples += num_examples
        for k in METRIC_KEYS:
            if k in m:
                totals[k] += num_examples * float(m[k])
    if examples == 0:
        return {}
    return {f"federated_{k}": totals[k] / examples for k in METRIC_KEYS}


def export_metrics(history, output_path="metrics.json", extra=None):
    losses = getattr(history, "losses_distributed", []) or []
    metrics_dist = getattr(history, "metrics_distributed", {}) or {}
    loss_per_round = [float(v) for _, v in losses]

    def series(name):
        return [float(v) for _, v in metrics_dist.get(name, [])]

    acc, f1 = series("federated_accuracy"), series("federated_f1_score")
    precision, recall = series("federated_precision"), series("federated_recall")
    roc_auc = series("federated_roc_auc")

    payload = {
        "experiment_id": os.environ.get("EXPERIMENT_ID", "01_logistic_regression_geographic"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "flower_history",
        "num_rounds": len(loss_per_round) or len(acc),
        "results": {
            "federated": {
                "accuracy": acc[-1] if acc else None,
                "f1_score": f1[-1] if f1 else None,
                "precision": precision[-1] if precision else None,
                "recall": recall[-1] if recall else None,
                "roc_auc": roc_auc[-1] if roc_auc else None,
                "loss": loss_per_round[-1] if loss_per_round else None,
                "per_round": {
                    "accuracy": acc,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": roc_auc,
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
