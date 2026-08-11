"""Reference implementation of extended metrics aggregation + export.
Copied into each experiment server/main.py (kept here for documentation).
"""

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
    return {f"federated_{k}": totals[k] / examples for k in METRIC_KEYS if totals[k] or k in ("accuracy", "f1_score")}
