"""Unit tests for metrics export helpers (no Docker / Flower runtime required)."""
import json
from pathlib import Path
from types import SimpleNamespace


def export_metrics(history, output_path="metrics.json", extra=None, experiment_id="test_exp"):
    """Mirror of the server export_metrics logic used in experiment servers."""
    losses = getattr(history, "losses_distributed", []) or []
    metrics_dist = getattr(history, "metrics_distributed", {}) or {}

    loss_per_round = [float(v) for _, v in losses]
    acc_pairs = metrics_dist.get("federated_accuracy", [])
    f1_pairs = metrics_dist.get("federated_f1_score", [])
    acc_per_round = [float(v) for _, v in acc_pairs]
    f1_per_round = [float(v) for _, v in f1_pairs]

    payload = {
        "experiment_id": experiment_id,
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

    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def test_export_metrics_writes_file(tmp_path: Path):
    history = SimpleNamespace(
        losses_distributed=[(1, 0.4), (2, 0.3)],
        metrics_distributed={
            "federated_accuracy": [(1, 0.6), (2, 0.7)],
            "federated_f1_score": [(1, 0.55), (2, 0.65)],
        },
    )
    out = tmp_path / "metrics.json"
    payload = export_metrics(history, output_path=str(out), experiment_id="unit_test")

    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["experiment_id"] == "unit_test"
    assert data["source"] == "flower_history"
    assert data["results"]["federated"]["accuracy"] == 0.7
    assert data["results"]["federated"]["f1_score"] == 0.65
    assert data["results"]["federated"]["loss"] == 0.3
    assert data["results"]["federated"]["per_round"]["accuracy"] == [0.6, 0.7]
    assert payload["num_rounds"] == 2


def test_export_metrics_handles_empty_history(tmp_path: Path):
    history = SimpleNamespace(losses_distributed=[], metrics_distributed={})
    out = tmp_path / "empty.json"
    payload = export_metrics(history, output_path=str(out))

    fed = payload["results"]["federated"]
    assert fed["accuracy"] is None
    assert fed["per_round"]["accuracy"] == []
