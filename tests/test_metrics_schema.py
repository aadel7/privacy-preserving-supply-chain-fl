"""Validate metrics.json files under experiments/."""
import json
from pathlib import Path


def _iter_metrics_files(experiments_dir: Path):
    return sorted(experiments_dir.glob("*/metrics*.json"))


def test_at_least_one_metrics_file(experiments_dir: Path):
    files = list(_iter_metrics_files(experiments_dir))
    assert files, "No metrics*.json found under experiments/"


def test_each_metrics_file_is_valid_json_and_has_core_fields(experiments_dir: Path):
    files = list(_iter_metrics_files(experiments_dir))
    assert files

    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), path
        assert "experiment_id" in data or "results" in data, f"Missing keys in {path}"
        assert "results" in data, f"Missing results in {path}"

        results = data["results"]
        assert isinstance(results, dict)

        known = {"federated", "local", "centralized"}
        nested_locals = {
            "local_client_1_express",
            "local_client_2_standard",
            "local_client_1_europe",
            "local_client_2_latam",
        }
        assert known.intersection(results.keys()) or nested_locals.intersection(results.keys()), (
            f"No recognized result block in {path}: {list(results.keys())}"
        )


def test_federated_metrics_values_in_range(experiments_dir: Path):
    for path in _iter_metrics_files(experiments_dir):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        fed = data.get("results", {}).get("federated")
        if not fed:
            continue

        for key in ("accuracy", "f1_score"):
            if fed.get(key) is not None:
                val = float(fed[key])
                assert 0.0 <= val <= 1.0, f"{path} federated.{key}={val}"

        per_round = fed.get("per_round") or {}
        for key in ("accuracy", "f1_score"):
            for i, val in enumerate(per_round.get(key, [])):
                assert 0.0 <= float(val) <= 1.0, f"{path} per_round.{key}[{i}]={val}"
