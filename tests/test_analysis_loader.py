"""Tests for the analysis notebook metrics loading logic."""
import json
from pathlib import Path


def load_metrics(experiments_dir: Path):
    records = []
    for path in sorted(experiments_dir.glob("*/metrics*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["_path"] = str(path)
        data["_folder"] = path.parent.name
        records.append(data)
    return records


def test_load_metrics_returns_list(experiments_dir: Path):
    records = load_metrics(experiments_dir)
    assert isinstance(records, list)
    assert len(records) >= 1


def test_load_metrics_includes_folder_metadata(experiments_dir: Path):
    records = load_metrics(experiments_dir)
    for r in records:
        assert "_folder" in r
        assert "_path" in r
        assert "results" in r
