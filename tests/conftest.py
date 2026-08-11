"""Shared fixtures for project tests."""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def experiments_dir() -> Path:
    return EXPERIMENTS_DIR
