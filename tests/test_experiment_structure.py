"""Verify experiments/ archive layout is complete."""
from pathlib import Path

EXPECTED_EXPERIMENTS = [
    "01_logistic_regression_geographic",
    "02_logistic_regression_strong_non_iid",
    "03_logistic_regression_increased_epochs",
    "04_neural_network_3rounds",
    "05_neural_network_local_baselines",
    "06_centralized_neural_network",
    "07_neural_network_5rounds",
]

FEDERATED_EXPERIMENTS = [
    "01_logistic_regression_geographic",
    "02_logistic_regression_strong_non_iid",
    "03_logistic_regression_increased_epochs",
    "04_neural_network_3rounds",
    "07_neural_network_5rounds",
]


def test_experiments_dir_exists(experiments_dir: Path):
    assert experiments_dir.is_dir()


def test_expected_experiment_folders_exist(experiments_dir: Path):
    for name in EXPECTED_EXPERIMENTS:
        assert (experiments_dir / name).is_dir(), f"Missing experiment folder: {name}"


def test_federated_experiments_have_server_and_compose(experiments_dir: Path):
    for name in FEDERATED_EXPERIMENTS:
        folder = experiments_dir / name
        assert (folder / "docker-compose.yml").is_file()
        assert (folder / "server" / "main.py").is_file()
        assert (folder / "requirements.txt").is_file()
        assert (folder / "Dockerfile").is_file()


def test_analysis_notebook_exists(experiments_dir: Path):
    nb = experiments_dir / "analysis" / "experiment_analysis.ipynb"
    assert nb.is_file()


def test_dockerignore_present_for_docker_experiments(experiments_dir: Path):
    for name in FEDERATED_EXPERIMENTS + ["05_neural_network_local_baselines"]:
        assert (experiments_dir / name / ".dockerignore").is_file()
