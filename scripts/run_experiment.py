#!/usr/bin/env python3
"""
Run a single isolated experiment from experiments/<name>/

Examples:
  python scripts/run_experiment.py 04_neural_network_3rounds
  python scripts/run_experiment.py 01_logistic_regression_geographic
  python scripts/run_experiment.py 06_centralized_neural_network
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DATA_DIR = REPO_ROOT / "data"

FEDERATED = {
    "01_logistic_regression_geographic",
    "02_logistic_regression_strong_non_iid",
    "03_logistic_regression_increased_epochs",
    "04_neural_network_3rounds",
    "07_neural_network_5rounds",
}
LOCAL_BASELINES = {"05_neural_network_local_baselines"}
CENTRALIZED = {"06_centralized_neural_network"}

# Which partition each experiment must use
EXPERIMENT_DATA = {
    "01_logistic_regression_geographic": {
        "partition": "geographic",
        "client_1": DATA_DIR / "geographic" / "client_1" / "data.csv",
        "client_2": DATA_DIR / "geographic" / "client_2" / "data.csv",
    },
    "02_logistic_regression_strong_non_iid": {
        "partition": "shipping_mode",
        "client_1": DATA_DIR / "shipping_mode" / "client_1" / "data.csv",
        "client_2": DATA_DIR / "shipping_mode" / "client_2" / "data.csv",
    },
    "03_logistic_regression_increased_epochs": {
        "partition": "shipping_mode",
        "client_1": DATA_DIR / "shipping_mode" / "client_1" / "data.csv",
        "client_2": DATA_DIR / "shipping_mode" / "client_2" / "data.csv",
    },
    "04_neural_network_3rounds": {
        "partition": "shipping_mode",
        "client_1": DATA_DIR / "shipping_mode" / "client_1" / "data.csv",
        "client_2": DATA_DIR / "shipping_mode" / "client_2" / "data.csv",
    },
    "05_neural_network_local_baselines": {
        "partition": "shipping_mode",
        "client_1": DATA_DIR / "shipping_mode" / "client_1" / "data.csv",
        "client_2": DATA_DIR / "shipping_mode" / "client_2" / "data.csv",
    },
    "06_centralized_neural_network": {
        "partition": "shipping_mode",
        "client_1": DATA_DIR / "shipping_mode" / "client_1" / "data.csv",
        "client_2": DATA_DIR / "shipping_mode" / "client_2" / "data.csv",
    },
    "07_neural_network_5rounds": {
        "partition": "shipping_mode",
        "client_1": DATA_DIR / "shipping_mode" / "client_1" / "data.csv",
        "client_2": DATA_DIR / "shipping_mode" / "client_2" / "data.csv",
    },
}


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd})")
    return subprocess.run(cmd, cwd=str(cwd), check=check)


def docker_compose(exp_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["docker", "compose", *args], cwd=exp_dir, check=check)


def ensure_data_present(experiment_name: str) -> bool:
    """Fail fast if the correct partition CSVs are missing."""
    meta = EXPERIMENT_DATA.get(experiment_name)
    if not meta:
        print(f"No data mapping for experiment: {experiment_name}")
        return False

    c1, c2 = meta["client_1"], meta["client_2"]
    print(f"Data partition required: {meta['partition']}")
    print(f"  client_1 → {c1}")
    print(f"  client_2 → {c2}")

    missing = [p for p in (c1, c2) if not p.is_file()]
    if missing:
        print("\nERROR: required data files are missing:")
        for p in missing:
            print(f"  - {p}")
        print("\nGenerate partitions with:")
        print("  python data/partition_data.py")
        print("  # or only one mode:")
        print("  python data/partition_data.py --mode geographic")
        print("  python data/partition_data.py --mode shipping_mode")
        return False

    # Basic non-empty check
    for p in (c1, c2):
        if p.stat().st_size < 1000:
            print(f"WARNING: {p} looks very small ({p.stat().st_size} bytes)")
    return True


def wait_for_file(path: Path, timeout_sec: int, poll_sec: float = 2.0) -> bool:
    print(f"Waiting for {path.name} (timeout={timeout_sec}s)...")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            try:
                text = path.read_text(encoding="utf-8")
                if "flower_history" in text or "local_baseline" in text or "centralized_baseline" in text:
                    print(f"Found metrics file: {path}")
                    return True
                if path.stat().st_mtime > time.time() - timeout_sec:
                    print(f"Found updated metrics file: {path}")
                    return True
            except OSError:
                pass
        time.sleep(poll_sec)
    return False


def cleanup_mount_leftovers(exp_dir: Path) -> None:
    for name in ("data.csv", "data"):
        p = exp_dir / name
        if p.exists():
            if p.is_dir():
                import shutil

                shutil.rmtree(p, ignore_errors=True)
                print(f"Removed leftover directory: {p}")
            elif p.is_file() and p.stat().st_size == 0:
                p.unlink(missing_ok=True)


def run_federated(exp_dir: Path, timeout_sec: int, keep_up: bool) -> int:
    cleanup_mount_leftovers(exp_dir)
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()

    docker_compose(exp_dir, "down", check=False)
    docker_compose(exp_dir, "up", "--build", "-d")
    time.sleep(3)

    print("Starting client-1 and client-2...")
    p1 = subprocess.Popen(
        ["docker", "compose", "exec", "-T", "client-1", "python", "client_node.py"],
        cwd=str(exp_dir),
    )
    time.sleep(2)
    p2 = subprocess.Popen(
        ["docker", "compose", "exec", "-T", "client-2", "python", "client_node.py"],
        cwd=str(exp_dir),
    )

    rc1 = p1.wait()
    rc2 = p2.wait()
    print(f"client-1 exit={rc1}, client-2 exit={rc2}")

    ok = wait_for_file(metrics_path, timeout_sec=timeout_sec)
    if not ok:
        print("WARNING: metrics.json was not produced in time. Check server logs:")
        docker_compose(exp_dir, "logs", "central-server", check=False)
        if not keep_up:
            docker_compose(exp_dir, "down", check=False)
        return 1

    print("\n===== metrics.json =====")
    print(metrics_path.read_text(encoding="utf-8"))

    if not keep_up:
        docker_compose(exp_dir, "down", check=False)
    else:
        print("Containers left running (--keep-up).")

    return 0 if rc1 == 0 and rc2 == 0 else 1


def run_local_baselines(exp_dir: Path, timeout_sec: int, keep_up: bool) -> int:
    cleanup_mount_leftovers(exp_dir)
    for name in ("metrics_client_1.json", "metrics_client_2.json", "metrics.json"):
        p = exp_dir / name
        if p.exists():
            p.unlink()

    docker_compose(exp_dir, "down", check=False)
    docker_compose(exp_dir, "up", "--build", "-d")
    time.sleep(2)

    r1 = docker_compose(
        exp_dir, "exec", "-T", "client-1", "python", "local_baseline_nn.py", check=False
    )
    r2 = docker_compose(
        exp_dir, "exec", "-T", "client-2", "python", "local_baseline_nn.py", check=False
    )

    m1 = exp_dir / "metrics_client_1.json"
    m2 = exp_dir / "metrics_client_2.json"
    ok1 = wait_for_file(m1, timeout_sec=min(120, timeout_sec))
    ok2 = wait_for_file(m2, timeout_sec=min(120, timeout_sec))

    if ok1:
        print("\n===== metrics_client_1.json =====")
        print(m1.read_text(encoding="utf-8"))
    if ok2:
        print("\n===== metrics_client_2.json =====")
        print(m2.read_text(encoding="utf-8"))

    if not keep_up:
        docker_compose(exp_dir, "down", check=False)

    return 0 if r1.returncode == 0 and r2.returncode == 0 and ok1 and ok2 else 1


def run_centralized(exp_dir: Path) -> int:
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()

    script = exp_dir / "centralized_baseline_nn.py"
    if not script.is_file():
        print(f"Missing {script}")
        return 1

    # Prefer shipping_mode paths via env for the centralized script
    env = os.environ.copy() if (os := __import__("os")) else {}
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(exp_dir),
        env=env,
    )
    if metrics_path.is_file():
        print("\n===== metrics.json =====")
        print(metrics_path.read_text(encoding="utf-8"))
    else:
        print("WARNING: metrics.json was not written.")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated experiment")
    parser.add_argument(
        "experiment",
        help="Experiment folder name under experiments/ (e.g. 04_neural_network_3rounds)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Seconds to wait for metrics.json (default: 600)",
    )
    parser.add_argument(
        "--keep-up",
        action="store_true",
        help="Do not docker compose down after the run",
    )
    args = parser.parse_args()

    name = args.experiment.strip().rstrip("/")
    exp_dir = EXPERIMENTS_DIR / name
    if not exp_dir.is_dir():
        print(f"Experiment not found: {exp_dir}")
        print("Available:")
        for p in sorted(EXPERIMENTS_DIR.iterdir()):
            if p.is_dir() and p.name[0].isdigit():
                print(f"  - {p.name}")
        return 1

    print(f"Running experiment: {name}")
    print(f"Path: {exp_dir}")

    if not ensure_data_present(name):
        return 1

    if name in FEDERATED:
        return run_federated(exp_dir, timeout_sec=args.timeout, keep_up=args.keep_up)
    if name in LOCAL_BASELINES:
        return run_local_baselines(exp_dir, timeout_sec=args.timeout, keep_up=args.keep_up)
    if name in CENTRALIZED:
        return run_centralized(exp_dir)

    print(f"Unknown experiment type for: {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
