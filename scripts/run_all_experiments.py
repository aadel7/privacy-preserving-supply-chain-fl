#!/usr/bin/env python3
"""
Run multiple isolated experiments sequentially using run_experiment.py.

Examples:
  python scripts/run_all_experiments.py
  python scripts/run_all_experiments.py --only 01,08
  python scripts/run_all_experiments.py --only 08
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_experiment.py"

DEFAULT_ORDER = [
    "01_logistic_regression_geographic",
    "08_geographic_local_baselines",
    "02_logistic_regression_strong_non_iid",
    "03_logistic_regression_increased_epochs",
    "04_neural_network_3rounds",
    "05_neural_network_local_baselines",
    "06_centralized_neural_network",
    "07_neural_network_5rounds",
]

SHORT_IDS = {f"{i:02d}" for i in range(1, 9)}


def parse_list(value: str | None) -> set[str] | None:
    if not value:
        return None
    items = set()
    for part in value.split(","):
        part = part.strip().rstrip("/")
        if not part:
            continue
        key = part.zfill(2) if part.isdigit() else part
        if key in SHORT_IDS or (len(part) <= 2 and part.isdigit()):
            matched = [e for e in DEFAULT_ORDER if e.startswith(key)]
            if matched:
                items.add(matched[0])
                continue
        items.add(part)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all (or selected) experiments sequentially")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--skip", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--keep-up", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    only = parse_list(args.only)
    skip = parse_list(args.skip) or set()

    experiments = []
    for name in DEFAULT_ORDER:
        if only is not None and name not in only:
            continue
        if name in skip:
            continue
        experiments.append(name)

    if not experiments:
        print("No experiments selected.")
        return 1

    print("Experiments to run:")
    for e in experiments:
        print(f"  - {e}")
    print()

    results: list[tuple[str, int, float]] = []
    overall_start = time.time()

    for name in experiments:
        print("\n" + "=" * 72)
        print(f"START {name}")
        print("=" * 72)
        cmd = [sys.executable, str(RUNNER), name, "--timeout", str(args.timeout)]
        if args.keep_up:
            cmd.append("--keep-up")

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        elapsed = time.time() - t0
        results.append((name, proc.returncode, elapsed))

        status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
        print(f"\nEND {name}: {status} in {elapsed/60:.1f} min")

        if proc.returncode != 0 and args.stop_on_error:
            print("Stopping batch due to --stop-on-error")
            break

    total = time.time() - overall_start
    print("\n" + "=" * 72)
    print("BATCH SUMMARY")
    print("=" * 72)
    failed = 0
    for name, rc, elapsed in results:
        flag = "OK" if rc == 0 else "FAIL"
        if rc != 0:
            failed += 1
        print(f"  [{flag}] {name:45s}  {elapsed/60:6.1f} min  (rc={rc})")
    print(f"\nTotal: {len(results)} run(s), {failed} failed, wall time {total/60:.1f} min")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
