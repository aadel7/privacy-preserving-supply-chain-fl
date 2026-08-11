# Scripts

## `run_experiment.py`

Run **one** isolated experiment end-to-end.

```bash
# From repository root
python scripts/run_experiment.py 04_neural_network_3rounds
python scripts/run_experiment.py 05_neural_network_local_baselines
python scripts/run_experiment.py 06_centralized_neural_network
```

Options:

```bash
python scripts/run_experiment.py 04_neural_network_3rounds --timeout 900
python scripts/run_experiment.py 04_neural_network_3rounds --keep-up
```

| Flag | Meaning |
|------|--------|
| `--timeout N` | Max seconds to wait for `metrics.json` (default 600) |
| `--keep-up` | Leave containers running after the run |

### What it does (federated)

1. Cleans accidental `data/` / `data.csv/` mount leftovers
2. `docker compose down` / `up --build -d`
3. Starts `client-1` and `client-2`
4. Waits for `metrics.json` (auto-exported by the server)
5. Prints metrics and tears down (unless `--keep-up`)

---

## `run_all_experiments.py`

Run **all** (or a subset of) experiments **sequentially** via `run_experiment.py`.

```bash
# Full batch (slow — can take a long time)
python scripts/run_all_experiments.py

# Only selected experiments (by number or full name)
python scripts/run_all_experiments.py --only 04,05,06
python scripts/run_all_experiments.py --only 04_neural_network_3rounds,07_neural_network_5rounds

# Skip early LR experiments
python scripts/run_all_experiments.py --skip 01,02

# Stop at first failure
python scripts/run_all_experiments.py --stop-on-error

# Longer per-experiment timeout
python scripts/run_all_experiments.py --timeout 900
```

Default order:

1. `01_logistic_regression_geographic`
2. `02_logistic_regression_strong_non_iid`
3. `03_logistic_regression_increased_epochs`
4. `04_neural_network_3rounds`
5. `05_neural_network_local_baselines`
6. `06_centralized_neural_network`
7. `07_neural_network_5rounds`

At the end it prints a batch summary (OK/FAIL + time per experiment).

---

## Requirements

- Docker & Docker Compose on `PATH`
- Port **8080** free for federated experiments
- Data at `data/client_1/data.csv` and `data/client_2/data.csv`

## After runs

Refresh charts:

```bash
cd experiments/analysis
python -m jupyter notebook experiment_analysis.ipynb
```
