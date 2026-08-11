# Scripts

## `run_experiment.py`

Run **one** isolated experiment end-to-end.

### Usage

From the repository root:

```bash
# Federated experiment (Docker server + 2 clients)
python scripts/run_experiment.py 04_neural_network_3rounds

# Logistic regression / increased epochs
python scripts/run_experiment.py 03_logistic_regression_increased_epochs

# Local NN baselines
python scripts/run_experiment.py 05_neural_network_local_baselines

# Centralized NN (no Docker clients)
python scripts/run_experiment.py 06_centralized_neural_network
```

### Options

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

### Requirements

- Docker & Docker Compose available on `PATH`
- Port **8080** free for federated experiments
- Data files at `data/client_1/data.csv` and `data/client_2/data.csv`

### After a successful run

Refresh charts:

```bash
cd experiments/analysis
python -m jupyter notebook experiment_analysis.ipynb
```
