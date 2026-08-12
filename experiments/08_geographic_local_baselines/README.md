# Exp 08 — Geographic Local Baselines

Isolated **logistic regression** baselines on the geographic partition:

- Client 1: **Europe** (`data/geographic/client_1`)
- Client 2: **LATAM** (`data/geographic/client_2`)

These runs complete the mild non-IID story: compare local Europe / local LATAM against federated LR (Exp 01).

## Prerequisites

```bash
python data/partition_data.py --mode geographic
# or full:
python data/partition_data.py
```

## Run

```bash
# from repo root
python scripts/run_experiment.py 08_geographic_local_baselines
```

Manual:

```bash
cd experiments/08_geographic_local_baselines
docker compose up --build -d
docker compose exec client-1 python local_baseline.py
docker compose exec client-2 python local_baseline.py
cat metrics_client_1.json metrics_client_2.json
docker compose down
```

## Expected pattern (mild non-IID)

Local Europe ≈ Local LATAM ≈ Federated Exp 01 (~0.69 accuracy).
