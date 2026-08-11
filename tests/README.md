# Tests

Lightweight pytest suite for the project. These tests **do not** start Docker or run full federated training. They verify:

- experiment folder structure
- `metrics.json` schema and value ranges
- metrics export helper behaviour
- FedAvg weighted metrics aggregation
- MLP forward shapes
- analysis metrics loader

## Setup

```bash
pip install -r tests/requirements.txt
# torch is required for test_mlp_model.py
```

## Run

From the repository root:

```bash
pytest tests/ -v
```

Run a single module:

```bash
pytest tests/test_metrics_schema.py -v
pytest tests/test_aggregation.py -v
```

## Notes

- Full Flower/Docker integration tests are intentionally out of scope here (slow, environment-heavy).
- After changing experiment servers or metrics format, re-run this suite.
- If `torch` is not installed, `test_mlp_model.py` will fail; install it or skip with:\n  `pytest tests/ -v --ignore=tests/test_mlp_model.py`
