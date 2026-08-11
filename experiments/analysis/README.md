# Experiment Analysis

This folder contains a Jupyter notebook that **reads metrics dynamically** from each experiment's `metrics.json` files.

## Setup

```bash
cd experiments/analysis
pip install -r requirements.txt
jupyter notebook experiment_analysis.ipynb
```

Or with VS Code / Cursor: open `experiment_analysis.ipynb` and run all cells.

## What it does

1. Scans `experiments/*/metrics*.json`
2. Builds a results table
3. Generates comparison bar charts
4. Plots learning curves when per-round data exists
5. Exports `figures/metrics_summary.csv` and PNG charts

## Important

- Charts are driven by **whatever is in the metrics files**, not hardcoded numbers.
- After re-running an experiment, re-run the notebook to refresh figures.
- Historical metrics files (written earlier) work the same as newly generated ones.
