# Experiment Analysis

Jupyter notebook that **reads metrics dynamically** from `experiments/*/metrics*.json`.

## Metrics supported

- accuracy
- f1_score
- precision
- recall
- roc_auc
- loss

Older `metrics.json` files without precision/recall/roc_auc still load; those columns will be empty until you re-run experiments with the updated evaluation code.

## Setup

```bash
cd experiments/analysis
pip install -r requirements.txt
jupyter notebook experiment_analysis.ipynb
```

## Outputs

Written under `figures/`:

- `comparison_from_metrics.png` — Accuracy / F1 / ROC-AUC bars
- `precision_recall_from_metrics.png` — Precision / Recall bars
- `learning_curves_from_metrics.png` — per-round curves including ROC-AUC when present
- `metrics_summary.csv` — flat table of all loaded metrics
