# Architecture - IDS v14

## Data Flow

1. Load UNSW-NB15 CSV files from `data/`.
2. Normalize labels and engineer numeric/categorical flow features.
3. Split known classes for supervised training and hold out selected classes as anomalous/OOD traffic.
4. Train a hybrid model with classifier, contrastive projection and autoencoder heads.
5. Calibrate thresholds on validation data.
6. Save model weights plus preprocessing pipeline metadata in `checkpoints/`.
7. FastAPI loads the artifacts and serves flow predictions to the React frontend.
8. The legacy Streamlit dashboard remains available for research workflows.

## Model

`IDSBackbone`

```text
Linear(n_features -> 256) -> LayerNorm -> GELU -> ResBlock x 3
```

Main heads:

| Component | Output | Purpose |
| --- | --- | --- |
| `classifier` | `n_classes` | Known-attack classification |
| `proj_head` | 64 | Supervised contrastive representation |
| `ae` | `n_features` | Reconstruction error for anomaly/OOD scoring |

Training objective:

```text
FocalLoss + 0.3 * SupConLoss + 0.5 * AE_MSE
```

Runtime anomaly score:

```text
learned hybrid meta-score(ae_re, 1 - max_prob)
```

The legacy compatibility wrapper `src/ids_v14_unswnb15.py` re-exports the split modules under `src/ids/`; new code should import from `ids.models`, `ids.dataset`, `ids.trainer`, `ids.evaluator` and related modules directly.

## Runtime Guards

- `ids.artifact_validator` checks feature count, class count, scaler metadata and threshold metadata before inference.
- `ids.inference_runtime` contains pure verdict, anomaly decision and risk helpers used by the dashboard.
- `ids.api` exposes `/health`, `/predict`, and `/predict/flow` for v14 artifact-backed FastAPI inference.
- `ids.evaluator.predict_with_uncertainty()` runs Monte Carlo Dropout for served uncertainty without changing training behavior.
- `scripts/smoke_check.py` compiles the code and runs unit smoke tests.
- `ids.llm_agent` is lazy-initialized so importing the dashboard does not require an API key.
