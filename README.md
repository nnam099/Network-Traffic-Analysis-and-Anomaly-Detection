# Network Traffic Analysis and Anomaly Detection IDS

![IDS overview](assets/readme/ids-overview.png)

Hybrid network intrusion detector for UNSW-NB15. It combines known-attack
classification with autoencoder-based anomaly scoring and exposes the result
through a Streamlit dashboard and FastAPI service.

> Research prototype only. Treat detections as analyst leads, not final verdicts.

## Highlights

- Known-attack classification and zero-day/OOD scoring.
- Single-flow and CSV batch analysis.
- SHAP, uncertainty, MITRE ATT&CK context, and optional LLM triage.
- Local SQLite alert queue with analyst status and notes.
- Reproducible training, artifact validation, and automated tests.

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run dashboard/app.py
```

Start the API:

```bash
export IDS_MODEL_PATH=checkpoints/ids_v14_model.pth
export IDS_PIPELINE_PATH=checkpoints/ids_v14_pipeline.pkl
uvicorn ids.api:app --app-dir src --host 0.0.0.0 --port 8080
```

The API provides `GET /health`, `POST /predict`, and `POST /predict-flow`.

## How it works

```text
Flow CSV / JSON
      │
      ▼
Schema normalization and feature engineering
      │
      ▼
Classifier + contrastive backbone + autoencoder
      │
      ▼
Calibrated OOD score and uncertainty
      │
      ├── Streamlit analyst dashboard
      └── FastAPI inference service
```

The checked-in v14 model is the operational demo. The v15 experiment remains
training-only until its offline and runtime OOD scoring paths are equivalent.

## Training

Place UNSW-NB15 CSV files in `data/`, then run:

```bash
python train.py \
  --data_dir data \
  --save_dir checkpoints \
  --plot_dir plots \
  --seed 42
```

For a short synthetic run:

```bash
python train.py --demo --epochs 2
```

Training produces a model checkpoint, preprocessing pipeline, evaluation report,
and diagnostic plots. See [data/README.md](data/README.md) for expected inputs.

## Project structure

```text
dashboard/       Streamlit interface
src/ids/         Models, training, inference, API, and runtime helpers
scripts/         Evaluation and maintenance commands
tests/           Unit, API, and research-protocol tests
checkpoints/     Demo v14 artifacts
data/            Local datasets and small committed samples
plots/           Reference figures
results/         Reports and reproducibility metadata
docs/            Architecture and operating notes
```

## Reference output

These figures come from the checked-in demo artifacts and are illustrative, not
current scientific benchmarks.

| Confusion matrix | OOD ROC curves |
| --- | --- |
| ![Confusion matrix](plots/v14_confusion_matrix.png) | ![OOD ROC curves](plots/v14_roc_curves.png) |

## Development

```bash
python -m pytest -q
python -m ruff check .
python scripts/smoke_check.py
```

Useful evaluation commands:

```bash
python scripts/evaluate_csv.py input.csv --output-dir results/csv_eval
python scripts/evaluate_baselines.py
python scripts/drift_report.py
```

## Limitations

- UNSW-NB15 does not represent current production traffic.
- A held-out attack family is only a proxy for zero-day behavior.
- Metrics depend heavily on split, deduplication, and calibration policy.
- Only load PyTorch and pickle artifacts from trusted sources.

Further reading: [architecture](docs/architecture.md),
[operations](docs/operations.md), and
[CSV normalization](docs/real_world_csv.md).

## License

MIT — see [LICENSE](LICENSE).
