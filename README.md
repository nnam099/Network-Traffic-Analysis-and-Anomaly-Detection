# Network Traffic Analysis and Anomaly Detection

Hybrid network intrusion detection research prototype. A React and Vite frontend
analyzes flow CSV files through a FastAPI service backed by a classifier and an
autoencoder. The training and evaluation code is written in Python.

> Detection results are investigation leads. An anomalous flow is not proof of an
> attack or of a previously unknown vulnerability.

## What the application shows

| Verdict | Meaning |
| --- | --- |
| Normal Traffic | The flow was not flagged by the anomaly rule and the classifier predicted Normal. |
| Known Attack | The flow was not flagged by the anomaly rule and the classifier predicted an attack class. |
| Anomalous Traffic | The calibrated anomaly rule flagged the flow for review. |

The displayed **confidence** is the classifier's highest class probability;
it is not the probability that an anomaly verdict is correct. The displayed
**anomaly score** combines autoencoder reconstruction error and classifier
uncertainty. A higher score alone does not determine the verdict; the model's
calibrated decision rule does.

## Run the application

Use Python 3.11 or 3.12 and Node.js for the frontend. In the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export IDS_MODEL_PATH=checkpoints/ids_v14_model.pth
export IDS_PIPELINE_PATH=checkpoints/ids_v14_pipeline.pkl
uvicorn ids.api:app --app-dir src --host 127.0.0.1 --port 8080
```

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>. Check the API at
<http://127.0.0.1:8080/health>. If the API runs on another port, start Vite
with `VITE_API_URL=http://127.0.0.1:8081 npm run dev`.

For a quick demo, upload `data/samples/cicflowmeter_sample.csv` or
`data/samples/firewall_flow_sample.csv`. The browser analyzes at most the first
100 flows; use the command line for full evaluation.

```bash
python scripts/evaluate_csv.py data/UNSW_NB15_testing-set.csv \
  --label-col label --output-dir results/csv_eval --scores-csv
```

For an external dataset, use a labeled CIC-IDS2017 flow CSV and specify its
`Label` column. Keep the model fixed when measuring transfer to another dataset:

```bash
python scripts/evaluate_csv.py /path/to/cic-ids2017.csv \
  --label-col Label --output-dir results/external_eval --scores-csv
```

Cross-dataset results need label distribution and normalization checks. A run
that loads successfully does not demonstrate detection quality.

## Architecture

```text
React + Vite (frontend/) -> FastAPI (src/ids/api.py)
                                |
                                v
                     Flow schema normalization
                                |
                                v
                   Classifier + autoencoder
                                |
                                v
                    Calibrated anomaly decision
```

The `dashboard/` directory contains the older Streamlit interface and is kept
for compatibility with existing research workflows. The React frontend is the
primary demonstration interface. See [architecture](docs/architecture.md) and
[operations](docs/operations.md) for further detail.

## Development

```bash
python -m pytest -q
python -m ruff check .
cd frontend && npm run build
```

Versioned model artifacts and historical experiment reports retain their
original field names for reproducibility. The application uses
**Anomalous Traffic** and **anomaly detection** in its public wording.

## License

MIT — see [LICENSE](LICENSE).
