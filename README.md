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

A labeled external flow CSV can be scored for engineering diagnostics only:

```bash
python scripts/evaluate_csv.py /path/to/external-flows.csv \
  --label-col Label --output-dir results/csv_eval --scores-csv
```

This is **not** cross-dataset scientific validation. The P10 audit
([docs/p10_external_validation.md](docs/p10_external_validation.md)) found that
CIC-IDS2017 and the other reviewed public datasets do not reproduce the frozen
61-field semantic contract, so their scores cannot be used to claim zero-day or
transfer performance. Treat any external run as a schema/normalization check,
not a benchmark.

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

## Repository layout

| Path | Contents |
| --- | --- |
| `src/ids/` | Runtime inference plus the versioned P1–P13 research protocol modules. |
| `src/ids_v14_unswnb15.py` | Compatibility wrapper re-exporting the split v14 modules. |
| `src/ids_v15_unswnb15.py` | Disabled v15 experiment; the runtime rejects v15 artifacts. |
| `frontend/` | React + Vite demonstration client. |
| `dashboard/` | Legacy Streamlit analyst workspace kept for research workflows. |
| `scripts/` | Training, evaluation, audit and operations entry points. |
| `tests/` | Unit, protocol and API tests (frozen artifacts are read directly). |
| `docs/` | Architecture, operations and per-phase research protocols. |
| `results/` | Frozen protocol reports and evidence used by tests and audits. |
| `plots/` | Reference figures cited by the documentation. |
| `checkpoints/` | Committed v14 demo artifacts; other checkpoints are local only. |
| `data/` | Local datasets; only small parser fixtures under `data/samples/` are committed. |

Regenerated outputs such as `results/csv_eval/`, local databases and plot
rebuilds are ignored by Git. Frozen protocol artifacts are intentionally
versioned because tests and audit scripts read them directly.

## Development

```bash
python -m pytest -q
python -m ruff check .
cd frontend && npm run build
```

To remove local caches and build output without touching datasets, checkpoints or
frozen protocol evidence:

```bash
./scripts/clean_local.sh
```

Versioned model artifacts and historical experiment reports retain their
original field names for reproducibility. The application uses
**Anomalous Traffic** and **anomaly detection** in its public wording.

## License

MIT — see [LICENSE](LICENSE).
