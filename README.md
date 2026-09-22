# Zero-Day Detection AutoEncoder IDS

Prototype IDS for UNSW-NB15 research and SOC-style demonstrations. The project combines known-attack classification, reconstruction-error based zero-day/OOD detection, Streamlit dashboard analysis, FastAPI inference serving, SHAP explainability, heuristic MITRE ATT&CK mapping and optional LLM triage.

> This is a research/demo project, not a production IDS. Analyst review and supporting SIEM/firewall/endpoint evidence are still required.

## Overview

The training pipeline learns from selected known attack classes and holds out other classes to simulate zero-day or out-of-distribution traffic. The dashboard and FastAPI server load saved artifacts and support single-alert analysis, CSV batch uploads and real-time JSON prediction.

Main flow:

1. Load and clean UNSW-NB15 CSV files.
2. Engineer flow features and encode categorical fields.
3. Train a hybrid supervised classifier + contrastive representation + autoencoder model.
4. Calibrate anomaly/zero-day thresholds.
5. Save checkpoint and preprocessing pipeline artifacts.
6. Analyze alerts in the Streamlit dashboard or call the FastAPI `/predict` endpoint.
7. Attach SHAP, MITRE, uncertainty and optional LLM context for SOC triage.

## Current Status

- v14 is the operational default because local artifacts exist in `checkpoints/`.
- v15 training code remains available for experiments, but v15 runtime selection is disabled until KNN/OOD-ensemble scoring has offline/runtime parity tests.
- Runtime dependencies are pinned in `requirements.txt`; smoke-check dependency ranges are in `requirements-smoke.txt`; developer tooling is in `requirements-dev.txt`.
- Phase 2/3 and P1 verification passes 111 tests with one v15-artifact skip, including the full API suite through direct ASGI transport and explicit application lifespan handling. This avoids a Starlette `TestClient`/AnyIO blocking-portal deadlock specific to the current Python 3.13 test environment while exercising the same FastAPI app.
- Smoke coverage includes artifact contract validation, threshold metadata validation, artifact manifest hashing, duplicate feature-name rejection, environment readiness checks, export config handling, checkpoint metadata patch logic, SQLite alert store persistence, CSV input guardrails, CSV normalization quality checks, dashboard preprocessing/context contracts, dashboard UI helper contracts, AI context selection, alert queue filtering, top-N batch alert selection, alert entity enrichment, lightweight correlation, time-window incident grouping, labeled evaluation reporting, API edge cases, Recon/DoS prototype separation, LLM fallback behavior, MITRE mapping and v14 artifact loading.
- `llm_agent.py` lazy-loads provider clients, so importing dashboard code does not require an API key.
- A Windows GitHub Actions smoke workflow is available at `.github/workflows/smoke.yml`.
- A scheduled and PR-triggered dependency audit workflow is available at `.github/workflows/dependency-audit.yml`.
- A Dockerfile is available for FastAPI inference on port `8080`.
- The checked-in operational and first-pass Step 2 metrics are retained only as historical/demo records. Neither is valid for a scientific claim; see the invalidation notice below.

## Features

- Known-attack classification for classes such as `Normal`, `DoS`, `Exploits`, `Reconnaissance` and `Generic`.
- Zero-day/OOD detection using reconstruction error, classifier confidence and a learned logistic-regression hybrid score. Label-assisted adaptive AE thresholding is experimental/demo-only and is disabled in scientific benchmarks.
- Streamlit SOC dashboard for single alert review, CSV batch analysis, alert history and AI follow-up.
- FastAPI inference server for real-time JSON predictions from saved v14 artifacts.
- Monte Carlo Dropout uncertainty for served predictions, with `LOW_CONFIDENCE` labeling when entropy is high.
- SQLite alert store for persisted queue history, status, analyst notes, queue filtering, top-N batch alert persistence and lightweight correlation groups.
- Time-window incident grouping across repeated source IP, service, class and attack-family signals in the analyst queue.
- Real-world CSV normalization for common firewall/flow/Zeek/Suricata-like exports.
- CSV upload guardrails for empty, oversized or malformed files.
- SHAP top-feature explanation.
- Heuristic MITRE ATT&CK mapping.
- Optional LLM triage through Groq, Gemini, OpenAI or Anthropic.

## Repository Layout

```text
src/
  train.py                 # v14 training entry point
  ids/                     # v14 model, dataset, losses, trainer, evaluator, thresholds and plots
  ids_v14_unswnb15.py      # compatibility wrapper for older imports/commands
  ids_v15_unswnb15.py      # experimental v15 pipeline
  inference_runtime.py     # pure verdict/zero-day/risk/CSV-quality helpers used by dashboard
  serve.py                 # FastAPI v14 inference server
  dashboard_runtime.py     # Streamlit-free dashboard preprocessing, AI context and fallback helpers
  alert_store.py           # SQLite alert history, status and analyst note persistence
  batch_evaluator.py       # CLI-friendly batch inference, CSV reporting and threshold calibration
  input_guard.py           # uploaded CSV size/shape validation
  artifact_validator.py    # checkpoint/pipeline contract validation
  explainer.py             # SHAP explainer
  log_normalizer.py        # real-world CSV normalization
  mitre_mapper.py          # heuristic MITRE ATT&CK mapping
  llm_agent.py             # optional lazy-loaded LLM triage
dashboard/
  app.py                   # Streamlit SOC dashboard
  views_queue.py           # alert queue, correlation and incident-window UI
  views_analysis.py        # alert-analysis safety UI helpers
  views_batch.py           # CSV batch summary UI helpers
  views_ood.py             # OOD candidate detail table helpers
  views_ai.py              # Ask AI context/suggestion UI helpers
  views_report.py          # JSON report/export helpers
  views_setup.py           # setup/status UI helpers
  ui_safety.py             # shared limitations and report safety copy
configs/
  config_default.yaml      # default v15 config
docs/
  architecture.md
  operations.md
  real_world_csv.md
  roadmap_real_world.md
  project_audit.md
tests/
  test_smoke.py
scripts/
  check_environment.py
  smoke_check.py
  install_requirements.ps1
  quality_check.ps1
  run_dashboard.ps1
  evaluate_csv.py
  evaluate_baselines.py
  drift_report.py
  regenerate_v14_report.py
  train_v14.ps1
  train_v15.ps1
  train.sh
  train_v15.sh
```

## Setup

Recommended:

- Python 3.11 for pinned runtime work. Python 3.11 is used by CI.
- Windows PowerShell or a Unix-like shell.

Clone the correct repository:

```powershell
git clone https://github.com/nnam099/ZeroDay-Detection-AutoEncoder-IDS.git
cd ZeroDay-Detection-AutoEncoder-IDS
```

Create/activate a virtual environment, then install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\install_requirements.ps1
```

Manual install:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
python -m pip install --progress-bar off -r requirements.txt
```

Minimal dependencies for smoke checks only:

```powershell
python -m pip install --progress-bar off -r requirements-smoke.txt
```

Developer test dependencies:

```powershell
python -m pip install --progress-bar off -r requirements-dev.txt
```

Dependency policy:

- `requirements.txt` is the lock-style runtime set for dashboard/API/demo execution.
- `requirements-smoke.txt` keeps broad lower bounds so CI can run compile, lint and unit smoke checks without installing optional dashboard/LLM extras.
- `requirements-dev.txt` contains local developer tools such as `ruff`, `pytest` and `pre-commit`.

The UTF-8 environment variables avoid console encoding failures when the project path contains Vietnamese characters.

Optional LLM setup:

```powershell
Copy-Item .env.example .env
# then fill the provider key you actually use
```

## Data

Place UNSW-NB15 CSV files in `data/`, for example:

```text
data/
  UNSW-NB15_1.csv
  UNSW-NB15_2.csv
  UNSW-NB15_3.csv
  UNSW-NB15_4.csv
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv
```

`data/` is ignored by git because it is large and local.

## Reproducibility

- Runtime packages are pinned in `requirements.txt`; test-only packages are pinned in `requirements-dev.txt`.
- v14 training defaults to seed `42`. Override it with `--seed` when running `train.py` or `src/ids_v14_unswnb15.py`.
- Data splits, PyTorch, NumPy, Python hashing and DataLoader workers are seeded through `seed_everything()` and seeded worker initialization.
- The Kaggle-trained v14 demo artifacts are committed under `checkpoints/` so the dashboard/API can run after clone. Additional retrained checkpoints, CSV datasets and SQLite files remain ignored because they can be large or environment-specific.
- Use `scripts/artifact_manifest.py` after retraining to record local artifact hashes.

For production-like deployments, prefer publishing signed artifacts through a release, Hugging Face artifact, Google Drive file or internal model registry object, then point runtime tools at it with `IDS_MODEL_PATH` and `IDS_PIPELINE_PATH`.

## Run Dashboard

Default dashboard mode uses v14:

```powershell
$env:IDS_MODEL_VERSION="v14"
streamlit run dashboard/app.py
```

PowerShell launcher:

```powershell
.\scripts\run_dashboard.ps1
```

Important environment variables:

| Variable | Meaning |
| --- | --- |
| `IDS_MODEL_VERSION` | Runtime model version. Only `v14` is enabled; `v15` is training-only until scoring parity is implemented. |
| `IDS_MODEL_PATH` | Path to model `.pth` |
| `IDS_PIPELINE_PATH` | Path to pipeline `.pkl` |
| `IDS_DATA_DIR` | Dataset directory |
| `IDS_SAMPLE_DATA_PATH` | Sample CSV path for dashboard |
| `IDS_ALERT_DB_PATH` | SQLite alert history path, default `results/alerts.sqlite3` |
| `IDS_DASHBOARD_MAX_CSV_BYTES` | Max uploaded CSV size for dashboard parsing, default `52428800` |
| `IDS_DASHBOARD_MAX_CSV_ROWS` | Max uploaded CSV rows for dashboard batch inference, default `100000` |
| `IDS_DASHBOARD_PREVIEW_MAX_ROWS` | Max raw CSV preview rows, default `1000` |
| `IDS_DASHBOARD_SESSION_RAW_MAX_ROWS` | Max raw CSV rows retained in Streamlit session state, default `100000` |
| `LLM_PROVIDER` | `none`, `groq`, `gemini`, `openai` or `anthropic` |

If model or pipeline artifacts are missing, the dashboard falls back to demo mode. Alert history is persisted locally in SQLite; database files are ignored by git.

## Run Inference API

The FastAPI server is inference-only. It loads the saved v14 checkpoint and preprocessing pipeline at startup from environment variables:

```powershell
$env:IDS_MODEL_PATH="checkpoints/ids_v14_model.pth"
$env:IDS_PIPELINE_PATH="checkpoints/ids_v14_pipeline.pkl"
uvicorn src.serve:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

Prediction request. Replace `61` with the feature count in your checkpoint if you retrain with a different feature set:

```powershell
$body = @{ features = @(0..60 | ForEach-Object { 0.0 }) } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://localhost:8080/predict" -Method Post -ContentType "application/json" -Body $body
```

The `features` array must match the checkpoint feature count. The server applies the saved `RobustScaler` from `ids_v14_pipeline.pkl`, then returns:

```json
{
  "label": "Normal",
  "confidence": 0.99,
  "ae_re": 0.01,
  "hybrid_score": 0.02,
  "is_anomaly": false,
  "uncertainty": {
    "entropy": 0.1,
    "std_max_class": 0.01
  }
}
```

Uncertainty is estimated with Monte Carlo Dropout over 30 stochastic forward passes. If `uncertainty.entropy > 1.5`, the API response uses `"label": "LOW_CONFIDENCE"` so clients can route the alert for analyst review.

For production-style integrations, use `/predict/flow` with a single firewall, NetFlow, Zeek or Suricata-like event. The API maps common field names into the saved feature contract before scoring:

```powershell
$event = @{
  event = @{
    src_ip = "10.0.0.5"
    dst_ip = "172.16.1.10"
    src_port = 52512
    dst_port = 443
    protocol = "tcp"
    service = "https"
    duration = 1.25
    bytes = 2048
    packets = 12
  }
} | ConvertTo-Json -Depth 3 -Compress
Invoke-RestMethod -Uri "http://localhost:8080/predict/flow" -Method Post -ContentType "application/json" -Body $event
```

The response includes the IDS verdict plus normalization quality so downstream SOC tooling can decide whether to auto-ticket or send the event to manual review:

```json
{
  "label": "Zero-Day Candidate",
  "classifier_class": "Normal",
  "confidence": 0.42,
  "ae_re": 0.73,
  "hybrid_score": 0.86,
  "is_anomaly": true,
  "zero_day_rule": "vote_2_of_3",
  "risk": 88,
  "normalization": {
    "schema": "firewall_or_flow_csv",
    "quality": "MEDIUM",
    "feature_coverage": 0.72,
    "mapped_columns": {
      "srcip": "src_ip",
      "dstip": "dst_ip"
    },
    "warnings": [
      "Feature coverage is moderate; treat scores as triage signals."
    ]
  }
}
```

Docker:

```bash
docker build -t ids-v14-serve .
docker run --rm -p 8080:8080 ids-v14-serve
```

The image includes the tracked v14 demo artifacts by default. For deployment-specific artifacts, mount a trusted `checkpoints/` directory and override `IDS_MODEL_PATH` and `IDS_PIPELINE_PATH`.

## Train

Train v14:

```bash
python train.py --data_dir data/ --save_dir checkpoints/ --plot_dir plots/
```

The default v14 training config now gives extra sampler/loss weight to the weak
known classes observed in artifact evaluation (`Exploits` and
`Reconnaissance`) while reducing the previous DoS sampler bias. Override these
without code changes:

```bash
python train.py --class_loss_weights "Exploits=3.0,Reconnaissance=3.0" \
  --class_sampler_weights "Exploits=4.0,Reconnaissance=4.0" \
  --dos_sampler_weight 1.5
```

PowerShell launcher:

```powershell
.\scripts\train_v14.ps1
```

Quick export helper for dashboard artifacts:

```bash
python export_model.py --data_dir data --epochs 5 --patience 3
```

Train v15:

```bash
python src/ids_v15_unswnb15.py --config configs/config_default.yaml
```

PowerShell launcher:

```powershell
.\scripts\train_v15.ps1
```

Training outputs:

- model weights `.pth`
- pipeline `.pkl`
- plots in `plots/`
- metric summary in `results/`

### Kaggle Retraining

On Kaggle, upload this repository as a working notebook dataset or clone it into
`/kaggle/working`, then attach UNSW-NB15/CICIDS datasets as Kaggle inputs:

```bash
cd /kaggle/working/ZeroDay-Detection-AutoEncoder-IDS
python train.py \
  --data_dir /kaggle/input/datasets/mrwellsdavid/unsw-nb15 \
  --save_dir /kaggle/working/checkpoints \
  --plot_dir /kaggle/working/plots \
  --target_fpr 0.01
```

After training, download the generated `.pth`, `.pkl`, plots and result JSONs,
then point the dashboard at those artifacts with `IDS_MODEL_PATH` and
`IDS_PIPELINE_PATH`.

## Verify

Check local environment and artifact availability:

```powershell
python scripts/check_environment.py
```

Create or verify local artifact hashes:

```powershell
python scripts/artifact_manifest.py
python scripts/artifact_manifest.py --verify
```

Run the full smoke check:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
python scripts/smoke_check.py
```

Run lint only:

```powershell
python -m ruff check .
```

Run dependency audit locally when `pip-audit` is available:

```powershell
python -m pip install --progress-bar off pip-audit
python -m pip_audit -r requirements.txt
python -m pip_audit -r requirements-smoke.txt
python -m pip_audit -r requirements-dev.txt
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `IDS_MODEL_PATH must be set` | Set `IDS_MODEL_PATH` and `IDS_PIPELINE_PATH` before starting `uvicorn src.serve:app`. |
| `expected N features, received M` | Send exactly the feature count saved in the checkpoint/pipeline; current local v14 artifacts use 66 features. |
| Dashboard starts in demo mode | Confirm `checkpoints/ids_v14_model.pth` and `checkpoints/ids_v14_pipeline.pkl` exist or set explicit artifact paths. |
| Too many zero-day candidates on a new CSV | Run `scripts/evaluate_csv.py --calibrate-thresholds` on labeled benign rows and load the generated threshold profile. |
| Import or console encoding errors on Windows paths | Set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`; `scripts/smoke_check.py` already does this for subprocesses. |
| Docker container cannot find artifacts | Mount `checkpoints/` into `/app/checkpoints` or build a deployment-specific image that includes trusted artifacts. |

## Regenerate v14 Evaluation Report

Evaluate the current saved v14 artifacts without retraining:

```powershell
python scripts/regenerate_v14_report.py --csv-path data\UNSW_NB15_training-set.csv --label-col attack_cat
```

This refreshes `results/ids_v14_results.json` and writes evaluation plots in `plots/`:

- `v14_eval_verdict_distribution.png`
- `v14_eval_score_distribution.png`
- `v14_eval_known_class_recall.png`

The report separates Normal-row false positives into `normal_ood_fpr` (flagged as OOD), `known_classifier_fpr` (classified as a known attack) and `normal_alert_fpr` (either path creates an alert). It also includes detection accuracy, known-class recall, OOD detection rate and the active threshold profile. The current Kaggle-trained v14 artifacts use 66 features. This command evaluates the current operational checkpoint; its metrics are historical/demo-only and are not a corrected Step 2 benchmark.

If `checkpoints/local_thresholds.json` exists, the report also includes a calibrated-threshold what-if section. Keep this file local and regenerate it for the active model; stale threshold profiles from older artifacts can override the saved thresholds in the dashboard.

### Historical Step 2 results — invalidated

The existing seed 42/43/44 reports and their combined report are **HISTORICAL / NON-INDEPENDENT / NOT VALID FOR SCIENTIFIC CLAIMS**. They were generated before the Phase 2 P0 correctness fixes. In particular, `__feature_fingerprint` entered the model feature matrix, feature selection varied by seed, and LOFO surrogate pools removed the target label without purging rows that shared target-family fingerprints.

The invalidated reports contain **68.84% ± 9.91% macro recall** and **84.38% ± 0.34% macro AUROC** across seeds. These numbers are preserved for audit history only and must not be quoted as current performance. Corrected results must be regenerated with the Phase 2 pipeline; the repository does not substitute or fabricate replacement values.

The operational values `91.52%` detection accuracy, `36.05%` OOD detection rate and `3.98%` Normal OOD FPR are also **HISTORICAL / NON-INDEPENDENT / NOT VALID FOR SCIENTIFIC CLAIMS**. The operational checkpoint remains available only for runtime compatibility and demonstrations.

### Operational checkpoint recommendation and known gap

The recommended path is to rerun seeds 42, 43 and 44 with the corrected Step 2 protocol: explicit train/calibration/test roles, a seed-invariant metadata-free schema, fingerprint-disjoint fitting partitions, target-fingerprint purge, and calibration-data-only thresholds. Promote nothing to the dashboard/API until the regenerated reports and runtime scoring parity pass review.

Until that retraining and review are completed, the dashboard/API may continue using the old operational checkpoint for compatibility and demonstrations, but must not present its values—or the invalidated first-pass Step 2 values—as current research benchmarks.

### Phase 3 target-isolated LOFO gate

Target-specific canonical purging and the 5-target × 3-seed data-only audit are
implemented in `scripts/audit_target_isolated_lofo.py`. The current UNSW-NB15
dry run is **BLOCKED** before neural training: known-role model-input vectors
still collide after preprocessing (primarily from `[-10, 10]` clipping), and
Backdoors also has target-equivalent vectors caused by unknown-category
encoding. See `results/data_quality/target_isolated_lofo_dry_run.json` and
`results/data_quality/label_collision_report.json`; do not bypass the gate or
start corrected training until the transform policy is explicitly resolved.

The saved operational pipeline was produced with scikit-learn 1.6.1. A 2,048-row A/B check found byte-identical `RobustScaler.transform()` output under 1.6.1 and 1.9.1, but reproducible deployment should still load artifacts with their pinned training dependencies.

### P1 collision-safe preprocessing audit

The P1 target-isolated audit path removes `[-10, 10]` hard clipping and uses
fold-local RobustScaler followed by finite float32 model input. Unknown
categorical tokens receive deterministic, feature-namespaced SHA-256 fallback
codes outside the fitted known-category code domain; the vocabulary still comes
only from the target-purged backbone. The canonical identity contract remains
`canonical-model-features-v1`; P1's model-input contract is
`post-robustscale-float32-sha256fallback-v2`. Saved v1 checkpoint inference and
the historical Phase 3 audit retain their original clipping behavior.

The P1 5 × 3 dry-run remains **BLOCKED**, with no neural training. Target-vs-fit
model-input collisions are zero for all 15 cells. Encoding and float64 scaling
introduce no new full-vector merges; the final float32 conversion introduces
merges across statistically separate known roles. Canonical identities shared
by known and surrogate OOD families with different labels also remain, as do
same-role cross-label identities. No
colliding model-input rows were removed. Full counts and gate reasons are in
`results/data_quality/p1/target_isolated_lofo_dry_run.json`; stage and feature
provenance is in `results/data_quality/p1/collision_provenance.json`.

To reproduce the data-only evidence, run these commands individually; the
first exits with status 2 while the gate is blocked:

```bash
python scripts/audit_target_isolated_lofo.py
python scripts/explain_p1_collisions.py
python scripts/recheck_backdoors_phase3.py
```

All archived Backdoors target collisions are reproduced from saved Phase 3
backbone IDs and shared fingerprints, then traced through both pipelines in
`results/data_quality/p1/backdoors_archived_recheck_summary.json`. Phase 3 did
not save held-out row IDs, so the trace records all matching held-out candidates
without claiming a unique assignment. The option to exclude entire ambiguous
canonical groups is reported but inactive; activating it would change the test
population and requires a declared evaluation protocol. Scalar category codes
retain arbitrary ordinal geometry, so the 61-feature P1 representation is an
audit candidate. A revised categorical schema is required before scientific
retraining; the audit alone does not validate distances between category codes.
The old operational artifacts remain demonstration-only.

### P2 representation and ambiguity protocol dry-run

P2 preserves the historical P0/P1 contracts and remains **BLOCKED** for
scientific retraining. `scripts/p2_float32_forensics.py` reconstructs every
float64 → float32 merge from the P1 evidence. `scripts/audit_p2_protocol.py`
compares P1 float32, P1 float64, a new 58-continuous-plus-3-opaque-category
schema, and the same new schema with candidate whole-ambiguous-group exclusion
across all five target families and three seeds. Its nonzero exit status means
at least one dry-run integrity gate fails; it is not a training entry point.

The P2 contract, exact support losses, collision reasons, and approval caveats
are documented in [`docs/p2_protocol.md`](docs/p2_protocol.md). In particular,
using combined train/test labels to curate ambiguous groups changes the
training population, even though scaler/vocabulary fitting remains train-only.
The proposed category IDs are identity-audit inputs, not an implemented
embedding architecture. Neither float64 nor group exclusion alone approves
neural training or scientific performance claims.

### P3 test-blind scientific data protocol

P3 constructs and freezes all 15 development cells from official train only,
then opens official test for a non-mutating external audit. Train-only
whole-group ambiguity exclusion, canonical-group splitting, target isolation,
fold-local vocabulary/scaler fitting, and structured float64 input auditing
pass in all target/seed cells. Historical train/test overlap is reported on a
diagnostic surface rather than used to alter development after test inspection.

The data protocol is **approved for the model-design phase only**. No training,
threshold tuning, checkpoint migration, or performance metric was run. Schema
V2 requires a new structured model interface before training can be considered.
Exact counts, the freeze hash, clean/diagnostic test surfaces, rare-family
support, and reproduction commands are in
[`docs/p3_protocol.md`](docs/p3_protocol.md).

### P4 structured model-interface design

P4 binds a design-only structured PyTorch interface to each frozen P3
manifest: 58 float64 continuous values, plus separate proto/service/state
integer-index tensors and independent embedding tables. Each feature reserves
PAD index 0 and a dedicated OOV index 1; unseen tokens never expand a frozen
vocabulary or modulo-hash into learned-token buckets. The baseline exposes a
representation only and rejects legacy flat 61-value tensors.

All 15 manifest-bound interface gates pass. No optimizer, checkpoint,
threshold, model metric, or training run is created by P4. See
[`docs/p4_model_design.md`](docs/p4_model_design.md) and
`results/model_design/p4/p4_matrix.json` for the architecture decision,
provenance hashes, compatibility inventory, and small-support evaluation plan.

### P5 development-only training protocol

P5 freezes the first schema-V2 training protocol without using official test.
All 15 target/seed cells pass dry validation. The only authorized initial
training is a controlled `Fuzzers/42/10042` smoke pair with a temporary
known-family classification head; validation selects checkpoints. It does not
measure zero-day detection effectiveness or authorize the full training
matrix. The scorer candidates and known-only calibration rule are declared
but not yet fitted or selected. See
[`docs/p5_training_protocol.md`](docs/p5_training_protocol.md) and
`results/training_protocol/p5/p5_matrix.json` for role permissions, hashes,
run manifests, and the reproducibility gate. The new entry point is
`scripts/train_p5_development.py`; training requires `--enable-training`.

### P6 development-only scorer selection and calibration

P6 compares only the two P5-frozen scorers: softmax uncertainty and
train-centroid representation distance. Selection uses canonical-identity mean
scores and surrogate-development AUROC on `meta_known` versus the fixed
`surrogate_ood` role. Each target/seed cell selects independently; only known
`calibration` rows fit the fixed 0.99-quantile threshold after scorer freeze.
P6 never opens target-held-out or official-test data and does not claim
zero-day effectiveness. The staged CLI is `scripts/run_p6_selection.py`;
details and gates are in
[`docs/p6_scorer_selection.md`](docs/p6_scorer_selection.md) and
`results/model_selection/p6/p6_matrix.json`.

### P7 external-evaluation preregistration

P7 Stage A freezes the scientific question, CLEAN canonical-identity
population, metrics, bootstrap intervals, threshold semantics, small-N
language, diagnostic-surface policy, output schema, provenance and a one-way
test gate. It verifies P3–P6 and synthetic metric tests, then emits
`P7_PREREGISTERED` and a protocol SHA-256. **No official-test evaluation has
run.** Stage B requires a separate explicit authorization plus both the exact
`--protocol-hash` and `--enable-official-test` flags; do not run it as part of
preregistration. See [`docs/p7_external_protocol.md`](docs/p7_external_protocol.md)
and `results/external_evaluation/p7/preregistration/`.

## Evaluate and Calibrate CSV Drift

Use the CSV evaluator before retraining when a new dataset produces too many
OOD/zero-day candidate hypotheses. It runs the saved model/pipeline outside Streamlit, writes a
score distribution report and can calibrate a local threshold profile.

```powershell
python scripts/evaluate_csv.py "path\to\Tuesday-WorkingHours.pcap_ISCX.csv" --scores-csv
```

If a dedicated calibration/validation CSV has a benign/attack label column,
calibrate thresholds from its benign rows at the requested false-positive rate:

```powershell
python scripts/evaluate_csv.py "data\calibration\representative-benign-validation.csv" `
  --label-col Label `
  --calibrate-thresholds `
  --target-fpr 0.01 `
  --threshold-output checkpoints\local_thresholds.json
```

The dashboard automatically loads `checkpoints/local_thresholds.json` when it is
present. Override with `IDS_THRESHOLD_PROFILE` if you want to test another
profile. Local profiles use vote-based OOD candidate decisions by default, so a row
must cross multiple calibrated signals instead of only the hybrid score.

Calibration and final reporting must use different files. Never calibrate on the
final test set used for published metrics. When `--calibrate-thresholds` is used,
`evaluate_csv.py` emits a threshold-leakage warning and records
`independent_evaluation: false` because its calibration profile and descriptive
report come from the same CSV:

```powershell
python scripts/evaluate_csv.py data\calibration\UNSW_NB15_calibration.csv --label-col attack_cat --calibrate-thresholds --target-fpr 0.05
python scripts/regenerate_v14_report.py --csv-path data\UNSW_NB15_testing-set.csv --label-col attack_cat
```

The dashboard loads `checkpoints/local_thresholds.json` automatically. Keep this file local unless you intentionally publish a calibrated deployment profile.

## Practical Readiness Automation

Use the real-world roadmap in [docs/roadmap_real_world.md](docs/roadmap_real_world.md) to move beyond UNSW-NB15-only evaluation. Several steps are automated and can be run as soon as you have a CSV export.

Compare the saved IDS scores against classical anomaly-detection baselines:

```powershell
python scripts/evaluate_baselines.py data\UNSW_NB15_testing-set.csv `
  --label-col attack_cat `
  --target-fpr 0.05 `
  --scores-csv
```

This writes a baseline report under `results/baseline_eval/` with IDS hybrid, AE-only, softmax-confidence, Isolation Forest, Local Outlier Factor and One-Class SVM comparisons.

Check score and normalization drift before trusting a new environment or dataset:

```powershell
python scripts/drift_report.py "path\to\new_environment.csv" `
  --label-col Label `
  --scores-csv
```

This writes a drift report under `results/drift/` and recommends recalibration when feature coverage, score distributions or zero-day candidate rates shift materially from the saved v14 reference report.

## Prepare Production Flow Data

For a more realistic deployment workflow, collect traffic with CICFlowMeter and
export CSV files with columns such as `Flow ID`, `Source IP`, `Destination IP`,
`Source Port`, `Destination Port`, `Protocol`, `Timestamp`, `Flow Duration`,
packet counters, byte counters and `Label`.

Normalize those files into the fixed production schema and split them
chronologically:

```powershell
python scripts/prepare_production_flow_data.py data\Monday-WorkingHours.csv `
  data\Tuesday-WorkingHours.csv `
  --source cicflowmeter `
  --output-dir results\production_flow_data
```

The output directory contains:

- `production_flows.csv`: all normalized rows with stable operational columns.
- `train.csv`, `validation.csv`, `test.csv`: time-ordered splits.
- `manifest.json`: schema columns, source reports, time range and label counts.

The production schema includes analyst workflow labels:

- `normal`
- `known_attack`
- `suspicious`
- `false_positive`
- `unknown`

Dataset labels such as `BENIGN`, `DoS Hulk` or `PortScan` are mapped to initial
`analyst_label` values. To label only reviewed rows, provide an override CSV:

```csv
flow_id,analyst_label,attack_category
10.0.0.8-172.16.1.20-44444-80-6,suspicious,Needs review
```

Then run:

```powershell
python scripts/prepare_production_flow_data.py data\Monday-WorkingHours.csv `
  --label-overrides labels\reviewed_flows.csv `
  --label-key flow_id `
  --output-dir results\production_flow_data
```

Committed samples for parser and smoke tests are in `data/samples/`.

Run the full local quality gate:

```powershell
.\scripts\quality_check.ps1
```

Or run commands separately:

```bash
python -m ruff check .
python -m compileall src dashboard scripts export_model.py patch_checkpoint.py tests
python -m unittest discover -s tests
```

For demo and runtime procedures, see [docs/operations.md](docs/operations.md).

## Limitations

- MITRE mapping is heuristic and should be treated as triage support.
- `Zero-Day Candidate` means the row crossed OOD/anomaly rules and needs analyst review. It is not proof of a novel attack, attribution or compromise.
- Zero-day results depend strongly on feature quality, scaler compatibility and threshold calibration. Recalibrate thresholds on representative benign traffic before operational demos that claim realistic SOC precision.
- Real-world CSV normalization is approximate when directional counters or timing fields are missing.
- LLM triage is optional decision support and must not be treated as the detection engine or a final verdict.
- v15 is training-only and runtime-disabled until its KNN/OOD-ensemble scoring has offline/runtime parity tests.
- The dashboard is not hardened for production deployment: no authentication, no realtime packet capture and no deployment security controls.

## License

See [LICENSE](LICENSE).
