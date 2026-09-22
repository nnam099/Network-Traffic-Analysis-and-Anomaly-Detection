# Project Audit

Updated: 2026-05-29

## Verified

- `scripts/smoke_check.py` now runs with UTF-8 subprocess settings, so it works on Windows paths containing Vietnamese characters.
- `scripts/check_environment.py` now configures UTF-8 console output before printing JSON, so direct execution works from Windows paths containing Vietnamese characters.
- Python compile passes for `src/`, `dashboard/`, `export_model.py`, `patch_checkpoint.py` and `tests/`.
- Smoke tests pass locally, with the v15 artifact smoke test skipped until v15 artifacts exist.
- Checkpoint `ids_v14_model.pth` loads into `IDSModel` v14 without missing or unexpected weights.
- Pipeline v14 has 66 features and matches `n_features` in the checkpoint.
- `log_normalizer.py` maps common firewall/flow CSV columns into an UNSW-like flow schema.
- Dashboard validates the checkpoint/pipeline contract before inference.
- Pure inference verdict/risk/CSV-quality helpers are split into `src/inference_runtime.py` and covered by smoke tests.
- Dashboard batch inference is split into `src/inference_runtime.py` and covered by a dashboard contract smoke test.
- Dashboard preprocessing, batch alert context construction, AI context option selection and LLM fallback handling are split into `src/dashboard_runtime.py` and covered by smoke tests.
- Alert history, status and analyst notes can persist across Streamlit reloads through `src/alert_store.py` backed by local SQLite.
- The alert queue supports status/severity/OOD/search filters, and batch uploads can persist the top N highest-risk rows.
- Persisted batch alerts retain source/destination/service entities when available and the dashboard surfaces lightweight correlation groups.
- The dashboard now surfaces 15-minute incident windows for repeated correlated signals across source IPs, destination/service context, classifier classes and zero-day families.
- Dashboard UI rendering has started moving into smaller modules: queue view, alert-analysis safety, batch upload summaries, Ask AI helpers, setup status and shared safety/report copy.
- `scripts/regenerate_v14_report.py` regenerates artifact evaluation metrics and plots from current saved v14 artifacts without retraining.
- `results/ids_v14_results.json` now includes detection accuracy, known-class recall, OOD detection rate, normal false-positive rate and threshold profile for `UNSW_NB15_testing-set.csv`.
- The old Kaggle-trained v14 values (91.52% detection accuracy, 3.98% normal false-positive rate and 33.27% OOD detection rate) are retained only as historical/demo records. They are non-independent and **not valid for scientific claims**.
- v14 training now supports class-specific sampler/loss weight overrides, with defaults focused on the currently weak `Exploits` and `Reconnaissance` classes.
- v14 training now reduces DoS focal over-weighting and explicitly penalizes Recon/DoS cross-confusion in focal and contrastive objectives.
- v14 hybrid anomaly scoring now fits a validation meta-learner from `ae_re` and classifier uncertainty, and the weak energy OOD comparator is removed from v14 reports.
- Label-assisted adaptive AE thresholding remains demo-only and is disabled in the scientific benchmark.
- v14 training code is split into `src/ids/` modules with a `train.py` entry point while retaining the legacy `ids_v14_unswnb15.py` wrapper.
- `llm_agent.py` no longer initializes the provider client on import; it initializes lazily when LLM output is requested.
- `patch_checkpoint.py` now validates checkpoint structure, accepts a path argument and creates a backup by default.
- `artifact_validator.py` rejects duplicate/empty feature names and invalid threshold metadata while accepting vote-mode threshold controls such as `decision_mode` and `min_votes`.
- Dashboard dataframe rendering now uses `use_container_width=True` for compatibility with the pinned Streamlit runtime.
- Dashboard batch-result preview handles very small CSV batches without creating an invalid slider range.
- `export_model.py` now exposes a CLI instead of hard-coding `data/quick_train`.
- PowerShell train launchers are available for v14 and v15.
- CI uses `requirements-smoke.txt` so optional dashboard/explainability packages do not slow down core smoke checks.
- `scripts/check_environment.py` reports package/artifact/data readiness without exposing secret values and is covered for UTF-8 console output.
- `input_guard.py` validates uploaded CSV size/shape before dashboard batch inference.
- `.env.example` documents optional provider/runtime environment variables without storing secrets.
- `scripts/artifact_manifest.py` can create/verify SHA-256 manifests for local model artifacts.
- A Windows GitHub Actions smoke workflow is available in `.github/workflows/smoke.yml`.
- Runtime dependencies are pinned in `requirements.txt`; pytest dependencies are pinned in `requirements-dev.txt`.
- A FastAPI inference server is available at `src/serve.py` and is covered by API smoke tests.
- Served predictions include Monte Carlo Dropout uncertainty and label high-entropy predictions as `LOW_CONFIDENCE`.

## Strengths

- Training, inference support modules and dashboard are separated at the folder level.
- Artifact v14 stores the metadata needed for inference: scaler, label encoder, feature list and thresholds.
- Dashboard has demo fallback when artifacts are missing and supports common real-world CSV uploads through the normalizer.
- FastAPI serving reuses saved v14 artifacts and the same persisted `RobustScaler` contract as batch/dashboard inference.
- Dashboard alert queue can now survive reloads through a local SQLite store.
- Time-window incident grouping helps analysts distinguish repeated bursts from isolated alerts in the persisted queue.
- MITRE mapping is packaged separately and is easy to extend with more techniques/evidence rules.
- Smoke tests cover the highest-risk runtime contracts: artifact compatibility, threshold metadata validation, artifact hash drift detection, duplicate feature metadata rejection, environment readiness reporting, export config handling, checkpoint metadata patching, SQLite alert persistence, CSV input guardrails, normalizer behavior, dashboard preprocessing/context helpers, dashboard UI helpers, AI context selection, alert queue filtering, top-N batch alert selection, alert entity enrichment, lightweight correlation and time-window incident grouping, labeled evaluation reporting, API edge cases, Recon/DoS prototype separation, adaptive threshold windowing, CSV quality classification, MITRE mapping and LLM import/fallback behavior.

## Current Risks

- `dashboard/app.py` is still a large orchestration file. Major queue rendering and several UI helper areas have been split out, but single-alert investigation and OOD log detail can still be modularized further.
- Some code comments/docstrings still contain mojibake or ASCII-only Vietnamese text from earlier encoding issues.
- v14 artifact evaluation has been regenerated from Kaggle-trained artifacts, but those values remain historical/demo-only and non-independent. Threshold choice must still be tied to analyst capacity and demo goals, especially before applying the model to non-UNSW traffic.
- v15 is experimental. The dashboard can select it, but stable v15 use requires separately trained/exported artifacts. The v15 smoke test is present and skipped until artifacts are available.
- LLM provider packages remain optional and are not installed unless the selected provider is needed.
- Pretrained artifacts are not yet published as a release/model-registry asset; users must train locally or receive artifacts out of band.
- Phase 3 target-isolated LOFO dry-run (15 target/seed cells) is blocked before neural training because post-transform model-input collisions remain; see `results/data_quality/target_isolated_lofo_dry_run.json` and `results/data_quality/label_collision_report.json`.
- P1 audit removes hard clipping, distinguishes unknown categories with deterministic feature-namespaced SHA-256 codes, and keeps canonical purge unchanged. Its 15-cell dry-run is still blocked: float32 conversion merges distinct known-role vectors, while canonical known/surrogate and same-role identities retain conflicting labels. Archived Backdoors target collisions are independently reconstructed from saved Phase 3 evidence. See `results/data_quality/p1/target_isolated_lofo_dry_run.json`, `collision_provenance.json`, and `backdoors_archived_recheck_summary.json`. No retraining was run. The scalar categorical coordinate retains arbitrary ordinal geometry; a revised categorical schema is required before scientific retraining.

## Next Priorities

1. Validate the Kaggle-trained v14 artifacts on an external flow dataset or real firewall/Zeek/Suricata export.
2. Calibrate a fresh local threshold profile only for the active artifact bundle and representative benign traffic.
3. Continue splitting `dashboard/app.py`, especially single-alert investigation and OOD log detail pages.
4. Publish trusted v14 demo artifacts as release assets or an external model artifact bundle.
5. Train/export v15 artifacts if v15 will be demonstrated, then enable the existing v15 artifact smoke test.
