# Operations Guide

This project is suitable for research demonstrations and SOC triage prototypes. It is not a hardened production IDS.

## Pre-Demo Checklist

Run these commands before a demo:

```powershell
.\scripts\quality_check.ps1
```

Expected minimum state:

- readiness status is `READY` or `WARN` with only expected v15 artifact warnings
- v14 model and pipeline are present
- smoke checks pass
- artifact manifest verifies cleanly

## Application Use

Start the FastAPI backend from the repository root:

```bash
export IDS_MODEL_PATH=checkpoints/ids_v14_model.pth
export IDS_PIPELINE_PATH=checkpoints/ids_v14_pipeline.pkl
uvicorn ids.api:app --app-dir src --host 127.0.0.1 --port 8080
```

Start the React frontend in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>. The Streamlit interface below is retained for
older research demonstrations.

## Legacy Streamlit Interface

Start v14 dashboard:

```powershell
$env:IDS_MODEL_VERSION="v14"
streamlit run dashboard/app.py
```

Or use the PowerShell launcher:

```powershell
.\scripts\run_dashboard.ps1
```

FastAPI Docker option:

```powershell
docker build -t ids-v14-serve .
docker run --rm -p 8080:8080 `
  -v ${PWD}/checkpoints:/app/checkpoints:ro `
  -e IDS_MODEL_PATH=/app/checkpoints/ids_v14_model.pth `
  -e IDS_PIPELINE_PATH=/app/checkpoints/ids_v14_pipeline.pkl `
  ids-v14-serve
```

The Dockerfile runs `uvicorn ids.api:app --app-dir src` on port `8080`. Run the
Streamlit dashboard directly with `streamlit run dashboard/app.py` only when
using its legacy analyst workflow.

Upload CSV files with flow-like fields when possible:

- source/destination IP
- source/destination port
- protocol/service/state
- duration or timestamps
- directional bytes and packets

The dashboard rejects empty, oversized or malformed CSVs and warns when feature coverage is low.

## Artifact Handling

Checkpoint and pipeline files are local runtime artifacts and are not committed. For reproducible demos, keep artifacts in one of these places:

- local `checkpoints/` plus a checked manifest
- GitHub release assets
- Hugging Face, Google Drive or an internal model registry

After training or replacing artifacts, regenerate the manifest:

```powershell
python scripts/artifact_manifest.py
```

Before a demo, verify the manifest:

```powershell
python scripts/artifact_manifest.py --verify
```

## LLM Handling

LLM triage is optional. Copy `.env.example` to `.env` and fill only the provider key you intend to use. Never commit `.env`.

If no provider key is configured, the dashboard still runs and returns deterministic fallback triage text.

## Production Gaps

Before production use, add:

- authentication and authorization for the dashboard
- authentication and rate limiting for the FastAPI server
- signed/trusted artifact storage
- monitored deployment and audit logging
- calibrated metrics from the current training code
- real-time ingestion integration
- analyst feedback loop and false-positive review process
