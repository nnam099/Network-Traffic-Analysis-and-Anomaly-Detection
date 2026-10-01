# -*- coding: utf-8 -*-
"""Streamlit dashboard for IDS alert review and triage."""

import streamlit as st
import sys
import os
import importlib.util
import numpy as np
import pandas as pd
import json
import uuid
import pickle
import hashlib
from datetime import datetime

# ── Page config (phai dat truoc bat ky st.* nao) ─────────────────
st.set_page_config(
    page_title="Network IDS",
    page_icon="[SOC]",
    layout="wide"
)

st.markdown("""
<style>
    :root {
        --soc-bg: #0b0e11;
        --soc-bg-2: #11161b;
        --soc-panel: #161b21;
        --soc-panel-2: #1d242c;
        --soc-panel-3: #232b34;
        --soc-line: #303943;
        --soc-line-strong: #46515d;
        --soc-text: #e8edf2;
        --soc-muted: #9ba7b4;
        --soc-dim: #6f7b88;
        --soc-cyan: #39b7e8;
        --soc-orange: #ff9f1c;
        --soc-green: #42d392;
        --soc-yellow: #f6c343;
        --soc-red: #ff5f6d;
    }
    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
    }
    .stApp {
        background:
            linear-gradient(180deg, rgba(22, 27, 33, 0.95) 0%, rgba(11, 14, 17, 1) 36%),
            var(--soc-bg);
        color: var(--soc-text);
    }
    .block-container {
        padding: 1rem 1.5rem 2.2rem 1.5rem;
        max-width: 100%;
    }
    section[data-testid="stSidebar"] {
        background: #0f1318;
        border-right: 1px solid var(--soc-line);
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: var(--soc-muted);
    }
    h1, h2, h3 {
        color: var(--soc-text);
        letter-spacing: 0;
    }
    h3 {
        font-size: 1.02rem;
        margin-top: 1.1rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--soc-line);
    }
    hr {
        border-color: var(--soc-line);
    }
    div[data-testid="stMetric"] {
        background: var(--soc-panel);
        border: 1px solid var(--soc-line);
        border-left: 3px solid var(--soc-cyan);
        border-radius: 4px;
        padding: 13px 15px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--soc-muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
    }
    div[data-testid="stMetricValue"] {
        color: var(--soc-text);
        font-weight: 750;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--soc-line);
        border-radius: 4px;
        overflow: hidden;
        background: var(--soc-panel);
    }
    div[data-testid="stTabs"] button {
        color: var(--soc-muted);
        border-radius: 3px 3px 0 0;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--soc-text);
        border-bottom-color: var(--soc-orange);
    }
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 4px;
        border: 1px solid var(--soc-line-strong);
        background: var(--soc-panel-2);
        color: var(--soc-text);
        font-weight: 650;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--soc-cyan);
        color: #ffffff;
        background: #26313b;
    }
    .stButton > button[kind="primary"] {
        background: #1f5f7a;
        border-color: var(--soc-cyan);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea,
    input {
        background: #11161b !important;
        border-color: var(--soc-line) !important;
        border-radius: 4px !important;
        color: var(--soc-text) !important;
    }
    div[data-testid="stAlert"] {
        border-radius: 4px;
        border: 1px solid var(--soc-line);
        background: var(--soc-panel);
    }
    .soc-header {
        border: 1px solid var(--soc-line);
        border-left: 4px solid var(--soc-orange);
        background: linear-gradient(90deg, #171d23 0%, #10151a 100%);
        border-radius: 4px;
        padding: 18px 18px 16px 18px;
        margin-bottom: 16px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        overflow: visible;
    }
    .soc-title {
        font-size: 1.34rem;
        font-weight: 760;
        margin: 0;
    }
    .soc-subtitle {
        color: var(--soc-muted);
        margin-top: 4px;
        font-size: 0.88rem;
    }
    .soc-topline {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }
    .soc-kicker {
        display: block;
        color: var(--soc-orange);
        font-size: 0.70rem;
        line-height: 1.25;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 7px;
        white-space: nowrap;
    }
    .soc-header-actions {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 4px;
    }
    .soc-panel {
        border: 1px solid var(--soc-line);
        background: var(--soc-panel);
        border-radius: 4px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.025);
    }
    .soc-badge {
        display: inline-block;
        border: 1px solid var(--soc-line);
        border-radius: 3px;
        padding: 3px 8px;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--soc-text);
        background: var(--soc-panel-2);
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .soc-detail-title {
        color: var(--soc-muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 2px;
    }
    .soc-detail-value {
        color: var(--soc-text);
        font-size: 1.02rem;
        font-weight: 650;
        margin-bottom: 10px;
    }
    .soc-pill-red { border-color: rgba(255,95,109,0.55); color: #ff9aa4; background: rgba(255,95,109,0.10); }
    .soc-pill-blue { border-color: rgba(57,183,232,0.55); color: #8bdcff; background: rgba(57,183,232,0.10); }
    .soc-pill-green { border-color: rgba(66,211,146,0.55); color: #93efc5; background: rgba(66,211,146,0.10); }
    .soc-pill-orange { border-color: rgba(255,159,28,0.55); color: #ffc66f; background: rgba(255,159,28,0.10); }
    .sidebar-brand {
        border: 1px solid var(--soc-line);
        border-left: 4px solid var(--soc-orange);
        background: #151a20;
        border-radius: 4px;
        padding: 14px 12px 12px 12px;
        margin: 6px 0 14px 0;
    }
    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #f4f8fc;
    }
    .sidebar-brand-sub {
        color: var(--soc-muted);
        font-size: 0.74rem;
        margin-top: 4px;
    }
    .health-grid {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 8px 10px;
        align-items: center;
        border: 1px solid var(--soc-line);
        border-radius: 4px;
        padding: 12px;
        background: #151a20;
    }
    .health-label { color: var(--soc-muted); font-size: 0.75rem; text-transform: uppercase; }
    .health-ok, .health-warn, .health-bad {
        border-radius: 3px;
        padding: 2px 7px;
        font-size: 0.70rem;
        font-weight: 700;
    }
    .health-ok { color: #93efc5; background: rgba(66,211,146,0.10); border: 1px solid rgba(66,211,146,0.35); }
    .health-warn { color: #ffd77a; background: rgba(246,195,67,0.10); border: 1px solid rgba(246,195,67,0.35); }
    .health-bad { color: #ff9ca6; background: rgba(255,95,109,0.10); border: 1px solid rgba(255,95,109,0.35); }
    .sev-critical { color: var(--soc-red); font-weight: 700; }
    .sev-high { color: #ff9b55; font-weight: 700; }
    .sev-medium { color: var(--soc-yellow); font-weight: 700; }
    .sev-low { color: var(--soc-green); font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Them src/ vao path de import module ──────────────────────────
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC_DIR)

# ── Kiem tra thu vien can thiet ───────────────────────────────────
def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False

MISSING_REQUIRED = []
try:
    import torch
except ImportError:
    MISSING_REQUIRED.append("torch")

HAS_SHAP = _has_module("shap")
if not HAS_SHAP:
    st.warning("[!] Thieu shap — SHAP se bi tat")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("[!] Thieu python-dotenv — se bo qua load .env")

LLM_PROVIDER = None
try:
    from ids.llm_agent import LLM_PROVIDER as _LLM_PROVIDER
    LLM_PROVIDER = (_LLM_PROVIDER or "").lower()
except Exception:
    LLM_PROVIDER = None

PROVIDER_DEPS = {
    "groq": "groq",
    "gemini": "google.generativeai",
    "openai": "openai",
    "anthropic": "anthropic",
}
LLM_DEP = PROVIDER_DEPS.get(LLM_PROVIDER or "")
HAS_LLM_DEPS = True
if LLM_DEP and not _has_module(LLM_DEP):
    HAS_LLM_DEPS = False
    st.warning(f"[!] Thieu thu vien cho LLM provider '{LLM_PROVIDER}': {LLM_DEP}")

if MISSING_REQUIRED:
    st.error(f"Thieu thu vien bat buoc: {', '.join(MISSING_REQUIRED)}")
    st.code(f"pip install {' '.join(MISSING_REQUIRED)}")
    st.stop()

PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
LLM_KEY_ENV = PROVIDER_KEYS.get(LLM_PROVIDER or "")
LLM_KEY_OK = bool(os.getenv(LLM_KEY_ENV, "").strip()) if LLM_KEY_ENV else False

# ── Runtime modules ───────────────────────────────────────────────
HAS_EXPLAINER = False
if HAS_SHAP:
    try:
        from ids.explainer import SHAPExplainer
        HAS_EXPLAINER = True
    except ImportError:
        st.warning("[!] Khong tim thay explainer.py trong src/ - SHAP se bi tat")

try:
    from ids.mitre_mapper import MITREMapper
    HAS_MITRE = True
except ImportError:
    HAS_MITRE = False
    st.warning("[!] Khong tim thay mitre_mapper.py trong src/")

try:
    from ids.log_normalizer import normalize_real_world_logs
    HAS_LOG_NORMALIZER = True
except ImportError:
    HAS_LOG_NORMALIZER = False
    normalize_real_world_logs = None
    st.warning("[!] Khong tim thay log_normalizer.py trong src/ - CSV thuc te se khong duoc chuan hoa nang cao")

try:
    from ids.artifact_validator import validate_artifact_contract
    HAS_ARTIFACT_VALIDATOR = True
except ImportError:
    HAS_ARTIFACT_VALIDATOR = False
    validate_artifact_contract = None

try:
    from ids.input_guard import CSVInputPolicy, validate_uploaded_csv
    HAS_INPUT_GUARD = True
except ImportError:
    HAS_INPUT_GUARD = False
    CSVInputPolicy = None
    validate_uploaded_csv = None

from ids.inference_runtime import (
    assess_normalization_quality as runtime_assess_normalization_quality,
    ground_truth_verdict as runtime_ground_truth_verdict,
    hybrid_score_from_meta as runtime_hybrid_score_from_meta,
    risk_score as runtime_risk_score,
    run_batch_inference as runtime_run_batch_inference,
    severity_class as runtime_severity_class,
    severity_rank as runtime_severity_rank,
    traffic_verdict as runtime_traffic_verdict,
    zero_day_decision as runtime_zero_day_decision,
)
from ids.dashboard_runtime import (
    answer_analyst_question as runtime_answer_analyst_question,
    build_alert_context_from_log as runtime_build_alert_context_from_log,
    build_ai_context_options as runtime_build_ai_context_options,
    build_top_batch_alerts as runtime_build_top_batch_alerts,
    default_ai_context_index as runtime_default_ai_context_index,
    preprocess_dashboard_df as runtime_preprocess_dashboard_df,
    triage_alert_with_fallback as runtime_triage_alert_with_fallback,
)
from ui_safety import render_safety_notice
from views_ai import render_ai_context_card, render_question_suggestions
from views_analysis import render_analysis_safety_notice
from views_batch import render_batch_safety_notice, render_bulk_detection_summary, render_ground_truth_summary
from views_ood import build_feature_table, build_score_table, enrich_ood_row
from views_queue import queue_summary
from views_report import render_raw_report, render_report_download
from views_setup import render_setup_status
from ids.alert_store import (
    init_alert_store,
    list_alerts as alert_store_list_alerts,
    save_alert as alert_store_save_alert,
    update_alert_status as alert_store_update_alert_status,
)

HAS_LLM = False
if LLM_DEP and HAS_LLM_DEPS:
    try:
        from ids.llm_agent import SOCTriageAgent
        HAS_LLM = True
    except ImportError:
        st.warning("[!] Khong tim thay llm_agent.py trong src/")

# ── Runtime config ────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CKPT_DIR = os.getenv("IDS_CHECKPOINT_DIR", os.path.join(BASE_DIR, 'checkpoints'))
DATA_DIR = os.getenv("IDS_DATA_DIR", os.path.join(BASE_DIR, 'data'))
ALERT_DB_PATH = os.getenv("IDS_ALERT_DB_PATH", os.path.join(BASE_DIR, 'results', 'alerts.sqlite3'))
CSV_UPLOAD_MAX_BYTES = int(os.getenv("IDS_DASHBOARD_MAX_CSV_BYTES", str(50 * 1024 * 1024)))
CSV_UPLOAD_MAX_ROWS = int(os.getenv("IDS_DASHBOARD_MAX_CSV_ROWS", "100000"))
CSV_UPLOAD_MAX_COLUMNS = int(os.getenv("IDS_DASHBOARD_MAX_CSV_COLUMNS", "250"))
CSV_PREVIEW_MAX_ROWS = int(os.getenv("IDS_DASHBOARD_PREVIEW_MAX_ROWS", "1000"))
CSV_SESSION_RAW_MAX_ROWS = int(os.getenv("IDS_DASHBOARD_SESSION_RAW_MAX_ROWS", "100000"))

# v14 is the operational default because this repo currently ships v14 artifacts.
MODEL_VERSION = os.getenv("IDS_MODEL_VERSION", "v14").strip().lower()
if MODEL_VERSION == "v15":
    st.warning(
        "v15 disabled: runtime does not implement KNN/ood_ensemble scoring used at evaluation time. "
        "Fallback ve v14."
    )
    MODEL_VERSION = "v14"
elif MODEL_VERSION != "v14":
    st.warning(f"IDS_MODEL_VERSION khong hop le: {MODEL_VERSION}. Fallback ve v14.")
    MODEL_VERSION = "v14"

DEFAULT_MODEL = f"ids_{MODEL_VERSION}_model.pth"
DEFAULT_PIPE = f"ids_{MODEL_VERSION}_pipeline.pkl"
MODEL_PATH = os.getenv("IDS_MODEL_PATH", os.path.join(CKPT_DIR, DEFAULT_MODEL))
PIPE_PATH = os.getenv("IDS_PIPELINE_PATH", os.path.join(CKPT_DIR, DEFAULT_PIPE))
DATA_PATH = os.getenv("IDS_SAMPLE_DATA_PATH", os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv'))
THRESHOLD_PROFILE_PATH = os.getenv("IDS_THRESHOLD_PROFILE", os.path.join(CKPT_DIR, "local_thresholds.json"))

# CLASS_NAMES se duoc load tu label_encoder sau khi model load
# De tranh sai thu tu do LabelEncoder sort theo alphabet
CLASS_NAMES  = ["Normal", "DoS", "Exploits", "Reconnaissance", "Generic"]  # fallback
AE_THRESHOLD = 0.5
HYBRID_THRESHOLD = 0.5
PIPELINE_THRESHOLDS = None
PIPELINE_META = {}

def _load_threshold_profile(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    thresholds = profile.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("threshold profile missing thresholds dict")
    return profile

def _build_categorical_maps_from_sample() -> dict:
    if not os.path.exists(DATA_PATH):
        return {}
    try:
        cols = pd.read_csv(DATA_PATH, nrows=1).columns
        usecols = [c for c in ['proto', 'service', 'state'] if c in cols]
        if not usecols:
            return {}
        df = pd.read_csv(DATA_PATH, usecols=usecols)
        maps = {}
        for col in usecols:
            values = df[col].astype(str).fillna('unk')
            classes = sorted(values.unique().tolist())
            if 'unk' not in classes:
                classes.append('unk')
            maps[col] = {v: i for i, v in enumerate(classes)}
        return maps
    except Exception:
        return {}

# ── Sidebar navigation ────────────────────────────────────────────
# Keep the student-facing workflow focused on the two core IDS tasks.
# Advanced triage, OOD logs and setup diagnostics remain available in code
# for development, but are intentionally removed from the primary UI.
NAV_ITEMS = ["[1] Dashboard", "[2] Analyze Traffic"]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = NAV_ITEMS[0]

st.sidebar.markdown(
    f"""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">Network IDS</div>
        <div class="sidebar-brand-sub">IDS {MODEL_VERSION.upper()} | Traffic Analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Navigation", NAV_ITEMS, key="nav_page")

def _health(value, warn=False):
    if value:
        return '<span class="health-ok">OK</span>'
    if warn:
        return '<span class="health-warn">WARN</span>'
    return '<span class="health-bad">OFF</span>'

llm_ok = HAS_LLM and LLM_KEY_OK
st.sidebar.markdown("**System Health**")
st.sidebar.markdown(
    f"""
    <div class="health-grid">
        <div class="health-label">CSV Normalize</div><div>{_health(HAS_LOG_NORMALIZER)}</div>
        <div class="health-label">Model</div><div>{_health(os.path.exists(MODEL_PATH))}</div>
        <div class="health-label">Pipeline</div><div>{_health(os.path.exists(PIPE_PATH))}</div>
        <div class="health-label">Data</div><div>{_health(os.path.exists(DATA_PATH))}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption(f"Model version: {MODEL_VERSION.upper()}")

# ── Pipeline guard/autogen ─────────────────────────────────────────
def ensure_pipeline():
    if os.path.exists(PIPE_PATH):
        try:
            with open(PIPE_PATH, 'rb') as f:
                pipeline = pickle.load(f)
            if not pipeline.get('categorical_maps'):
                maps = _build_categorical_maps_from_sample()
                if maps:
                    pipeline['categorical_maps'] = maps
                    pipeline['categorical_maps_source'] = 'sample_data_fallback'
            return pipeline
        except Exception as e:
            st.warning(f"[!] Loi doc pipeline: {e}")
            return None

    if not os.path.exists(DATA_DIR):
        st.warning("[!] Khong tim thay data/ de tao pipeline")
        return None

    try:
        train_mod = __import__(f"ids_{MODEL_VERSION}_unswnb15", fromlist=["load_unsw_csvs"])
        load_unsw_csvs = train_mod.load_unsw_csvs
        clean_df = train_mod.clean_df
        prepare_splits = train_mod.prepare_splits
        with st.spinner("Dang tao pipeline tu data/ (chi 1 lan)..."):
            df = load_unsw_csvs(DATA_DIR)
            df = clean_df(df)
            splits = prepare_splits(df, seed=42)
        pipeline = {
            'scaler':        splits['scaler'],
            'label_encoder': splits['label_encoder'],
            'feat_cols':     splits['feat_cols'],
            'feature_names': splits.get('feature_names', splits['feat_cols']),
            'known_cats':    splits['known_cats'],
            'zd_cats':       splits['zd_cats'],
            'categorical_maps': splits.get('categorical_maps', {}),
            'version':       MODEL_VERSION,
            'n_features':    splits['n_features'],
            'n_classes':     splits['n_classes'],
        }
        with open(PIPE_PATH, 'wb') as f:
            pickle.dump(pipeline, f)
        st.success("[OK] Pipeline da duoc tao va luu")
        return pipeline
    except Exception as e:
        st.error(f"Loi tao pipeline: {e}")
        return None

# ── Load model (cache) ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    global PIPELINE_THRESHOLDS, PIPELINE_META
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None
    try:
        train_mod = __import__(f"ids_{MODEL_VERSION}_unswnb15", fromlist=["IDSModel"])
        IDSModel = train_mod.IDSModel

        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        n_feat     = checkpoint['n_features']
        n_cls      = checkpoint['n_classes']
        hidden     = checkpoint.get('hidden', 256)
        ae_hidden  = checkpoint.get('ae_hidden', 128)

        if MODEL_VERSION == "v15":
            latent_dim = checkpoint.get('latent_dim', 32)
            model = IDSModel(
                n_features=n_feat, n_classes=n_cls,
                hidden=hidden, ae_hidden=ae_hidden, latent_dim=latent_dim,
            )
        else:
            model = IDSModel(n_features=n_feat, n_classes=n_cls, hidden=hidden, ae_hidden=ae_hidden)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        pipeline = ensure_pipeline()
        if pipeline is None:
            st.warning("[!] Thieu pipeline — chuyen sang DEMO MODE")
            return None, None, None, None

        if HAS_ARTIFACT_VALIDATOR and validate_artifact_contract is not None:
            validation = validate_artifact_contract(checkpoint, pipeline)
            for warning in validation.warnings:
                st.warning(f"[Artifact] {warning}")
            validation.raise_for_errors()

        PIPELINE_THRESHOLDS = dict(pipeline.get('thresholds') or {})
        hybrid_meta = pipeline.get('hybrid_meta') or checkpoint.get('hybrid_meta') or PIPELINE_THRESHOLDS.get('hybrid_meta')
        if hybrid_meta:
            PIPELINE_THRESHOLDS['hybrid_meta'] = hybrid_meta
            pipeline['hybrid_meta'] = hybrid_meta
        PIPELINE_META = pipeline
        threshold_profile = _load_threshold_profile(THRESHOLD_PROFILE_PATH)
        if threshold_profile is not None:
            PIPELINE_THRESHOLDS.update(threshold_profile["thresholds"])
            PIPELINE_META["threshold_profile"] = threshold_profile
            st.info(
                "Da load threshold profile: "
                f"{os.path.basename(THRESHOLD_PROFILE_PATH)} "
                f"(target FPR={threshold_profile.get('target_fpr', 'N/A')})"
            )

        scaler        = pipeline['scaler']
        feature_names = pipeline.get('feature_names', pipeline.get('feat_cols', []))
        label_encoder = pipeline['label_encoder']

        return model, scaler, feature_names, label_encoder

    except Exception as e:
        st.error(f"Loi load model: {e}")
        return None, None, None, None

model, scaler, feature_names, label_encoder = load_model()

if PIPELINE_THRESHOLDS:
    AE_THRESHOLD = float(PIPELINE_THRESHOLDS.get('ae_re', AE_THRESHOLD))
    HYBRID_THRESHOLD = float(PIPELINE_THRESHOLDS.get('hybrid', HYBRID_THRESHOLD))

# Lay CLASS_NAMES dung thu tu tu label_encoder (tranh sai do sort alphabet)
if label_encoder is not None:
    CLASS_NAMES = list(label_encoder.classes_)

def _uploaded_file_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()

def _reset_bulk_results_for_new_file(file_hash: str) -> None:
    if st.session_state.get('bulk_source_file_hash') == file_hash:
        return
    for key in [
        'bulk_result_df',
        'bulk_raw_df',
        'bulk_source_file_hash',
        'last_log_normalization_report',
        'bulk_score_summary',
    ]:
        st.session_state.pop(key, None)
    st.session_state['current_upload_file_hash'] = file_hash

def _zero_day_decision(ae_score, max_prob, hybrid_score):
    return runtime_zero_day_decision(
        ae_score,
        max_prob,
        hybrid_score,
        thresholds=PIPELINE_THRESHOLDS,
        ae_threshold=AE_THRESHOLD,
        hybrid_threshold=HYBRID_THRESHOLD,
    )

def _hybrid_score(ae_score, max_prob):
    return runtime_hybrid_score_from_meta(ae_score, max_prob, thresholds=PIPELINE_THRESHOLDS)

def _traffic_verdict(is_zeroday, classifier_class) -> str:
    return runtime_traffic_verdict(is_zeroday, classifier_class)

def _ground_truth_verdict(label) -> str:
    return runtime_ground_truth_verdict(label)

def preprocess_raw_df(df_raw: pd.DataFrame, feat_cols: list) -> np.ndarray:
    """
    Ap dung dung cac buoc tien xu ly nhu trong prepare_splits().
    Output LUON CO DUNG SO LUONG FEATURES = len(feat_cols), thu tu khop voi scaler.
    """
    normalizer = None
    if HAS_LOG_NORMALIZER and normalize_real_world_logs is not None:
        def normalizer(frame):
            return normalize_real_world_logs(frame, expected_features=feat_cols)

    result = runtime_preprocess_dashboard_df(
        df_raw,
        feat_cols,
        MODEL_VERSION,
        pipeline_meta=PIPELINE_META,
        normalizer=normalizer,
    )
    if result.normalization_report is not None:
        st.session_state['last_log_normalization_report'] = result.normalization_report
    return result.features


@st.cache_resource
def load_background(_scaler, _feature_names):
    if not os.path.exists(DATA_PATH):
        return None
    try:
        df        = pd.read_csv(DATA_PATH)
        label_col = 'label' if 'label' in df.columns else df.columns[-1]
        normal    = df[df[label_col] == 0].sample(
            min(200, len(df[df[label_col] == 0])), random_state=42
        )
        bg_arr = preprocess_raw_df(normal, _feature_names or [])
        if bg_arr.shape[1] == 0:
            st.warning("[SHAP] Khong tim thay feature khop — SHAP se bi tat")
            return None
        return bg_arr
    except Exception as e:
        st.error(f"Loi load background data: {e}")
        return None



@st.cache_resource
def get_components(_model, _scaler, _feature_names, _bg):
    comps = {}
    if HAS_EXPLAINER and _model and _scaler and _bg is not None:
        try:
            comps['explainer'] = SHAPExplainer(_model, _scaler, _feature_names, _bg)
        except Exception as e:
            st.warning(f"SHAP init loi: {type(e).__name__}: {e}")
    if HAS_MITRE:
        comps['mapper'] = MITREMapper()
    if HAS_LLM:
        comps['agent'] = SOCTriageAgent()
    return comps

# ── Demo mode neu chua co model ───────────────────────────────────
DEMO_MODE = (model is None or scaler is None)

def mock_inference(n_features=49):
    """Tao du lieu gia lap de demo khi chua co model."""
    ae_score     = float(np.random.uniform(0.3, 0.95))
    max_prob     = float(np.random.uniform(0.4, 0.99))
    pred_idx     = int(np.random.randint(0, len(CLASS_NAMES)))
    classifier_class = CLASS_NAMES[pred_idx]
    hybrid_score = float(np.asarray(_hybrid_score(ae_score, max_prob)).reshape(-1)[0])
    is_zeroday   = ae_score > AE_THRESHOLD and max_prob < 0.6
    verdict       = _traffic_verdict(is_zeroday, classifier_class)

    # SHAP gia lap
    if feature_names:
        fnames = feature_names[:10]
    else:
        fnames = [f"feature_{i}" for i in range(10)]

    shap_summary_lines = []
    for fname in fnames:
        sv  = float(np.random.uniform(-0.3, 0.3))
        fv  = float(np.random.uniform(0, 100))
        dir_str = "tang nguy co" if sv > 0 else "giam nguy co"
        shap_summary_lines.append(f"  - {fname}: gia_tri={fv:.2f}, SHAP={sv:+.4f} ({dir_str})")

    shap_summary = "\n".join(shap_summary_lines)

    return {
        "alert_id"       : str(uuid.uuid4())[:8].upper(),
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hybrid_score"   : hybrid_score,
        "ae_score"       : ae_score,
        "max_prob"       : max_prob,
        "predicted_class": verdict,
        "classifier_class": classifier_class,
        "is_zeroday"     : is_zeroday,
        "shap_summary"   : shap_summary,
        "mitre_summary"  : "MITRE mapping: [T1595] Active Scanning (Reconnaissance)",
        "top_features"   : [(fnames[i], np.random.uniform(-0.3,0.3), np.random.uniform(0,100)) for i in range(min(5,len(fnames)))],
        "probs"          : np.random.dirichlet(np.ones(len(CLASS_NAMES))).tolist(),
        "demo_mode"      : True,
    }

def run_full_pipeline(raw_features: np.ndarray, comps: dict):
    """Chay toan bo pipeline that: IDS -> SHAP -> MITRE -> LLM."""
    # 1. Inference — IDSModel.forward() tra ve (logits, fv)
    scaled = scaler.transform(raw_features)
    tensor = torch.FloatTensor(scaled)
    with torch.no_grad():
        outputs = model(tensor)
        logits  = outputs[0] if isinstance(outputs, tuple) else outputs
        probs   = torch.softmax(logits, dim=1).numpy()[0]

    max_prob   = float(probs.max())
    pred_idx   = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Unknown"

    # AE score tu AutoEncoder
    with torch.no_grad():
        if hasattr(model, 'ae'):
            ae_score = float(model.ae.recon_error(tensor).mean().item())
        elif hasattr(model, 'vae'):
            ae_score = float(model.vae.recon_error(tensor).mean().item())
        elif hasattr(model, 'autoencoder'):
            recon    = model.autoencoder(tensor)
            ae_score = float(torch.mean((tensor - recon) ** 2).item())
        else:
            ae_score = float(1.0 - max_prob + np.random.uniform(0, 0.1))

    hybrid_score = float(np.asarray(_hybrid_score(ae_score, max_prob)).reshape(-1)[0])
    is_zeroday, decision_rule = _zero_day_decision(ae_score, max_prob, hybrid_score)
    is_zeroday = bool(is_zeroday)
    verdict = _traffic_verdict(is_zeroday, pred_class)

    result = {
        "alert_id"       : str(uuid.uuid4())[:8].upper(),
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hybrid_score"   : hybrid_score,
        "ae_score"       : ae_score,
        "max_prob"       : max_prob,
        "predicted_class": verdict,
        "classifier_class": pred_class,
        "is_zeroday"     : is_zeroday,
        "zero_day_rule"   : decision_rule,
        "probs"          : probs.tolist(),
        "demo_mode"      : False,
    }

    # 2. SHAP
    if 'explainer' in comps:
        try:
            shap_res = comps['explainer'].explain_alert(raw_features)
            result['shap_summary']  = shap_res['summary_text']
            result['top_features']  = shap_res['top_features']
            result['shap_values']   = shap_res['shap_values']
        except Exception as e:
            result['shap_summary'] = f"SHAP loi: {e}"
            result['top_features'] = []
    else:
        result['shap_summary'] = "SHAP chua duoc kich hoat"
        result['top_features'] = []

    # 3. MITRE
    if 'mapper' in comps:
        try:
            mapper = comps['mapper']
            if is_zeroday:
                mitre_res = mapper.map_zeroday(ae_score, result.get('top_features', []))
            else:
                mitre_res = mapper.map_known_attack(pred_idx, class_names=CLASS_NAMES, top_features=result.get('top_features', []))
            result['mitre_result']  = mitre_res
            result['mitre_summary'] = mapper.format_for_llm(mitre_res)
        except Exception as e:
            result['mitre_summary'] = f"MITRE loi: {e}"
            result['mitre_result']  = None
    else:
        result['mitre_summary'] = "MITRE chua duoc kich hoat"
        result['mitre_result']  = None

    return result


def run_batch_inference(raw_features: np.ndarray, batch_size: int = 512) -> pd.DataFrame:
    """Chay inference cho toan bo file CSV (khong SHAP/LLM)."""
    return runtime_run_batch_inference(
        model,
        scaler,
        raw_features,
        CLASS_NAMES,
        thresholds=PIPELINE_THRESHOLDS,
        ae_threshold=AE_THRESHOLD,
        hybrid_threshold=HYBRID_THRESHOLD,
        batch_size=batch_size,
    )

def severity_rank(severity: str) -> int:
    return runtime_severity_rank(severity)

def severity_class(severity: str) -> str:
    return runtime_severity_class(severity)

def risk_score(result: dict, severity: str | None = None) -> int:
    return runtime_risk_score(result, severity)

def load_alert_history(force: bool = False) -> list[dict]:
    if force or not st.session_state.get("alert_history_loaded"):
        try:
            init_alert_store(ALERT_DB_PATH)
            st.session_state["alert_history"] = alert_store_list_alerts(ALERT_DB_PATH, limit=200)
            st.session_state["alert_history_loaded"] = True
        except Exception as e:
            st.session_state["alert_history"] = st.session_state.get("alert_history", [])
            st.warning(f"Khong load duoc alert store: {e}")
    return st.session_state.get("alert_history", [])

def persist_alert(result: dict, llm: dict | None = None, source: str = "single") -> None:
    if result.get("demo_mode"):
        return
    result["llm_severity"] = (llm or {}).get("severity", result.get("llm_severity", "N/A"))
    result["risk"] = risk_score(result, result.get("llm_severity"))
    try:
        alert_store_save_alert(ALERT_DB_PATH, result, llm=llm, source=source)
        st.session_state["alert_history"] = alert_store_list_alerts(ALERT_DB_PATH, limit=200)
        st.session_state["alert_history_loaded"] = True
    except Exception as e:
        st.warning(f"Khong luu duoc alert vao SQLite store: {e}")
        history = st.session_state.setdefault("alert_history", [])
        if not any(item.get("alert_id") == result.get("alert_id") for item in history):
            history.append(result)

def update_persisted_alert_status(alert_id: str, status: str, analyst_note: str = "") -> None:
    try:
        alert_store_update_alert_status(ALERT_DB_PATH, alert_id, status, analyst_note)
        st.session_state["alert_history"] = alert_store_list_alerts(ALERT_DB_PATH, limit=200)
        st.session_state["alert_history_loaded"] = True
    except Exception as e:
        st.warning(f"Khong cap nhat duoc alert store: {e}")

def render_soc_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="soc-header">
            <div class="soc-kicker">SOC Workbench</div>
            <div class="soc-topline">
                <div>
                    <div class="soc-title">{title}</div>
                    <div class="soc-subtitle">{subtitle}</div>
                </div>
                <div class="soc-header-actions">
                    <span class="soc-badge soc-pill-orange">IDS {MODEL_VERSION.upper()}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_alert_context_from_log(row_scores: dict) -> dict:
    return runtime_build_alert_context_from_log(row_scores)

def get_llm_analysis(result: dict, comps: dict):
    """Goi LLM triage agent."""
    return runtime_triage_alert_with_fallback(result, comps.get('agent'))

# ── Display result ────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═════════════════════════════════════════════════════════════════
def display_result(result: dict, llm: dict):
    """SOC-style alert detail view. Overrides the earlier basic renderer."""
    sev = str(llm.get('severity', 'HIGH')).upper()
    risk = risk_score(result, sev)

    render_safety_notice(level="warning" if result.get('is_zeroday') else "info")

    if result.get('demo_mode'):
        st.info("DEMO SANDBOX - Du lieu gia lap, khong dung de ket luan an ninh.")

    st.markdown(
        f"""
        <div class="soc-panel">
            <span class="soc-badge">ALERT {result['alert_id']}</span>
            <span class="soc-badge">{result['timestamp']}</span>
            <span class="soc-badge {severity_class(sev)}">{sev}</span>
            <span class="soc-badge">Risk {risk}/100</span>
            <div style="margin-top:10px;font-size:1.05rem;font-weight:600;">{llm.get('verdict', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Risk Score", f"{risk}/100")
    col2.metric("Hybrid", f"{result['hybrid_score']:.3f}")
    col3.metric("AE / VAE", f"{result['ae_score']:.3f}")
    col4.metric("Detection", result['predicted_class'])
    col5.metric("Classifier", result.get('classifier_class', result['predicted_class']))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Triage", "Model Evidence", "MITRE ATT&CK", "Response", "Raw Report"])

    with tab1:
        st.markdown("**Attack Summary**")
        st.write(llm.get('attack_summary', 'N/A'))
        a, b, c = st.columns(3)
        a.metric("Severity", sev)
        b.metric("False Positive Risk", llm.get('false_positive_risk', 'N/A'))
        c.metric("MITRE Confidence", (result.get("mitre_result") or {}).get("confidence", "N/A"))
        st.markdown("**Analyst Note**")
        st.write(llm.get('analyst_note', 'N/A'))
        if llm.get('false_positive_reason'):
            st.caption(f"False-positive rationale: {llm['false_positive_reason']}")

    with tab2:
        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES[:len(result['probs'])],
            "Probability": [round(p, 4) for p in result['probs']]
        })
        st.markdown("**Class Probabilities**")
        st.dataframe(prob_df.sort_values("Probability", ascending=False), use_container_width=True, hide_index=True)
        st.bar_chart(prob_df.set_index("Class"))

        feats = result.get('top_features', [])
        if feats:
            df_shap = pd.DataFrame(feats, columns=["Feature", "SHAP Value", "Actual Value"])
            df_shap["SHAP Value"] = df_shap["SHAP Value"].round(4)
            df_shap["Actual Value"] = df_shap["Actual Value"].round(4)
            df_shap["Direction"] = df_shap["SHAP Value"].apply(lambda x: "Tang nguy co" if x > 0 else "Giam nguy co")
            st.dataframe(df_shap, use_container_width=True, hide_index=True)
            st.bar_chart(df_shap.set_index("Feature")["SHAP Value"])
        else:
            st.write(result.get('shap_summary', 'SHAP chua chay'))

    with tab3:
        mitre = result.get('mitre_result')
        if mitre:
            techniques = mitre.get('techniques') or mitre.get('suspected_techniques', [])
            st.markdown(
                f"**Mode:** `{mitre.get('mapping_mode', 'unknown')}` | "
                f"**Primary tactic:** `{mitre.get('primary_tactic', 'N/A')}` | "
                f"**Confidence:** `{mitre.get('confidence', 'LOW')}`"
            )
            if techniques:
                mitre_df = pd.DataFrame([{
                    "Technique": f"{t['id']} - {t['name']}",
                    "Tactic": t.get('tactic', ''),
                    "Confidence": t.get('confidence', ''),
                    "Rationale": t.get('rationale', ''),
                    "Evidence": ", ".join(t.get('evidence', [])),
                    "ATT&CK": t.get('url', f"https://attack.mitre.org/techniques/{t['id']}/"),
                } for t in techniques])
                st.dataframe(mitre_df, use_container_width=True, hide_index=True)
                checks = []
                for t in techniques:
                    for action in t.get("response_actions", []):
                        checks.append({"Technique": t["id"], "Check": action})
                if checks:
                    st.markdown("**Recommended MITRE-driven checks**")
                    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
            st.caption(mitre.get("coverage_note", ""))
        else:
            st.write(result.get('mitre_summary', 'MITRE chua chay'))

    with tab4:
        mitre = result.get('mitre_result') or {}
        actions = list(llm.get('recommended_actions', []))
        for t in mitre.get("techniques", []) or mitre.get("suspected_techniques", []):
            actions.extend(t.get("response_actions", []))
        actions = list(dict.fromkeys(actions))
        st.markdown("**Analyst Response Checklist**")
        for i, action in enumerate(actions, 1):
            st.checkbox(f"{i}. {action}", key=f"act_{result['alert_id']}_{i}")
        render_report_download(result, llm, key=str(result["alert_id"]))

    with tab5:
        render_raw_report(result, llm)

    st.session_state['last_alert'] = result
    st.session_state['last_llm'] = llm


if page == "[1] Dashboard":
    render_soc_header(
        "Network Intrusion Detection",
        "Network Traffic Analysis and Anomaly Detection",
    )

    history = load_alert_history()
    summary = queue_summary(history)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analyzed Alerts", summary["alerts"])
    c2.metric("High-risk Alerts", summary["critical_high"])
    c3.metric("Anomalous Traffic", summary["ood"])
    c4.metric("Model", "DEMO" if DEMO_MODE else MODEL_VERSION.upper())
    threshold_profile = PIPELINE_META.get("threshold_profile") if isinstance(PIPELINE_META, dict) else None
    threshold_badge = "LOCAL" if threshold_profile else "ARTIFACT"

    st.markdown(
        f"""
        <div class="soc-panel">
            <span class="soc-badge">Model: {os.path.basename(MODEL_PATH)}</span>
            <span class="soc-badge">Pipeline: {os.path.basename(PIPE_PATH)}</span>
            <span class="soc-badge">Thresholds: {threshold_badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Upload a CSV to classify network flows and detect anomalous traffic.")

    if DEMO_MODE:
        st.warning("Model artifacts are not loaded. Check the model and pipeline paths before analysis.")

# ═════════════════════════════════════════════════════════════════
# PAGE: Analyze Alert
# ═════════════════════════════════════════════════════════════════
elif page == "[2] Analyze Traffic":
    render_soc_header(
        "Analyze Network Traffic",
        "Upload flow data to classify traffic and detect anomalous activity.",
    )

    render_analysis_safety_notice(DEMO_MODE)

    bg_data = None
    comps   = {}

    if not DEMO_MODE:
        bg_data = load_background(scaler, feature_names)
        comps   = get_components(model, scaler, feature_names, bg_data)

    mode = st.radio(
        "Input source",
        ["Random sample tu dataset", "Nhap thu cong (demo)", "Upload CSV"],
        horizontal=True,
    )

    if mode == "Random sample tu dataset":
        if not os.path.exists(DATA_PATH) and not DEMO_MODE:
            st.error(f"Khong tim thay data tai: {DATA_PATH}")
            st.info("Tai UNSW-NB15 tu Kaggle va dat vao thu muc data/")
        else:
            prefer_attack = st.checkbox("Uu tien lay mau attack (de demo ro hon)", value=True)
            if st.button("Lay sample ngau nhien + Phan tich", type="primary"):
                if DEMO_MODE:
                    with st.spinner("Dang phan tich (DEMO)..."):
                        result = mock_inference()
                        llm    = get_llm_analysis(result, comps)
                else:
                    with st.spinner("Dang phan tich... (SHAP mat ~30 giay)"):
                        df = pd.read_csv(DATA_PATH)
                        label_col = 'label' if 'label' in df.columns else df.columns[-1]
                        pool   = df[df[label_col] == 1] if prefer_attack else df
                        sample = pool.sample(1, random_state=np.random.randint(0, 9999))
                        # Tien xu ly dung nhu luc train (encode + engineer features)
                        raw    = preprocess_raw_df(sample, feature_names or [])
                        result = run_full_pipeline(raw, comps)
                        llm    = get_llm_analysis(result, comps)

                display_result(result, llm)

                persist_alert(result, llm, source="single")

    elif mode == "Nhap thu cong (demo)":
        st.info("Score sandbox chi de minh hoa rule; khong luu vao analyst queue.")
        col1, col2 = st.columns(2)
        ae_val  = col1.slider("AE Reconstruction Error", 0.0, 1.0, 0.75, 0.01)
        max_p   = col2.slider("Classifier Max Probability", 0.0, 1.0, 0.45, 0.01)
        atk_cls = st.selectbox("Predicted Class", CLASS_NAMES)

        if st.button("Phan tich", type="primary"):
            hybrid = float(np.asarray(_hybrid_score(ae_val, max_p)).reshape(-1)[0])
            is_zd, decision_rule = _zero_day_decision(ae_val, max_p, hybrid)
            is_zd = bool(is_zd)
            verdict = _traffic_verdict(is_zd, atk_cls)
            result = {
                "alert_id"       : str(uuid.uuid4())[:8].upper(),
                "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "hybrid_score"   : hybrid,
                "ae_score"       : ae_val,
                "max_prob"       : max_p,
                "predicted_class": verdict,
                "classifier_class": atk_cls,
                "is_zeroday"     : is_zd,
                "zero_day_rule"   : decision_rule,
                "shap_summary"   : "Manual input - khong co SHAP data",
                "mitre_summary"  : "",
                "top_features"   : [],
                "probs"          : [1/len(CLASS_NAMES)] * len(CLASS_NAMES),
                "demo_mode"      : True,
            }
            if HAS_MITRE:
                mapper = MITREMapper()
                if is_zd:
                    mr = mapper.map_zeroday(ae_val, [])
                else:
                    mr = mapper.map_known_attack(
                        CLASS_NAMES.index(atk_cls) if atk_cls in CLASS_NAMES else 1,
                        class_names=CLASS_NAMES,
                    )
                result['mitre_result']  = mr
                result['mitre_summary'] = mapper.format_for_llm(mr)

            with st.spinner("LLM dang phan tich..."):
                llm = get_llm_analysis(result, comps)
            display_result(result, llm)

    else:  # Upload CSV
        render_batch_safety_notice()
        persist_top_n = st.number_input(
            "Persist top batch alerts",
            min_value=0,
            max_value=100,
            value=25,
            step=5,
            help="Save the highest-risk batch rows into the SQLite alert queue. Use 0 to disable persistence.",
        )
        uploaded = st.file_uploader(
            "Upload CSV (UNSW-NB15 raw hoac log thuc te: firewall, NetFlow, Zeek, Suricata)", type="csv"
        )
        if uploaded:
            uploaded_size = getattr(uploaded, "size", None)
            if uploaded_size is not None and uploaded_size > CSV_UPLOAD_MAX_BYTES:
                st.error(
                    "CSV qua lon de phan tich trong dashboard: "
                    f"{uploaded_size:,} bytes > {CSV_UPLOAD_MAX_BYTES:,} bytes. "
                    "Hay dung scripts/evaluate_csv.py cho batch lon hoac tang IDS_DASHBOARD_MAX_CSV_BYTES."
                )
                st.stop()
            file_hash = _uploaded_file_hash(uploaded)
            _reset_bulk_results_for_new_file(file_hash)
            try:
                uploaded.seek(0)
                raw_df = pd.read_csv(uploaded)
            except UnicodeDecodeError:
                uploaded.seek(0)
                try:
                    raw_df = pd.read_csv(uploaded, encoding="latin1")
                    st.warning("CSV khong doc duoc bang UTF-8; da fallback sang latin1.")
                except Exception as e:
                    st.error(f"Khong doc duoc CSV: {e}")
                    st.stop()
            except Exception as e:
                st.error(f"Khong doc duoc CSV: {e}")
                st.stop()
            input_validation = None
            if HAS_INPUT_GUARD and validate_uploaded_csv is not None:
                input_policy = CSVInputPolicy(
                    max_bytes=CSV_UPLOAD_MAX_BYTES,
                    max_rows=CSV_UPLOAD_MAX_ROWS,
                    max_columns=CSV_UPLOAD_MAX_COLUMNS,
                )
                input_validation = validate_uploaded_csv(raw_df, size_bytes=uploaded_size, policy=input_policy)
                st.session_state["last_csv_input_validation"] = input_validation.as_dict()
            st.caption(f"File: {uploaded.name} | SHA256: {file_hash[:12]}")
            if input_validation is not None:
                if input_validation.ok:
                    st.success(
                        f"CSV input OK: {input_validation.rows:,} rows x {input_validation.columns:,} columns"
                    )
                else:
                    st.error("CSV input khong hop le: " + "; ".join(input_validation.errors))
                for warning in input_validation.warnings[:4]:
                    st.warning(warning)
            with st.expander("Preview full CSV", expanded=True):
                st.caption(f"{len(raw_df):,} rows x {len(raw_df.columns):,} columns")
                preview_limit = min(len(raw_df), CSV_PREVIEW_MAX_ROWS)
                if preview_limit < 10:
                    preview_rows = preview_limit
                else:
                    preview_rows = st.slider(
                        "So dong preview",
                        min_value=10,
                        max_value=preview_limit,
                        value=min(200, preview_limit),
                        step=10,
                        key=f"raw_preview_rows_{file_hash[:8]}",
                    )
                st.dataframe(
                    raw_df.head(preview_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=560,
                )

            if st.button("Phan tich TOAN BO file"):
                if input_validation is not None and not input_validation.ok:
                    st.error("Khong the phan tich file CSV khi validation con loi.")
                elif DEMO_MODE:
                    st.warning("DEMO MODE khong ho tro phan tich toan bo file.")
                else:
                    raw = preprocess_raw_df(raw_df, feature_names or [])
                    norm_report = st.session_state.get('last_log_normalization_report')
                    if isinstance(norm_report, dict):
                        with st.expander("Bao cao chuan hoa CSV", expanded=True):
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Schema", norm_report.get("schema", "unknown"))
                            c2.metric("Rows", norm_report.get("rows", len(raw_df)))
                            c3.metric("Mapped cols", len(norm_report.get("mapped_columns", {})))
                            cov = norm_report.get("feature_coverage")
                            c4.metric("Feature coverage", f"{float(cov) * 100:.1f}%" if isinstance(cov, (int, float)) else "N/A")
                            if norm_report.get("missing_core_features"):
                                st.caption("Missing core features: " + ", ".join(norm_report["missing_core_features"][:12]))
                            if norm_report.get("derived_columns"):
                                st.caption("Derived features: " + ", ".join(norm_report["derived_columns"][:18]))
                            quality = runtime_assess_normalization_quality(norm_report)
                            if quality["level"] in {"LOW", "UNKNOWN"}:
                                st.warning(quality["message"])
                            elif quality["level"] == "MEDIUM":
                                st.info(quality["message"])
                            else:
                                st.success(quality["message"])
                            for warning in quality["warnings"][:4]:
                                st.caption(warning)
                    if raw.shape[1] == 0 or (raw == 0).all():
                        st.error(
                            "Khong tim thay features hop le trong file CSV nay. "
                            "Vui long upload CSV co truong flow co ban nhu src/dst IP, port, protocol, bytes, packets, duration/timestamp."
                        )
                    else:
                        with st.spinner("Dang phan tich toan bo file..."):
                            result_df = run_batch_inference(raw)
                        if result_df.empty:
                            st.error("Khong the chay inference cho file nay.")
                        else:
                            result_df = result_df.copy()
                            result_df.insert(0, "source_row", np.arange(len(result_df)))
                            family_col = next(
                                (c for c in ["true_label", "attack_cat", "zero_day_family", "label"] if c in raw_df.columns),
                                None,
                            )
                            if family_col:
                                result_df["zero_day_family"] = raw_df[family_col].astype(str).reset_index(drop=True)
                                result_df["ground_truth"] = result_df["zero_day_family"].map(_ground_truth_verdict)
                            else:
                                result_df["zero_day_family"] = ""
                                result_df["ground_truth"] = ""
                            result_df["detection"] = result_df.apply(
                                lambda r: _traffic_verdict(r["is_zeroday"], r.get("classifier_class", r.get("predicted_class", "Unknown"))),
                                axis=1,
                            )
                            if "ground_truth" in result_df.columns:
                                result_df["correct_vs_ground_truth"] = np.where(
                                    result_df["ground_truth"].astype(str).str.len() > 0,
                                    result_df["detection"].astype(str) == result_df["ground_truth"].astype(str),
                                    np.nan,
                                )
                            st.session_state['bulk_score_summary'] = {
                                "ae_min": float(result_df["ae_score"].min()),
                                "ae_median": float(result_df["ae_score"].median()),
                                "ae_max": float(result_df["ae_score"].max()),
                                "hybrid_min": float(result_df["hybrid_score"].min()),
                                "hybrid_median": float(result_df["hybrid_score"].median()),
                                "hybrid_max": float(result_df["hybrid_score"].max()),
                                "hybrid_threshold": HYBRID_THRESHOLD,
                                "ae_threshold": AE_THRESHOLD,
                                "decision_rule": str(result_df["zero_day_rule"].iloc[0]) if "zero_day_rule" in result_df else "unknown",
                            }
                            st.session_state['bulk_result_df'] = result_df
                            if len(raw_df) <= CSV_SESSION_RAW_MAX_ROWS:
                                st.session_state['bulk_raw_df'] = raw_df.reset_index(drop=True)
                            else:
                                st.session_state['bulk_raw_df'] = None
                                st.warning(
                                    "Raw CSV khong duoc giu trong session_state vi vuot "
                                    f"{CSV_SESSION_RAW_MAX_ROWS:,} dong. Bang diem van duoc giu, "
                                    "nhung chi tiet raw feature sau khi chuyen trang se bi gioi han."
                                )
                            st.session_state['bulk_source_file_hash'] = file_hash

            result_df = st.session_state.get('bulk_result_df')
            result_hash = st.session_state.get('bulk_source_file_hash')
            if isinstance(result_df, pd.DataFrame) and not result_df.empty and result_hash == file_hash:
                st.success(f"Da phan tich {len(result_df):,} dong.")

                # Luu alert gan nhat de tab Ask AI co the su dung
                try:
                    batch_alerts = runtime_build_top_batch_alerts(
                        result_df,
                        file_hash=file_hash,
                        limit=int(persist_top_n),
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        raw_df=st.session_state.get('bulk_raw_df'),
                    )
                    if batch_alerts:
                        st.session_state['last_alert'] = batch_alerts[0]
                    persist_key = f"{file_hash}:{int(persist_top_n)}"
                    if batch_alerts and st.session_state.get("bulk_alert_saved_hash") != persist_key:
                        for alert in batch_alerts:
                            persist_alert(alert, None, source="batch_top")
                        st.session_state["bulk_alert_saved_hash"] = persist_key
                        st.caption(f"Saved top {len(batch_alerts)} batch alerts to SQLite queue.")
                except Exception:
                    pass

                max_rows = min(1000, len(result_df))
                if max_rows <= 10:
                    rows_to_show = max_rows
                else:
                    rows_to_show = st.slider(
                        "So dong muon xem",
                        min_value=10,
                        max_value=max_rows,
                        value=min(100, max_rows),
                        step=10,
                    )
                st.dataframe(
                    result_df.head(rows_to_show),
                    use_container_width=True,
                    hide_index=True,
                    height=620,
                )

                render_bulk_detection_summary(result_df)
                render_ground_truth_summary(result_df)
                score_summary = st.session_state.get('bulk_score_summary')
                if isinstance(score_summary, dict):
                    with st.expander("Kiem tra score va nguong phat hien", expanded=False):
                        st.json(score_summary)

                st.download_button(
                    "Download ket qua (CSV)",
                    data=result_df.to_csv(index=False).encode('utf-8'),
                    file_name="ids_bulk_results.csv",
                    mime="text/csv"
                )

# ═════════════════════════════════════════════════════════════════
# PAGE: OOD Candidate Logs
# ═════════════════════════════════════════════════════════════════
elif page == "[3] OOD Candidate Logs":
    render_soc_header(
        "OOD Candidate Logs",
        "Anomaly flags are hypotheses for analyst review. Dataset family values are optional reference metadata.",
    )

    result_df = st.session_state.get('bulk_result_df')
    raw_df = st.session_state.get('bulk_raw_df')

    if not isinstance(result_df, pd.DataFrame) or result_df.empty:
        st.info("Chua co batch log. Vao Analyze Alert -> Upload CSV -> Phan tich TOAN BO file truoc.")
    else:
        logs = result_df.copy()
        if "source_row" not in logs.columns:
            logs.insert(0, "source_row", np.arange(len(logs)))
        if "zero_day_family" not in logs.columns:
            family_col = None
            if isinstance(raw_df, pd.DataFrame):
                family_col = next(
                    (c for c in ["true_label", "attack_cat", "zero_day_family", "label"] if c in raw_df.columns),
                    None,
                )
            logs["zero_day_family"] = (
                raw_df[family_col].astype(str).reset_index(drop=True)
                if family_col else ""
            )
        # Luon recompute de dam bao nhat quan voi is_zeroday moi nhat
        logs["detection"] = logs.apply(
            lambda r: _traffic_verdict(r.get("is_zeroday"), r.get("classifier_class", r.get("predicted_class", "Unknown"))),
            axis=1,
        )

        zd_logs = logs[logs["is_zeroday"].astype(bool)].copy()
        total = len(logs)
        zd_total = len(zd_logs)
        verdicts = sorted(logs["detection"].astype(str).unique().tolist())
        classifier_classes = sorted(logs.get("classifier_class", logs["predicted_class"]).astype(str).unique().tolist())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Batch Rows", f"{total:,}")
        c2.metric("Anomalous Traffic Logs", f"{zd_total:,}")
        c3.metric("Anomaly Rate", f"{(zd_total / total * 100):.2f}%" if total else "0.00%")
        c4.metric("Verdict Labels", len(verdicts))

        st.markdown("### Filters")
        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        show_only_zd = f1.toggle("Only anomalous traffic", value=True)
        selected_verdict = f2.selectbox("Verdict", ["All"] + verdicts)
        selected_class = f3.selectbox("Classifier class", ["All"] + classifier_classes)
        min_score = f4.number_input("Min anomaly score", min_value=0.0, value=0.0, step=0.1)

        view_df = zd_logs if show_only_zd else logs
        if selected_verdict != "All":
            view_df = view_df[view_df["detection"].astype(str) == selected_verdict]
        if selected_class != "All":
            view_df = view_df[view_df.get("classifier_class", view_df["predicted_class"]).astype(str) == selected_class]
        view_df = view_df[view_df["hybrid_score"] >= min_score].copy()
        view_df["risk"] = view_df.apply(lambda r: risk_score({
            "hybrid_score": r.get("hybrid_score", 0),
            "ae_score": r.get("ae_score", 0),
            "is_zeroday": bool(r.get("is_zeroday", False)),
        }, "HIGH" if r.get("is_zeroday", False) else "MEDIUM"), axis=1)
        view_df = view_df.sort_values(["risk", "hybrid_score", "ae_score"], ascending=False)

        left_col, right_col = st.columns([1.55, 1.0], gap="large")
        selected_source_row = None

        with left_col:
            st.markdown("### Detection Log")
            if view_df.empty:
                st.info("Khong co log nao khop filter hien tai.")
            else:
                display_cols = [c for c in [
                    "source_row", "detection", "ground_truth", "correct_vs_ground_truth", "classifier_class",
                    "risk", "hybrid_score", "ae_score", "max_prob", "is_zeroday"
                ] if c in view_df.columns]
                try:
                    event = st.dataframe(
                        view_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        selection_mode="single-row",
                        on_select="rerun",
                    )
                    selected_rows = event.selection.rows if event and hasattr(event, "selection") else []
                    if selected_rows:
                        selected_source_row = int(view_df.iloc[selected_rows[0]]["source_row"])
                except TypeError:
                    st.dataframe(view_df[display_cols], use_container_width=True, hide_index=True)

                if selected_source_row is None:
                    labels = []
                    for _, r in view_df.head(500).iterrows():
                        labels.append(
                            f"row={int(r['source_row'])} | {r.get('detection', r.get('predicted_class'))} | "
                            f"hybrid={float(r['hybrid_score']):.3g} | ae={float(r['ae_score']):.3g}"
                        )
                    selected_label = st.selectbox("Select log row", labels)
                    selected_pos = labels.index(selected_label)
                    selected_source_row = int(view_df.head(500).iloc[selected_pos]["source_row"])

            st.download_button(
                "Download filtered anomaly log",
                data=view_df.to_csv(index=False).encode("utf-8"),
                file_name="ood_candidate_logs_filtered.csv",
                mime="text/csv",
                disabled=view_df.empty,
            )

        with right_col:
            st.markdown("### Event Detail")
            if selected_source_row is None:
                st.markdown(
                    """
                    <div class="soc-panel">
                        Select a row on the left to inspect classifier output, OOD hypothesis, MITRE hypothesis and full features.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                row_scores = enrich_ood_row(logs[logs["source_row"] == selected_source_row].iloc[0].to_dict())
                classifier_class = str(row_scores.get("classifier_class", "Unknown"))
                detection = str(row_scores.get("detection", "Unknown"))
                verdict_badge = "ANOMALOUS TRAFFIC" if row_scores.get("is_zeroday") else "KNOWN"
                badge_class = "soc-pill-red" if row_scores.get("is_zeroday") else "soc-pill-green"

                st.markdown(
                    f"""
                    <div class="soc-panel">
                        <span class="soc-badge {badge_class}">{verdict_badge}</span>
                        <span class="soc-badge soc-pill-blue">row {selected_source_row}</span>
                        <div class="soc-detail-title" style="margin-top:12px;">Detection</div>
                        <div class="soc-detail-value">{detection}</div>
                        <div class="soc-detail-title">Known-Class Classifier Output</div>
                        <div class="soc-detail-value">{classifier_class}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                d1, d2 = st.columns(2)
                d1.metric("Hybrid", f"{float(row_scores.get('hybrid_score', 0)):.3g}")
                d2.metric("AE / VAE", f"{float(row_scores.get('ae_score', 0)):.3g}")
                d3, d4 = st.columns(2)
                d3.metric("Max Prob", f"{float(row_scores.get('max_prob', 0)):.3f}")
                d4.metric("Risk", f"{int(row_scores.get('risk', 0))}/100")
                if st.button("Ask AI about this log", type="primary", key=f"ask_ai_log_{selected_source_row}"):
                    alert_context = build_alert_context_from_log(row_scores)
                    if HAS_MITRE:
                        mapper = MITREMapper()
                        if alert_context["is_zeroday"]:
                            mitre_res = mapper.map_zeroday(alert_context["ae_score"], [])
                        else:
                            class_idx = CLASS_NAMES.index(classifier_class) if classifier_class in CLASS_NAMES else 0
                            mitre_res = mapper.map_known_attack(class_idx, class_names=CLASS_NAMES)
                        alert_context["mitre_result"] = mitre_res
                        alert_context["mitre_summary"] = mapper.format_for_llm(mitre_res)
                    st.session_state["last_alert"] = alert_context
                    st.session_state["ask_ai_context_source"] = "zero_day_log"
                    st.session_state["messages"] = []
                    st.session_state["nav_page"] = "[4] Ask AI"
                    st.rerun()

                tabs = st.tabs(["Full Features", "Scores", "MITRE"])
                with tabs[0]:
                    if isinstance(raw_df, pd.DataFrame) and selected_source_row < len(raw_df):
                        feature_row = raw_df.iloc[selected_source_row]
                        search = st.text_input("Search feature", "")
                        feature_table = build_feature_table(feature_row, search)
                        st.dataframe(feature_table, use_container_width=True, hide_index=True, height=430)
                    else:
                        st.warning("Khong tim thay raw feature row tu batch upload.")

                with tabs[1]:
                    score_table = build_score_table(row_scores)
                    st.dataframe(score_table, use_container_width=True, hide_index=True, height=360)

                with tabs[2]:
                    if HAS_MITRE:
                        mapper = MITREMapper()
                        if bool(row_scores.get("is_zeroday")):
                            mitre_res = mapper.map_zeroday(float(row_scores.get("ae_score", 0)), [])
                        else:
                            class_idx = CLASS_NAMES.index(classifier_class) if classifier_class in CLASS_NAMES else 0
                            mitre_res = mapper.map_known_attack(class_idx, class_names=CLASS_NAMES)
                        techniques = mitre_res.get("techniques") or mitre_res.get("suspected_techniques", [])
                        st.markdown(
                            f"**Mode:** `{mitre_res.get('mapping_mode', 'unknown')}` | "
                            f"**Confidence:** `{mitre_res.get('confidence', 'LOW')}`"
                        )
                        if techniques:
                            st.dataframe(pd.DataFrame([{
                                "Technique": f"{t['id']} - {t['name']}",
                                "Tactic": t.get("tactic", ""),
                                "Confidence": t.get("confidence", ""),
                                "Rationale": t.get("rationale", ""),
                                "ATT&CK": t.get("url", ""),
                            } for t in techniques]), use_container_width=True, hide_index=True)
                        st.caption(mitre_res.get("coverage_note", ""))
                    else:
                        st.warning("MITRE module chua duoc kich hoat.")

# ═════════════════════════════════════════════════════════════════
# PAGE: Ask AI
# ═════════════════════════════════════════════════════════════════
elif page == "[4] Ask AI":
    render_soc_header(
        "Triage Assistant",
        "Ask follow-up questions against the latest alert context, MITRE mapping and model evidence.",
    )

    context_options = runtime_build_ai_context_options(
        load_alert_history(),
        st.session_state.get("bulk_result_df"),
    )

    if context_options:
        labels = [option.label for option in context_options]
        current_alert_id = st.session_state.get("last_alert", {}).get("alert_id")
        default_idx = runtime_default_ai_context_index(context_options, current_alert_id)
        selected_context = st.selectbox("AI context", labels, index=default_idx)
        selected_ctx = context_options[labels.index(selected_context)].context
        if st.button("Use selected context", key="use_ai_context"):
            st.session_state["last_alert"] = selected_ctx
            st.session_state["messages"] = []
            st.rerun()

    if 'last_alert' not in st.session_state:
        st.warning("Chua co alert/log nao. Hay phan tich 1 alert hoac upload CSV batch truoc.")
        st.stop()

    last = st.session_state['last_alert']
    render_ai_context_card(last)
    render_question_suggestions()

    st.markdown("---")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role_label = "Analyst" if msg["role"] == "user" else "Assistant"
        with st.chat_message(msg["role"]):
            st.write(f"**{role_label}:** {msg['content']}")

    # Xu ly cau hoi tu goi y
    pending = st.session_state.pop('pending_question', None)

    question = st.chat_input("Hoi AI ve alert nay...") or pending

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(f"**Analyst:** {question}")

        with st.chat_message("assistant"):
            with st.spinner("Dang suy nghi..."):
                answer = runtime_answer_analyst_question(
                    question,
                    last,
                    HAS_LLM,
                    agent_factory=SOCTriageAgent if HAS_LLM else None,
                    llm_provider=LLM_PROVIDER,
                    llm_dependency=LLM_DEP,
                    has_llm_dependency=HAS_LLM_DEPS,
                )

            st.write(f"**Assistant:** {answer}")
            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("Xoa lich su chat"):
        st.session_state.messages = []
        st.rerun()

# ═════════════════════════════════════════════════════════════════
# PAGE: Setup Guide
# ═════════════════════════════════════════════════════════════════
elif page == "[5] Setup Guide":
    render_soc_header(
        "Deployment Checklist",
        "Runtime paths, model artifacts, data availability and optional provider configuration.",
    )

    st.markdown("### Buoc 1: Cau truc thu muc")
    st.code("""
ZeroDay-Detection-AutoEncoder-IDS/
├── dashboard/
│   └── app.py          <- file nay
├── src/
│   └── ids/
│       ├── explainer.py
│       ├── mitre_mapper.py
│       └── llm_agent.py
├── checkpoints/
│   ├── ids_v14_model.pth
│   └── ids_v14_pipeline.pkl
├── data/
│   └── UNSW_NB15_training-set.csv
└── .env
    """)

    st.markdown("### Buoc 2: Export model tu IDS v14")
    st.code("""
# QUAN TRONG: Phai save dung format sau de dashboard load duoc
import pickle, torch

torch.save({
    'model_state_dict': model.state_dict(),   # weights cua model
    'n_features'      : X_train.shape[1],     # so luong features
    'n_classes'       : len(le.classes_),     # so luong class
    'hidden'          : 256,                  # hidden dim backbone
    'ae_hidden'       : 128,                  # hidden dim autoencoder
}, 'checkpoints/ids_v14_model.pth')

# Save pipeline
pipeline = {
    'scaler'       : scaler,
    'feature_names': feat_cols,
    'label_encoder': le,
    'n_features'   : X_train.shape[1],
    'n_classes'    : len(le.classes_),
}
with open('checkpoints/ids_v14_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Da luu model va pipeline!")
    """, language="python")

    st.markdown("### Buoc 3: Cau hinh API Key cho LLM")
    st.markdown("1. LLM mac dinh tat bang `LLM_PROVIDER=none`; day chi la tro ly phan tich, khong phai engine quyet dinh verdict.")
    st.markdown("2. Neu can LLM, dang ky API Key tai nha cung cap ban chon va set provider trong `.env`.")
    st.markdown("3. Tao file `.env` trong thu muc goc va them key tuong ung:")
    st.code("LLM_PROVIDER=groq\nGROQ_API_KEY=gsk_...\n# GEMINI_API_KEY=AIzaSy...\n# OPENAI_API_KEY=sk-...", language="bash")

    st.markdown("**Ghi chu:** Neu thieu pipeline, app se co gang tao tu data/ khi khoi dong.")

    st.markdown("### Buoc 4: Chay app")
    st.code("""
cd dashboard
streamlit run app.py
    """, language="bash")

    render_setup_status(
        MODEL_PATH,
        PIPE_PATH,
        DATA_PATH,
        ALERT_DB_PATH,
        HAS_EXPLAINER,
        HAS_MITRE,
        HAS_LLM,
    )
