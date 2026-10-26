from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from artifact_manifest import verify_manifest  # noqa: E402


def configure_console_encoding() -> None:
    """Keep JSON output printable from Windows consoles with non-UTF-8 codepages."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def python_version_status(version_info: tuple[int, int, int] | None = None) -> dict:
    version = version_info or sys.version_info[:3]
    major, minor, patch = version
    supported = (major, minor) >= (3, 11) and (major, minor) < (3, 13)
    return {
        "version": f"{major}.{minor}.{patch}",
        "requirement": ">=3.11,<3.13",
        "supported": supported,
    }


def collect_environment() -> dict:
    packages = [
        "torch",
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "fastapi",
        "uvicorn",
        "httpx",
        "streamlit",
        "shap",
        "dotenv",
    ]
    provider_keys = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    key_name = provider_keys.get(provider)

    python = python_version_status()
    env = {
        "python": python["version"],
        "python_requirement": python["requirement"],
        "python_supported": python["supported"],
        "root": str(ROOT_DIR),
        "packages": {name: module_available(name) for name in packages},
        "artifacts": {
            "v14_model": (ROOT_DIR / "checkpoints" / "ids_v14_model.pth").exists(),
            "v14_pipeline": (ROOT_DIR / "checkpoints" / "ids_v14_pipeline.pkl").exists(),
        },
        "data": {
            "data_dir": (ROOT_DIR / "data").exists(),
            "training_csv": (ROOT_DIR / "data" / "UNSW_NB15_training-set.csv").exists(),
            "testing_csv": (ROOT_DIR / "data" / "UNSW_NB15_testing-set.csv").exists(),
        },
        "llm": {
            "provider": provider or None,
            "key_env": key_name,
            "key_present": bool(os.getenv(key_name, "").strip()) if key_name else False,
        },
    }
    manifest_path = ROOT_DIR / "results" / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        env["artifact_manifest"] = verify_manifest(ROOT_DIR, manifest)
    else:
        env["artifact_manifest"] = {"ok": None, "errors": [], "warnings": ["results/artifact_manifest.json is missing"]}
    env["readiness"] = assess_readiness(env)
    return env


def assess_readiness(env: dict) -> dict:
    blockers: list[str] = []
    warnings: list[str] = []

    if env.get("python_supported") is False:
        blockers.append(
            f"Unsupported Python version: {env.get('python')} "
            f"(required {env.get('python_requirement', '>=3.11,<3.13')})"
        )

    packages = env.get("packages", {})
    for package in ["torch", "numpy", "pandas", "sklearn", "matplotlib", "fastapi", "uvicorn", "httpx", "dotenv"]:
        if not packages.get(package):
            blockers.append(f"Missing required package: {package}")

    artifacts = env.get("artifacts", {})
    if not (artifacts.get("v14_model") and artifacts.get("v14_pipeline")):
        warnings.append("v14 artifacts are missing; dashboard will fall back to demo mode.")

    data = env.get("data", {})
    if not data.get("data_dir"):
        blockers.append("data/ directory is missing.")
    elif not (data.get("training_csv") and data.get("testing_csv")):
        warnings.append("UNSW train/test CSV files are incomplete; training workflows may fail.")

    if not packages.get("streamlit"):
        warnings.append("streamlit is missing; dashboard cannot run until installed.")
    if not packages.get("shap"):
        warnings.append("shap is missing; dashboard explainability will be disabled.")

    llm = env.get("llm", {})
    if llm.get("provider") and not llm.get("key_present"):
        warnings.append(f"LLM provider is set to {llm['provider']} but its API key is missing.")

    manifest = env.get("artifact_manifest", {})
    if manifest.get("ok") is False:
        blockers.extend(manifest.get("errors", []))
    elif manifest.get("ok") is None:
        warnings.extend(manifest.get("warnings", []))
    else:
        warnings.extend(manifest.get("warnings", []))

    if blockers:
        status = "BLOCKED"
    elif warnings:
        status = "WARN"
    else:
        status = "READY"

    return {
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report local IDS project environment readiness.")
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return a non-zero exit code when readiness status is BLOCKED.",
    )
    args = parser.parse_args()

    configure_console_encoding()
    env = collect_environment()
    print(json.dumps(env, indent=2, ensure_ascii=False))
    if args.fail_on_blocked and env.get("readiness", {}).get("status") == "BLOCKED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
