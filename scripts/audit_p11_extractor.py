#!/usr/bin/env python3
"""Create or verify immutable P11 extractor-validation artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p11_extractor.audit import build_artifacts, validate_artifacts  # noqa: E402


OUTPUT = ROOT / "results/extractor/p11"


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def timestamp(value: str | None) -> str:
    if value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("--created-at must be an ISO-8601 UTC timestamp")
        return parsed.isoformat().replace("+00:00", "Z")
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_new(artifacts: dict[str, dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        path = OUTPUT / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable P11 artifact: {path}")
        path.write_bytes(json_bytes(value))


def read_existing() -> dict[str, dict]:
    names = (
        "schema_contract.json", "feature_semantic_inventory.json",
        "original_extractor_lineage.json", "reference_corpus.json",
        "extractor_environment.json", "feature_validation_matrix.json",
        "categorical_validation.json", "continuous_validation.json",
        "determinism_validation.json", "portable_schema_gap_analysis.json", "p11_gate.json",
    )
    return {name: json.loads((OUTPUT / name).read_text()) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--created-at")
    args = parser.parse_args()
    if args.write:
        artifacts = build_artifacts(ROOT, timestamp(args.created_at))
        validate_artifacts(ROOT, artifacts)
        write_new(artifacts)
    else:
        artifacts = read_existing()
        validate_artifacts(ROOT, artifacts)
    print("P11_REFERENCE_CORPUS_BLOCKED")
    print("No immutable row-alignable raw-capture/authoritative-feature pair is locally available.")
    print("No P9 checkpoint, score, metric, scaler, vocabulary, or threshold was accessed or changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
