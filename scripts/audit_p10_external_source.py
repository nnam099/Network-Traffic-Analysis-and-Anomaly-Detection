#!/usr/bin/env python3
"""Create or verify the P10A metadata-only compatibility audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p10_external_source import (  # noqa: E402
    build_artifacts, json_bytes, validate_artifacts,
)


OUTPUT = ROOT / "results/loafo/p10"


def _timestamp(value: str | None) -> str:
    if value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ValueError("--audited-at must be an ISO-8601 UTC timestamp")
        return parsed.isoformat().replace("+00:00", "Z")
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_new(artifacts: dict[str, dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        path = OUTPUT / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable P10A artifact: {path}")
        path.write_bytes(json_bytes(value))


def read_existing() -> dict[str, dict]:
    names = (
        "candidate_dataset_audit.json", "feature_mapping.json", "label_mapping.json",
        "identity_contract.json", "external_source_reservation.json",
        "external_population_audit.json", "p10_preregistration.json",
    )
    return {name: json.loads((OUTPUT / name).read_text()) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="create immutable P10A artifacts")
    parser.add_argument("--audited-at", help="fixed ISO-8601 UTC timestamp for creation")
    args = parser.parse_args()
    if args.write:
        artifacts = build_artifacts(ROOT, _timestamp(args.audited_at))
        validate_artifacts(ROOT, artifacts)
        write_new(artifacts)
    else:
        artifacts = read_existing()
        validate_artifacts(ROOT, artifacts)
    print("P10_EXTERNAL_SOURCE_BLOCKED")
    print("No candidate reproduces all 58 continuous plus 3 categorical frozen semantics.")
    print("No external raw data or P9 checkpoint was opened; no model score was computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
