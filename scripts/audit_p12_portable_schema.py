#!/usr/bin/env python3
"""Build and validate the P12 design-only portable schema artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="validate existing artifacts without rewriting")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from ids.p12_portable_schema import (  # noqa: PLC0415
        build_artifacts, canonical_json_bytes, validate_artifacts,
    )

    output = root / "results/portable_schema/p12"
    if args.check:
        import json

        artifacts = {path.name: json.loads(path.read_text()) for path in sorted(output.glob("*.json"))}
        validate_artifacts(root, artifacts)
    else:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        artifacts = build_artifacts(root, timestamp)
        validate_artifacts(root, artifacts)
        output.mkdir(parents=True, exist_ok=True)
        for name, value in artifacts.items():
            (output / name).write_bytes(canonical_json_bytes(value, pretty=True))
        # Read back and independently validate the exact serialized artifacts.
        import json

        reread = {path.name: json.loads(path.read_text()) for path in sorted(output.glob("*.json"))}
        validate_artifacts(root, reread)
    gate = artifacts["p12_gate.json"]
    print(gate["status"])
    print(f"P12 protocol SHA-256: {gate['p12_protocol_sha256']}")
    print(f"Baseline semantic dimension: {gate['baseline_semantic_dimension']}")
    print(f"Complete capture families: {gate['complete_capture_family_count']}")
    print("No neural network was trained and no P7/P9 detector was evaluated during P12.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
