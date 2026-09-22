#!/usr/bin/env python3
"""P7 Stage A: preregister external evaluation without loading official test."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p5_training import json_bytes, stable_hash, write_immutable  # noqa: E402
from ids.p7_external_protocol import frozen_contract, p7_path  # noqa: E402


def main() -> int:
    if p7_path(ROOT, "primary").exists() or p7_path(ROOT, "diagnostics").exists():
        raise FileExistsError("Stage A refuses a namespace with external results")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_p7_external_protocol.py"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"P7 synthetic tests failed before freeze:\n{result.stdout}\n{result.stderr}")
    contract = frozen_contract(ROOT)
    digest = stable_hash(contract)
    base = "preregistration/"
    artifacts = {
        "external_protocol.json": contract,
        "metric_contract.json": contract["metric_contract"],
        "uncertainty_contract.json": contract["uncertainty_contract"],
        "surface_contract.json": contract["surface_contract"],
        "result_schema.json": contract["result_schema"],
        "preregistration_gate.json": {
            "status": "P7_PREREGISTERED", "protocol_sha256": digest,
            "upstream_aggregate_sha256": contract["upstream_snapshot"]["aggregate_sha256"],
            "source_code_sha256": contract["source_code_sha256"],
            "synthetic_tests_passed": True,
            "synthetic_test_command": "python -m pytest -q -p no:cacheprovider tests/test_p7_external_protocol.py",
            "official_test_loaded": False, "external_scores_computed": False,
            "stage_b_executed": False,
        },
    }
    for name, payload in artifacts.items():
        write_immutable(p7_path(ROOT, base + name), json_bytes(payload))
    write_immutable(p7_path(ROOT, base + "protocol_hash.txt"), (digest + "\n").encode())
    print(f"[P7] P7_PREREGISTERED protocol SHA-256 {digest}; OFFICIAL TEST NOT OPENED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
