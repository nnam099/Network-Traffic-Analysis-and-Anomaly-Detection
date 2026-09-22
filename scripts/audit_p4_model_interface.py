#!/usr/bin/env python3
"""Verify P4 structured interfaces against frozen P3 manifests; never train."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p4_model_interface import (  # noqa: E402
    ARCHITECTURE_VERSION, CATEGORICAL_FIELDS, CONTINUOUS_FEATURE_COUNT,
    MODEL_DTYPE_NAME, MODEL_INTERFACE_VERSION, OOV_INDEX, PAD_INDEX,
    RepresentationBackbone, StructuredBatchAdapter, bound_model_config,
    stable_hash, vocabularies_from_manifest,
)


EXPECTED_P3_COLLECTION_SHA256 = (
    "1443746a4f024b3d7c1569ba9bbd31987933b5f9e2fb0ce6507ba5e57ddbab21"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False,
                                allow_nan=False).encode() + b"\n")


def p3_snapshot(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): file_sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def legacy_inventory() -> dict:
    entries = [
        ("src/ids/models.py", r"def forward\(self, x\)", "LEGACY-ONLY",
         "v14 flat-tensor models remain available only for historical/demo artifacts."),
        ("src/train.py", r"from ids\.models import IDSModel", "BLOCKING",
         "The legacy training entry point must never be used for schema V2."),
        ("src/ids/dataset.py", r"astype\(np\.float32\)", "LEGACY-ONLY",
         "Legacy flat preprocessing casts before its old model boundary."),
        ("src/ids/target_isolation.py", r"P0_FEATURE_COUNT = 61", "LEGACY-ONLY",
         "Immutable P0/P1 audit contract; not a P4 neural interface."),
        ("src/artifact_validator.py", r"checkpoint", "LEGACY-ONLY",
         "Existing checkpoint validation covers old artifact families only."),
        ("src/batch_evaluator.py", r"feature_names", "MUST-REFACTOR",
         "Future schema-V2 runtime evaluation needs a structured adapter."),
        ("src/ids/p2_protocol.py", r"scientific-feature-schema-v2", "SAFE",
         "Data audit defines separated continuous and categorical semantics."),
        ("src/ids/p4_model_interface.py", r"flat tensors are forbidden", "SAFE",
         "P4 structured path rejects legacy flat tensors at runtime."),
    ]
    records = []
    for relative, pattern, category, rationale in entries:
        path = ROOT / relative
        text = path.read_text()
        lines = [index for index, line in enumerate(text.splitlines(), 1)
                 if re.search(pattern, line)]
        records.append({
            "path": relative, "pattern": pattern, "category": category,
            "matching_lines": lines, "rationale": rationale,
        })
    p4_source = (ROOT / "src/ids/p4_model_interface.py").read_text()
    forbidden_imports = [name for name in ("ids.models", "ids.trainer", "src.train")
                         if name in p4_source]
    return {
        "version": "p4-legacy-compatibility-audit-v1",
        "categories": ["SAFE", "MUST-REFACTOR", "LEGACY-ONLY", "BLOCKING"],
        "entries": records,
        "p4_forbidden_legacy_imports": forbidden_imports,
        "blocking_code_isolated_from_p4": not forbidden_imports,
        "legacy_checkpoint_migration_performed": False,
        "conclusion": (
            "Historical flat paths are retained but are not imported by P4. "
            "src/train.py is BLOCKING if reused for scientific-feature-schema-v2."
        ),
    }


def model_interface_artifact() -> dict:
    return {
        "version": MODEL_INTERFACE_VERSION,
        "p3_collection_sha256": EXPECTED_P3_COLLECTION_SHA256,
        "schema_version": "scientific-feature-schema-v2",
        "components": [
            {"name": "continuous", "shape": ["batch", 58],
             "source_dtype": "numpy.float64", "model_dtype": MODEL_DTYPE_NAME},
            *({"name": name, "shape": ["batch"], "model_dtype": "torch.int64"}
              for name in CATEGORICAL_FIELDS),
        ],
        "continuous_contract": {
            "preprocessing": "P3 frozen fold-local RobustScaler; no clipping",
            "source": "post-P3-audit scaled NumPy float64",
            "framework_boundary": "copy to torch.float64 without numerical conversion",
            "float64_to_float32": (
                "PROHIBITED in P4 baseline; requires a newly versioned collision audit"
            ),
            "nan_inf": "fail closed", "continuous_feature_count": 58,
        },
        "categorical_contract": {
            "fields": list(CATEGORICAL_FIELDS), "separate_embedding_tables": True,
            "pad_index": PAD_INDEX, "pad_use": "reserved; not emitted for tabular rows",
            "oov_index": OOV_INDEX, "known_indices": "contiguous from 2",
            "unseen_behavior": "one dedicated OOV embedding per feature",
            "modulo_hashing": False, "python_hash": False,
            "test_vocabulary_expansion": False,
            "identity_caveat": (
                "Distinct unseen tokens intentionally share a model OOV representation, "
                "but remain distinct in canonical/audit metadata."
            ),
        },
        "audit_metadata": "never passed into neural tensors",
        "generic_flat_tensor_accepted": False,
        "training_performed": False,
    }


def architecture_artifact() -> dict:
    return {
        "version": "p4-architecture-decision-v1",
        "decision_basis": ["simplicity", "auditability", "scientific defensibility"],
        "performance_evidence_used": False,
        "alternatives": [
            {
                "id": "A", "name": "Concatenated embeddings + numeric MLP",
                "advantages": ["simple", "explicit typed branches", "low engineering risk"],
                "risks": ["limited interaction inductive bias"],
                "decision": "SELECTED P4 BASELINE",
            },
            {
                "id": "B", "name": "Feature-token transformer",
                "advantages": ["explicit inter-feature attention"],
                "risks": ["greater complexity", "overfitting risk with small support"],
                "decision": "NOT SELECTED; no performance comparison was run",
            },
            {
                "id": "C", "name": "Hybrid controlled-interaction encoder",
                "advantages": ["can add constrained interactions later"],
                "risks": ["more design degrees of freedom", "harder attribution"],
                "decision": "DEFERRED",
            },
        ],
        "baseline": {
            "architecture_version": ARCHITECTURE_VERSION,
            "continuous": "58→64→32 float64 MLP",
            "categorical": "three independent per-feature embeddings",
            "fusion": "concatenate numeric latent and embeddings; MLP to 32-D representation",
            "classifier": None, "open_set_scorer": None,
            "zero_day_compatibility": (
                "The backbone exposes a representation without committing to a classifier or "
                "score. Future prototype/density/reconstruction scorers require separate approval."
            ),
            "autoencoder": (
                "Optional continuous reconstruction head is isolated and not selected by P4; "
                "categorical reconstruction and scoring semantics remain future decisions."
            ),
        },
        "small_support_evaluation_plan": {
            "clean_identity_counts": {
                "Fuzzers": 4317, "Analysis": 58, "Backdoors": 57,
                "Shellcode": 364, "Worms": 43,
            },
            "requirements": [
                "report raw rows and unique canonical identities",
                "report each target and seed separately",
                "include uncertainty intervals only where statistically meaningful",
                "attach explicit small-N warnings for Analysis, Backdoors, and Worms",
                "never aggregate targets in a way that hides target-specific support",
                "predeclare future scores/thresholds without using official-test-clean",
            ],
            "warning": "Nonzero support does not imply adequate statistical power.",
            "thresholds_or_metrics_selected": False,
        },
    }


def main() -> int:
    p3_dir = ROOT / "results/data_quality/p3"
    destination = ROOT / "results/model_design/p4"
    before = p3_snapshot(p3_dir)
    freeze = json.loads((p3_dir / "development_freeze.json").read_text())
    if canonical_hash(freeze) != EXPECTED_P3_COLLECTION_SHA256:
        raise AssertionError("P3 collection hash differs from the approved frozen contract")
    schema = json.loads((p3_dir / "schema_v2.json").read_text())
    interface = model_interface_artifact()
    design = architecture_artifact()
    legacy = legacy_inventory()
    if legacy["p4_forbidden_legacy_imports"]:
        raise AssertionError("P4 imports a legacy flat model/trainer")
    source_files = (ROOT / "src/ids/p4_model_interface.py",
                    ROOT / "scripts/audit_p4_model_interface.py")
    source_hashes = {str(path.relative_to(ROOT)): file_sha256(path)
                     for path in source_files}
    aggregate_source_hash = stable_hash(source_hashes)
    cells = []
    for entry in freeze["cells"]:
        path = p3_dir / entry["manifest_filename"]
        raw = gzip.decompress(path.read_bytes())
        manifest_hash = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        reasons = []
        if manifest_hash != entry["manifest_sha256"]:
            reasons.append("p3_manifest_hash_mismatch")
        try:
            config = bound_model_config(
                manifest, manifest_sha256=manifest_hash,
                p3_collection_sha256=EXPECTED_P3_COLLECTION_SHA256,
                schema=schema, source_code_sha256=aggregate_source_hash,
            )
            vocabularies = vocabularies_from_manifest(manifest)
            adapter = StructuredBatchAdapter(vocabularies)
            batch = adapter.transform(
                np.zeros((2, CONTINUOUS_FEATURE_COUNT), dtype=np.float64),
                {name: [None, "p4-unseen-probe"] for name in CATEGORICAL_FIELDS},
            )
            model = RepresentationBackbone(vocabularies)
            with torch.no_grad():
                output = model(batch)
            if tuple(output.shape) != (2, 32) or output.dtype != torch.float64:
                reasons.append("structured_architecture_output_contract")
            if any(int(getattr(batch, name)[1]) != OOV_INDEX for name in CATEGORICAL_FIELDS):
                reasons.append("finite_oov_contract")
            if config["continuous_feature_count"] != 58:
                reasons.append("continuous_schema")
            if len(config["categorical_fields"]) != 3:
                reasons.append("categorical_schema")
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            config = {"construction_error": f"{type(error).__name__}: {error}"}
            reasons.append("bound_configuration_construction")
        cells.append({
            "target": entry["target"], "seed": entry["seed"],
            "p3_manifest_filename": entry["manifest_filename"],
            "p3_manifest_sha256": manifest_hash,
            "bound_model_config": config,
            "checks": {
                "p3_manifest_verified": manifest_hash == entry["manifest_sha256"],
                "continuous_schema_58": not reasons or "continuous_schema" not in reasons,
                "categorical_schema_3": not reasons or "categorical_schema" not in reasons,
                "finite_oov_contract": "finite_oov_contract" not in reasons,
                "legacy_flattening": False,
                "dtype_boundary_explicit": True,
                "structured_architecture_accepted": (
                    "structured_architecture_output_contract" not in reasons
                ),
                "no_test_vocabulary_fitting": True,
                "training_performed": False,
                "performance_metrics_generated": False,
                "checkpoint_created": False,
            },
            "gate": "MODEL_INTERFACE_PASS" if not reasons else "FAIL",
            "fail_reasons": reasons,
        })
    after = p3_snapshot(p3_dir)
    if before != after:
        raise AssertionError("P4 mutated the frozen P3 artifact tree")
    matrix = {
        "version": "p4-model-interface-matrix-v1",
        "status": "MODEL_INTERFACE_PASS" if all(
            cell["gate"] == "MODEL_INTERFACE_PASS" for cell in cells
        ) else "FAIL",
        "p3_collection_sha256": EXPECTED_P3_COLLECTION_SHA256,
        "p3_tree_unchanged": True,
        "model_interface_version": MODEL_INTERFACE_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "source_code_sha256": source_hashes,
        "aggregate_source_code_sha256": aggregate_source_hash,
        "neural_training_performed": False, "optimizer_steps": 0,
        "threshold_tuning_performed": False,
        "performance_metrics_generated": False,
        "checkpoint_created": False,
        "cells": cells,
        "training_authorized_by_p4": all(
            cell["gate"] == "MODEL_INTERFACE_PASS" for cell in cells
        ),
        "authorization_scope": (
            "future training implementation may be designed against this interface; "
            "P4 itself performs no training"
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "model_interface.json", interface)
    write_json(destination / "architecture_design.json", design)
    write_json(destination / "legacy_compatibility_audit.json", legacy)
    write_json(destination / "p4_matrix.json", matrix)
    print(destination / "p4_matrix.json")
    print(f"[P4 STATUS] {matrix['status']}; NO TRAINING", flush=True)
    return 0 if matrix["status"] == "MODEL_INTERFACE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
