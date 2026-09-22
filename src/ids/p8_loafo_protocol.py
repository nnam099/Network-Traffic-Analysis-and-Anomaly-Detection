"""P8 train-only, no-training LOAFO protocol and fail-closed data audit.

The consumed official-test CSV and P7 primary results are not inputs to this
module. Historical test support is read only from the already-frozen P3 audit.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import ROW_ID_COLUMN, SOURCE_ROLE_COLUMN
from .p3_protocol import (
    _role_lineage, development_model_audit, sha256_json,
)
from .p4_model_interface import (
    CATEGORICAL_FIELDS, ContinuousEncoder, FrozenVocabulary, MODEL_DTYPE,
    StructuredModelInput,
)
from .target_isolation import (
    CANONICAL_FINGERPRINT_COLUMN as CANONICAL, _split_canonical_known,
)


VERSION = "loafo-p8-train-only-protocol-v1"
SEEDS = (42, 43, 44)
ROLE_NAMES = ("train", "val", "meta_known", "calibration", "surrogate_ood")
NORMAL = "Normal"
SURROGATE_POLICY = "S2_top_two_remaining_train_identity_support_v1"
CONSUMED_TEST_LABEL = "REUSED TEST — NOT INDEPENDENT EXTERNAL VALIDATION"
P3_FREEZE_SHA256 = "1443746a4f024b3d7c1569ba9bbd31987933b5f9e2fb0ce6507ba5e57ddbab21"


class P8ReferenceBackbone(nn.Module):
    """Versioned fixed-4 embedding baseline, matching the concrete P4/P5 44-D fusion.

    The generic P4 vocabulary-size heuristic is deliberately not reused: it
    would silently enlarge the proto embedding when LOAFO changes class mix.
    This class defines architecture only; it does not fit any parameter.
    """

    def __init__(self, vocabularies: dict[str, FrozenVocabulary]):
        super().__init__()
        if tuple(vocabularies) != CATEGORICAL_FIELDS:
            raise ValueError("P8 requires ordered proto/service/state vocabularies")
        self.vocabularies = dict(vocabularies)
        self.continuous_encoder = ContinuousEncoder(hidden_dim=64, output_dim=32)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocabulary.cardinality, 4,
                               padding_idx=vocabulary.pad_index, dtype=MODEL_DTYPE)
            for name, vocabulary in vocabularies.items()
        })
        self.fusion = nn.Sequential(
            nn.Linear(44, 64, dtype=MODEL_DTYPE),
            nn.LayerNorm(64, dtype=MODEL_DTYPE), nn.GELU(),
            nn.Linear(64, 32, dtype=MODEL_DTYPE),
        )
        self.representation_dim = 32

    def forward(self, inputs: StructuredModelInput) -> torch.Tensor:
        if not isinstance(inputs, StructuredModelInput):
            raise TypeError("P8 requires the P4 structured input, not a flat tensor")
        categorical = []
        for name in CATEGORICAL_FIELDS:
            indices = getattr(inputs, name)
            if bool((indices < 0).any()) or bool((indices >= self.vocabularies[name].cardinality).any()):
                raise ValueError(f"{name} index outside frozen vocabulary")
            categorical.append(self.embeddings[name](indices))
        return self.fusion(torch.cat((self.continuous_encoder(inputs.continuous),
                                      *categorical), dim=1))


class P8KnownClassModel(nn.Module):
    """Architecture declaration with a cell-specific head; no training methods."""

    def __init__(self, vocabularies: dict[str, FrozenVocabulary], class_order: list[str]):
        super().__init__()
        if not class_order or class_order[0] != NORMAL or len(class_order) != len(set(class_order)):
            raise ValueError("invalid deterministic known-class order")
        self.class_order = tuple(class_order)
        self.backbone = P8ReferenceBackbone(vocabularies)
        self.head = nn.Linear(32, len(class_order), dtype=MODEL_DTYPE)

    def forward(self, inputs: StructuredModelInput) -> torch.Tensor:
        return self.head(self.backbone(inputs))


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"P8 artifact already exists with different content: {path}")
        return
    with path.open("xb") as output:
        output.write(raw)


def verify_frozen_p3(root: Path) -> tuple[dict, dict, dict, dict]:
    """Verify upstream scientific bindings without opening any test CSV."""
    base = root / "results/data_quality/p3"
    freeze = json.loads((base / "development_freeze.json").read_text())
    if sha256_json(freeze) != P3_FREEZE_SHA256 or freeze["official_test_consulted"] is not False:
        raise ValueError("P3 development freeze binding mismatch")
    for relative, expected in freeze["code_source_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"P3 source binding mismatch: {relative}")
    ambiguity = json.loads((base / "train_only_ambiguity.json").read_text())
    schema = json.loads((base / "schema_v2.json").read_text())
    if sha256_json(ambiguity) != freeze["train_ambiguity_report_sha256"]:
        raise ValueError("P3 ambiguity artifact binding mismatch")
    if schema["sha256"] != freeze["schema_sha256"]:
        raise ValueError("P3 schema binding mismatch")
    for cell in freeze["cells"]:
        path = base / cell["manifest_filename"]
        if hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() != cell["manifest_sha256"]:
            raise ValueError(f"P3 manifest binding mismatch: {path}")
    p4 = json.loads((root / "results/model_design/p4/p4_matrix.json").read_text())
    if p4["status"] != "MODEL_INTERFACE_PASS" or p4["p3_collection_sha256"] != P3_FREEZE_SHA256:
        raise ValueError("P4 interface binding mismatch")
    for relative, expected in p4["source_code_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"P4 source binding mismatch: {relative}")
    # This is frozen historical metadata, not a fresh read of consumed test rows.
    historical = json.loads((base / "official_test_external_audit.json").read_text())
    if historical["development_freeze_sha256"] != P3_FREEZE_SHA256:
        raise ValueError("P3 historical test-support provenance mismatch")
    return freeze, ambiguity, schema, historical


def verified_family_universe(context: dict, ambiguity: dict, historical: dict) -> dict:
    """Derive labels from frozen source metadata, then cross-check official train."""
    train = context["train"]
    meta = ambiguity["per_family"]
    labels = sorted(meta)
    if NORMAL not in meta or sorted(train.attack_cat.unique().tolist()) != labels:
        raise ValueError("actual official-train labels differ from frozen P3 metadata")
    if set(historical["per_family"]) != set(meta):
        raise ValueError("historical support labels differ from train metadata")
    families = {}
    for label in labels:
        frame = train[train.attack_cat == label]
        frozen = meta[label]
        if len(frame) != frozen["original_rows"]:
            raise ValueError(f"frozen source row count mismatch: {label}")
        hist = historical["per_family"][label]
        families[label] = {
            "is_attack_family": label != NORMAL,
            "official_train_source_rows": len(frame),
            "official_train_source_canonical_identities": int(frame[CANONICAL].nunique()),
            "official_train_ambiguity_excluded_rows": frozen["excluded_rows"],
            "official_train_retained_rows": frozen["remaining_rows"],
            "official_train_retained_canonical_identities": frozen[
                "remaining_unique_canonical_identities"],
            "historical_consumed_official_test_rows": sum(v["rows"] for v in hist.values()),
            "historical_consumed_official_test_clean_rows": hist["CLEAN"]["rows"],
            "historical_consumed_official_test_clean_canonical_identities": hist[
                "CLEAN"]["unique_canonical_identities"],
        }
    attack_labels = [label for label in labels if label != NORMAL]
    if len(attack_labels) < 4 or any(families[x]["official_train_retained_canonical_identities"] == 0
                                     for x in attack_labels):
        raise ValueError("insufficient attack-family universe for S2 LOAFO")
    return {
        "version": VERSION, "source": "P3 frozen official-train ambiguity metadata, verified against train",
        "historical_test_support_source": "P3 frozen official_test_external_audit.json only",
        "official_test_csv_accessed": False, "normal_label": NORMAL,
        "attack_families": attack_labels, "families": families,
    }


def surrogate_families(target: str, universe: dict, *, n: int = 2) -> list[str]:
    attacks = universe["attack_families"]
    if target not in attacks:
        raise ValueError("target is not a verified attack family")
    candidates = sorted((x for x in attacks if x != target), key=lambda x: (
        -universe["families"][x]["official_train_retained_canonical_identities"], x
    ))
    if len(candidates) < n:
        raise ValueError("too few remaining attack families for surrogate policy")
    return candidates[:n]


def known_class_order(target: str, universe: dict, surrogates: list[str]) -> list[str]:
    attacks = set(universe["attack_families"])
    if target not in attacks or len(surrogates) != len(set(surrogates)) or not set(surrogates) <= attacks - {target}:
        raise ValueError("invalid target/surrogate family assignment")
    return [NORMAL, *sorted(attacks - {target} - set(surrogates))]


def support_tier(identities: int) -> str:
    """Descriptive train-only flag; it never excludes a target."""
    if identities >= 1000:
        return "HIGH SUPPORT"
    if identities >= 200:
        return "MODERATE SUPPORT"
    if identities >= 50:
        return "SMALL-N"
    return "INSUFFICIENT"


def construct_roles(context: dict, ambiguous: set[int], target: str, seed: int,
                    universe: dict) -> tuple[dict, dict, list[str], list[str]]:
    if seed not in SEEDS:
        raise ValueError("unapproved split seed")
    surrogates = surrogate_families(target, universe)
    class_order = known_class_order(target, universe, surrogates)
    retained = context["train"][~context["train"][CANONICAL].isin(ambiguous)]
    target_frame = retained[retained.attack_cat == target]
    known = retained[retained.attack_cat.isin(class_order)]
    surrogate = retained[retained.attack_cat.isin(surrogates)]
    if target_frame.empty or known.empty or surrogate.empty:
        raise ValueError("empty target, known, or surrogate population")
    parts = _split_canonical_known(known, seed)
    roles = {
        "train": parts["backbone_train"], "val": parts["validation"],
        "meta_known": parts["meta_known"], "calibration": parts["calibration"],
        "surrogate_ood": surrogate,
    }
    population = {"retained": retained, "target_train": target_frame,
                  "target_identity_hashes": set(int(x) for x in target_frame[CANONICAL])}
    return population, roles, class_order, surrogates


def validate_cell(context: dict, schema: dict, population: dict, roles: dict,
                  target: str, class_order: list[str], surrogates: list[str]) -> tuple[dict, dict]:
    """Assert row/canonical/model-input separation and fit-only provenance."""
    target_ids = population["target_identity_hashes"]
    frames = {**roles, "target_held_out": population["target_train"]}
    identity_sets = {name: set(int(x) for x in frame[CANONICAL]) for name, frame in frames.items()}
    row_sets = {name: set(int(x) for x in frame[ROW_ID_COLUMN]) for name, frame in frames.items()}
    overlap = {}
    for left, right in itertools.combinations(frames, 2):
        overlap[f"{left}__{right}"] = len(identity_sets[left] & identity_sets[right])
        if row_sets[left] & row_sets[right]:
            raise ValueError(f"row overlap: {left}/{right}")
    if any(overlap.values()) or any(target_ids & identity_sets[name] for name in ROLE_NAMES):
        raise ValueError("target or cross-role canonical isolation failure")
    for name in ("train", "val", "meta_known", "calibration"):
        if not set(roles[name].attack_cat) <= set(class_order) or target in set(roles[name].attack_cat):
            raise ValueError(f"forbidden known label in {name}")
        if set(roles[name].attack_cat) != set(class_order):
            raise ValueError(f"known class missing in {name}")
    if set(roles["surrogate_ood"].attack_cat) != set(surrogates):
        raise ValueError("surrogate role composition mismatch")
    if any(set(frame[SOURCE_ROLE_COLUMN]) != {"official_train"} for frame in frames.values()):
        raise ValueError("non-train source in P8 development roles")
    audit, fitting, vocab = development_model_audit(context, schema, population, roles)
    if audit["gate"]["gate"] != "PASS":
        raise ValueError(f"structured input audit failed: {audit['gate']['fail_reasons']}")
    train_rows = sorted(row_sets["train"])
    expected_hash = hashlib.sha256(np.asarray(train_rows, dtype="<i8").tobytes()).hexdigest()
    if fitting["scaler_fit_row_ids_sha256"] != expected_hash or fitting["scaler_fit_row_count"] != len(train_rows):
        raise ValueError("scaler fit row provenance mismatch")
    if fitting["vocabulary_sha256"] != sha256_json(vocab) or fitting["target_used"] is not False:
        raise ValueError("vocabulary/target fit provenance mismatch")
    if tuple(vocab) != CATEGORICAL_FIELDS:
        raise ValueError("P8 vocabulary field order differs from P4 interface")
    for name, mapping in vocab.items():
        frozen = FrozenVocabulary.from_p3_mapping(name, mapping)
        if frozen.encode("__P8_UNSEEN_TOKEN__") != 1 or frozen.pad_index != 0:
            raise ValueError("P4 OOV/PAD semantics mismatch")
    return audit, fitting


def cell_manifest(context: dict, schema: dict, universe: dict, population: dict,
                  roles: dict, target: str, seed: int, class_order: list[str],
                  surrogates: list[str], audit: dict, fitting: dict,
                  provenance: dict) -> dict:
    return {
        "version": VERSION, "target": target, "split_seed": seed,
        "model_seed": 10000 + seed, "surrogate_policy": SURROGATE_POLICY,
        "known_attack_families": class_order[1:], "known_class_order": class_order,
        "num_known_classes": len(class_order), "temporary_head": {
            "type": "Linear", "in_features": 32, "out_features": len(class_order),
            "dtype": "torch.float64", "target_in_labels": False,
        },
        "surrogate_families": surrogates,
        "roles": {name: _role_lineage(frame) for name, frame in roles.items()},
        "target_held_out": _role_lineage(population["target_train"]),
        "canonical_role_intersections": audit["canonical_role_intersections"],
        "canonical_target_intersections": audit["canonical_target_intersections"],
        "structured_model_input_audit": audit,
        "vocabulary_provenance": {k: v for k, v in fitting.items() if k.startswith("vocabulary_")},
        "scaler_provenance": {k: v for k, v in fitting.items() if k.startswith("scaler_")},
        "source_provenance": provenance, "feature_schema_sha256": schema["sha256"],
        "canonical_contract_version": context["canonical_identity_version"],
        "official_test_consulted": False, "neural_training_performed": False,
        "performance_metrics_generated": False, "gate": "LOAFO_PROTOCOL_PASS",
    }
