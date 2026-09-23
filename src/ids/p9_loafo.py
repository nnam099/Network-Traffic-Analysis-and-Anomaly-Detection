"""P9 development-only LOAFO execution under immutable P8 cell bindings.

Only the P8-approved official training source and P8/P9 artifacts are inputs.
No target-held-out role is materialized or scored by this module.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from .dataset import (
    DEFAULT_TRAIN_FILE, ROW_ID_COLUMN, SOURCE_ROLE_COLUMN, load_unsw_csvs,
)
from .p2_protocol import continuous_matrix, fit_opaque_vocabulary
from .p3_protocol import sha256_json, train_only_context
from .p4_model_interface import (
    CATEGORICAL_FIELDS, FrozenVocabulary, StructuredBatchAdapter,
    StructuredModelInput,
)
from .p8_loafo_protocol import (
    P8KnownClassModel, file_sha256, known_class_order, surrogate_families,
    verify_frozen_p3,
)
from .target_isolation import CANONICAL_FINGERPRINT_COLUMN as CANONICAL


P8_COMMIT = "397954fd792f5309a26e2a0e784bac1d1a686860"
P8_PROTOCOL_HASH = "62916073ef6048503415d45e2605bbf475b729e032c838a60cc7e94c43768216"
P9_VERSION = "p9-loafo-development-v1"
TARGETS = ("Analysis", "Backdoors", "DoS", "Exploits", "Fuzzers", "Generic",
           "Reconnaissance", "Shellcode", "Worms")
SEEDS = (42, 43, 44)
ALLOWED_ROLES = ("train", "val", "meta_known", "surrogate_ood", "calibration")
SCORERS = ("negative_max_softmax", "nearest_centroid_l2")
TIE_TOLERANCE = 1e-12
TIE_PREFERENCE = "nearest_centroid_l2"
TRAINING_CONFIG = {
    "version": "p9-loafo-known-class-ce-v1", "optimizer": "AdamW",
    "learning_rate": 0.001, "weight_decay": 0.01, "batch_size": 512,
    "max_epochs": 8, "gradient_clip_norm": 1.0,
    "dtype": "torch.float64", "amp": False, "scheduler": "none",
    "class_weighting": "none", "objective": "known_class_cross_entropy",
    "device": "cpu", "patience": 2, "min_delta": 0.0,
    "checkpoint_metric": "sample_mean_val_cross_entropy",
    "checkpoint_tie_break": "lowest_epoch_exact_tie",
    "model_seed_rule": "10000 + split_seed",
    "batch_order": "single_process_numpy_PCG64_epoch_permutation",
    "deterministic_algorithms": True,
}
ROLE_PERMISSIONS = {
    "gradient_fit": {"train"}, "checkpoint_selection": {"val"},
    "centroid_fit": {"train"},
    "scorer_selection": {"meta_known", "surrogate_ood"},
    "threshold_fit": {"calibration"},
}


def require_role(operation: str, role: str) -> None:
    if operation not in ROLE_PERMISSIONS or role not in ROLE_PERMISSIONS[operation]:
        raise PermissionError(f"P9 forbids {operation} consuming {role}")


def checked_input(root: Path, path: Path, kind: str) -> Path:
    """Fail closed on test, P7, symlink, and outside-namespace inputs."""
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink() or not candidate.is_file():
        raise PermissionError("P9 requires an existing regular input file")
    resolved = candidate.resolve()
    if kind == "train":
        allowed = root / "data" / DEFAULT_TRAIN_FILE
        if resolved != allowed or candidate.name != DEFAULT_TRAIN_FILE:
            raise PermissionError("P9 may load only the frozen official training CSV")
    elif kind == "p8":
        if not resolved.is_relative_to(root / "results/loafo/p8"):
            raise PermissionError("P9 may read only frozen P8 artifacts")
    elif kind == "p9":
        if not resolved.is_relative_to(root / "results/loafo/p9"):
            raise PermissionError("P9 may read only its own development artifacts")
    elif kind == "p3_schema":
        if resolved != root / "results/data_quality/p3/schema_v2.json":
            raise PermissionError("P9 may read only frozen P3 schema")
    else:
        raise PermissionError("unapproved P9 input kind")
    return resolved


def read_json(root: Path, relative: str, kind: str) -> dict:
    return json.loads(checked_input(root, Path(relative), kind).read_text())


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_p8(root: Path) -> dict:
    """Read-only P8 bundle and all 27 manifests, before any optimizer exists."""
    root = root.resolve()
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != P8_COMMIT:
        raise ValueError("P9 requires exact frozen P8 repository revision")
    base = "results/loafo/p8/"
    names = ("attack_family_universe.json", "target_support_audit.json",
             "loafo_role_contract.json", "surrogate_policy_design.json",
             "model_contract.json", "external_validation_plan.json",
             "provenance.json")
    bundle = {name: read_json(root, base + name, "p8") for name in names}
    if sha256_json(bundle) != P8_PROTOCOL_HASH:
        raise ValueError("frozen P8 protocol design hash mismatch")
    matrix = read_json(root, base + "loafo_matrix.json", "p8")
    universe = bundle["attack_family_universe.json"]
    if (matrix["status"] != "LOAFO_PROTOCOL_PASS" or
            matrix["protocol_design_sha256"] != P8_PROTOCOL_HASH or
            matrix["passed_cells"] != 27 or matrix["expected_cells"] != 27 or
            len(matrix["cells"]) != 27):
        raise ValueError("P8 27-cell freeze gate mismatch")
    if tuple(universe["attack_families"]) != TARGETS or universe["normal_label"] != "Normal":
        raise ValueError("P8 family universe differs from frozen P9 scope")
    if set((x["target"], x["split_seed"]) for x in matrix["cells"]) != {
            (target, seed) for target in TARGETS for seed in SEEDS}:
        raise ValueError("P8 target x seed matrix mismatch")
    provenance = matrix["source_provenance"]
    if provenance["official_test_csv_accessed"] is not False or provenance["p7_primary_results_accessed"] is not False:
        raise ValueError("P8 provenance violates test boundary")
    for relative, expected in provenance["p8_source_sha256"].items():
        if file_sha256(checked_input(root, Path(relative), "p8") if relative.startswith(base)
                       else root / relative) != expected:
            raise ValueError(f"P8 source binding mismatch: {relative}")
    freeze, _, schema, _ = verify_frozen_p3(root)
    if provenance["p3_development_freeze_sha256"] != sha256_json(freeze):
        raise ValueError("P8/P3 development freeze binding mismatch")
    if schema["sha256"] != provenance["p3_schema_sha256"]:
        raise ValueError("P8/P3 schema binding mismatch")
    return {"matrix": matrix, "universe": universe,
            "model_contract": bundle["model_contract.json"], "schema": schema,
            "provenance": provenance}


def load_manifest(root: Path, cell: dict) -> dict:
    relative = "results/loafo/p8/" + cell["manifest_filename"]
    compressed = checked_input(root, Path(relative), "p8").read_bytes()
    if _sha256(compressed) != cell["manifest_artifact_sha256"]:
        raise ValueError("P8 compressed manifest artifact hash mismatch")
    raw = gzip.decompress(compressed)
    if _sha256(raw) != cell["manifest_sha256"]:
        raise ValueError("P8 manifest content hash mismatch")
    return json.loads(raw)


def verify_manifest(cell: dict, manifest: dict, p8: dict) -> dict:
    """No target-held-out values are materialized; only frozen role lineage is checked."""
    target, seed = cell["target"], cell["split_seed"]
    if (target not in TARGETS or seed not in SEEDS or manifest["target"] != target or
            manifest["split_seed"] != seed or manifest["model_seed"] != 10000 + seed or
            manifest["gate"] != "LOAFO_PROTOCOL_PASS" or
            manifest["official_test_consulted"] is not False or
            manifest["neural_training_performed"] is not False):
        raise ValueError("P8 cell identity/gate mismatch")
    if manifest["feature_schema_sha256"] != p8["schema"]["sha256"]:
        raise ValueError("P8 feature schema mismatch")
    if set(manifest["roles"]) != set(ALLOWED_ROLES):
        raise ValueError("P8 role set mismatch")
    surrogates = surrogate_families(target, p8["universe"])
    classes = known_class_order(target, p8["universe"], surrogates)
    if (manifest["surrogate_families"] != surrogates or
            manifest["known_class_order"] != classes or len(classes) != 7 or
            manifest["num_known_classes"] != 7 or
            manifest["temporary_head"]["out_features"] != 7 or
            target in classes or target in surrogates):
        raise ValueError("P8 surrogate/class ordering mismatch")
    identities, row_ids = {}, {}
    for name, payload in {**manifest["roles"], "target_held_out": manifest["target_held_out"]}.items():
        rows, canonical = payload["row_ids"], payload["canonical_identity_hashes"]
        if not rows or len(rows) != len(canonical) or len(rows) != len(set(rows)):
            raise ValueError(f"P8 {name} role lineage invalid")
        identities[name], row_ids[name] = set(canonical), set(rows)
        if name in ("train", "val", "meta_known", "calibration"):
            if set(payload["label_counts"]) != set(classes):
                raise ValueError(f"P8 known class support missing in {name}")
        elif name == "surrogate_ood" and set(payload["label_counts"]) != set(surrogates):
            raise ValueError("P8 surrogate support mismatch")
        elif name == "target_held_out" and set(payload["label_counts"]) != {target}:
            raise ValueError("P8 target-held-out label mismatch")
    names = list(identities)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if identities[left] & identities[right] or row_ids[left] & row_ids[right]:
                raise ValueError(f"P8 row/canonical role overlap: {left}/{right}")
    audit = manifest["structured_model_input_audit"]
    if (audit["gate"]["gate"] != "PASS" or audit["model_input_cross_identity_final"] != 0 or
            audit["new_scaling_merges"] != 0 or any(manifest["canonical_target_intersections"].values())):
        raise ValueError("P8 structured input/target isolation mismatch")
    center = np.asarray(manifest["scaler_provenance"]["scaler_center"], dtype="<f8")
    scale = np.asarray(manifest["scaler_provenance"]["scaler_scale"], dtype="<f8")
    if (center.shape != (58,) or scale.shape != (58,) or not np.isfinite(center).all() or
            not np.isfinite(scale).all() or bool((scale <= 0).any())):
        raise ValueError("P8 scaler shape/value mismatch")
    if _sha256(np.concatenate((center, scale)).astype("<f8").tobytes()) != manifest[
            "scaler_provenance"]["scaler_center_scale_sha256"]:
        raise ValueError("P8 scaler hash mismatch")
    train_ids = np.asarray(sorted(row_ids["train"]), dtype="<i8")
    if (_sha256(train_ids.tobytes()) != manifest["scaler_provenance"]["scaler_fit_row_ids_sha256"] or
            len(train_ids) != manifest["scaler_provenance"]["scaler_fit_row_count"]):
        raise ValueError("P8 scaler fit population mismatch")
    maps = manifest["vocabulary_provenance"]["vocabulary_maps"]
    if tuple(maps) != CATEGORICAL_FIELDS or sha256_json(maps) != manifest[
            "vocabulary_provenance"]["vocabulary_sha256"]:
        raise ValueError("P8 vocabulary hash/field mismatch")
    vocab = {name: FrozenVocabulary.from_p3_mapping(name, maps[name])
             for name in CATEGORICAL_FIELDS}
    return {"target": target, "split_seed": seed, "model_seed": 10000 + seed,
            "class_order": classes, "surrogate_families": surrogates,
            "p8_manifest_sha256": cell["manifest_sha256"],
            "p8_manifest_artifact_sha256": cell["manifest_artifact_sha256"],
            "scaler_sha256": manifest["scaler_provenance"]["scaler_center_scale_sha256"],
            "vocabulary_sha256": manifest["vocabulary_provenance"]["vocabulary_sha256"],
            "vocabulary_feature_sha256": {name: vocab[name].sha256 for name in CATEGORICAL_FIELDS},
            "role_rows": {name: len(manifest["roles"][name]["row_ids"]) for name in ALLOWED_ROLES},
            "role_identities": {name: len(identities[name]) for name in ALLOWED_ROLES},
            "target_held_out_row_count_from_p8": len(row_ids["target_held_out"]),
            "target_held_out_identity_count_from_p8": len(identities["target_held_out"]),
            "status": "P9_CELL_PROTOCOL_PASS"}


def load_train_context(root: Path, p8: dict) -> dict:
    source = checked_input(root, Path("data") / DEFAULT_TRAIN_FILE, "train")
    if file_sha256(source) != p8["provenance"]["official_train_sha256"]:
        raise ValueError("official train source differs from frozen P8 provenance")
    frame = load_unsw_csvs(source.parent, files=[source], source_role="official_train")
    context = train_only_context(frame)
    if context["feature_schema_sha256"] != p8["schema"]["contract"]["source_schema_sha256"]:
        raise ValueError("P9 official-train source feature schema mismatch")
    return context


def selected_role_frame(context: dict, manifest: dict, role: str):
    if role not in ALLOWED_ROLES:
        raise PermissionError("P9 may not materialize target-held-out or external roles")
    frozen = manifest["roles"][role]
    rows = frozen["row_ids"]
    selected = context["train"].iloc[rows]
    if (selected[ROW_ID_COLUMN].tolist() != rows or
            set(selected[SOURCE_ROLE_COLUMN]) != {"official_train"} or
            [str(int(x)) for x in selected[CANONICAL]] != frozen["canonical_identity_hashes"] or
            selected.attack_cat.value_counts().to_dict() != frozen["label_counts"]):
        raise ValueError(f"P9 source rows differ from frozen {role} lineage")
    return selected


def verify_fold_fit(context: dict, schema: dict, manifest: dict) -> None:
    """Recompute scaler/vocabulary from TRAIN only; never fit on target or surrogates."""
    train = selected_role_frame(context, manifest, "train")
    if fit_opaque_vocabulary(train) != manifest["vocabulary_provenance"]["vocabulary_maps"]:
        raise ValueError("fold-local train vocabulary differs from P8 freeze")
    scaler = RobustScaler().fit(continuous_matrix(train, context, schema))
    center = np.asarray(manifest["scaler_provenance"]["scaler_center"], dtype=np.float64)
    scale = np.asarray(manifest["scaler_provenance"]["scaler_scale"], dtype=np.float64)
    if not np.array_equal(scaler.center_, center) or not np.array_equal(scaler.scale_, scale):
        raise ValueError("fold-local train scaler differs from P8 freeze")


def materialize_role(context: dict, schema: dict, manifest: dict, role: str) -> dict:
    if role not in ALLOWED_ROLES:
        raise PermissionError("P9 role forbidden")
    frame = selected_role_frame(context, manifest, role)
    center = np.asarray(manifest["scaler_provenance"]["scaler_center"], dtype=np.float64)
    scale = np.asarray(manifest["scaler_provenance"]["scaler_scale"], dtype=np.float64)
    raw = continuous_matrix(frame, context, schema)
    scaled = np.asarray((raw - center) / scale, dtype=np.float64)
    maps = manifest["vocabulary_provenance"]["vocabulary_maps"]
    vocabulary = {name: FrozenVocabulary.from_p3_mapping(name, maps[name])
                  for name in CATEGORICAL_FIELDS}
    adapter = StructuredBatchAdapter(vocabulary)
    batch = adapter.transform(scaled, {name: frame[name].tolist() for name in CATEGORICAL_FIELDS})
    labels = frame.attack_cat.tolist()
    targets = None if role == "surrogate_ood" else torch.tensor(
        [manifest["known_class_order"].index(label) for label in labels], dtype=torch.long)
    return {"role": role, "batch": batch, "targets": targets, "labels": labels,
            "row_ids": frame[ROW_ID_COLUMN].tolist(),
            "identities": [str(int(x)) for x in frame[CANONICAL]],
            "rows": len(frame), "identity_count": int(frame[CANONICAL].nunique())}


def set_determinism(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def batch_order(size: int, model_seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng(model_seed + epoch).permutation(size)


def select_batch(batch: StructuredModelInput, positions: np.ndarray) -> StructuredModelInput:
    index = torch.from_numpy(np.asarray(positions, dtype=np.int64))
    return StructuredModelInput(*(getattr(batch, field)[index]
                                  for field in ("continuous", *CATEGORICAL_FIELDS)))


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode() + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_cell(model: P8KnownClassModel, train: dict, val: dict, model_seed: int) -> dict:
    require_role("gradient_fit", train["role"])
    require_role("checkpoint_selection", val["role"])
    if any(parameter.dtype != torch.float64 for parameter in model.parameters()):
        raise TypeError("P9 requires float64 network parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"],
                                  weight_decay=TRAINING_CONFIG["weight_decay"])
    size = TRAINING_CONFIG["batch_size"]
    history, best_loss, best_epoch, best_state, stale = [], float("inf"), 0, None, 0
    steps = 0
    for epoch in range(1, TRAINING_CONFIG["max_epochs"] + 1):
        model.train()
        permutation = batch_order(train["rows"], model_seed, epoch)
        for start in range(0, train["rows"], size):
            indices = permutation[start:start + size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(select_batch(train["batch"], indices))
            loss = F.cross_entropy(logits, train["targets"][indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite P9 train loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clip_norm"])
            optimizer.step()
            steps += 1
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, val["rows"], size):
                indices = np.arange(start, min(start + size, val["rows"]))
                logits = model(select_batch(val["batch"], indices))
                total += F.cross_entropy(logits, val["targets"][indices], reduction="sum").item()
        value = total / val["rows"]
        if not np.isfinite(value):
            raise ValueError("nonfinite P9 validation loss")
        history.append({"epoch": epoch, "development_val_cross_entropy": value})
        if value < best_loss - TRAINING_CONFIG["min_delta"]:
            best_loss, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= TRAINING_CONFIG["patience"]:
            break
    if best_state is None:
        raise ValueError("P9 failed to select validation checkpoint")
    model.load_state_dict(best_state)
    return {"state_dict": best_state, "selected_epoch": best_epoch,
            "validation_value": best_loss, "epoch_history": history,
            "optimizer_steps": steps, "model_state_sha256": state_hash(best_state)}


def representations_logits(model: P8KnownClassModel, role: dict) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    reps, logits = [], []
    with torch.no_grad():
        for start in range(0, role["rows"], TRAINING_CONFIG["batch_size"]):
            indices = np.arange(start, min(start + TRAINING_CONFIG["batch_size"], role["rows"]))
            batch = select_batch(role["batch"], indices)
            representation = model.backbone(batch)
            output = model.head(representation)
            if representation.dtype != torch.float64 or output.dtype != torch.float64:
                raise TypeError("P9 representation/logits must remain float64")
            reps.append(representation.numpy())
            logits.append(output.numpy())
    return np.concatenate(reps), np.concatenate(logits)


def softmax_uncertainty(logits: np.ndarray, class_count: int) -> np.ndarray:
    values = np.asarray(logits)
    if values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != class_count:
        raise TypeError("P9 softmax scorer requires float64[rows,known_classes]")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite logits")
    stable = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(stable)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.asarray(1.0 - probabilities.max(axis=1), dtype=np.float64)


def centroid_hash(centroids: np.ndarray, class_order: list[str], checkpoint_hash: str) -> str:
    state = np.asarray(centroids)
    if state.dtype != np.float64 or state.shape != (len(class_order), 32):
        raise TypeError("P9 centroid state shape/dtype mismatch")
    header = {"scorer_type": "nearest_centroid_l2", "class_order": class_order,
              "representation_dim": 32, "checkpoint_state_sha256": checkpoint_hash,
              "formula_version": "minimum-euclidean-distance-v1"}
    return _sha256(json.dumps(header, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":")).encode() +
                   np.ascontiguousarray(state.astype("<f8")).tobytes())


def fit_centroids(representations: np.ndarray, labels: list[str], *, role: str,
                  class_order: list[str], checkpoint_hash: str) -> tuple[np.ndarray, str]:
    require_role("centroid_fit", role)
    values = np.asarray(representations)
    if values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != 32:
        raise TypeError("P9 centroids require float64[rows,32]")
    if len(labels) != len(values) or not np.isfinite(values).all():
        raise ValueError("invalid centroid input")
    centroids = []
    observed = np.asarray(labels)
    for label in class_order:
        selected = values[observed == label]
        if len(selected) == 0:
            raise ValueError(f"missing train class for centroid: {label}")
        centroids.append(selected.mean(axis=0, dtype=np.float64))
    state = np.ascontiguousarray(np.stack(centroids).astype("<f8"))
    return state, centroid_hash(state, class_order, checkpoint_hash)


def centroid_distance(representations: np.ndarray, centroids: np.ndarray,
                      class_count: int) -> np.ndarray:
    values, centers = np.asarray(representations), np.asarray(centroids)
    if (values.dtype != np.float64 or centers.dtype != np.float64 or values.ndim != 2 or
            values.shape[1] != 32 or centers.shape != (class_count, 32)):
        raise TypeError("P9 centroid score shape/dtype mismatch")
    if not np.isfinite(values).all() or not np.isfinite(centers).all():
        raise ValueError("nonfinite centroid input")
    return np.asarray(np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2).min(axis=1),
                      dtype=np.float64)


def identity_means(scores: np.ndarray, identities: list[str]) -> tuple[list[str], np.ndarray]:
    values = np.asarray(scores)
    if values.dtype != np.float64 or values.ndim != 1 or len(values) != len(identities):
        raise TypeError("P9 identity aggregation requires aligned float64 rows")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite identity scores")
    grouped: dict[str, list[float]] = {}
    for identity, score in zip(identities, values, strict=True):
        grouped.setdefault(str(identity), []).append(float(score))
    keys = sorted(grouped)
    means = np.asarray([math.fsum(sorted(grouped[key])) / len(grouped[key]) for key in keys],
                       dtype=np.float64)
    return keys, means


def development_auroc(known: np.ndarray, surrogate: np.ndarray) -> float:
    left, right = np.asarray(known), np.asarray(surrogate)
    if (left.dtype != np.float64 or right.dtype != np.float64 or not len(left) or
            not len(right) or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("P9 AUROC requires nonempty finite identity scores")
    labels = np.concatenate((np.zeros(len(left), dtype=np.int8), np.ones(len(right), dtype=np.int8)))
    return float(roc_auc_score(labels, np.concatenate((left, right))))


def choose_scorer(softmax_auroc: float, centroid_auroc: float) -> dict:
    if not np.isfinite(softmax_auroc) or not np.isfinite(centroid_auroc):
        raise ValueError("nonfinite scorer-selection statistic")
    margin = centroid_auroc - softmax_auroc
    tie = abs(margin) <= TIE_TOLERANCE
    return {"selected_scorer": TIE_PREFERENCE if tie or margin > 0 else SCORERS[0],
            "selection_margin_centroid_minus_softmax": margin,
            "tie_break_used": tie, "tie_tolerance": TIE_TOLERANCE,
            "tie_preference": TIE_PREFERENCE}


def fit_identity_threshold(row_scores: np.ndarray, identities: list[str], *, role: str) -> dict:
    require_role("threshold_fit", role)
    keys, scores = identity_means(row_scores, identities)
    if not len(scores):
        raise ValueError("empty calibration identity population")
    threshold = float(np.quantile(scores, 0.99, method="higher"))
    return {"calibration_row_count": len(row_scores), "calibration_identity_count": len(keys),
            "quantile": 0.99, "numpy_method": "higher", "threshold": threshold,
            "comparison": "score > threshold", "known_only": True,
            "identity_scores_sha256": _sha256(np.asarray(scores, dtype="<f8").tobytes())}


def score_summary(values: np.ndarray) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(data), "min": float(data.min()), "max": float(data.max()),
            "mean": float(data.mean())}


def source_hashes(root: Path) -> dict:
    names = ("src/ids/p9_loafo.py", "scripts/run_p9_loafo.py",
             "src/ids/p8_loafo_protocol.py", "src/ids/p4_model_interface.py")
    return {name: file_sha256(root / name) for name in names}
