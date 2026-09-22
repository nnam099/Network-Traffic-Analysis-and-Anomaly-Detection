"""Fail-closed, development-only schema-V2 training protocol.

Only explicit P3 train/val row IDs can reach the neural path. This module
never opens an official-test file or artifact and never computes test metrics.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import (DEFAULT_TRAIN_FILE, ROW_ID_COLUMN, SOURCE_ROLE_COLUMN,
                      KNOWN_ATTACK_CATS, load_unsw_csvs)
from .p2_protocol import continuous_matrix
from .p3_protocol import train_only_context, sha256_json
from .p4_model_interface import (CATEGORICAL_FIELDS, RepresentationBackbone,
                                 StructuredBatchAdapter, StructuredModelInput,
                                 bound_model_config, stable_hash,
                                 vocabularies_from_manifest)
from .target_isolation import CANONICAL_FINGERPRINT_COLUMN


P3_HASH = "1443746a4f024b3d7c1569ba9bbd31987933b5f9e2fb0ce6507ba5e57ddbab21"
TARGETS = ("Fuzzers", "Analysis", "Backdoors", "Shellcode", "Worms")
SPLIT_SEEDS = (42, 43, 44)
ROLES = ("train", "val", "meta_known", "calibration", "surrogate_ood",
         "target_development_held_out", "official_test")
PERMISSIONS = {
    "scaler_vocabulary_fit": ["train"],
    "gradient_fit": ["train"],
    "checkpoint_selection": ["val"],
    "scorer_parameter_fit": ["train"],
    "scorer_family_selection": ["meta_known", "surrogate_ood"],
    "threshold_fit": ["calibration"],
    "final_external_evaluation": [],
}
TRAINING_CONFIG = {
    "version": "p5-known-family-ce-v1", "objective": "known_family_cross_entropy",
    "class_order": list(KNOWN_ATTACK_CATS), "temporary_head": "linear_32_to_5_float64",
    "backbone": "p4-structured-mlp-embedding-representation-v1",
    "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.01,
    "batch_size": 512, "max_epochs": 8, "gradient_clip_norm": 1.0,
    "initialization": "pytorch_default_after_model_seed", "class_weighting": "none",
    "scheduler": "none", "early_stopping_metric": "sample_mean_val_cross_entropy",
    "patience": 2, "min_delta": 0.0, "tie_break": "lowest_epoch_exact_tie",
    "model_seed_rule": "10000 + split_seed", "device": "cpu",
    "dtype": "torch.float64", "amp": False, "deterministic_algorithms": True,
    "dataloader": "single_process_numpy_PCG64_epoch_permutation",
    "smoke_cell": {"target": "Fuzzers", "split_seed": 42, "model_seed": 10042},
}


def require_role(operation: str, role: str) -> None:
    if operation not in PERMISSIONS or role not in PERMISSIONS[operation]:
        raise PermissionError(f"P5 forbids {operation} consuming {role}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(root: Path) -> tuple[dict, str]:
    names = ("src/ids/p5_training.py", "scripts/train_p5_development.py",
             "src/ids/p4_model_interface.py")
    hashes = {name: file_hash(root / name) for name in names}
    return hashes, stable_hash(hashes)


def checked_output(root: Path, relative: str) -> Path:
    base = (root / "results/training_protocol/p5").resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or any(part in {"official_test", "test_results"}
                                                for part in path.parts):
        raise PermissionError("P5 output must remain inside its development namespace")
    return path


def write_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"immutable P5 artifact differs: {path}")
        return
    with path.open("xb") as output:
        output.write(raw)


def json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False).encode() + b"\n"


def _identity_set(role: dict) -> set[str]:
    return set(role["canonical_identity_hashes"])


def validate_roles(manifest: dict) -> dict:
    if set(manifest["roles"]) != set(ROLES[:5]):
        raise ValueError("P3 role set differs from frozen contract")
    if manifest["official_test_consulted"] is not False:
        raise ValueError("test-informed P3 manifest refused")
    if manifest["development_gate"]["gate"] != "PASS":
        raise ValueError("P3 development gate failed")
    all_roles = dict(manifest["roles"])
    all_roles["target_development_held_out"] = manifest["target_development_held_out"]
    sets = {}
    row_sets = {}
    for name, role in all_roles.items():
        ids, identities = role["row_ids"], role["canonical_identity_hashes"]
        if len(ids) != len(identities) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid P3 {name} row/identity alignment")
        sets[name] = _identity_set(role)
        row_sets[name] = set(ids)
    names = list(all_roles)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if sets[left] & sets[right] or row_sets[left] & row_sets[right]:
                raise ValueError(f"P3 role overlap: {left}/{right}")
    return {name: {"rows": len(role["row_ids"]),
                   "canonical_identities": len(sets[name]),
                   "label_counts": role["label_counts"]}
            for name, role in all_roles.items()}


def validate_cell(root: Path, target: str, split_seed: int, model_seed: int) -> dict:
    if target not in TARGETS or split_seed not in SPLIT_SEEDS:
        raise ValueError("unsupported P5 target/split seed")
    if model_seed != 10000 + split_seed:
        raise ValueError("model seed must be distinct and equal 10000 + split seed")
    p3_dir = root / "results/data_quality/p3"
    p4_dir = root / "results/model_design/p4"
    freeze = json.loads((p3_dir / "development_freeze.json").read_text())
    if sha256_json(freeze) != P3_HASH or freeze["official_test_consulted"] is not False:
        raise ValueError("P3 frozen collection hash/status mismatch")
    p4 = json.loads((p4_dir / "p4_matrix.json").read_text())
    if p4["status"] != "MODEL_INTERFACE_PASS" or p4["p3_collection_sha256"] != P3_HASH:
        raise ValueError("P4 model interface not approved for P3 collection")
    for name, digest in p4["source_code_sha256"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"frozen P4 source changed: {name}")
    p3_entry = next((x for x in freeze["cells"] if x["target"] == target
                     and x["seed"] == split_seed), None)
    p4_entry = next((x for x in p4["cells"] if x["target"] == target
                     and x["seed"] == split_seed), None)
    if p3_entry is None or p4_entry is None or p4_entry["gate"] != "MODEL_INTERFACE_PASS":
        raise ValueError("cell absent or failed in P3/P4")
    manifest_path = p3_dir / p3_entry["manifest_filename"]
    if not manifest_path.resolve().is_relative_to((p3_dir / "manifests").resolve()):
        raise PermissionError("manifest outside frozen development directory")
    raw = gzip.decompress(manifest_path.read_bytes())
    manifest_hash = hashlib.sha256(raw).hexdigest()
    if manifest_hash != p3_entry["manifest_sha256"] or manifest_hash != p4_entry["p3_manifest_sha256"]:
        raise ValueError("P3 manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest["target"] != target or manifest["seed"] != split_seed:
        raise ValueError("P3 manifest cell mismatch")
    counts = validate_roles(manifest)
    config = p4_entry["bound_model_config"]
    if stable_hash({k: v for k, v in config.items() if k != "model_config_sha256"}) != config["model_config_sha256"]:
        raise ValueError("P4 bound config hash mismatch")
    if config["p3_manifest_sha256"] != manifest_hash:
        raise ValueError("P4 manifest binding mismatch")
    schema = json.loads((p3_dir / "schema_v2.json").read_text())
    rebuilt = bound_model_config(
        manifest, manifest_sha256=manifest_hash, p3_collection_sha256=P3_HASH,
        schema=schema, source_code_sha256=p4["aggregate_source_code_sha256"],
    )
    if rebuilt != config:
        raise ValueError("P4 bound config differs from frozen P3/P4 reconstruction")
    vocab = vocabularies_from_manifest(manifest)
    if {name: item.sha256 for name, item in vocab.items()} != config["vocabulary_sha256"]:
        raise ValueError("P4 vocabulary hash mismatch")
    scaler = manifest["scaler_provenance"]
    values = np.concatenate((np.asarray(scaler["scaler_center"], dtype=np.float64),
                             np.asarray(scaler["scaler_scale"], dtype=np.float64)))
    if hashlib.sha256(values.astype("<f8").tobytes()).hexdigest() != config["scaler_center_scale_sha256"]:
        raise ValueError("P4 scaler hash mismatch")
    if config["architecture"]["parameter_dtype"] != "torch.float64" or config["architecture"]["representation_dim"] != 32:
        raise ValueError("P4 dtype/architecture mismatch")
    source_files, source_digest = source_hash(root)
    output = checked_output(root, f"dry_runs/{target.lower()}_seed{split_seed}.json")
    return {"target": target, "split_seed": split_seed, "model_seed": model_seed,
            "p3_manifest_sha256": manifest_hash, "p4_model_config_sha256": config["model_config_sha256"],
            "training_config_sha256": stable_hash(TRAINING_CONFIG),
            "p3_collection_sha256": P3_HASH, "source_hashes": source_files,
            "source_hash": source_digest, "vocabulary_sha256": config["vocabulary_sha256"],
            "scaler_sha256": config["scaler_center_scale_sha256"],
            "roles": counts, "device": TRAINING_CONFIG["device"],
            "dtype": TRAINING_CONFIG["dtype"], "output_path": str(output),
            "official_test_access": False, "optimizer_created": False,
            "checkpoint_written": False, "status": "DRY_RUN_PASS"}


def _load_train_val(root: Path, manifest: dict) -> tuple:
    require_role("gradient_fit", "train")
    require_role("checkpoint_selection", "val")
    source = root / "data" / DEFAULT_TRAIN_FILE
    if source.name != DEFAULT_TRAIN_FILE or source.is_symlink() or not source.is_file():
        raise PermissionError("only the regular official training CSV is permitted")
    if file_hash(source) != manifest["source_dataset_hashes"]["official_train_sha256"]:
        raise ValueError("official training CSV hash mismatch")
    frame = load_unsw_csvs(source.parent, files=[source], source_role="official_train")
    context = train_only_context(frame)
    if context["feature_schema_sha256"] != manifest["source_feature_schema_sha256"]:
        raise ValueError("source feature schema mismatch")
    schema = json.loads((root / "results/data_quality/p3/schema_v2.json").read_text())
    if schema["sha256"] != manifest["feature_schema_sha256"]:
        raise ValueError("P3 schema mismatch")
    vocab = vocabularies_from_manifest(manifest)
    adapter = StructuredBatchAdapter(vocab)
    center = np.asarray(manifest["scaler_provenance"]["scaler_center"], dtype=np.float64)
    scale = np.asarray(manifest["scaler_provenance"]["scaler_scale"], dtype=np.float64)
    result = []
    for role in ("train", "val"):
        row_ids = manifest["roles"][role]["row_ids"]
        selected = context["train"].iloc[row_ids]
        if selected[ROW_ID_COLUMN].tolist() != row_ids or set(selected[SOURCE_ROLE_COLUMN]) != {"official_train"}:
            raise ValueError(f"{role} row/source mismatch")
        if [str(x) for x in selected[CANONICAL_FINGERPRINT_COLUMN]] != manifest["roles"][role]["canonical_identity_hashes"]:
            raise ValueError(f"{role} canonical identity mismatch")
        labels = selected["attack_cat"].tolist()
        if any(label not in KNOWN_ATTACK_CATS for label in labels):
            raise ValueError(f"{role} contains target/surrogate label")
        if dict(selected["attack_cat"].value_counts()) != manifest["roles"][role]["label_counts"]:
            raise ValueError(f"{role} label counts mismatch")
        continuous = continuous_matrix(selected, context, schema)
        scaled = np.asarray((continuous - center) / scale, dtype=np.float64)
        categories = {name: selected[name].tolist() for name in CATEGORICAL_FIELDS}
        batch = adapter.transform(scaled, categories)
        targets = torch.tensor([KNOWN_ATTACK_CATS.index(x) for x in labels], dtype=torch.long)
        result.append((batch, targets))
    return vocab, result[0], result[1]


class TrainingOnlyModel(nn.Module):
    def __init__(self, vocabularies: dict):
        super().__init__()
        self.backbone = RepresentationBackbone(vocabularies)
        self.training_head = nn.Linear(32, len(KNOWN_ATTACK_CATS), dtype=torch.float64)

    def forward(self, batch: StructuredModelInput) -> torch.Tensor:
        representation = self.backbone(batch)
        if representation.dtype != torch.float64:
            raise TypeError("P5 representation must stay float64")
        return self.training_head(representation)


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


def select_batch(batch: StructuredModelInput, indices: np.ndarray) -> StructuredModelInput:
    index = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    return StructuredModelInput(*(getattr(batch, name)[index]
                                  for name in ("continuous", *CATEGORICAL_FIELDS)))


def _environment(root: Path) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda_version": torch.version.cuda,
            "hardware": platform.processor() or platform.machine(), "platform": platform.platform(),
            "device": "cpu", "python_seed": None, "numpy_seed": None,
            "pytorch_seed": None, "cuda_seed": None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "source_commit": commit}


def _state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode() + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_smoke(root: Path, run_tag: str) -> dict:
    if run_tag not in {"smoke-a", "smoke-b"}:
        raise ValueError("P5 authorizes only two named smoke repetitions")
    dry = validate_cell(root, "Fuzzers", 42, 10042)
    manifest_path = root / "results/data_quality/p3/manifests/fuzzers_seed42.json.gz"
    manifest = json.loads(gzip.decompress(manifest_path.read_bytes()))
    set_determinism(10042)
    vocab, train_data, val_data = _load_train_val(root, manifest)
    model = TrainingOnlyModel(vocab)
    if any(parameter.dtype != torch.float64 for parameter in model.parameters()):
        raise TypeError("P5 model parameters must all remain float64")
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"],
                                  weight_decay=TRAINING_CONFIG["weight_decay"])
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    train_batch, train_targets = train_data
    val_batch, val_targets = val_data
    size = TRAINING_CONFIG["batch_size"]
    for epoch in range(1, TRAINING_CONFIG["max_epochs"] + 1):
        model.train()
        for indices in np.array_split(batch_order(len(train_targets), 10042, epoch),
                                      range(size, len(train_targets), size)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(select_batch(train_batch, indices))
            loss = F.cross_entropy(logits, train_targets[indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clip_norm"])
            optimizer.step()
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(val_targets), size):
                indices = np.arange(start, min(start + size, len(val_targets)))
                logits = model(select_batch(val_batch, indices))
                total += F.cross_entropy(logits, val_targets[indices], reduction="sum").item()
        val_loss = total / len(val_targets)
        if not np.isfinite(val_loss):
            raise ValueError("nonfinite validation loss")
        history.append({"epoch": epoch, "development_val_cross_entropy": val_loss})
        if val_loss < best_loss - TRAINING_CONFIG["min_delta"]:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= TRAINING_CONFIG["patience"]:
            break
    if best_state is None:
        raise AssertionError("no checkpoint selected")
    run_id = f"p5-fuzzers-s42-m10042-{run_tag}"
    checkpoint = checked_output(root, f"checkpoints/{run_id}.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError("smoke checkpoint already exists; immutable run")
    payload = {"state_dict": best_state, "provenance": {
        "run_id": run_id, "target": "Fuzzers", "split_seed": 42, "model_seed": 10042,
        "p3_manifest_sha256": dry["p3_manifest_sha256"],
        "p4_model_config_sha256": dry["p4_model_config_sha256"],
        "training_config_sha256": dry["training_config_sha256"],
        "vocabulary_sha256": dry["vocabulary_sha256"],
        "scaler_sha256": dry["scaler_sha256"], "source_hash": dry["source_hash"],
        "epoch": best_epoch, "validation_selection_value": best_loss}}
    torch.save(payload, checkpoint)
    environment = _environment(root)
    for key in ("python_seed", "numpy_seed", "pytorch_seed", "cuda_seed"):
        environment[key] = 10042
    run = {"version": "p5-smoke-run-v1", "run_id": run_id,
           "target": "Fuzzers", "split_seed": 42, "model_seed": 10042,
           "p3_manifest_sha256": dry["p3_manifest_sha256"],
           "p4_model_config_sha256": dry["p4_model_config_sha256"],
           "training_config_sha256": dry["training_config_sha256"],
           "source_hash": dry["source_hash"], "source_hashes": dry["source_hashes"],
           "environment": environment, "role_counts": {k: v["rows"] for k, v in dry["roles"].items()},
           "identity_counts": {k: v["canonical_identities"] for k, v in dry["roles"].items()},
           "optimizer_config": {k: TRAINING_CONFIG[k] for k in ("optimizer", "learning_rate",
                                   "weight_decay", "batch_size", "max_epochs", "gradient_clip_norm")},
           "optimizer_steps": len(history) * ((len(train_targets) + size - 1) // size),
           "epoch_history": history, "selected_epoch": best_epoch,
           "validation_selection_value": best_loss,
           "selection_rule": "minimum development val cross entropy; exact tie -> earliest epoch",
           "checkpoint_sha256": file_hash(checkpoint), "checkpoint_state_sha256": _state_digest(best_state),
           "checkpoint_path": str(checkpoint.relative_to(root)),
           "vocabulary_sha256": dry["vocabulary_sha256"], "scaler_sha256": dry["scaler_sha256"],
           "official_test_access": False, "development_metrics": {"val_cross_entropy": best_loss},
           "scientific_effectiveness_claim": False}
    write_immutable(checked_output(root, f"training_runs/{run_id}.json"), json_bytes(run))
    return run


def assert_checkpoint_provenance(payload: dict, run: dict, dry: dict) -> None:
    expected = {
        "target": "Fuzzers", "split_seed": 42, "model_seed": 10042,
        "p3_manifest_sha256": dry["p3_manifest_sha256"],
        "p4_model_config_sha256": dry["p4_model_config_sha256"],
        "training_config_sha256": dry["training_config_sha256"],
        "vocabulary_sha256": dry["vocabulary_sha256"],
        "scaler_sha256": dry["scaler_sha256"],
        "source_hash": dry["source_hash"],
        "epoch": run["selected_epoch"],
        "validation_selection_value": run["validation_selection_value"],
        "run_id": run["run_id"],
    }
    if payload["provenance"] != expected or any(
        run.get(name) != dry[name] for name in (
            "p3_manifest_sha256", "p4_model_config_sha256", "training_config_sha256",
            "vocabulary_sha256", "scaler_sha256", "source_hash")
    ):
        raise ValueError("checkpoint provenance mismatch")


def verify_smoke(root: Path) -> dict:
    dry = validate_cell(root, "Fuzzers", 42, 10042)
    runs = []
    for tag in ("smoke-a", "smoke-b"):
        path = checked_output(root, f"training_runs/p5-fuzzers-s42-m10042-{tag}.json")
        run = json.loads(path.read_text())
        checkpoint = root / run["checkpoint_path"]
        if file_hash(checkpoint) != run["checkpoint_sha256"]:
            raise ValueError("smoke checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert_checkpoint_provenance(payload, run, dry)
        if _state_digest(payload["state_dict"]) != run["checkpoint_state_sha256"]:
            raise ValueError("checkpoint state hash mismatch")
        runs.append((run, payload))
    left, right = runs
    if left[0]["selected_epoch"] != right[0]["selected_epoch"]:
        raise ValueError("smoke selected epochs differ")
    keys = left[1]["state_dict"].keys()
    max_diff = max(float(torch.max(torch.abs(left[1]["state_dict"][key] - right[1]["state_dict"][key])))
                   for key in keys)
    if max_diff > 1e-12:
        raise ValueError("smoke state exceeds frozen 1e-12 absolute tolerance")
    return {"selected_epoch": left[0]["selected_epoch"], "bitwise_state_equal": max_diff == 0,
            "max_absolute_parameter_difference": max_diff,
            "checkpoint_file_hash_equal": left[0]["checkpoint_sha256"] == right[0]["checkpoint_sha256"],
            "status": "P5_SMOKE_PASS"}
