"""P4 design-only structured model interface bound to frozen P3 manifests.

This module intentionally defines no trainer, optimizer, loss, threshold, metric,
checkpoint writer, or official-test loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .p2_protocol import SCHEMA_VERSION
from .p3_protocol import sha256_json
from .preprocessing_p1 import normalized_category


MODEL_INTERFACE_VERSION = "structured-scientific-model-input-v1"
ARCHITECTURE_VERSION = "structured-mlp-embedding-representation-v1"
VOCABULARY_VERSION = "per-feature-pad-oov-known-v1"
CONTINUOUS_FEATURE_COUNT = 58
CATEGORICAL_FIELDS = ("proto", "service", "state")
PAD_INDEX = 0
OOV_INDEX = 1
MODEL_DTYPE = torch.float64
MODEL_DTYPE_NAME = "torch.float64"


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def stable_hash(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class FrozenVocabulary:
    """Per-feature immutable finite vocabulary with a dedicated OOV index."""

    feature: str
    known_token_to_index: Mapping[str, int]
    pad_index: int = PAD_INDEX
    oov_index: int = OOV_INDEX
    version: str = VOCABULARY_VERSION

    def __post_init__(self) -> None:
        if self.feature not in CATEGORICAL_FIELDS:
            raise ValueError(f"unsupported categorical feature: {self.feature}")
        mapping = dict(self.known_token_to_index)
        values = sorted(mapping.values())
        if self.pad_index != 0 or self.oov_index != 1:
            raise ValueError("P4 reserves PAD=0 and OOV=1")
        if values != list(range(2, len(mapping) + 2)):
            raise ValueError("known category indices must be contiguous from 2")
        if any(not isinstance(token, str) for token in mapping):
            raise TypeError("normalized categorical tokens must be strings")
        object.__setattr__(self, "known_token_to_index", MappingProxyType(mapping))

    @classmethod
    def from_p3_mapping(cls, feature: str, mapping: Mapping[str, int]):
        ordered = sorted(mapping.items(), key=lambda item: item[1])
        if [index for _, index in ordered] != list(range(len(ordered))):
            raise ValueError("P3 vocabulary indices are not contiguous from zero")
        return cls(feature, {token: index + 2 for token, index in ordered})

    @classmethod
    def from_payload(cls, payload: dict):
        if payload.get("version") != VOCABULARY_VERSION:
            raise ValueError("unsupported vocabulary payload version")
        return cls(
            feature=payload["feature"],
            known_token_to_index=payload["known_token_to_index"],
            pad_index=payload["pad_index"], oov_index=payload["oov_index"],
        )

    @property
    def known_count(self) -> int:
        return len(self.known_token_to_index)

    @property
    def cardinality(self) -> int:
        return self.known_count + 2

    def encode(self, value: str | None) -> int:
        if value is not None and not isinstance(value, str):
            raise TypeError("categorical source values must be strings or None")
        token = normalized_category(value)
        return self.known_token_to_index.get(token, self.oov_index)

    def to_payload(self) -> dict:
        return {
            "version": self.version, "feature": self.feature,
            "pad_index": self.pad_index, "oov_index": self.oov_index,
            "known_token_to_index": dict(self.known_token_to_index),
            "known_count": self.known_count, "cardinality": self.cardinality,
            "unknown_behavior": (
                "All unseen normalized tokens intentionally share this feature's "
                "dedicated OOV embedding; this is not canonical identity equivalence."
            ),
        }

    @property
    def sha256(self) -> str:
        return stable_hash(self.to_payload())


@dataclass(frozen=True)
class StructuredModelInput:
    """Only accepted P4 neural boundary; canonical/audit metadata stays outside."""

    continuous: torch.Tensor
    proto: torch.Tensor
    service: torch.Tensor
    state: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.continuous, torch.Tensor):
            raise TypeError("continuous input must be a torch.Tensor")
        if self.continuous.ndim != 2 or self.continuous.shape[1] != CONTINUOUS_FEATURE_COUNT:
            raise ValueError("continuous input must have shape [batch, 58]")
        if self.continuous.dtype != MODEL_DTYPE:
            raise TypeError("P4 continuous model boundary requires torch.float64")
        if not bool(torch.isfinite(self.continuous).all()):
            raise ValueError("continuous input contains NaN or Inf")
        batch = self.continuous.shape[0]
        for name in CATEGORICAL_FIELDS:
            values = getattr(self, name)
            if not isinstance(values, torch.Tensor):
                raise TypeError(f"{name} input must be a torch.Tensor")
            if values.ndim != 1 or values.shape[0] != batch:
                raise ValueError(f"{name} input must have shape [batch]")
            if values.dtype != torch.long:
                raise TypeError(f"{name} indices must use torch.long, never floating point")

    @property
    def batch_size(self) -> int:
        return self.continuous.shape[0]


class StructuredBatchAdapter:
    """Explicit NumPy-float64 to torch-float64 boundary; performs no fitting."""

    def __init__(self, vocabularies: Mapping[str, FrozenVocabulary]):
        if tuple(vocabularies) != CATEGORICAL_FIELDS:
            raise ValueError("adapter requires ordered proto/service/state vocabularies")
        self._vocabularies = MappingProxyType(dict(vocabularies))

    @property
    def vocabularies(self) -> Mapping[str, FrozenVocabulary]:
        return self._vocabularies

    def transform(self, scaled_continuous: np.ndarray,
                  categorical: Mapping[str, Sequence[str | None]]) -> StructuredModelInput:
        if not isinstance(scaled_continuous, np.ndarray):
            raise TypeError("scaled continuous source must be a NumPy array")
        if scaled_continuous.dtype != np.float64:
            raise TypeError("scaled continuous source must remain NumPy float64")
        if scaled_continuous.ndim != 2 or scaled_continuous.shape[1] != 58:
            raise ValueError("scaled continuous source must have shape [batch, 58]")
        if not np.isfinite(scaled_continuous).all():
            raise ValueError("scaled continuous source contains NaN or Inf")
        if tuple(categorical) != CATEGORICAL_FIELDS:
            raise ValueError("categorical source requires ordered proto/service/state")
        batch = len(scaled_continuous)
        encoded = {}
        for feature in CATEGORICAL_FIELDS:
            values = categorical[feature]
            if len(values) != batch:
                raise ValueError(f"{feature} batch length mismatch")
            encoded[feature] = torch.tensor(
                [self._vocabularies[feature].encode(value) for value in values],
                dtype=torch.long,
            )
        # Copy into a framework-owned tensor without dtype conversion.
        continuous = torch.from_numpy(np.ascontiguousarray(scaled_continuous).copy())
        return StructuredModelInput(continuous=continuous, **encoded)


def embedding_dimension(known_count: int) -> int:
    """Conservative deterministic policy, independent of labels/test behavior."""
    if known_count < 0:
        raise ValueError("known category count cannot be negative")
    return min(16, max(4, math.ceil(math.sqrt(known_count + 2))))


class ContinuousEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(58, hidden_dim, dtype=MODEL_DTYPE),
            nn.LayerNorm(hidden_dim, dtype=MODEL_DTYPE), nn.GELU(),
            nn.Linear(hidden_dim, output_dim, dtype=MODEL_DTYPE),
            nn.LayerNorm(output_dim, dtype=MODEL_DTYPE), nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != 58 or values.dtype != MODEL_DTYPE:
            raise ValueError("ContinuousEncoder requires float64[batch,58]")
        return self.network(values)


class CategoricalEncoder(nn.Module):
    def __init__(self, vocabularies: Mapping[str, FrozenVocabulary]):
        super().__init__()
        if tuple(vocabularies) != CATEGORICAL_FIELDS:
            raise ValueError("three ordered feature-specific vocabularies are required")
        self.vocabularies = dict(vocabularies)
        self.embedding_dims = {
            name: embedding_dimension(vocabulary.known_count)
            for name, vocabulary in vocabularies.items()
        }
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(
                vocabulary.cardinality, self.embedding_dims[name],
                padding_idx=vocabulary.pad_index, dtype=MODEL_DTYPE,
            )
            for name, vocabulary in vocabularies.items()
        })
        self.output_dim = sum(self.embedding_dims.values())

    def forward(self, inputs: StructuredModelInput) -> torch.Tensor:
        if not isinstance(inputs, StructuredModelInput):
            raise TypeError("CategoricalEncoder requires StructuredModelInput")
        outputs = []
        for name in CATEGORICAL_FIELDS:
            indices = getattr(inputs, name)
            vocabulary = self.vocabularies[name]
            if bool((indices < 0).any()) or bool((indices >= vocabulary.cardinality).any()):
                raise ValueError(f"{name} index outside frozen vocabulary")
            outputs.append(self.embeddings[name](indices))
        return torch.cat(outputs, dim=1)


class StructuredFeatureEncoder(nn.Module):
    def __init__(self, vocabularies: Mapping[str, FrozenVocabulary]):
        super().__init__()
        self.continuous_encoder = ContinuousEncoder()
        self.categorical_encoder = CategoricalEncoder(vocabularies)
        self.output_dim = (
            self.continuous_encoder.output_dim + self.categorical_encoder.output_dim
        )

    def forward(self, inputs: StructuredModelInput) -> torch.Tensor:
        if not isinstance(inputs, StructuredModelInput):
            raise TypeError("flat tensors are forbidden; pass StructuredModelInput")
        numeric = self.continuous_encoder(inputs.continuous)
        categorical = self.categorical_encoder(inputs)
        return torch.cat((numeric, categorical), dim=1)


class RepresentationBackbone(nn.Module):
    """Baseline representation only; no classifier or open-set score is selected."""

    def __init__(self, vocabularies: Mapping[str, FrozenVocabulary],
                 hidden_dim: int = 64, representation_dim: int = 32):
        super().__init__()
        self.feature_encoder = StructuredFeatureEncoder(vocabularies)
        self.representation_dim = representation_dim
        self.head = nn.Sequential(
            nn.Linear(self.feature_encoder.output_dim, hidden_dim, dtype=MODEL_DTYPE),
            nn.LayerNorm(hidden_dim, dtype=MODEL_DTYPE), nn.GELU(),
            nn.Linear(hidden_dim, representation_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, inputs: StructuredModelInput) -> torch.Tensor:
        if not isinstance(inputs, StructuredModelInput):
            raise TypeError("legacy flat tensor input is not accepted")
        return self.head(self.feature_encoder(inputs))


class ContinuousReconstructionHead(nn.Module):
    """Optional isolated component; not selected or instantiated by P4 baseline."""

    def __init__(self, representation_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim, dtype=MODEL_DTYPE), nn.GELU(),
            nn.Linear(hidden_dim, 58, dtype=MODEL_DTYPE),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.dtype != MODEL_DTYPE or representation.ndim != 2:
            raise TypeError("reconstruction input must be a float64 representation batch")
        return self.network(representation)


def vocabularies_from_manifest(manifest: dict) -> dict[str, FrozenVocabulary]:
    provenance = manifest["vocabulary_provenance"]
    if provenance["vocabulary_source_role"] != (
        "official_train target-purged backbone train"
    ):
        raise ValueError("P3 vocabulary provenance is not permitted")
    mappings = provenance["vocabulary_maps"]
    if tuple(mappings) != CATEGORICAL_FIELDS:
        raise ValueError("P3 manifest must contain exactly three ordered vocabularies")
    if sha256_json(mappings) != provenance["vocabulary_sha256"]:
        raise ValueError("P3 vocabulary hash mismatch")
    return {name: FrozenVocabulary.from_p3_mapping(name, mappings[name])
            for name in CATEGORICAL_FIELDS}


def architecture_config(vocabularies: Mapping[str, FrozenVocabulary]) -> dict:
    embedding_dims = {
        name: embedding_dimension(vocabularies[name].known_count)
        for name in CATEGORICAL_FIELDS
    }
    fusion_dim = 32 + sum(embedding_dims.values())
    return {
        "version": ARCHITECTURE_VERSION,
        "selected_baseline": "numeric MLP + independent embeddings + concatenation MLP",
        "continuous_encoder_dims": [58, 64, 32],
        "categorical_embedding_dims": embedding_dims,
        "categorical_cardinalities": {
            name: vocabularies[name].cardinality for name in CATEGORICAL_FIELDS
        },
        "fusion_input_dim": fusion_dim,
        "representation_head_dims": [fusion_dim, 64, 32],
        "representation_dim": 32,
        "parameter_dtype": MODEL_DTYPE_NAME,
        "classifier_head": None, "open_set_scorer": None,
        "reconstruction_head_selected": False,
    }


def bound_model_config(manifest: dict, *, manifest_sha256: str,
                       p3_collection_sha256: str, schema: dict,
                       source_code_sha256: str) -> dict:
    if manifest.get("official_test_consulted") is not False:
        raise ValueError("model config cannot bind a test-informed development manifest")
    if manifest["development_gate"]["gate"] != "PASS":
        raise ValueError("P3 development gate did not pass")
    if manifest["feature_schema_version"] != SCHEMA_VERSION:
        raise ValueError("P3 schema version mismatch")
    if manifest["feature_schema_sha256"] != schema["sha256"]:
        raise ValueError("P3 schema hash mismatch")
    if len(schema["contract"]["continuous_features"]) != 58:
        raise ValueError("P4 requires exactly 58 continuous features")
    if tuple(schema["contract"]["categorical_features"]) != CATEGORICAL_FIELDS:
        raise ValueError("P4 requires exactly proto/service/state")
    scaler = manifest["scaler_provenance"]
    center = np.asarray(scaler["scaler_center"], dtype=np.float64)
    scale = np.asarray(scaler["scaler_scale"], dtype=np.float64)
    scaler_hash = hashlib.sha256(
        np.concatenate([center, scale]).astype("<f8").tobytes()
    ).hexdigest()
    if scaler_hash != scaler["scaler_center_scale_sha256"]:
        raise ValueError("P3 scaler hash mismatch")
    vocabularies = vocabularies_from_manifest(manifest)
    architecture = architecture_config(vocabularies)
    payload = {
        "model_interface_version": MODEL_INTERFACE_VERSION,
        "p3_collection_sha256": p3_collection_sha256,
        "p3_manifest_sha256": manifest_sha256,
        "target": manifest["target"], "seed": manifest["seed"],
        "schema_version": manifest["feature_schema_version"],
        "schema_sha256": manifest["feature_schema_sha256"],
        "continuous_feature_count": 58,
        "continuous_source_dtype": "numpy.float64",
        "continuous_model_dtype": MODEL_DTYPE_NAME,
        "categorical_fields": list(CATEGORICAL_FIELDS),
        "vocabularies": {
            name: vocabularies[name].to_payload() for name in CATEGORICAL_FIELDS
        },
        "vocabulary_sha256": {
            name: vocabularies[name].sha256 for name in CATEGORICAL_FIELDS
        },
        "p3_combined_vocabulary_sha256": manifest[
            "vocabulary_provenance"
        ]["vocabulary_sha256"],
        "scaler_center_scale_sha256": scaler_hash,
        "scaler_fit_row_ids_sha256": scaler["scaler_fit_row_ids_sha256"],
        "architecture": architecture,
        "architecture_config_sha256": stable_hash(architecture),
        "source_code_sha256": source_code_sha256,
        "training_performed": False, "checkpoint_created": False,
        "performance_metrics_generated": False,
    }
    return {**payload, "model_config_sha256": stable_hash(payload)}
