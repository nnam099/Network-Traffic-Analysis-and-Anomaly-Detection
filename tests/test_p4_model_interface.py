"""P4 structured architecture contract tests; no training or metric execution."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p4_model_interface import (  # noqa: E402
    CATEGORICAL_FIELDS, OOV_INDEX, FrozenVocabulary, RepresentationBackbone,
    StructuredBatchAdapter, StructuredFeatureEncoder, StructuredModelInput,
    architecture_config, bound_model_config, stable_hash,
    vocabularies_from_manifest,
)


P3 = ROOT / "results/data_quality/p3"


def frozen_case():
    freeze = json.loads((P3 / "development_freeze.json").read_text())
    entry = freeze["cells"][0]
    raw = gzip.decompress((P3 / entry["manifest_filename"]).read_bytes())
    manifest = json.loads(raw)
    schema = json.loads((P3 / "schema_v2.json").read_text())
    return freeze, entry, raw, manifest, schema


def structured_batch(batch_size=3):
    _, _, _, manifest, _ = frozen_case()
    vocabularies = vocabularies_from_manifest(manifest)
    adapter = StructuredBatchAdapter(vocabularies)
    categories = {name: [None] * batch_size for name in CATEGORICAL_FIELDS}
    return vocabularies, adapter.transform(
        np.zeros((batch_size, 58), dtype=np.float64), categories
    )


def test_exact_structured_schema_and_wrong_inputs_rejected():
    vocabularies, valid = structured_batch()
    assert valid.continuous.shape == (3, 58)
    assert valid.continuous.dtype == torch.float64
    assert all(getattr(valid, name).dtype == torch.long for name in CATEGORICAL_FIELDS)
    with pytest.raises(ValueError, match="batch, 58"):
        StructuredModelInput(
            torch.zeros(3, 57, dtype=torch.float64), valid.proto,
            valid.service, valid.state,
        )
    with pytest.raises(TypeError, match="torch.float64"):
        StructuredModelInput(
            torch.zeros(3, 58, dtype=torch.float32), valid.proto,
            valid.service, valid.state,
        )
    with pytest.raises(TypeError, match="torch.long"):
        StructuredModelInput(
            valid.continuous, valid.proto.float(), valid.service, valid.state,
        )
    with pytest.raises(TypeError):
        StructuredModelInput(valid.continuous, valid.proto, valid.service)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="ordered proto/service/state"):
        StructuredBatchAdapter({"state": vocabularies["state"],
                                "proto": vocabularies["proto"],
                                "service": vocabularies["service"]})


def test_vocabulary_oov_is_deterministic_immutable_and_round_trips():
    _, _, _, manifest, _ = frozen_case()
    vocabularies = vocabularies_from_manifest(manifest)
    for feature, vocabulary in vocabularies.items():
        size = vocabulary.known_count
        assert vocabulary.encode("definitely-unseen-p4-token") == OOV_INDEX
        assert vocabulary.encode("another-unseen-p4-token") == OOV_INDEX
        assert vocabulary.known_count == size
        assert "value:definitely-unseen-p4-token" not in vocabulary.known_token_to_index
        with pytest.raises(TypeError):
            vocabulary.known_token_to_index["test-derived-token"] = 999  # type: ignore[index]
        payload = vocabulary.to_payload()
        restored = FrozenVocabulary.from_payload(payload)
        assert restored.to_payload() == payload
        assert restored.sha256 == vocabulary.sha256
        assert restored.feature == feature


def test_float64_framework_boundary_is_exact_and_categorical_never_floats():
    _, _, _, manifest, _ = frozen_case()
    adapter = StructuredBatchAdapter(vocabularies_from_manifest(manifest))
    rng = np.random.default_rng(42)
    source = rng.normal(size=(4, 58)).astype(np.float64)
    categories = {name: ["tcp", "udp", None, "unseen"] for name in CATEGORICAL_FIELDS}
    first = adapter.transform(source, categories)
    second = adapter.transform(source.copy(), categories)
    np.testing.assert_array_equal(first.continuous.numpy(), source)
    assert torch.equal(first.continuous, second.continuous)
    assert all(torch.equal(getattr(first, name), getattr(second, name))
               for name in CATEGORICAL_FIELDS)
    assert all(not getattr(first, name).dtype.is_floating_point
               for name in CATEGORICAL_FIELDS)
    with pytest.raises(TypeError, match="NumPy float64"):
        adapter.transform(source.astype(np.float32), categories)
    source[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        adapter.transform(source, categories)


def test_architecture_shapes_embeddings_and_flat_input_rejection():
    vocabularies, batch = structured_batch(batch_size=5)
    config = architecture_config(vocabularies)
    encoder = StructuredFeatureEncoder(vocabularies)
    model = RepresentationBackbone(vocabularies)
    with torch.no_grad():
        fused = encoder(batch)
        representation = model(batch)
    assert fused.shape == (5, config["fusion_input_dim"])
    assert representation.shape == (5, config["representation_dim"])
    assert representation.dtype == torch.float64
    assert encoder.categorical_encoder.embedding_dims == config[
        "categorical_embedding_dims"
    ]
    for feature in CATEGORICAL_FIELDS:
        embedding = encoder.categorical_encoder.embeddings[feature]
        assert embedding.num_embeddings == vocabularies[feature].cardinality
        assert embedding.embedding_dim == config["categorical_embedding_dims"][feature]
    with pytest.raises(TypeError, match="legacy flat"):
        model(torch.zeros(5, 61, dtype=torch.float64))  # type: ignore[arg-type]


def test_bound_config_matches_exact_p3_manifest_provenance():
    freeze, entry, raw, manifest, schema = frozen_case()
    manifest_hash = hashlib.sha256(raw).hexdigest()
    assert manifest_hash == entry["manifest_sha256"]
    config = bound_model_config(
        manifest, manifest_sha256=manifest_hash,
        p3_collection_sha256=(
            "1443746a4f024b3d7c1569ba9bbd31987933b5f9e2fb0ce6507ba5e57ddbab21"
        ),
        schema=schema, source_code_sha256="unit-test-source-hash",
    )
    payload = {key: value for key, value in config.items()
               if key != "model_config_sha256"}
    assert stable_hash(payload) == config["model_config_sha256"]
    assert config["p3_manifest_sha256"] == manifest_hash
    assert config["schema_sha256"] == manifest["feature_schema_sha256"]
    assert config["scaler_center_scale_sha256"] == manifest[
        "scaler_provenance"
    ]["scaler_center_scale_sha256"]
    assert config["p3_collection_sha256"] == freeze.get(
        "p3_collection_sha256",
        "1443746a4f024b3d7c1569ba9bbd31987933b5f9e2fb0ce6507ba5e57ddbab21",
    )
    assert config["continuous_feature_count"] == 58
    assert len(config["vocabulary_sha256"]) == 3


def test_p4_sources_and_artifacts_prohibit_training_and_metrics():
    sources = "\n".join((ROOT / name).read_text() for name in (
        "src/ids/p4_model_interface.py", "scripts/audit_p4_model_interface.py"
    ))
    for forbidden in ("torch.optim", ".backward(", "torch.save(", "roc_auc_score",
                      "f1_score", "accuracy_score", "precision_recall_curve"):
        assert forbidden not in sources
    output = ROOT / "results/model_design/p4"
    if output.exists():
        names = {path.name for path in output.iterdir() if path.is_file()}
        assert names == {
            "model_interface.json", "architecture_design.json",
            "legacy_compatibility_audit.json", "p4_matrix.json",
        }
        assert not any(path.suffix in {".pt", ".pth", ".ckpt", ".pkl"}
                       for path in output.iterdir())
        matrix = json.loads((output / "p4_matrix.json").read_text())
        assert matrix["optimizer_steps"] == 0
        assert matrix["performance_metrics_generated"] is False
        assert matrix["checkpoint_created"] is False
