"""P8 design-only safety tests; no model training or external-test access."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.dataset import ROW_ID_COLUMN, SOURCE_ROLE_COLUMN  # noqa: E402
from ids.p8_loafo_protocol import (  # noqa: E402
    CONSUMED_TEST_LABEL, P8KnownClassModel, known_class_order, surrogate_families,
    support_tier, validate_cell, verified_family_universe,
)
from ids.p4_model_interface import FrozenVocabulary  # noqa: E402
from ids.target_isolation import CANONICAL_FINGERPRINT_COLUMN as CANONICAL  # noqa: E402


def universe_fixture():
    return {"attack_families": ["Alpha", "Beta", "Gamma", "Delta"], "families": {
        "Alpha": {"official_train_retained_canonical_identities": 100},
        "Beta": {"official_train_retained_canonical_identities": 200},
        "Gamma": {"official_train_retained_canonical_identities": 200},
        "Delta": {"official_train_retained_canonical_identities": 50},
    }}


def test_surrogate_policy_uses_train_identity_support_and_deterministic_tie_break():
    universe = universe_fixture()
    assert surrogate_families("Alpha", universe) == ["Beta", "Gamma"]
    assert surrogate_families("Beta", universe) == ["Gamma", "Alpha"]
    assert known_class_order("Alpha", universe, ["Beta", "Gamma"]) == ["Normal", "Delta"]
    with pytest.raises(ValueError):
        known_class_order("Alpha", universe, ["Alpha", "Beta"])


def test_variable_classifier_head_contract_has_no_target_label():
    order = known_class_order("Alpha", universe_fixture(), ["Beta", "Gamma"])
    vocabs = {name: FrozenVocabulary.from_p3_mapping(name, {"value:tcp": 0})
              for name in ("proto", "service", "state")}
    model = P8KnownClassModel(vocabs, order)
    assert model.head.in_features == 32 and model.head.out_features == len(order) == 2
    assert {name: layer.embedding_dim for name, layer in model.backbone.embeddings.items()} == {
        "proto": 4, "service": 4, "state": 4}
    assert "Alpha" not in order


def _role_frame(label: str, canonical: int, row: int) -> pd.DataFrame:
    return pd.DataFrame({"attack_cat": [label], CANONICAL: [canonical],
                         ROW_ID_COLUMN: [row], SOURCE_ROLE_COLUMN: ["official_train"]})


def test_target_and_duplicate_canonical_cannot_cross_roles():
    roles = {"train": _role_frame("Normal", 1, 1),
             "val": _role_frame("Normal", 2, 2),
             "meta_known": _role_frame("Normal", 3, 3),
             "calibration": _role_frame("Normal", 4, 4),
             "surrogate_ood": _role_frame("Beta", 5, 5)}
    target = _role_frame("Alpha", 1, 6)
    population = {"target_train": target, "target_identity_hashes": {1}}
    with pytest.raises(ValueError, match="canonical isolation"):
        validate_cell({}, {}, population, roles, "Alpha", ["Normal"], ["Beta"])
    population = {"target_train": _role_frame("Alpha", 6, 6), "target_identity_hashes": {6}}
    roles["val"] = _role_frame("Normal", 1, 7)
    with pytest.raises(ValueError, match="canonical isolation"):
        validate_cell({}, {}, population, roles, "Alpha", ["Normal"], ["Beta"])
    roles["val"] = _role_frame("Normal", 2, 1)
    with pytest.raises(ValueError, match="row overlap"):
        validate_cell({}, {}, population, roles, "Alpha", ["Normal"], ["Beta"])


def test_consumed_official_test_source_cannot_enter_development():
    roles = {"train": _role_frame("Normal", 1, 1),
             "val": _role_frame("Normal", 2, 2),
             "meta_known": _role_frame("Normal", 3, 3),
             "calibration": _role_frame("Normal", 4, 4),
             "surrogate_ood": _role_frame("Beta", 5, 5)}
    roles["val"].loc[0, SOURCE_ROLE_COLUMN] = "official_test"
    population = {"target_train": _role_frame("Alpha", 6, 6),
                  "target_identity_hashes": {6}}
    with pytest.raises(ValueError, match="non-train source"):
        validate_cell({}, {}, population, roles, "Alpha", ["Normal"], ["Beta"])


def test_vocab_is_fold_local_and_unseen_target_token_is_oov():
    vocab = FrozenVocabulary.from_p3_mapping("proto", {"value:tcp": 0})
    assert vocab.encode("tcp") == 2
    assert vocab.encode("target-only-token") == 1
    assert vocab.encode("another-target-only-token") == 1
    assert vocab.cardinality == 3 and vocab.pad_index == 0


def test_support_tiers_never_drop_small_target():
    assert support_tier(1000) == "HIGH SUPPORT"
    assert support_tier(200) == "MODERATE SUPPORT"
    assert support_tier(50) == "SMALL-N"
    assert support_tier(49) == "INSUFFICIENT"


def test_verified_universe_fails_on_unfrozen_or_missing_family():
    frame = pd.DataFrame({"attack_cat": ["Normal", "Alpha", "Beta", "Gamma", "Delta"],
                          CANONICAL: [1, 2, 3, 4, 5]})
    meta = {label: {"original_rows": 1, "excluded_rows": 0, "remaining_rows": 1,
                    "remaining_unique_canonical_identities": 1}
            for label in frame.attack_cat}
    historical = {"per_family": {label: {"CLEAN": {"rows": 1,
                        "unique_canonical_identities": 1}} for label in frame.attack_cat}}
    universe = verified_family_universe({"train": frame}, {"per_family": meta}, historical)
    assert universe["attack_families"] == ["Alpha", "Beta", "Delta", "Gamma"]
    frame.loc[0, "attack_cat"] = "unfrozen"
    with pytest.raises(ValueError, match="differ"):
        verified_family_universe({"train": frame}, {"per_family": meta}, historical)


def test_consumed_test_is_diagnostic_only_and_runner_names_only_train_loader():
    assert CONSUMED_TEST_LABEL == "REUSED TEST — NOT INDEPENDENT EXTERNAL VALIDATION"
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/audit_p8_loafo.py").read_text()
    protocol = (root / "src/ids/p8_loafo_protocol.py").read_text()
    assert "DEFAULT_TEST_FILE" not in runner
    assert "p7/primary" not in runner and "p7/primary" not in protocol
    assert 'files=[train_path]' in runner
    assert "backward(" not in runner and "optimizer.step(" not in runner
