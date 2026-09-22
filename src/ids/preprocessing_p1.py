"""Versioned P1 transform; legacy checkpoint transforms remain separate.

Scalar categorical coordinates retain P0's arbitrary ordinal geometry. The hash
fallback is an identity-preserving compatibility experiment, not a meaningful
distance between nominal categories or a demonstrated improvement in detection.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata

import numpy as np
import pandas as pd


LEGACY_MODEL_INPUT_IDENTITY_VERSION = "post-scaling-clipping-float32-v1"
MODEL_INPUT_IDENTITY_VERSION = "post-robustscale-float32-sha256fallback-v2"
CATEGORY_ENCODING_VERSION = "normalized-known-index-sha256-negative23-v1"
CATEGORY_COLUMNS = ("proto", "service", "state")
REQUIRED_UNICODE_DATA_VERSION = "15.1.0"
if unicodedata.unidata_version != REQUIRED_UNICODE_DATA_VERSION:
    raise RuntimeError(
        "P1 canonical/category normalization requires Unicode data 15.1.0; "
        f"got {unicodedata.unidata_version}. Re-audit before changing this contract."
    )


def normalized_category(value) -> str:
    if pd.isna(value):
        return "missing:"
    return "value:" + unicodedata.normalize("NFKC", str(value)).strip().casefold()


def unknown_category_code(feature: str, normalized_token: str) -> float:
    """23-bit SHA-256 code in [-2, -1), exactly representable as float32.

    JSON framing avoids namespace ambiguity; digest parsing is big-endian.
    No vocabulary size, row order, label, seed or process hash enters the code.
    Hash collisions are possible and must be audited, never resolved using test
    data. Nonnegative fitted indices cannot collide with these fallback codes.
    """
    payload = json.dumps(
        [CATEGORY_ENCODING_VERSION, feature, normalized_token],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    bits = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**23)
    return -(1.0 + (bits + 1) / 2**23)


def fit_categorical_maps(backbone: pd.DataFrame) -> dict:
    return {
        feature: {
            token: index for index, token in enumerate(sorted(
                backbone[feature].map(normalized_category).unique()
            ))
        }
        for feature in CATEGORY_COLUMNS if feature in backbone
    }


def encode_categories(frame: pd.DataFrame, maps: dict) -> None:
    """Transform only; a missing map must never trigger vocabulary fitting."""
    for feature in CATEGORY_COLUMNS:
        if feature not in frame:
            continue
        if feature not in maps:
            raise ValueError(f"missing backbone vocabulary: {feature}")
        mapping = maps[feature]
        codes = np.asarray(list(mapping.values()), dtype=np.float64)
        if not np.isfinite(codes).all() or (codes < 0).any() or (codes >= 2**24).any():
            raise ValueError("known category codes must be finite nonnegative float32 indices")
        values = frame[feature].map(normalized_category)
        lookup = {
            token: mapping[token] if token in mapping else unknown_category_code(feature, token)
            for token in values.unique()
        }
        frame[f"{feature}_num"] = values.map(lookup).astype(np.float64)


def transform_stages(values: np.ndarray, scaler) -> tuple[np.ndarray, np.ndarray]:
    """No imputation or saturation: non-finite inputs/results fail closed."""
    encoded = np.asarray(values, dtype=np.float64)
    if not np.isfinite(encoded).all():
        raise ValueError("P1 encoded input contains non-finite values; explicit missing policy required")
    scaled = np.asarray(scaler.transform(encoded), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        final = scaled.astype(np.float32)
    if not np.isfinite(scaled).all() or not np.isfinite(final).all():
        raise ValueError("P1 transform is non-finite or overflows float32; no saturation permitted")
    return scaled, final


def matrix_fingerprints(values: np.ndarray) -> np.ndarray:
    """Preserve stage precision (in particular, never cast float64 to float32)."""
    matrix = np.array(values, copy=True)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("stage fingerprinting requires a finite 2D matrix")
    matrix[matrix == 0] = 0
    return pd.util.hash_pandas_object(pd.DataFrame(matrix), index=False).to_numpy(np.uint64)
