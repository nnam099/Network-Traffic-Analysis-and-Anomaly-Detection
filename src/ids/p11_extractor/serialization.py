"""Canonical serialization for deterministic extractor intermediates."""

from __future__ import annotations

import hashlib
import json
import math


def _reject_nonfinite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values require an explicit missing representation")
    if isinstance(value, dict):
        return {str(key): _reject_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_reject_nonfinite(item) for item in value]
    return value


def deterministic_json_bytes(value) -> bytes:
    clean = _reject_nonfinite(value)
    return (json.dumps(clean, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def serialized_sha256(value) -> str:
    return hashlib.sha256(deterministic_json_bytes(value)).hexdigest()
