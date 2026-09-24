"""Dataset-independent P13 projection onto frozen portable schema v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

from ids.p12_portable_schema import (
    CONTINUOUS_FEATURES,
    PortableFlow,
    canonical_flow_identity_v3,
    canonical_json_bytes,
    extract_core_features,
)


@dataclass(frozen=True, slots=True)
class ExtractedRow:
    flow_id: str
    continuous: tuple[float | None, ...]
    missing_mask: tuple[bool, ...]
    ip_protocol: str
    canonical_identity_v3: str

    def as_serializable(self) -> dict[str, object]:
        return asdict(self)


def extract_rows(flows: list[PortableFlow]) -> list[ExtractedRow]:
    rows = []
    for ordinal, flow in enumerate(flows):
        features = extract_core_features(flow)
        continuous = tuple(
            None if features[name] is None else float(features[name]) for name in CONTINUOUS_FEATURES
        )
        rows.append(ExtractedRow(
            flow_id=f"flow-{ordinal:012d}",
            continuous=continuous,
            missing_mask=tuple(value is None for value in continuous),
            ip_protocol=str(features["ip_protocol"]),
            canonical_identity_v3=canonical_flow_identity_v3(features),
        ))
    return rows


def semantic_content_hash(rows: list[ExtractedRow]) -> str:
    payload = {
        "format": "p13-portable-semantic-rows-v1",
        "rows": [row.as_serializable() for row in rows],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
