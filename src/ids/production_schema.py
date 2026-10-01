from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .log_normalizer import normalize_real_world_logs


PRODUCTION_FLOW_COLUMNS = [
    "event_time",
    "source",
    "source_file",
    "source_row",
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "service",
    "state",
    "duration",
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
    "sload",
    "dload",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "sintpkt",
    "dintpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "dataset_label",
    "attack_category",
    "analyst_label",
    "split",
]

ALLOWED_ANALYST_LABELS = {"normal", "known_attack", "suspicious", "false_positive", "unknown"}


@dataclass
class ProductionSchemaResult:
    data: pd.DataFrame
    report: dict[str, Any]


def normalize_to_production_schema(
    df_raw: pd.DataFrame,
    source: str = "cicflowmeter",
    source_file: str = "",
) -> ProductionSchemaResult:
    """Normalize flow CSV rows into a stable production contract.

    The output is intentionally wider than the model feature vector. It keeps
    operational fields analysts need while preserving the model-ready flow
    counters produced by ``log_normalizer``.
    """
    normalized, normalization_report = normalize_real_world_logs(df_raw)
    raw = df_raw.copy()
    raw.columns = [_clean_col(col) for col in raw.columns]

    out = pd.DataFrame(index=raw.index)
    out["event_time"] = _event_time(raw, normalized)
    out["source"] = source
    out["source_file"] = source_file
    out["source_row"] = np.arange(len(raw), dtype=int)
    out["flow_id"] = _first_present_series(raw, ["flow_id", "flowid", "id"], default="")
    out["src_ip"] = _pick(normalized, "srcip", raw, ["src_ip", "source_ip", "source.address", "id.orig_h"])
    out["dst_ip"] = _pick(normalized, "dstip", raw, ["dst_ip", "destination_ip", "dest_ip", "destination.address", "id.resp_h"])
    out["src_port"] = _pick(normalized, "sport", raw, ["src_port", "source_port", "source.port", "id.orig_p"])
    out["dst_port"] = _pick(normalized, "dsport", raw, ["dst_port", "destination_port", "dest_port", "destination.port", "id.resp_p"])
    out["protocol"] = _pick(normalized, "proto", raw, ["protocol", "proto", "transport"])
    out["service"] = _pick(normalized, "service", raw, ["service", "app_proto", "application"])
    out["state"] = _pick(normalized, "state", raw, ["state", "conn_state", "tcp_state"])

    numeric_map = {
        "duration": "dur",
        "sbytes": "sbytes",
        "dbytes": "dbytes",
        "spkts": "spkts",
        "dpkts": "dpkts",
        "sload": "sload",
        "dload": "dload",
        "sttl": "sttl",
        "dttl": "dttl",
        "sloss": "sloss",
        "dloss": "dloss",
        "sintpkt": "sintpkt",
        "dintpkt": "dintpkt",
        "tcprtt": "tcprtt",
        "synack": "synack",
        "ackdat": "ackdat",
    }
    for target, source_col in numeric_map.items():
        out[target] = _numeric(normalized[source_col]) if source_col in normalized.columns else 0.0

    label = _dataset_label(raw, normalized)
    out["dataset_label"] = label
    out["attack_category"] = _attack_category(raw, normalized, label)
    out["analyst_label"] = label.map(_default_analyst_label)
    out["split"] = ""

    out = out[PRODUCTION_FLOW_COLUMNS].copy()
    report = {
        "schema_version": 1,
        "rows": int(len(out)),
        "source": source,
        "source_file": source_file,
        "columns": PRODUCTION_FLOW_COLUMNS,
        "normalization_report": normalization_report.as_dict(),
        "analyst_label_distribution": _value_counts(out["analyst_label"]),
        "dataset_label_distribution": _value_counts(out["dataset_label"]),
    }
    return ProductionSchemaResult(data=out, report=report)


def apply_label_overrides(
    flows: pd.DataFrame,
    overrides: pd.DataFrame,
    key: str = "flow_id",
) -> pd.DataFrame:
    """Apply partial analyst labels from a review CSV.

    ``overrides`` may use ``flow_id`` or ``source_row`` as the key and can update
    ``analyst_label`` plus optional ``attack_category``.
    """
    if overrides.empty:
        return flows.copy()
    if key not in flows.columns:
        raise ValueError(f"flows do not include key column: {key}")
    if key not in overrides.columns:
        raise ValueError(f"overrides do not include key column: {key}")
    if "analyst_label" not in overrides.columns:
        raise ValueError("overrides must include analyst_label")

    out = flows.copy()
    patch = overrides.copy()
    patch["analyst_label"] = patch["analyst_label"].map(_clean_label)
    invalid = sorted(set(patch["analyst_label"]) - ALLOWED_ANALYST_LABELS)
    if invalid:
        raise ValueError("invalid analyst_label values: " + ", ".join(invalid))

    patch = patch.drop_duplicates(subset=[key], keep="last").set_index(key)
    mask = out[key].isin(patch.index)
    if not bool(mask.any()):
        return out

    out.loc[mask, "analyst_label"] = out.loc[mask, key].map(patch["analyst_label"])
    if "attack_category" in patch.columns:
        mapped_attack = out.loc[mask, key].map(patch["attack_category"]).fillna(out.loc[mask, "attack_category"])
        out.loc[mask, "attack_category"] = mapped_attack
    return out


def split_by_event_time(
    flows: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    total = train_ratio + validation_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_ratio = train_ratio / total
    validation_ratio = validation_ratio / total

    out = flows.copy()
    sort_key = pd.to_datetime(out["event_time"], errors="coerce", utc=True)
    out["_sort_time"] = sort_key
    out["_sort_row"] = np.arange(len(out), dtype=int)
    out = out.sort_values(["_sort_time", "_sort_row"], na_position="last").reset_index(drop=True)

    n = len(out)
    train_end = int(round(n * train_ratio))
    validation_end = int(round(n * (train_ratio + validation_ratio)))
    train_end = max(0, min(train_end, n))
    validation_end = max(train_end, min(validation_end, n))

    out["split"] = "test"
    out.loc[: train_end - 1, "split"] = "train"
    out.loc[train_end : validation_end - 1, "split"] = "validation"
    out = out.drop(columns=["_sort_time", "_sort_row"])
    return out[PRODUCTION_FLOW_COLUMNS]


def summarize_production_flows(flows: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(flows)),
        "columns": list(flows.columns),
        "split_distribution": _value_counts(flows.get("split", pd.Series(dtype=str))),
        "analyst_label_distribution": _value_counts(flows.get("analyst_label", pd.Series(dtype=str))),
        "dataset_label_distribution": _value_counts(flows.get("dataset_label", pd.Series(dtype=str))),
        "source_distribution": _value_counts(flows.get("source", pd.Series(dtype=str))),
        "time_range": {
            "start": _min_time(flows.get("event_time")),
            "end": _max_time(flows.get("event_time")),
        },
    }


def _clean_col(col: Any) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def _first_present_series(df: pd.DataFrame, names: list[str], default: Any = "") -> pd.Series:
    for name in names:
        clean = _clean_col(name)
        if clean in df.columns:
            return df[clean]
    return pd.Series(default, index=df.index)


def _pick(normalized: pd.DataFrame, normalized_col: str, raw: pd.DataFrame, raw_names: list[str]) -> pd.Series:
    if normalized_col in normalized.columns:
        return normalized[normalized_col]
    return _first_present_series(raw, raw_names, default="")


def _event_time(raw: pd.DataFrame, normalized: pd.DataFrame) -> pd.Series:
    candidates = [
        _first_present_series(raw, ["timestamp", "time", "start_time", "ts", "@timestamp"], default=""),
        normalized["stime"] if "stime" in normalized.columns else None,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        parsed = pd.to_datetime(candidate, errors="coerce", utc=True)
        if bool(parsed.notna().any()):
            return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
    return pd.Series("", index=raw.index)


def _dataset_label(raw: pd.DataFrame, normalized: pd.DataFrame) -> pd.Series:
    if "label" in raw.columns:
        label = raw["label"]
    elif "label" in normalized.columns:
        label = normalized["label"]
    else:
        label = pd.Series("", index=raw.index)
    return label.astype(str).fillna("").str.strip()


def _attack_category(raw: pd.DataFrame, normalized: pd.DataFrame, label: pd.Series) -> pd.Series:
    if "attack_cat" in raw.columns:
        attack = raw["attack_cat"]
    elif "attack_cat" in normalized.columns:
        attack = normalized["attack_cat"]
    else:
        attack = label
    attack = attack.astype(str).fillna("").str.strip()
    return attack.where(~attack.str.lower().isin({"", "nan", "none", "benign", "normal", "0"}), "Normal")


def _default_analyst_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "nan", "none", "unknown"}:
        return "unknown"
    if text in {"0", "benign", "normal"}:
        return "normal"
    return "known_attack"


def _clean_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or "unknown"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0.0)


def _value_counts(series: pd.Series) -> dict[str, int]:
    if not isinstance(series, pd.Series) or series.empty:
        return {}
    counts = series.astype(str).fillna("").value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _min_time(series: pd.Series | None) -> str:
    if series is None:
        return ""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    value = parsed.min()
    return "" if pd.isna(value) else value.isoformat()


def _max_time(series: pd.Series | None) -> str:
    if series is None:
        return ""
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    value = parsed.max()
    return "" if pd.isna(value) else value.isoformat()
