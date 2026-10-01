from __future__ import annotations

from typing import Any

import pandas as pd

from ids.inference_runtime import risk_score, traffic_verdict


PRIORITY_FEATURES = [
    "true_label", "attack_cat", "label", "dur", "sbytes", "dbytes",
    "sload", "dload", "spkts", "dpkts", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "state_num", "proto_num", "service_num",
]


def enrich_ood_row(row_scores: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row_scores)
    classifier_class = str(enriched.get("classifier_class", enriched.get("predicted_class", "Unknown")))
    enriched["risk"] = risk_score(
        {
            "hybrid_score": enriched.get("hybrid_score", 0),
            "ae_score": enriched.get("ae_score", 0),
            "is_zeroday": bool(enriched.get("is_zeroday", False)),
        },
        "HIGH" if enriched.get("is_zeroday") else "MEDIUM",
    )
    enriched["classifier_class"] = classifier_class
    enriched["detection"] = str(
        enriched.get("detection") or traffic_verdict(enriched.get("is_zeroday"), classifier_class)
    )
    return enriched


def build_feature_table(feature_row: pd.Series, search: str = "") -> pd.DataFrame:
    feature_table = pd.DataFrame({
        "Feature": feature_row.index.astype(str),
        "Value": feature_row.astype(str).values,
    })
    priority = feature_table["Feature"].isin(PRIORITY_FEATURES)
    feature_table = pd.concat([feature_table[priority], feature_table[~priority]], ignore_index=True)
    if search:
        feature_table = feature_table[
            feature_table["Feature"].str.contains(search, case=False, na=False)
        ]
    return feature_table


def build_score_table(row_scores: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": key, "Value": value}
        for key, value in row_scores.items()
        if key != "source_row"
    ])
