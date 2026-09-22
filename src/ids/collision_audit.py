"""Stage-aware collision taxonomy and unapplied canonical ambiguity policy."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def collision_taxonomy(fingerprints: dict, labels: dict) -> dict:
    """Counts are unique fingerprints per category/pair, not row pair counts.

    Categories can overlap: a fingerprint can carry same-label duplicate rows
    and cross-label rows, or appear in more than two roles.
    """
    tables = {
        role: pd.DataFrame({"fp": values, "label": np.asarray(labels[role]).astype(str)})
        for role, values in fingerprints.items()
    }
    same_role = {}
    for role, table in tables.items():
        counts = table.groupby(["fp", "label"]).size()
        same = set(counts[counts > 1].index.get_level_values("fp"))
        label_counts = table.groupby("fp")["label"].nunique()
        cross = set(label_counts[label_counts > 1].index)
        same_role[role] = {
            "same_label_fingerprints": len(same),
            "same_label_duplicate_rows_beyond_first": int((counts - 1).clip(lower=0).sum()),
            "cross_label_fingerprints": len(cross),
            "cross_label_affected_rows": int(table["fp"].isin(cross).sum()),
        }
    role_labels = {
        role: table.drop_duplicates().groupby("fp")["label"].agg(frozenset).to_dict()
        for role, table in tables.items()
    }
    pairs = {}
    cross_same, cross_different = set(), set()
    for left, right in itertools.combinations(tables, 2):
        common = role_labels[left].keys() & role_labels[right].keys()
        same = {fp for fp in common if role_labels[left][fp] & role_labels[right][fp]}
        different = {
            fp for fp in common if len(role_labels[left][fp] | role_labels[right][fp]) > 1
        }
        cross_same.update(same)
        cross_different.update(different)
        pairs[f"{left}__{right}"] = {
            "shared_fingerprints": len(common),
            "same_label_fingerprints": len(same),
            "cross_label_fingerprints": len(different),
        }
    target_fps = set(fingerprints.get("target", []))
    target_vs_roles = {
        role: len(target_fps & set(values)) for role, values in fingerprints.items()
        if role != "target"
    }
    return {
        "count_unit": "unique fingerprints per category/pair; overlapping categories are not additive",
        "same_role": same_role,
        "cross_role_pairs": pairs,
        "cross_role_same_label_fingerprints_unique": len(cross_same),
        "cross_role_cross_label_fingerprints_unique": len(cross_different),
        "target_vs_fit": len(target_fps & set().union(*(
            set(v) for k, v in fingerprints.items() if k != "target"
        ))),
        "target_vs_roles": target_vs_roles,
    }


def collision_gate(taxonomy: dict, canonical_target_intersections: dict) -> dict:
    reasons = []
    for role, counts in taxonomy["same_role"].items():
        if counts["cross_label_fingerprints"]:
            reasons.append({"severity": "CRITICAL FAIL", "reason": "same_role_cross_label",
                            "role": role, "fingerprints": counts["cross_label_fingerprints"]})
    for role, count in canonical_target_intersections.items():
        if count:
            reasons.append({"severity": "CRITICAL FAIL", "reason": "canonical_target_contamination",
                            "role": role, "fingerprints": count})
    for role, count in taxonomy["target_vs_roles"].items():
        if count:
            reasons.append({"severity": "CRITICAL FAIL", "reason": "target_vs_fit",
                            "role": role, "fingerprints": count})
    for pair, counts in taxonomy["cross_role_pairs"].items():
        if counts["cross_label_fingerprints"]:
            reasons.append({"severity": "CRITICAL FAIL", "reason": "cross_role_cross_label",
                            "pair": pair, "fingerprints": counts["cross_label_fingerprints"]})
        if counts["same_label_fingerprints"]:
            reasons.append({"severity": "FAIL", "reason": "cross_role_same_label",
                            "pair": pair, "fingerprints": counts["same_label_fingerprints"]})
    return {
        "gate": "FAIL" if reasons else "PASS",
        "severity": "CRITICAL FAIL" if any(r["severity"] == "CRITICAL FAIL" for r in reasons)
                    else "FAIL" if reasons else "PASS",
        "fail_reasons": reasons,
        "same_role_same_label_policy": "REPORT ONLY",
        "same_role_cross_label_policy": "CRITICAL FAIL; unresolved ambiguity; no label selection",
    }


def stage_audit(stage_fingerprints: dict, labels: dict) -> dict:
    """Locate every newly merged identity group at its first transition.

    A transition merge is an output fingerprint containing >1 input-stage
    fingerprints. Already identical canonical events are not transform-created.
    Complete merged-group manifests are retained, with roles and semantic labels.
    """
    reports = {}
    previous = None
    for stage, parts in stage_fingerprints.items():
        taxonomy = collision_taxonomy(parts, labels)
        table = pd.concat([
            pd.DataFrame({"fp": values, "role": role, "label": labels[role]})
            for role, values in parts.items()
        ], ignore_index=True)
        counts = table.groupby("fp").size()
        merges = []
        if previous is not None:
            table["previous_fp"] = np.concatenate([previous[role] for role in parts])
            distinct = table.groupby("fp")["previous_fp"].nunique()
            selected = distinct[distinct > 1].index
            for fp, group in table[table["fp"].isin(selected)].groupby("fp"):
                merges.append({
                    "fingerprint": str(int(fp)), "rows": len(group),
                    "upstream_fingerprints": [str(int(v)) for v in sorted(group["previous_fp"].unique())],
                    "roles": sorted(group["role"].unique()),
                    "labels": sorted(group["label"].unique()),
                })
        reports[stage] = {
            "rows": len(table), "duplicate_fingerprints": int((counts > 1).sum()),
            "duplicate_rows_beyond_first": int((counts - 1).clip(lower=0).sum()),
            "newly_merged_fingerprints_from_previous_stage": len(merges),
            "newly_merged_groups": merges, "taxonomy": taxonomy,
        }
        previous = parts
    return reports


def ambiguity_policy_report(context: dict) -> dict:
    from .target_isolation import CANONICAL_FINGERPRINT_COLUMN as canonical

    combined = pd.concat([context["train"], context["test"]], ignore_index=True)
    counts = combined.groupby(canonical)["attack_cat"].nunique()
    ambiguous = counts[counts > 1].index
    removed = combined[combined[canonical].isin(ambiguous)]
    groups = []
    for fp, frame in removed.groupby(canonical):
        groups.append({
            "canonical_fingerprint": str(int(fp)), "rows": len(frame),
            "labels": frame["attack_cat"].value_counts().sort_index().to_dict(),
            "source_counts": frame["__source_role"].value_counts().sort_index().to_dict(),
        })
    return {
        "policy": "exclude_entire_ambiguous_canonical_group_v1",
        "active": False,
        "diagnostic_population": "official train + test; hypothetical loss only",
        "requires_explicit_activation": True,
        "scientific_caveat": (
            "Using test labels to define ambiguity exclusions changes the evaluation population; "
            "requires a declared test-aware curation protocol and separate original-population reporting."
        ),
        "groups": groups, "ambiguous_fingerprints": len(ambiguous), "rows_lost": len(removed),
        "per_family_rows_lost": removed["attack_cat"].value_counts().sort_index().to_dict(),
        "per_source_per_family_rows_lost": {
            str(source): frame["attack_cat"].value_counts().sort_index().to_dict()
            for source, frame in removed.groupby("__source_role")
        },
    }


def ambiguity_exclusion_mask(frame: pd.DataFrame, ambiguous_fingerprints, *, activate=False):
    """Prepared opt-in policy: exclude whole groups, never choose a winning label."""
    if not activate:
        return np.ones(len(frame), dtype=bool)
    return ~frame["__canonical_feature_fingerprint"].isin(
        np.asarray(list(ambiguous_fingerprints), dtype=np.uint64)
    ).to_numpy()
