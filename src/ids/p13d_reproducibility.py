"""Pure comparison helpers for P13D controlled-session reproducibility."""

from __future__ import annotations

from collections import Counter


def compare_identity_label_maps(
    session_a: dict[str, set[str]], session_b: dict[str, set[str]]
) -> dict[str, object]:
    """Surface exact semantic overlap and every cross-session label relation."""
    identities_a = set(session_a)
    identities_b = set(session_b)
    intersection = identities_a & identities_b
    union = identities_a | identities_b
    relationship_counts: Counter[str] = Counter()
    details = []
    for identity in sorted(intersection):
        labels_a = sorted(session_a[identity])
        labels_b = sorted(session_b[identity])
        relationship = f"A_{'+'.join(labels_a)}/B_{'+'.join(labels_b)}"
        relationship_counts[relationship] += 1
        details.append(
            {
                "canonical_identity_v3": identity,
                "session_a_labels": labels_a,
                "session_b_labels": labels_b,
                "relationship": relationship,
            }
        )
    cross_label_count = sum(
        count
        for relationship, count in relationship_counts.items()
        if relationship in {"A_NORMAL/B_ATTACK", "A_ATTACK/B_NORMAL"}
    )
    return {
        "session_a_unique_identities": len(identities_a),
        "session_b_unique_identities": len(identities_b),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard_overlap": len(intersection) / len(union) if union else 0.0,
        "cross_session_label_relationship_counts": dict(sorted(relationship_counts.items())),
        "cross_label_intersection_count": cross_label_count,
        "meaningful_cross_label_rule": "cross_label_intersection_count >= 1",
        "group_identity_review_required": cross_label_count >= 1,
        "identity_overlap_is_not_packet_duplication": True,
        "details": details,
    }
