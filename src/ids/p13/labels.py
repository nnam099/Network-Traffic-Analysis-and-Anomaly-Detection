"""Post-extraction label-join reference logic; never imported by the extractor."""

from __future__ import annotations

from dataclasses import dataclass

from ids.p12_portable_schema import PortableFlow


@dataclass(frozen=True, slots=True)
class GroundTruthRule:
    rule_id: str
    start_ns: int
    end_ns: int
    source_label: str
    portable_label: str
    ip_protocol: int | None = None
    source_address: str | None = None
    destination_address: str | None = None
    source_port: int | None = None
    destination_port: int | None = None

    def __post_init__(self) -> None:
        if self.end_ns < self.start_ns:
            raise ValueError("ground-truth interval end precedes start")
        if self.portable_label not in {"NORMAL", "ATTACK"}:
            raise ValueError("portable_label must be NORMAL or ATTACK")


@dataclass(frozen=True, slots=True)
class LabelJoinResult:
    status: str
    source_label: str | None
    portable_label: str | None
    matched_rule_ids: tuple[str, ...]


def _matches(flow: PortableFlow, rule: GroundTruthRule) -> bool:
    first = flow.packets[0]
    if not rule.start_ns <= flow.start_ns <= rule.end_ns:
        return False
    checks = (
        (rule.ip_protocol, first.ip_protocol),
        (rule.source_address, first.src_address),
        (rule.destination_address, first.dst_address),
        (rule.source_port, first.src_port),
        (rule.destination_port, first.dst_port),
    )
    return all(expected is None or expected == observed for expected, observed in checks)


def join_labels(flows: list[PortableFlow], rules: list[GroundTruthRule]) -> list[LabelJoinResult]:
    """Join after extraction using flow start and optional tuple constraints.

    This is a tested reference policy, not an active dataset contract.  A
    source-specific adapter must freeze its fields and tolerance after source
    reservation.
    """
    output = []
    for flow in flows:
        matches = sorted((rule for rule in rules if _matches(flow, rule)), key=lambda rule: rule.rule_id)
        if not matches:
            output.append(LabelJoinResult("UNMATCHED", None, None, ()))
            continue
        label_pairs = {(rule.source_label, rule.portable_label) for rule in matches}
        ids = tuple(rule.rule_id for rule in matches)
        if len(label_pairs) != 1:
            output.append(LabelJoinResult("AMBIGUOUS", None, None, ids))
            continue
        source_label, portable_label = next(iter(label_pairs))
        output.append(LabelJoinResult("MATCHED", source_label, portable_label, ids))
    return output
