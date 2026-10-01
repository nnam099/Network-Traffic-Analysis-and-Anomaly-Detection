"""Fail-closed boundary for historical last-N connection semantics."""

from __future__ import annotations


CONTEXT_FEATURES = (
    "ct_state_ttl", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
)


class UnverifiedContextSemantics(RuntimeError):
    pass


def historical_unsw_context_features(*_args, **_kwargs):
    raise UnverifiedContextSemantics(
        "original C# last-N ordering, tie, host/service and capture-boundary rules are not available"
    )
