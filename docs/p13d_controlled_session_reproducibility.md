# P13D controlled-session reproducibility

P13D validates the frozen portable-feature-schema-v3 extraction protocol across
two independently executed controlled loopback sessions. It is a data-protocol
reproducibility study, not a detector evaluation or external benchmark.

Session A is bound to P13C freeze commit
`ce0b6be7604acd748015834df3f024e62afb5d07`. Session B uses the same P12/P13
scientific source hashes, 60-second idle timeout, first-observed direction rule,
`IPPROTO_n` mapping, canonical-flow-identity-v3 and post-extraction label join.

The authoritative Session B raw PCAP remains outside Git under the existing
`data/*` policy:

```text
path: data/controlled_capture/p13d/session_b_raw.pcap
size: 70474 bytes
sha256: c2196509d774354f4f1d9b231886abae17f1d4a323d26b2b729c73d85dc9a4a3
mode: 0444
```

Retention policy: preserve the exact read-only file in controlled local or
content-addressed research storage. Restore only a byte-identical copy matching
the reservation hash. Never filter, rewrite, merge or reconstruct the reserved
source.

P13D produced `P13D_REPRODUCIBILITY_PASS`. Session B contains 728 packets and
84 flows; two independent extraction runs share semantic-content SHA-256
`27a4b0b7f56a07a3516f6a3aab3014d527d17a60a439e2925f16e4265c1da886`.
The combined controlled corpus has 164 flows and 164 semantic identities, with
no unmatched, ambiguous or mixed-label identity. There is no exact semantic
identity intersection between sessions.

No ML split, balancing, scaling, optimizer step, model forward pass, anomaly
score, threshold or detector metric was created during P13D.
