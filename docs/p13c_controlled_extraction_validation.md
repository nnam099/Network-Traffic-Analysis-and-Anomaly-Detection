# P13C controlled extraction validation

P13C attempt 002 is a controlled, independently generated loopback capture used
only to validate portable-feature-schema-v3 extraction. It is not a real-world
benchmark and contains no detector-performance result.

The capture used ordinary-user `dumpcap` access through the preconfigured
`wireshark` group. All packets were restricted to `127.0.0.2` and `127.0.0.3`
on `lo`. The frozen schedule, raw reservation, two independent extraction runs,
post-extraction label join, population audit, identity audit, feature audit and
safety audit are stored under `results/portable_schema/p13c/attempt_002/`.

The authoritative raw PCAP is intentionally outside Git under the repository's
existing `data/*` policy:

```text
path: data/controlled_capture/p13c/attempt_002/session_a_raw.pcap
size: 77284 bytes
sha256: 570fad15116845545d92914aa1caf1848a1d979030352f15c1bd4c6caa84b986
mode: 0444
```

Retention policy: preserve the exact read-only file in controlled local or
content-addressed research storage. Its recorded SHA-256 is authoritative. Do
not reconstruct, filter, rewrite, merge or replace the reserved PCAP. A missing
local copy must be restored only from a byte-identical retained copy whose hash
matches the reservation.

The validated result is `P13C_CONTROLLED_SOURCE_PASS`. It comprises 793 packet
records, 80 extracted flows, 12 NORMAL rows, 68 ATTACK rows, no unmatched or
ambiguous rows, and no mixed-label canonical identities. Both extraction runs
have semantic-content SHA-256
`f65f0a71d35f432011ab4abe7159319499ed6385809d3829b0066ce6d222a526`.

No scaler, checkpoint, neural network, anomaly score, threshold or performance
metric is used by P13C.
