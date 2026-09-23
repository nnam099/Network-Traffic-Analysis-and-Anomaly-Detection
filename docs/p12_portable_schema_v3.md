# P12 portable feature schema v3

## Scientific status

P12 starts a new scientific branch. `portable-feature-schema-v3` is not an adapter for the frozen 61-input P9 detector, and no P9 checkpoint, scaler, scorer, calibration threshold, or performance result transfers to this branch. A future detector must be trained from scratch under a separately preregistered protocol.

P12 is design-only. It does not reserve or open an external dataset, construct train/test populations, train a neural network, compute anomaly scores, or evaluate P7/P9.

## Design principles

The baseline keeps only quantities that can be computed from timestamped IPv4/IPv6 headers with deterministic, dataset-independent rules. Feature inclusion is based on semantic reproducibility, raw-capture availability, numerical stability, and network-behavior relevance—not predictive performance.

The flow direction is the source endpoint of the first observed packet after sorting by `(timestamp_ns, capture_index)`. A flow key is an unordered endpoint pair plus the IANA IP protocol number. An idle gap strictly greater than 60 seconds creates a new flow. P13 must audit non-IP frames, truncation, fragmentation, duplicate packet records, capture loss, and label joins; P12 does not silently guess those policies.

IP-layer bytes mean the IPv4 `total_length`, or 40 plus the IPv6 `payload_length`. They include the IP header and exclude link-layer bytes. This definition deliberately differs from historical Argus/Bro/CICFlowMeter byte fields where extractor-specific payload/header conventions can differ.

## Baseline feature set

The baseline has 19 semantic fields: 18 `float64` continuous fields and one controlled categorical token.

Continuous fields, in frozen order:

1. `duration_seconds`
2. `forward_packet_count`
3. `backward_packet_count`
4. `total_packet_count`
5. `forward_byte_count`
6. `backward_byte_count`
7. `total_byte_count`
8. `forward_packet_fraction`
9. `forward_byte_fraction`
10. `mean_packet_size_bytes`
11. `forward_mean_packet_size_bytes`
12. `backward_mean_packet_size_bytes`
13. `packets_per_second`
14. `bytes_per_second`
15. `forward_packets_per_second`
16. `backward_packets_per_second`
17. `forward_bytes_per_second`
18. `backward_bytes_per_second`

The categorical field is `ip_protocol`, normalized as `IPPROTO_0` through `IPPROTO_255`. Integer encoding is fixed and data-independent: PAD is 0, OOV is 1, and `IPPROTO_n` is `n + 2`. Valid IANA numbers never become OOV; malformed adapter tokens do.

No payload parsing, application service, source/destination identity, raw port, TTL, TCP sequence number, loss estimate, or historical `ct_*` context statistic is in the baseline. `service` is removed rather than approximated from a port.

## Optional transport extension

P12 freezes auditable reference formulas for IAT summaries, TCP flag counts, ordered three-way-handshake intervals, and `portable_connection_state`. They are excluded from the baseline input and cannot be added later without an explicit amendment.

The state vocabulary is `NOT_TCP`, `TCP_RESET`, `TCP_CLOSED`, `TCP_ESTABLISHED`, `TCP_PARTIAL_HANDSHAKE`, and `TCP_OBSERVED`, with that rule precedence. It is a local packet-observation state machine. It is explicitly neither an Argus state nor a Zeek `conn_state` mapping.

## Missing values

P12 distinguishes four semantic cases:

- `ZERO_IS_VALID_ABSENCE`: observed counts such as zero backward packets or zero SYN flags.
- `NULL_NOT_APPLICABLE`: rates for a zero-duration flow and handshake times for non-TCP.
- `NULL_INSUFFICIENT_PACKETS`: directional means/IAT/handshake quantities without enough observations.
- `NULL_EXTRACTION_FAILURE`: parser, truncation, or unresolved-fragment failure.

There is no epsilon denominator, minimum-duration clamp, or structural zero imputation. A future model receives continuous values and a same-width missing mask. Only after fold-local scaling may masked numeric slots be filled with zero, while the mask remains a separate input.

## Canonical identity

`canonical-flow-identity-v3` hashes the ordered 19 semantic inputs with version tags. Continuous values use big-endian binary64 hex (with canonical positive zero) or null; protocol uses the canonical token. Labels, scores, predictions, row numbers, timestamps, addresses, ports, file paths, and dataset names are excluded.

This semantic identity supports duplicate audits and group-aware splitting. It is distinct from the reconstruction/provenance key used to locate a flow in a capture.

## Cross-dataset design compatibility

The compatibility matrix covers UNSW-NB15, CIC-IDS2017, CSE-CIC-IDS2018, TON_IoT network, Bot-IoT, and CIC-DDoS2019. Their official project pages document raw PCAP availability. Therefore the baseline is classified as directly readable (`ip_protocol`) or derivable (flow aggregates) from a common normalized packet representation, without structural imputation.

This is a metadata-level implementability conclusion, not empirical validation. No raw file was opened. P13 must reserve exact files and hashes, verify capture/link-layer formats, implement deterministic adapters, audit label-to-flow joins, and demonstrate coverage before any population freeze or training.

Official metadata sources:

- <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- <https://www.unb.ca/cic/datasets/ids-2017.html>
- <https://www.unb.ca/cic/datasets/ids-2018.html>
- <https://research.unsw.edu.au/projects/toniot-datasets>
- <https://research.unsw.edu.au/projects/bot-iot-dataset>
- <https://www.unb.ca/cic/datasets/ddos-2019.html>

## Future model interface

A future model receives `float64[batch,18]` continuous values, `bool[batch,18]` missing masks, and `int64[batch]` protocol IDs. Preprocessing is fit only on development-training identities, using observed values. External data never influences scaling or vocabulary.

The untrained conservative design is a small continuous encoder over values plus mask, a four-dimensional protocol embedding, and compact fusion/reconstruction heads. P12 makes no architecture-performance claim. Within-dataset LOAFO and cross-dataset OOD remain separate future evaluation tracks.

## Phase boundary

P13 may implement extraction and data reservation against this frozen schema. P14 may define development-only training, P15 within-dataset LOAFO evaluation, and P16 cross-dataset OOD evaluation. Each requires a new gate; `P12_PORTABLE_SCHEMA_PASS` alone authorizes none of them.
