# P13 raw-source reservation and deterministic extraction protocol

## Status

P13 currently fails closed at `P13_SOURCE_BLOCKED`. No raw dataset object has been downloaded, opened, parsed, or hashed. The reference parser and extraction components are validated only on synthetic captures and are not activated for a dataset.

P12 is frozen at commit `720789fb90f36e1413cef47cd96f3126d2b8f3a4`, protocol SHA-256 `db6405fcac5d8fd94e406660ea60c22853a9c0073762df0e3f21881fe806fec3`, and schema SHA-256 `db430498147a0601f0d1db9ef5579e210ec9795fa871043d0ab1bf5d328adb7a`.

## Source triage

- UNSW-NB15 documents about 100 GB of PCAP, but the official download currently redirects to authenticated SharePoint.
- CIC-IDS2017 documents 7.8–13 GB per day and detailed attack intervals, but the official download workflow requires a personal-information form which returned a server error during this audit.
- CSE-CIC-IDS2018 exposes a public S3 bucket and documented attack intervals, but each daily raw archive is 38.5–59.1 GB. Objects expose multipart ETags, not SHA-256. Automatically transferring an archive merely to test suitability was rejected.
- TON_IoT and Bot-IoT document PCAP data but redirect to authenticated SharePoint; Bot-IoT documents 69.3 GB of PCAP.
- CIC-DDoS2019 documents PCAP and labeled flow products, but no direct immutable raw-object URL and SHA-256 were available outside its download workflow.

The source gate requires exact bytes, size, per-file SHA-256, aggregate SHA-256, license, source URL, and documentation bindings before parsing. Metadata listing is not reservation.

## Candidate parser contract

The unactivated parser candidate supports classic PCAP 2.4 with Ethernet (`DLT_EN10MB`), up to two VLAN tags, unfragmented IPv4, basic IPv6 without extension headers, TCP, UDP, ICMP, and other IP protocol numbers. All timestamps are normalized to integer nanoseconds.

IPv4 and IPv6 fragments are excluded and counted; P13 performs no implicit reassembly. Truncated captures, malformed packets, non-IP frames, unsupported IPv6 extensions and unsupported encapsulations receive separate audit counts. File-level corruption and unsupported PCAP/link formats fail closed.

This scope must be compared with the reserved capture before activation. PCAPNG, non-Ethernet link types, IPv6 extension headers, or fragment-dependent coverage require a protocol amendment or a new parser version.

## Flow and feature contract

Packets are ordered by timestamp and unique capture index. The first observed packet defines forward direction. The reconstruction key is the unordered endpoint pair plus IP protocol. A gap exactly equal to 60 seconds remains in the flow; a gap strictly greater than 60 seconds starts a new flow.

The extractor delegates formulas to the frozen P12 implementation and emits 18 nullable float64 values, an 18-position boolean missing mask, canonical `IPPROTO_n`, and `canonical-flow-identity-v3`. It performs no scaling or imputation. Label code is isolated from the portable extractor.

## Label-join activation

The tested generic join uses an inclusive flow-start interval and optional protocol/endpoint/port constraints. It preserves `source_label` and `portable_label`, emits `UNMATCHED` and `AMBIGUOUS`, and never applies majority voting. It is not an active scientific label contract: exact fields, timezone, tolerance, NAT interpretation and conflict handling must be frozen from the selected source documentation after reservation.

## Required continuation

Resume P13 only after one of the following:

1. Provide authenticated access to an official UNSW/TON/Bot source.
2. Complete the CIC download workflow and supply a stable raw URL or local raw files.
3. Explicitly authorize acquisition of a named CSE-CIC-IDS2018 archive of at least 38.5 GB.
4. Supply an independently obtained, provenance-documented raw candidate and its ground-truth files.

After reservation, run capture-format audit, activate or amend the parser, extract twice, compare semantic-content hashes, join labels, and generate feature, identity, ambiguity and population audits. No model training belongs in P13.
