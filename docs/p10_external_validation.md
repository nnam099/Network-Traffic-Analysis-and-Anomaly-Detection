# P10A independent external-source compatibility audit

## Outcome

P10A is fail-closed with status `P10_EXTERNAL_SOURCE_BLOCKED`. No reviewed public
candidate can supply all 58 continuous semantics and all three categorical
semantics required by the frozen P9 interface. No external raw file was
downloaded, opened, parsed, sampled, or hashed, and no P9 checkpoint was opened
or evaluated.

This is a metadata/schema audit, not external validation. It does not authorize
P10B.

## Frozen upstream contract

- P8 protocol SHA-256: `62916073ef6048503415d45e2605bbf475b729e032c838a60cc7e94c43768216`
- P9 protocol SHA-256: `c1c7352a9ca546a3e8e456b112d09fd42f29243e205a8760703b27f68b5baf87`
- P9 freeze commit: `3dd5f12a6bf3948efff817654111347d1fedc7ce`
- P9 artifact aggregate SHA-256: `0f11f518391a66a0ea320e0b3558ea5de9b7440910014f57457f0f0ae9908ed7`
- model boundary: finite `float64[B,58]`, plus frozen per-cell `proto`,
  `service`, and `state` vocabularies with PAD=0 and OOV=1
- selected scorer: `negative_max_softmax` in 27/27 cells
- thresholds: frozen P9 identity-level 0.99 higher quantiles, alert rule
  `score > threshold`

External data may not fit a scaler, vocabulary, embedding, centroid, scorer, or
threshold. Similar column names do not establish semantic equivalence.

## Candidate comparison

| Dataset | Extractor/schema | Independence | Family mapping | Categorical compatibility | Decision |
|---|---|---|---|---|---|
| CIC-IDS2017 | CICFlowMeter, >80 fields | independent UNB/CIC 2017 testbed | partial | protocol only; service/state absent or incompatible | reject |
| CSE-CIC-IDS2018 | CICFlowMeter-V3, 80 fields | independent CSE/CIC testbed | partial | protocol only; service/state absent or incompatible | reject |
| TON_IoT network | Zeek/Bro, 44 network fields plus labels | distinct Industry 4.0/IoT testbed | partial | proto/service usable only through frozen OOV; Zeek state is incompatible | reject |
| Bot-IoT | Argus/CSV | distinct UNSW Canberra botnet testbed | partial | proto/state only; service absent | reject |
| CIC-DDoS2019 | CICFlowMeter-V3, >80 fields | independent two-day UNB/CIC testbed | DDoS-specific | protocol only; service/state absent or incompatible | reject |

The full 61-field classification for every candidate is in
`results/loafo/p10/feature_mapping.json`. Every non-exact classification has a
rationale. Derived fields include their frozen formulas, dependencies, dtype,
clipping, and missing-value policy.

## Decisive compatibility gaps

The public processed products do not jointly reproduce the frozen Argus/Bro
semantics for direction-specific TTL, loss, jitter and load; TCP sequence and
handshake timing; FTP command/login fields; the UNSW last-100-connection context
counts; and all three categorical semantics. CICFlowMeter uses a different flow
construction and feature family. TON_IoT exposes useful Zeek service and HTTP
fields but lacks mandatory Argus fields, and its `conn_state` is not the UNSW
Argus state machine. Bot-IoT is closer for a small Argus subset but still omits
many mandatory fields and the frozen service semantic.

Zeros, medians, port-derived services, vocabulary extension, and score-guided
remapping are not permitted substitutes. Raw PCAP re-extraction also cannot be
approved yet: the repository lacks a frozen, versioned extractor that has been
shown to reproduce all 61 UNSW semantic fields against known reference records.

## Label contract

All label mappings are provisional and inactive because schema compatibility
failed. The audit explicitly does not equate CIC/TON/Bot DDoS with the
UNSW-NB15 `DoS` family. If a future source passes the schema gate, the default
mode is generic unseen attack versus external normal. Family-specific claims
require a direct, preregistered semantic mapping; missing P9 families are never
fabricated.

## Identity and population contracts

The proposed identity is a label-, row-index-, timestamp-, and score-independent
SHA-256 over canonical pre-scaled semantic features and normalized categorical
tokens. It remains inactive until a source passes compatibility and is reserved.
Consequently, row, identity, class-support, duplicate, and categorical-OOV counts
are intentionally `null`, not estimated. Computing them from an unselected
source would not cure the schema failure.

## Independence and contamination audit

Official metadata describes all five candidates as captures from testbeds other
than UNSW-NB15. No reviewed source documentation states that any candidate
incorporates UNSW-NB15 records. This is a metadata-level finding only: no
content-level overlap test was performed because no raw source passed the schema
gate or was reserved.

## Source documentation consulted

- https://www.unb.ca/cic/datasets/ids-2017.html
- https://www.unb.ca/cic/datasets/ids-2018.html
- https://www.unb.ca/cic/datasets/ddos-2019.html
- https://github.com/ahlashkari/CICFlowMeter/blob/master/ReadMe.txt
- https://research.unsw.edu.au/projects/toniot-datasets
- https://research.unsw.edu.au/projects/bot-iot-dataset
- https://arxiv.org/abs/1811.00701

These pages and papers were used only as provenance/schema metadata. They are
not reserved dataset files.

## What would unblock P10A

Collect or identify an independent raw PCAP source with defensible attack and
normal ground truth, then first freeze a versioned packet-to-P9 extractor. The
extractor must reproduce every frozen semantic, including capture-level context
features, on a separate reference corpus without fitting to external labels or
scores. Only after that precondition passes should the raw file list and SHA-256
hashes be reserved and population/OOV audits run. P9 model access must remain
disabled until the resulting P10 preregistration is immutable.
