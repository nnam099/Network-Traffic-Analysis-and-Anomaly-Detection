# P11 reproducible 61-semantic feature extractor

## Outcome

P11 returns `P11_REFERENCE_CORPUS_BLOCKED`. The official UNSW-NB15 page states
that PCAP, Argus, Bro, CSV, ground-truth and report files exist, but this
repository does not contain an immutable row-alignable raw-capture/expected-row
pair. The official download link currently redirects to Microsoft authentication.
The local train/test CSV partitions omit source/destination identities and
timestamps needed to associate their rows with raw flows.

No exact-schema claim is made from column names, synthetic tests, or distribution
similarity. P11 does not authorize external evaluation.

## Upstream scientific boundary

- P8 protocol: `62916073ef6048503415d45e2605bbf475b729e032c838a60cc7e94c43768216`
- P9 protocol: `c1c7352a9ca546a3e8e456b112d09fd42f29243e205a8760703b27f68b5baf87`
- P9 freeze commit: `3dd5f12a6bf3948efff817654111347d1fedc7ce`
- P10A: `P10_EXTERNAL_SOURCE_BLOCKED`

The extractor package imports neither PyTorch nor P9 code. It cannot load a
checkpoint, calculate anomaly scores/AUROC, alter a threshold, or fit a scaler or
vocabulary.

## Authoritative schema

`results/extractor/p11/schema_contract.json` records all 58 continuous and three
categorical semantics in frozen order. Every field includes type, unit,
directionality, aggregation window, flow semantic, source dependency, formula,
missing behavior, categorical meaning, provenance and semantic class.

Repository-engineered formulas are copied from `src/ids/dataset.py`. Their
`1e-8` denominator epsilon and `1e-6` duration floor are explicit. Candidate raw
rate/load calculations do not silently inject these constants: zero-duration raw
flows produce explicit null rates. P11 uses float64 before any P9 scaler.

## Original extraction lineage

Authoritative documentation identifies the pipeline as:

1. IXIA PerfectStorm generated normal and attack traffic.
2. `tcpdump` captured about 100 GB of PCAP.
3. Argus server/client created binary Argus and bidirectional-flow fields.
4. Bro-IDS produced connection, HTTP and FTP logs.
5. Argus and Bro outputs were loaded into SQL Server 2008 and matched using flow
   fields.
6. Twelve C# algorithms generated the additional/context fields.
7. IXIA event ground truth was attached using the GT/LIST_EVENTS artifacts.

The historical Argus/Bro versions, command lines, configs, merge rules and C#
source are not frozen. Modern Zeek state is not treated as historical Argus
state.

Sources:

- https://research.unsw.edu.au/projects/unsw-nb15-dataset
- https://doi.org/10.1109/MilCIS.2015.7348942

## Candidate extractor architecture

The standard-library-only package under `src/ids/p11_extractor/` contains:

- `packet_reader.py`: classic Ethernet PCAP reader; nanosecond integer
  timestamps; explicit rejection of PCAPNG, fragmented IPv4 and unparsed IPv6
  extension headers.
- `flow_builder.py`: deterministic bidirectional five-tuple grouping, first
  packet direction and explicit idle timeout. The timeout is a candidate policy,
  not an Argus-equivalence claim.
- `continuous_features.py`: candidate duration, directional packet/byte, TTL,
  timing, jitter, load/rate and TCP handshake computations; packet loss and
  historical context remain null.
- `categorical_features.py`: IANA protocol tokens, conservative payload-evidenced
  HTTP/FTP candidates, and a `LOCAL_*` TCP summary deliberately kept outside the
  Argus token namespace.
- `context_features.py`: fail-closed boundary for unavailable historical C#
  last-N semantics.
- `validation.py`: per-unit tolerances and row-wise exact/error diagnostics.
- `serialization.py`: sorted, finite-only deterministic JSON.

## Flow and temporal policy

Packets are sorted by `(timestamp_ns, capture_index)`. A bidirectional key uses
IP version, two address/port endpoints and IP protocol. Direction is fixed by the
first packet. Timeout is configurable and defaults to 60 seconds solely for the
candidate implementation. Singleton-direction IAT and jitter candidates are
zero; zero-duration rate/load is null. These behaviors require reference
validation before any compatibility claim.

## Categorical and application policy

`proto` is derived from the IANA protocol number. `service` is never guessed from
destination port; only recognizable payload supplies a candidate. `state` emits
`LOCAL_*`, not Argus or Zeek labels. FTP login/commands and HTTP depth/body fields
require payload; absent/encrypted/truncated payload remains unavailable.

## Context and loss gaps

The exact last-100 connection inclusion, ordering, tie, host/service and
capture-boundary rules are unavailable, so all contextual calls fail closed.
Packet loss is capture-point and tool-semantic dependent; it is null rather than
silently set to zero. These gaps alone prevent exact P9 compatibility.

## Validation and determinism

Synthetic tests cover ordering, bidirectional grouping, timeout, duration,
directional bytes/packets, TTL, explicit loss gaps, IAT, jitter, rate/load,
zero-duration behavior, TCP handshake timing, protocol, payload-based service,
local state, context failure, PCAP parsing and deterministic serialization.

Synthetic correctness is not reference semantic validation. All continuous
reference counts and errors remain empty until a cryptographically pinned,
row-alignable corpus is available.

## Required unblock

Acquire from the authoritative source and hash a bounded reference bundle that
contains matching PCAP, Argus/Bro intermediates, full UNSW-NB15 CSV rows,
GT/LIST_EVENTS, tool/config metadata and an explicit row association. Freeze the
subset before running the extractor. Then populate all 61 comparison rows under
the already-declared per-unit tolerances. If the historical context/state/service
semantics remain unrecoverable, close P11 as `P11_EXACT_SCHEMA_BLOCKED` and move
any portable subset to a separate protocol-amendment phase.
