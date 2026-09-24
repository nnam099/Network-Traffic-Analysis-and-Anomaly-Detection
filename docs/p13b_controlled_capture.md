# P13B controlled independent raw-PCAP capture

`P13B_CONTROLLED_CAPTURE` is a small extraction-validation corpus generated entirely on Linux loopback. It is not a real-world benchmark, production capture, deployment validation, or detector test.

The generator is `127.0.0.2`; disposable target services bind to `127.0.0.3`. `tcpdump` captures only packets exchanged between those two addresses on `lo`. No scenario sends traffic to a public or unrelated system.

The schedule is serialized and hashed before capture. It contains normal ICMP, HTTP, UDP echo, short TCP and long TCP scenarios, plus bounded local port enumeration, HTTP request burst, and failed authentication against a disposable custom service. Labels are joined after packet parsing, flow reconstruction, feature extraction, and canonical identity computation.

The authoritative PCAP is stored under `data/controlled_capture/p13b/`, excluded from Git by the existing data policy, SHA-256 reserved, and changed to read-only immediately after capture. Artifacts and separated model-input/private-provenance tables are under `results/portable_schema/p13b/`.

P13B reuses the frozen P12 formulas and P13 parser. It performs no scaling, model loading, scoring, thresholding, feature selection, or performance evaluation.
