# P2 representation and ambiguity dry-run

Status: **BLOCKED — no neural training, threshold tuning, or performance metrics.**
P2 is a data-only diagnostic and does not authorize scientific retraining. The
historical canonical (`canonical-model-features-v1`), legacy model-input
(`post-scaling-clipping-float32-v1`), and P1 model-input
(`post-robustscale-float32-sha256fallback-v2`) contracts remain unchanged.

## Evidence and reproduction

- `results/data_quality/p2/float32_forensics.json`: every P1 stage-D newly
  merged fingerprint, including roles, labels, canonical identities, full
  binary64/binary32 vectors and per-feature absolute/relative/ULP differences.
- `results/data_quality/p2/schema_v2.json`: content-hashed P2 source/order/dtype/
  preprocessing/categorical-ID candidate contract.
- `results/data_quality/p2/ambiguity_impact.json`: exact whole-canonical-group
  exclusion impact by source partition, family and target.
- `results/data_quality/p2/p2_configuration_matrix.json`: every 5 × 3 × 4
  dry-run cell with collision taxonomy, split comparison and fail reasons.

Run in the pinned project environment, with official UNSW-NB15 splits already
present under `data/`:

```bash
.venv/bin/python scripts/p2_float32_forensics.py
.venv/bin/python scripts/audit_p2_protocol.py
.venv/bin/python -m pytest -q -p no:cacheprovider
```

`audit_p2_protocol.py` returns code 2 when any candidate fails its integrity
gate; that is expected evidence, not a successful training gate. It verifies
the P1 scaler evidence checksum and reconstructs P1 model-input identities.

The audited 15 cells contain 1,041 new float32 merge-group incidents (the
same source rows may recur across seeds/targets). Every incident joins distinct
canonical identities; 1,022 incidents (98.17%) differ in `sinpkt`, all 1,022
in that feature alone. No new incident is cross-label, but 524 cross roles.
Float64 B removes the conversion-created merges; C and D introduce no new
scaling or dtype merges at their structured boundaries. These findings do not
resolve canonical label conflicts.

The candidate whole-group exclusion removes 2,445 ambiguous identities and
40,400 rows across official train and test. Backdoors retains 295 train and 57
official-test rows; Worms retains 116 and 43; Analysis retains 558 and 58.
All 60 A/B/C/D-by-target/seed data-integrity gates remain **FAIL**. In D,
within-fit cross-label and cross-role canonical conflicts disappear, but 738
canonical identities in known official test still occur in fit roles in every
cell (also 738 structured model-input identities). The original official-test
population remains in the audit rather than being silently removed.

## Candidate protocol

The ordering is: raw official partitions → canonical normalization and
fingerprint → detect conflicting canonical groups across train *and* test →
optionally exclude **all** members of those groups → construct target identity
set → purge target identities from known/surrogate training populations →
canonical-group split of known roles → fit categorical vocabulary and numeric
RobustScaler on backbone training only → transform → model-input identity audit.
No side of a conflicting group is relabelled or retained selectively.

Configuration A is P1's 61 scalar values at float32; B keeps the same P1
features and scaler with a float64 model boundary; C uses 58 continuous float64
values and three *opaque* categorical IDs with the unchanged P1 role
allocation; D adds whole-group exclusion and canonical-group splitting before
C's fitting and transformation. The D population depends on official test
**labels** to identify ambiguous groups. It does **not** use official test rows
for scaler or vocabulary fitting, but test-aware training-population curation
is a separate protocol issue that must be approved or replaced by a train-only
policy before a scientifically independent evaluation can be claimed.

Opaque known IDs are per-feature sorted backbone-vocabulary indexes. Unseen
values use the full, feature-namespaced SHA-256 digest of the normalized token;
IDs are never treated as continuous magnitudes. The report estimates and
measures collisions for hypothetical 1,024, 65,536, 1,048,576 and 16,777,216
OOV-bucket designs, but does not select one. A finite embedding table for
arbitrary unseen tokens cannot guarantee exact identity without an explicit
bounded-vocabulary/OOV policy; no embedding architecture is built here.

All configurations retain a distinct **data-integrity gate** and
**scientific-protocol approval** state. A PASS in the former is not approval to
train. Canonical label conflicts and model-input collisions are reported
separately. Nonzero support for rare families is not evidence of adequate
statistical power; no minimum sample threshold was invented.

Float64 arrays use twice the storage of float32 arrays for the same element
count. TensorFlow/Keras and PyTorch expose float64 dtypes, but a future model
must keep its compute/weights and inputs at a compatible dtype; GPU throughput
and operator support depend on hardware/backend and require separate validation.
Existing 61-scalar float32 checkpoints are not scientifically or structurally
compatible with the new structured P2 schema, and even B's dtype change would
require an explicit checkpoint/dtype migration decision. No neural benchmark
or migration was performed.
