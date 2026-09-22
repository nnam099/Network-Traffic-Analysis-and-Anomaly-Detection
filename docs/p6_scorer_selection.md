# P6 — development-only scorer selection and known-only calibration

P6 extends the frozen P3 data, P4 float64 model interface and P5 training
protocol without modifying them. It is a development model-selection study,
not an external detection study. The source CSV loader accepts only the exact
hash-checked official-training file. No official-test artifact is opened.

## Information flow and roles

```text
TRAIN ------------------> P5 representation/head fitting
  |                       P5 fixed CE objective and optimizer
  +---------------------> 32-D known-class centroids (train only)
VAL --------------------> checkpoint selection by P5 validation CE
                             |
                             v checkpoint frozen
META_KNOWN + SURROGATE --> identity-level AUROC for two frozen scorer families
                             |
                             v scorer family frozen
CALIBRATION_KNOWN ------> 0.99 known-only score quantile, method="higher"
                             |
                             v threshold frozen; strict score > threshold
TARGET_HELD_OUT --------X
OFFICIAL_TEST -----------X (future external-evaluation phase only)
```

No true target-family or official-test observation contributed to training,
checkpoint selection, scorer selection, scorer fitting, or threshold calibration.
Roles are checked at materialization and operation boundaries; canonical IDs,
row IDs, source role and labels are rechecked against each P3 manifest.

## Frozen scorer implementation

`negative_max_softmax` scores `1 - max softmax(logits)` from the temporary P5
known-family head. It fits no new parameters. Its logical state hash binds the
checkpoint state hash, known-class order and formula version.

`nearest_centroid_l2` fits one 32-D float64 mean representation per known class
from TRAIN alone, in the fixed P5 class order. Its score is the minimum
Euclidean distance to those centroids. Its state hash binds the ordered labels,
little-endian float64 centroid bytes, dimension and checkpoint state hash.
Both scores have the frozen direction “higher is more OOD-like.” No third
scorer, hyperparameter search or architecture change is authorized.

## Canonical-identity selection

The primary selection unit is canonical identity, consistent with P3's
group-isolated roles. For each identity, P6 takes the arithmetic mean of its
row scores. Input order cannot change that mean: each group's values are
sorted and summed with `math.fsum`. AUROC uses `meta_known` identities as 0
and the fixed four-family `surrogate_ood` identities as 1. This is labeled
**SURROGATE DEVELOPMENT AUROC**. Row-level AUROC and row-versus-identity
score summaries are diagnostics only; they never choose a scorer.

The higher identity-level AUROC wins. Numerical ties within `1e-12` choose
`nearest_centroid_l2`, preserving P5's frozen exact-tie preference while
making numerical equality explicit. P6 does not adopt the optional softmax
simplicity recommendation because that would reverse the P5 tie preference.
Each target × seed selects independently. Seed stability is descriptive and
never forces a global scorer.

Surrogate-based scorer selection estimates development discrimination against
the predefined surrogate unknown families only. It does not establish
performance on the held-out target family. Surrogate composition is fixed by
P3; it is not a target-family sample or an unbiased proxy for one.

## Calibration and diagnostics

Calibration is opened only after selection artifacts are persisted and the
pilot selection rerun has reproduced. For every cell, use only known
`calibration` scores. The threshold is
`np.quantile(scores, 0.99, method="higher")`; alerts require `score > threshold`,
so equality remains on the known side. This is a known-only calibration
operating point, not a target-optimized threshold. A nominal 1% upper-tail
construction does not guarantee an exact 1% future false-positive rate.

After threshold freeze, row and identity score summaries and alert fractions
may be recorded for meta, surrogate and calibration, explicitly labeled
`DEVELOPMENT_DIAGNOSTIC_ONLY`. They cannot alter scorer, training or threshold.

## Stages and provenance

Stage A validates all 15 cells without an optimizer. Stage B trains
Fuzzers/42/10042 with exact P5 settings and selects a scorer without opening
calibration. Stage C repeats the identical pilot, verifies selected epoch,
model state, centroid state, selection statistics and selected scorer, then
fits and compares known-only thresholds. Stage D executes the same frozen
protocol for all 15 cells; the verified pilot checkpoint is reused for its
Fuzzers/42 cell. No adaptive reruns or changes are permitted within this P6
namespace.

The separate `results/model_selection/p6/` namespace stores immutable per-run
training manifests, checkpoint references, scorer states and selections,
calibration artifacts and development diagnostics. Threshold provenance binds
target and seeds; P3, P4, P5 and P6 hashes; checkpoint/scorer state hashes;
calibration row/identity hashes; quantile, method and threshold. Any changed
component fails validation. P5 checkpoints and artifacts are never overwritten.

`P6_PROTOCOL_PASS`, `P6_PILOT_PASS` and `P6_DEVELOPMENT_PASS` certify staged
contract execution only. They do not establish zero-day effectiveness or
authorize use of official test before a separately approved external phase.
