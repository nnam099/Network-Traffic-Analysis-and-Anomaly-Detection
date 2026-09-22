# P7 — preregistered external evaluation, Stage A only

P7 freezes a future, one-way external-evaluation protocol for the 15 P6 target ×
seed detectors. Its current status is `P7_PREREGISTERED`: no new official-test
CSV access, model scoring, external metric, or Stage B execution is permitted
by this status. The frozen protocol and its SHA-256 are in
`results/external_evaluation/p7/preregistration/`. P3's prior external audit is
bound by file hash only during Stage A; it is not parsed to generate new
test-derived information. P3–P6 scientific artifacts remain immutable.

## Primary scientific question and population

For each target and split seed, does the frozen P6 detector rank previously
unseen **CLEAN official-test canonical identities** of that exact target above
the CLEAN known/background identities? The five frozen known families are the
negative class. The cell's own target family is the positive class. Other OOD
families are excluded from that cell's primary binary task. Only P3's CLEAN
surface is external evidence; `DEVELOPMENT-OVERLAP`,
`TEST-INTERNAL-AMBIGUOUS`, and `TRAIN-CONFLICT-RELATED` are separate,
diagnostic-only surfaces and never alter the primary result.

The primary unit is canonical identity. Each identity receives the arithmetic
mean of its row anomaly scores, calculated using P6's deterministic identity
aggregation. Its already-audited CLEAN family label must be unique; a label
conflict fails closed. Row-level metrics are explicitly marked `ROW-LEVEL
DIAGNOSTIC`, never substituted for identity-level estimates. Each of the three
seeds is reported separately. Per-target mean, sample SD, minimum and maximum
across seeds are descriptive only; no row/identity pooling or cross-target
headline metric is permitted.

## Frozen metric, threshold and uncertainty contract

Higher score means more OOD-like. The primary threshold-independent metric is
identity-level AUROC; identity Average Precision is also reported. At the
unchanged P6 known-only threshold, an alert is `score > threshold` (equality
does not alert). Report identity TP, FP, TN, FN, TPR, FPR, specificity,
precision, F1 and balanced accuracy. Undefined precision/F1 from zero
predicted positives is defined as zero. Row-level AUROC, Average Precision,
TPR and FPR are secondary diagnostics. External known-side FPR is an observed
test quantity, not guaranteed to equal P6's nominal 1% known-only calibration
tail; its difference from the development alert fraction is descriptive, not
a reason to recalibrate.

For each cell, compute percentile 95% intervals for identity AUROC, Average
Precision, TPR and FPR from 2,000 stratified bootstrap resamples, separately
resampling positive and negative canonical identities with replacement. NumPy
PCG64 uses seed `700000 + 100*target_index + split_seed`, with target order
`Fuzzers, Analysis, Backdoors, Shellcode, Worms` and seeds `42, 43, 44`.
Quantile tails are 0.025/0.975 with linear interpolation. Discard and count
only mathematically undefined replicates; fewer than 1,900 valid replicates
fails the cell. Do not alter this method after observing results.

`Analysis`, `Backdoors`, and `Worms` are labeled `SMALL-N EXTERNAL TARGET`;
their support and intervals must be interpreted cautiously. A finite
confidence interval does not imply adequate sample size or broad population
representativeness. No claim of universal zero-day effectiveness or deployment
readiness follows from this benchmark.

## One-way gate and output order

Stage A is `python scripts/freeze_p7_external_protocol.py`. It verifies all
P3–P6 bindings and synthetic P7 tests, serializes metric, uncertainty,
surface, result-schema and provenance contracts, and writes the immutable
preregistration hash. It does **not** open the official-test CSV. Repeating it
with changed inputs fails immutable-write validation.

Stage B is intentionally **not authorized or executed by Stage A**. After a
separate explicit decision, its CLI would require both
`--protocol-hash <exact frozen hash>` and `--enable-official-test`. Before any
official-test read, it revalidates the frozen protocol/source/upstream hashes
and atomically creates an exclusive `evaluation_intent.json` lock; a failed or
completed run cannot silently rerun. It then checks the official-test file
hash against the previously frozen P3 audit, reproduces P3 classification,
and verifies CLEAN canonical-identity isolation. No model, scaler, vocabulary,
scorer or threshold fitting occurs on test. Test-time categorical OOV rates
are descriptive only; the P4 OOV bucket remains frozen.

The Stage B implementation is unexecuted at this preregistration. Its declared
sequence is: produce all 15 CLEAN primary cell results and cross-seed
descriptives; freeze `primary/primary_hash.txt`; only then write clean score
summaries and diagnostic-surface reports. Diagnostic surfaces receive
descriptive scores, not external performance metrics, and cannot modify the
primary hash. Every per-cell result binds P3 collection/manifest, P4 model
config, P5 training config, P6 checkpoint/scorer/calibration, P7 protocol,
official-test source/CLEAN surface and P7 source-code hashes. Outputs live
only under `results/external_evaluation/p7/`; P3–P6 are never overwritten.

The P7 synthetic tests cover metric orientation and strict thresholding,
identity aggregation and duplicate rows, bootstrap determinism and invalid
replicates, forbidden primary surfaces and labels, changed detector bindings,
and missing CLI gate flags. These are implementation tests, **not** official
external-evaluation results.
