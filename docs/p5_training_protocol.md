# P5 — development-only training protocol freeze

P5 binds the frozen P3 collection SHA-256
`1443746a4f024b3d7c1569ba9bbd31987933b5f9e2fb0ce6507ba5e57ddbab21`
to the P4 `MODEL_INTERFACE_PASS` configurations. Neither an official-test
artifact nor the official-test CSV is an input to the P5 loader. P5 status is
about protocol execution, not detection effectiveness.

## Information flow

```text
frozen P3 official-train-only manifest + P4 config
                 |
       TRAIN known families only
                 |  frozen scaler/vocab from P3; CE gradients
                 v
      float64 structured backbone + temporary head
                 |
        VAL known families only
                 |  minimum sample-mean CE; earliest epoch on exact tie
                 v
          checkpoint frozen
                 |
     TRAIN scorer parameters (only if needed)
                 |
   META_KNOWN + fixed SURROGATE_OOD
                 |  scorer-family choice only; no threshold
                 v
          scorer family frozen
                 |
       CALIBRATION known only
                 |  fixed 0.99 quantile; no surrogate/target
                 v
          threshold frozen
                 |
                 X  OFFICIAL TEST — inaccessible until later external evaluation
```

The executable operation-by-role permissions are in
`results/training_protocol/p5/role_permission_matrix.json`. The P3 train role
alone fitted scaler and vocab. Only train rows fit the representation/head and
future scorer parameters; val only selects epochs. Meta known and surrogate
may jointly choose a scorer *family* after checkpoint freeze. Calibration may
only fit the predeclared known-only threshold. Target-held-out and official
test are forbidden for every P5 operation. Role and canonical-ID overlap is a
hard failure. Training accesses only the row IDs for train and val and
revalidates each ID, source role, label and canonical identity against P3.

## Initial objective and architecture

The sole initial objective is known-family cross entropy using the fixed five
known classes (`Normal`, `DoS`, `Exploits`, `Reconnaissance`, `Generic`). The
P4 baseline is `continuous[58] → 64 → 32`, three feature-specific 4-D
embeddings, then `concat[44] → 64 → representation[32]`; a temporary linear
32-to-5 head supplies the training loss. The temporary head is versioned with
the training config and has no automatic zero-day interpretation. No target
or surrogate row enters gradients. The initial run does not select an open-set
scorer. Continuous input, parameters, representation and logits remain
float64; categorical indices remain int64. AMP is disabled.

Reconstruction was deferred because it would privilege reconstruction error
before evidence justifies that anomaly assumption. Metric learning was
deferred because deterministic pair mining and margin choices add degrees of
freedom. No experiment or official-test observation chose among objectives.

Optimization is fixed in `training_config.json`: AdamW, learning rate 0.001,
weight decay 0.01, batch size 512, maximum eight epochs, gradient norm clip
1.0, PyTorch default initialization after model seed, no class weighting and
no scheduler. Validation sample-mean cross entropy uses patience two and
minimum delta zero. Exact ties retain the earlier epoch. Training uses CPU,
single-process deterministic epoch permutations, deterministic PyTorch
algorithms and model seed `10000 + split_seed`. The split seed remains a
separate P3 data-split provenance value.

## Surrogate, scoring and threshold

Surrogate is exactly the four non-target OOD families in each frozen P3
manifest, with no resampling. It never fits gradients, scorer parameters or
thresholds, and never selects architecture or checkpoint. Its sole reserved
purpose is simulating unknown-family behavior for scorer-family selection
against `meta_known`, consistently for all 15 cells. This introduces a
declared surrogate-family selection bias; future claims must state it. The
initial P5 stages do not consume surrogate features or perform scorer choice.

The only predeclared scorers are `1 - max softmax(logits)` and minimum
Euclidean distance to train-only known-family representation centroids. The
first has no fitted scorer parameters but depends on the temporary head; the
second fits one float64 centroid per known family on train only. A future
selection pass may compare development AUROC with surrogate as positive and
meta known as negative. Exact ties select nearest-centroid. This is not a
scientific performance result and cannot be optimized against calibration or
official test. The scorer remains unselected in this initial run.

After scorer choice and parameter freeze, threshold fitting uses only known
calibration scores. The rule is NumPy `quantile(..., 0.99, method="higher")`,
and flags scores strictly greater than that value. It is target-independent
known-only calibration, not a population false-positive-rate guarantee.
Surrogate-calibrated or target-calibrated thresholds are not authorized.

## Execution gates and provenance

Stage A runs `python scripts/train_p5_development.py --all-dry-runs` and
requires all five targets × three P3 split seeds to pass manifest, role,
P4 config, dtype, vocabulary/scaler hash, output-path and official-test
exclusion checks. No optimizer or checkpoint is created. Stage B is exactly
`Fuzzers/42/10042` with `--enable-training --run-tag smoke-a`. Stage C repeats
it with `--run-tag smoke-b` and runs `--verify-smoke`. A matching selected
epoch and parameter state within absolute tolerance `1e-12` is required;
bitwise equality and file hashes are also reported. No broader matrix
training is authorized by P5.

Each immutable smoke manifest binds target, split/model seed, P3 manifest
hash, P4 config hash, training/source hashes, vocab/scaler hashes, environment,
role row and identity counts, optimizer config, epoch validation history,
selected epoch and checkpoint/state hashes. Development val CE is recorded
only as a smoke diagnostic. Files are isolated under
`results/training_protocol/p5/`; legacy `src/ids/training.py` and flat schema-V1
artifacts are not imported or reused. The new CLI without
`--enable-training` only validates and exits. Unsupported cells and manifests
fail closed.

`P5_PROTOCOL_PASS` certifies only Stage A. `P5_SMOKE_PASS` additionally
certifies the deterministic Stage B/C smoke pair and provenance checks.
Neither certifies detection performance. Future development scorer selection
and calibration need their own guarded execution and tests before an external
evaluation phase. Low official-test support for Analysis, Backdoors and Worms
does not alter architecture, role allocation or threshold policy; future
external evaluation must report small-N uncertainty honestly.
