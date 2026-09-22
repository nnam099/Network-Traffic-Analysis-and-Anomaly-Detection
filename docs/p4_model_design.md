# P4 scientific model interface and architecture design

Status: **MODEL_INTERFACE_PASS — no training was performed**.

P4 is bound to the frozen P3 collection
`1443746a4f024b3d7c1569ba9bbd31987933b5f9e2fb0ce6507ba5e57ddbab21`.
It does not regenerate or modify P3, read official-test data, migrate a
checkpoint, construct an optimizer, select a threshold, or compute a model
metric. Passing P4 means a future training phase may implement this interface;
it is not evidence of predictive or open-set effectiveness.

## Structured input contract

The only accepted neural boundary is `StructuredModelInput`:

```text
NumPy float64[B,58] --exact copy--> torch.float64[B,58] --> ContinuousEncoder

proto raw identity   --> frozen proto vocabulary   --> int64[B] --> proto embedding
service raw identity --> frozen service vocabulary --> int64[B] --> service embedding
state raw identity   --> frozen state vocabulary   --> int64[B] --> state embedding

numeric latent + three embeddings --> concatenation --> representation head --> 32-D
```

Canonical identity, labels, row IDs, role, and source provenance remain audit
metadata and never become model tensors. A generic tensor, including
`float[B,61]`, is rejected. Exactly 58 continuous coordinates and three 1-D
categorical index tensors with a common batch dimension are required.

## Continuous dtype boundary

The source is the finite float64 array produced by the P3 manifest-bound
`RobustScaler`. The adapter accepts only NumPy float64 `[B,58]`, fails on
NaN/Inf, and copies it to a framework-owned `torch.float64` tensor without
numeric conversion. All baseline numeric layers and embeddings also use
float64.

Float64 → float32 is **not authorized in P4**. It would require a new,
versioned boundary-collision audit because P2 established that float32 can
collapse canonical distinctions. Categorical identities never pass through a
floating-point conversion.

## Categorical and finite OOV contract

Each field has an immutable vocabulary and independent embedding table:

- `PAD=0`, reserved but not emitted by ordinary tabular rows;
- `OOV=1`, one dedicated value per feature;
- known training tokens occupy contiguous indices starting at 2;
- known-token ordering is inherited deterministically from the frozen P3
  per-feature vocabulary;
- vocabulary fitting provenance is the P3 target-purged backbone train only;
- test or target tokens cannot add entries after freeze;
- neither modulo hashing nor Python `hash()` is used.

Multiple unseen tokens intentionally share the feature's OOV embedding. This
is a finite architectural abstraction, not a claim that their canonical
identities are equivalent. Exact canonical identity remains in separate audit
metadata.

The deterministic embedding-dimension policy is
`clip(ceil(sqrt(known_count + 2)), 4, 16)`. Current P3 cells therefore use
4-dimensional embeddings for all three fields. Cardinalities, including PAD
and OOV, are proto/service/state `8/15/10` for seeds 42 and 44, and `9/15/11`
for seed 43.

## Architecture decision record

### A. Numeric MLP plus concatenated embeddings — selected

The numeric encoder is `58→64→32`. Three independent embeddings are
concatenated with its 32-D latent, then a `44→64→32` representation head is
applied. This is the conservative baseline because its branch semantics and
dimensions are directly auditable and it introduces few design degrees of
freedom. Its risk is a limited interaction inductive bias.

### B. Feature-token transformer — not selected

It would make inter-feature interaction explicit, but adds substantially more
complexity and overfitting risk without any approved empirical basis,
especially for low-support targets.

### C. Hybrid controlled-interaction encoder — deferred

This could introduce constrained cross-branch interactions later, but adds
design and attribution choices that should be predeclared in a future phase.

No alternative was ranked using official-test behavior or model performance.

## Representation and open-set boundary

The baseline returns only a 32-D representation. It contains no classifier,
open-set scorer, score fusion, decision threshold, or selected reconstruction
objective. This permits future prototype, density, reconstruction, or other
open-set interfaces without quietly treating a closed-set classifier as the
scientific design.

`ContinuousReconstructionHead` is defined as an isolated optional component to
make encoder/reconstruction separation explicit, but it is not selected or
instantiated by the P4 baseline. Any reconstruction target and open-set scoring
semantics require later approval.

## Manifest-bound configuration

Every target/seed configuration records and verifies:

- P3 collection and manifest hashes;
- schema version/hash;
- scaler parameter and fitting-row hashes;
- combined P3 and per-feature P4 vocabulary hashes;
- categorical cardinalities and embedding dimensions;
- continuous count and model-interface version;
- architecture and source-code hashes;
- explicit false/zero values for training, checkpoints, metrics, and optimizer
  steps.

The 15 configurations are in `results/model_design/p4/p4_matrix.json`. P4 also
hashes the entire P3 artifact tree before and after its audit and requires it to
remain byte-identical.

## Legacy compatibility

The existing `IDSModel`, old dataset transforms, `src/train.py`, checkpoint
validator, and runtime evaluator belong to historical flat interfaces. They
are retained for reproducibility and demos but are not imported by P4.
`src/train.py` is explicitly BLOCKING if reused for schema V2. A future runtime
adapter must be implemented before structured checkpoints can be served.

The machine-readable inventory is
`results/model_design/p4/legacy_compatibility_audit.json`.

## Small-support evaluation plan

Future work must report each target and seed separately, including raw row and
unique canonical-identity counts. Analysis (58 clean identities), Backdoors
(57), and Worms (43) require explicit small-N warnings. Fuzzers (4,317) and
Shellcode (364) provide contrast but must not be pooled to hide target-level
support.

Uncertainty intervals may be reported only when their assumptions are stated
and meaningful for the observed support. Nonzero support is not evidence of
adequate statistical power. Future scores and thresholds must be predeclared
without official-test-clean and no aggregate may conceal a rare target.

## Reproduction

```bash
.venv/bin/python scripts/audit_p4_model_interface.py
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p4_model_interface.py
```

The audit creates only JSON design evidence under `results/model_design/p4/`.
It creates no `.pt`, `.pth`, `.ckpt`, or `.pkl` file.
