# P8A/P8B — train-only LOAFO design and freeze

Status is determined by `results/loafo/p8/loafo_matrix.json`. P8 performs data audit and protocol freeze only: no optimizer, gradient step, checkpoint, threshold fitting, target performance, or external evaluation. P9 may train only against the frozen P8 manifests and must not infer protocol choices from the already-consumed official test.

## Scientific question and source boundary

For each verified attack family T, can a model fitted and selected without any T family data detect T as unseen/OOD traffic? `Normal` is background/in-distribution and is never a target. Canonical model-feature identity, not row, is the independence unit. The attack-family universe is obtained from P3's frozen official-train ambiguity metadata, cross-checked against the exact P3-bound official-training CSV. Historical test counts come only from P3's frozen external-support audit and are descriptive. The official-test CSV is not opened by P8. P7 primary results are not an input.

The original official test is **consumed**. Any later use must be labeled exactly `REUSED TEST — NOT INDEPENDENT EXTERNAL VALIDATION`. No P8/P9 result from it can be claimed as fresh independent external validation.

The universe artifact also records whether each family was a target or surrogate in frozen P3 development manifests. P8 uses that history only as descriptive lineage, never to choose the new surrogate policy. Each P8 cell records known-train and surrogate rows and canonical identities per class so imbalance is visible before P9.

## Role construction

Whole train-side canonical groups with conflicting family labels are excluded using the frozen P3 ambiguity policy. For each target T, all retained T identities are assigned only to `target_held_out`. The policy selects surrogate attack families independently of target/test behavior (below). The remaining attack families plus `Normal` form the known classification population. `StratifiedGroupKFold` with 10 folds and seeds 42/43/44 assigns known canonical groups to train folds 0–6, val fold 7, meta_known fold 8, and calibration fold 9. Surrogate families enter only `surrogate_ood`. Every role uses official-train rows. The auditor requires zero row and canonical intersections across all six roles and zero final structured-model-input cross-identity collisions, including target against fit identities. Duplicated rows stay with their canonical group.

Allowed future P9 uses: train for scaler/vocabulary, gradients, representation, and known-class centroids; val for checkpoint selection; meta_known plus surrogate_ood for scorer-family selection; calibration known only for threshold. Target is forbidden for all those operations. The consumed official test is forbidden for model development/selection. These are permissions, not operations performed in P8.

## Surrogate selection, frozen before P9

Two policies were considered. S1 reserves one remaining attack family; this preserves more known classes but offers one potentially unrepresentative OOD contrast. S2 reserves two remaining attack families; this provides more than one development OOD family at the cost of fewer known classifier classes. Both require complete exclusion of surrogate identities from backbone fitting, preprocessing, checkpointing, centroid fitting, and calibration. P8 freezes S2 as primary. Candidates are ranked by retained official-train canonical identity support, descending, with exact-label ascending tie-break; T is omitted and the top two are reserved. This is a fixed, train-only deterministic rule. The same algorithm applies to every T and seed. Neither P7 results nor historical test-support counts choose surrogates. S1 is conceptual only, not a P9 search branch under this freeze.

Support tiers are descriptive and never remove a target: HIGH SUPPORT ≥1000 retained train identities; MODERATE SUPPORT 200–999; SMALL-N 50–199; INSUFFICIENT <50. New external identity support is unknown; historical P3 clean-test support is shown separately and is not available as a fresh validation population. Class imbalance and small-N must be reported, not hidden by exclusion.

## Structured model and variable head

The input remains `torch.float64[B,58]` continuous plus three `torch.int64[B]` categorical fields (`proto`, `service`, `state`). Each fold's vocabulary is fitted from train only, immutable, with PAD=0 and per-feature OOV=1; no modulo hashing. The numeric scaler is fitted from train only. Target-only tokens cannot expand these vocabularies. Canonical identity is tracked outside the tensor boundary.

The concrete P4/P5 reference architecture is retained: 58→64→32 numeric; three independent 4-dimensional embeddings; concatenated 44→64→32 representation. The generic P4 implementation's vocabulary-size heuristic would silently enlarge the `proto` embedding when LOAFO changes class composition. P8 therefore version-defines `P8ReferenceBackbone` with fixed four-dimensional embeddings to preserve the concrete 44-D reference, without changing P4/P5 code or artifacts. Any other width or architecture requires a separate versioned ablation/amendment before use. The P8 class is an untrained architecture declaration.

The classifier classes are `Normal` first and the fitted attack families in exact lexical order. The temporary head is `Linear(32, len(known_class_order))`, float64; T and surrogate labels are absent. The baseline objective for P9 is cross entropy over those known classes. Split and model seeds are separate: model seed = 10000 + split seed. P5's reference optimizer configuration is AdamW, learning rate 0.001, weight decay 0.01, batch size 512, max 8 epochs, gradient clipping 1.0, float64, no AMP. Checkpoint selection uses val sample-mean cross entropy only, patience 2, min_delta 0, lowest epoch on exact tie. If cell composition makes these settings untenable, a versioned protocol amendment is required before training.

The only baseline scorer families are `1 - max softmax` and nearest known-class centroid Euclidean distance in representation space; centroids are fitted from train only. Scores are averaged arithmetically within canonical identity before the primary meta_known vs surrogate_ood scorer-selection statistic (identity-level AUROC). As in P6, differences within 1e-12 select nearest centroid. The baseline threshold uses known-only calibration identity scores: `np.quantile(scores, 0.99, method="higher")`; alert iff `score > threshold`. Held-out T never tunes scorer or threshold. AUROC, AP, known alert fraction, and surrogate recall generated later are `DEVELOPMENT ONLY` and are not external zero-day performance.

## External validation after P9

P10 requires a new, untouched and preregistered source: a cryptographically reserved new partition, genuinely external dataset, or newly collected traffic. Candidate sources (CIC-IDS2017, CSE-CIC-IDS2018, CIC-DDoS2019, TON_IoT, Bot-IoT, new reserved UNSW partition, or collected traffic) are design possibilities only; P8 downloads none. Their feature schema, family semantics, distribution, background traffic, and categorical domains cannot be assumed equivalent to UNSW-NB15. Before any cross-dataset evaluation, preregister semantic attack mapping, all 58 continuous-feature mappings, missing-feature behavior, categorical OOV behavior, and normal/background definition. An incompatible source must not be forced through this interface or renamed to fake label equivalence.

## Reproduction and gate

`scripts/audit_p8_loafo.py` verifies P3/P4 frozen bindings, hashes the permitted official-train source, recomputes P3 train-only ambiguity, dynamically enumerates targets, and audits every target × seed cell. It writes candidates in a temporary directory and publishes P8 artifacts immutably only after all cells pass. Run with the project virtualenv and no external-test flags:

```bash
.venv/bin/python scripts/audit_p8_loafo.py
.venv/bin/pytest -q tests/test_p8_loafo_protocol.py
```

The pass gate requires every target to exist in the verified universe; valid surrogate/known composition; target absence from all fitting and selection roles; nonempty known class support; zero canonical/row/model-input cross-identity leakage; train-only scaler/vocabulary provenance; deterministic class order; P4 structured-input compatibility; complete source and manifest hashes; and no consumed-test selection. The status is `LOAFO_PROTOCOL_PASS` only when every dynamically generated cell passes. P8 artifacts are under `results/loafo/p8/`; P3–P7 remain immutable. No `results/loafo/p9/` artifact is created by this phase.
