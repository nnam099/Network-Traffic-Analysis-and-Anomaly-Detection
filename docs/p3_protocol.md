# P3 test-blind scientific data protocol

Status: **DATA PROTOCOL APPROVED FOR MODEL-DESIGN PHASE — NO TRAINING**.

P3 separates development construction from official-test inspection in code and
execution. The development phase accepts only `official_train`, writes 15
content-hashed manifests, writes a collection freeze, verifies it, and only then
opens `official_test`. Test findings cannot change development roles, ambiguity
exclusion, target purge, vocabulary, scaler, or schema.

## Train-only construction

The frozen order is:

1. Canonicalize official train with `canonical-model-features-v1`.
2. Identify label-conflicting groups using official-train labels only and
   exclude every member of each conflicting group.
3. Construct the held-out target identity set from retained official train and
   purge it from prohibited development roles.
4. Split by canonical identity group into train, validation, meta, calibration,
   and surrogate roles.
5. Fit the categorical vocabulary and `RobustScaler` only on the target-purged
   backbone train fold.
6. Transform to `scientific-feature-schema-v2` and audit the structured model
   input at float64 precision.

The official train contains 1,772 conflicting canonical groups with 30,528
rows. Whole-group exclusion retains 144,813 rows. Every one of the 15 target ×
seed development cells passes: all ten pairwise canonical-role intersections,
target-vs-fit intersections, cross-identity structured-input collisions,
scaling-created merges, and dtype-created merges are zero.

Target purge removes zero additional known/surrogate rows after train-only
conflict exclusion. This is evidence, not a skipped rule: after every
train-internal cross-label canonical group is wholly excluded, a retained
target canonical identity no longer occurs under a different family label.

The freeze collection SHA-256 is
`1443746a4f024b3d7c1569ba9bbd31987933b5f9e2fb0ce6507ba5e57ddbab21`.
Per-cell compressed manifests preserve stable official-train row IDs,
canonical identities, exclusion and purge identities, role assignments,
schema/code/source hashes, and the frozen vocabulary/scaler with provenance.

## Post-freeze official-test audit

The mutually exclusive classification priority is
`TRAIN-CONFLICT-RELATED` → `TEST-INTERNAL-AMBIGUOUS` →
`DEVELOPMENT-OVERLAP` → `CLEAN`. All 82,332 rows are retained in exactly one
descriptive surface:

| Surface | Rows | Unique canonical identities |
|---|---:|---:|
| CLEAN | 69,315 | 52,283 |
| DEVELOPMENT-OVERLAP | 3,875 | 953 |
| TEST-INTERNAL-AMBIGUOUS | 6,482 | 474 |
| TRAIN-CONFLICT-RELATED | 2,660 | 236 |

The primary future scientific evaluation surface is `official-test-clean`.
The other three surfaces are diagnostics and were not silently deleted. Frozen
scaler/vocabulary parameters create zero cross-identity model-input collision
rows on the clean surface in all 15 cells; no test row is used for fitting.

Clean target support is: Fuzzers 4,777 rows / 4,317 identities (78.80%),
Analysis 58 / 58 (8.57%), Backdoors 57 / 57 (9.78%), Shellcode 364 / 364
(96.30%), and Worms 43 / 43 (97.73%). The known/background clean surface has
64,016 rows / 47,444 identities (85.83%). Analysis, Backdoors, and Worms remain
small-sample regimes requiring a later statistical design decision; no minimum
support threshold or performance conclusion is inferred.

## Historical 738 identities

The value 738 is both the global union of affected canonical identities and the
per-cell count in every historical P2 D cell—not 15 disjoint sets. All 738 have
agreeing train/test labels, all enter P3's `DEVELOPMENT-OVERLAP` diagnostic,
and none belongs to train-only ambiguity exclusion. Families are Normal 281,
Generic 238, Reconnaissance 214, DoS 4, and Exploits 1. Their per-seed role
assignments and complete identity-level provenance are preserved in
`historical_738_investigation.json`.

## Reproduction and stop condition

```bash
.venv/bin/python scripts/audit_p3_test_blind.py --development-only
.venv/bin/python scripts/audit_p3_test_blind.py
.venv/bin/python -m pytest -q -p no:cacheprovider
```

`--development-only` completes without opening the official-test file. A full
run freezes development before its first official-test read. Existing frozen
manifest files fail closed if a later run attempts to overwrite them with
different content.

Approval is limited to beginning model-design work. Schema V2 is not honestly
representable by the previous flat 61-continuous-value architecture: it has 58
continuous float64 inputs plus three opaque nominal identities. A new model
interface needs per-feature categorical components (for example embeddings)
and an explicit finite OOV inference contract. No architecture implementation,
checkpoint migration, neural training, threshold tuning, or model metric was
performed.
