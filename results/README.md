# Evaluation Results and Reports

This directory contains two kinds of material:

1. **Frozen protocol evidence** — the JSON reports, manifests, matrices and
   checkpoints produced by P1–P13D. These files are intentionally versioned
   because the audit scripts and `tests/` read them directly to verify hashes,
   roles and provenance.
2. **Local regenerated output** — `results/csv_eval/`, external diagnostic runs,
   SQLite databases and `results/step2_seed*/` are ignored by Git.

## Authoritative summaries

| Path | Meaning |
| --- | --- |
| `artifact_manifest.json` | SHA-256 manifest for the committed v14 runtime artifacts. |
| `ids_v14_results.json` | Historical demo record; **not** a current benchmark. |
| `step2_multiseed_report.json` | Historical multiseed aggregate; **not valid for scientific claims**. |
| `data_quality/p1..p3/` | Preprocessing, ambiguity and leakage forensics used by tests. |
| `model_selection/p6/` | Frozen scorer-selection runs and checkpoints used by tests. |
| `loafo/p8/` and `loafo/p9/` | Train-only LOAFO protocol and development-only results. |
| `loafo/p10/` | External-source audit; status is `P10_EXTERNAL_SOURCE_BLOCKED`. |
| `portable_schema/p11..p13d/` | Portable extractor and controlled-capture evidence. |

Do not delete frozen evidence to "shrink" the repository without updating the
tests and the corresponding protocol document first.

![Reference OOD ROC curves](../plots/v14_roc_curves.png)

`ids_v14_results.json` and `step2_multiseed_report.json` are historical demo
records, not current benchmarks. Retrain and regenerate reports before making
model comparisons.
