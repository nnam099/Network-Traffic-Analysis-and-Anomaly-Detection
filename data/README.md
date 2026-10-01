# Dataset and Input Files

Put local UNSW-NB15 or flow CSV files in this directory. Large datasets are
ignored by Git; only small parser fixtures under `samples/` are committed.

Expected UNSW-NB15 files:

```text
UNSW-NB15_1.csv
UNSW-NB15_2.csv
UNSW-NB15_3.csv
UNSW-NB15_4.csv
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

Dataset source: [UNSW-NB15 on Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15).

Normalize a CICFlowMeter-style export with:

```bash
python scripts/prepare_production_flow_data.py input.csv \
  --output-dir results/production_flow_data
```

The command creates normalized data, train/validation/test splits, and a
manifest in the selected output directory.
