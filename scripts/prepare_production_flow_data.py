from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ids.production_schema import (  # noqa: E402
    PRODUCTION_FLOW_COLUMNS,
    apply_label_overrides,
    normalize_to_production_schema,
    split_by_event_time,
    summarize_production_flows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CICFlowMeter/flow CSV logs for production-like IDS deployment."
    )
    parser.add_argument("csv_paths", nargs="+", help="One or more CICFlowMeter/firewall/flow CSV files.")
    parser.add_argument("--source", default="cicflowmeter", help="Source name stored in the production dataset.")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "results" / "production_flow_data"))
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional cap for quick validation runs.")
    parser.add_argument(
        "--label-overrides",
        default=None,
        help="Optional CSV with flow_id or source_row plus analyst_label and optional attack_category.",
    )
    parser.add_argument("--label-key", default="flow_id", choices=["flow_id", "source_row"])
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    source_reports = []
    for csv_path_text in args.csv_paths:
        csv_path = Path(csv_path_text)
        raw_df = pd.read_csv(csv_path, nrows=args.max_rows_per_file)
        result = normalize_to_production_schema(raw_df, source=args.source, source_file=csv_path.name)
        frames.append(result.data)
        source_reports.append(result.report)

    if not frames:
        raise RuntimeError("no CSV rows were loaded")

    flows = pd.concat(frames, ignore_index=True)
    if args.label_overrides:
        overrides = pd.read_csv(args.label_overrides)
        flows = apply_label_overrides(flows, overrides, key=args.label_key)

    flows = split_by_event_time(
        flows,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    all_path = output_dir / "production_flows.csv"
    train_path = output_dir / "train.csv"
    validation_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"
    manifest_path = output_dir / "manifest.json"

    flows.to_csv(all_path, index=False)
    flows.loc[flows["split"] == "train", PRODUCTION_FLOW_COLUMNS].to_csv(train_path, index=False)
    flows.loc[flows["split"] == "validation", PRODUCTION_FLOW_COLUMNS].to_csv(validation_path, index=False)
    flows.loc[flows["split"] == "test", PRODUCTION_FLOW_COLUMNS].to_csv(test_path, index=False)

    manifest = {
        "schema_version": 1,
        "schema_columns": PRODUCTION_FLOW_COLUMNS,
        "inputs": [str(Path(path).resolve()) for path in args.csv_paths],
        "output_dir": str(output_dir.resolve()),
        "source_reports": source_reports,
        "summary": summarize_production_flows(flows),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"Production dataset: {all_path}")
    print(f"Train split: {train_path}")
    print(f"Validation split: {validation_path}")
    print(f"Test split: {test_path}")
    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
