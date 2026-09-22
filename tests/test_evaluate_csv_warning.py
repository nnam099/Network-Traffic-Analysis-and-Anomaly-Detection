from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "scripts" / "evaluate_csv.py"
SPEC = importlib.util.spec_from_file_location("evaluate_csv_script", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_calibration_warning_marks_same_file_as_non_independent() -> None:
    summary: dict = {}

    MODULE.add_threshold_leakage_warning(summary, "data/calibration.csv")

    assert summary["methodology_warnings"] == [MODULE.THRESHOLD_LEAKAGE_WARNING]
    provenance = summary["threshold_calibration_evaluation_independence"]
    assert provenance["calibration_input_csv"] == "data/calibration.csv"
    assert provenance["report_input_csv"] == "data/calibration.csv"
    assert provenance["same_input_file"] is True
    assert provenance["independent_evaluation"] is False
