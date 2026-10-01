from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CSVInputPolicy:
    max_bytes: int = 200 * 1024 * 1024
    max_rows: int = 1_000_000
    max_columns: int = 250
    min_columns: int = 3


@dataclass
class CSVInputValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: int = 0
    columns: int = 0
    size_bytes: int | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "rows": self.rows,
            "columns": self.columns,
            "size_bytes": self.size_bytes,
        }


def validate_uploaded_csv(df: pd.DataFrame, size_bytes: int | None = None, policy: CSVInputPolicy | None = None) -> CSVInputValidation:
    policy = policy or CSVInputPolicy()
    errors: list[str] = []
    warnings: list[str] = []

    rows = int(len(df)) if isinstance(df, pd.DataFrame) else 0
    columns = int(len(df.columns)) if isinstance(df, pd.DataFrame) else 0

    if size_bytes is not None and size_bytes > policy.max_bytes:
        errors.append(f"CSV file is too large: {size_bytes} bytes > {policy.max_bytes} bytes")
    if rows <= 0:
        errors.append("CSV has no rows")
    if columns < policy.min_columns:
        errors.append(f"CSV has too few columns: {columns} < {policy.min_columns}")
    if rows > policy.max_rows:
        errors.append(f"CSV has too many rows for dashboard batch inference: {rows} > {policy.max_rows}")
    if columns > policy.max_columns:
        errors.append(f"CSV has too many columns: {columns} > {policy.max_columns}")

    if isinstance(df, pd.DataFrame):
        duplicate_columns = sorted(name for name, count in Counter(str(col) for col in df.columns).items() if count > 1)
    else:
        duplicate_columns = []
    if duplicate_columns:
        errors.append("CSV contains duplicate column names: " + ", ".join(duplicate_columns[:8]))

    unnamed_columns = [str(col) for col in df.columns if str(col).strip() == "" or str(col).lower().startswith("unnamed:")]
    if unnamed_columns:
        warnings.append("CSV contains unnamed/index-like columns: " + ", ".join(unnamed_columns[:8]))

    if rows > 50_000:
        warnings.append("Large CSV upload; dashboard analysis may take time on CPU.")

    return CSVInputValidation(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        rows=rows,
        columns=columns,
        size_bytes=size_bytes,
    )
