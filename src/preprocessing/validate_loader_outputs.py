from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loaders._common import UNIFIED_COLUMNS, ensure_unified_columns  # noqa: E402
from loaders.load_calce import load_calce  # noqa: E402
from loaders.load_mit_severson import load_mit_severson  # noqa: E402
from loaders.load_nasa import load_nasa  # noqa: E402


LOGGER = logging.getLogger("validate_loader_outputs")
NUMERIC_CHECK_COLUMNS = ["voltage_v", "current_a", "temperature_c", "capacity_ah"]


@dataclass
class ValidationCase:
    name: str
    dataset_label: str
    loader: Callable[[], pd.DataFrame]
    special_check: Callable[[pd.DataFrame], list[str]] | None = None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _format_number(value) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_number(item) for item in row) + " |")
    return "\n".join(lines)


def check_required_columns(df: pd.DataFrame) -> list[str]:
    missing = [col for col in UNIFIED_COLUMNS if col not in df.columns]
    if missing:
        return [f"Missing unified columns: {', '.join(missing)}"]
    return ["All unified columns are present."]


def summarize_basic(df: pd.DataFrame) -> list[str]:
    lines = []
    if df.empty:
        return ["Rows: 0", "Battery count: 0", "Cycle index range: empty"]

    cycle = _safe_numeric(df["cycle_index"])
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Battery count: {df['battery_id'].nunique(dropna=True)}")
    lines.append(f"Cycle index range: {_format_number(cycle.min())} to {_format_number(cycle.max())}")
    return lines


def summarize_step_type(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty or "step_type" not in df.columns:
        return "No step_type data."
    counts = df["step_type"].fillna("<NaN>").astype(str).value_counts().head(max_rows)
    rows = [[step, count, count / len(df)] for step, count in counts.items()]
    return _markdown_table(["step_type", "rows", "ratio"], rows)


def summarize_missing_rates(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows to compute missing rates."
    rows = []
    for col in UNIFIED_COLUMNS:
        if col in df.columns:
            rows.append([col, df[col].isna().mean()])
        else:
            rows.append([col, 1.0])
    return _markdown_table(["field", "missing_rate"], rows)


def summarize_numeric_stats(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows to compute numeric stats."
    rows = []
    for col in NUMERIC_CHECK_COLUMNS:
        if col not in df.columns:
            rows.append([col, np.nan, np.nan, np.nan, 0])
            continue
        numeric = _safe_numeric(df[col])
        rows.append([col, numeric.min(), numeric.max(), numeric.mean(), numeric.notna().sum()])
    return _markdown_table(["field", "min", "max", "mean", "non_null_rows"], rows)


def cycle_quality_checks(df: pd.DataFrame) -> tuple[str, list[str]]:
    if df.empty:
        return "No rows to check cycle_index quality.", ["No cycle quality checks were run."]

    work = df[["battery_id", "cycle_index", "source_file"]].copy()
    work["cycle_index_num"] = _safe_numeric(work["cycle_index"])
    work = work.dropna(subset=["battery_id", "cycle_index_num"])
    if work.empty:
        return "No numeric cycle_index values found.", ["cycle_index is fully missing or non-numeric."]

    row_duplicate_count = int(work.duplicated(["battery_id", "cycle_index_num"]).sum())
    source_repeat = (
        work.groupby(["battery_id", "cycle_index_num"])["source_file"]
        .nunique(dropna=True)
        .reset_index(name="source_file_count")
    )
    cross_file_repeats = source_repeat[source_repeat["source_file_count"] > 1]

    discontinuities = []
    duplicate_cycle_ids = []
    for battery_id, group in work.groupby("battery_id"):
        cycles = np.sort(group["cycle_index_num"].dropna().unique())
        if len(cycles) == 0:
            continue

        duplicated_cycle_id_count = int(group["cycle_index_num"].duplicated().sum())
        if duplicated_cycle_id_count:
            duplicate_cycle_ids.append([battery_id, duplicated_cycle_id_count])

        rounded = np.round(cycles)
        if not np.allclose(cycles, rounded, equal_nan=False):
            continue
        gaps = np.diff(rounded.astype(int))
        large_gaps = gaps[gaps > 1]
        if len(large_gaps):
            first_gap_idx = int(np.where(gaps > 1)[0][0])
            discontinuities.append(
                [
                    battery_id,
                    int(rounded[first_gap_idx]),
                    int(rounded[first_gap_idx + 1]),
                    int(large_gaps.max()),
                    len(large_gaps),
                ]
            )

    rows = [
        ["row-level duplicate (battery_id, cycle_index)", row_duplicate_count],
        ["cross-source_file repeated cycle ids", len(cross_file_repeats)],
        ["battery ids with non-continuous integer cycles", len(discontinuities)],
    ]
    summary = _markdown_table(["check", "count"], rows)

    notes = []
    if row_duplicate_count:
        notes.append(
            "Row-level duplicate cycle_index is expected for detailed time-series data, "
            "because many time points belong to the same battery cycle."
        )
    if len(cross_file_repeats):
        sample = cross_file_repeats.head(5)
        notes.append(
            "Some battery_id/cycle_index pairs appear in multiple source files. "
            "For CALCE this can indicate Excel files restarting Cycle_Index from 1. "
            f"Sample: {sample.to_dict(orient='records')}"
        )
    if discontinuities:
        notes.append(
            "Some battery_id series have non-continuous integer cycle_index values. "
            f"Sample: {discontinuities[:5]}"
        )
    if not notes:
        notes.append("No cross-file cycle repeats or integer cycle discontinuities were detected.")

    return summary, notes


def capacity_anomaly_check(df: pd.DataFrame) -> str:
    if df.empty or "capacity_ah" not in df.columns:
        return "No capacity_ah data to check."
    capacity = _safe_numeric(df["capacity_ah"])
    checked = capacity.notna().sum()
    abnormal = (capacity <= 0).sum()
    ratio = abnormal / checked if checked else np.nan
    return _markdown_table(
        ["check", "count", "ratio_within_non_null_capacity"],
        [["capacity_ah <= 0", abnormal, ratio]],
    )


def special_mit_summary_check(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["MIT summary-only loader returned no rows."]
    step_values = set(df["step_type"].dropna().astype(str).unique())
    notes = []
    if "cycle_summary" in step_values:
        notes.append("MIT summary-only output uses step_type=cycle_summary as expected.")
    else:
        notes.append(f"MIT summary-only step_type does not include cycle_summary. Observed: {sorted(step_values)}")
    non_null_capacity = df["capacity_ah"].notna().sum()
    notes.append(f"MIT summary-only non-null capacity_ah rows: {non_null_capacity}")
    return notes


def special_mit_detailed_check(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["MIT detailed loader returned no rows."]
    step_values = set(df["step_type"].dropna().astype(str).unique())
    notes = [f"MIT detailed step_type values observed: {sorted(step_values)}"]
    if {"charge", "discharge"}.intersection(step_values):
        notes.append("MIT detailed curves include current-derived charge/discharge labels.")
    else:
        notes.append("MIT detailed curves did not expose charge/discharge labels in this sample.")
    notes.append(
        "MIT detailed output can be very large; validation samples only one file and one battery by default."
    )
    return notes


def special_calce_check(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["CALCE loader returned no rows."]

    notes = []
    source_files = df["source_file"].nunique(dropna=True)
    notes.append(f"CALCE sample contains {source_files} source_file entries.")

    work = df[["battery_id", "cycle_index", "source_file"]].copy()
    work["cycle_index_num"] = _safe_numeric(work["cycle_index"])
    cross_file = (
        work.dropna(subset=["cycle_index_num"])
        .groupby(["battery_id", "cycle_index_num"])["source_file"]
        .nunique()
        .reset_index(name="source_file_count")
    )
    repeats = cross_file[cross_file["source_file_count"] > 1]
    if repeats.empty:
        notes.append("No CALCE cycle_index reuse across Excel files was detected in this sample.")
    else:
        notes.append(
            "CALCE cycle_index reuse across Excel files was detected. "
            "This likely means each workbook restarts Cycle_Index; later preprocessing should build a global cycle index."
        )
        notes.append(f"CALCE cross-file repeat sample: {repeats.head(5).to_dict(orient='records')}")

    return notes


def special_nasa_check(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["NASA loader returned no rows."]

    observed = set(df["step_type"].dropna().astype(str).str.lower().unique())
    expected = {"charge", "discharge", "impedance"}
    notes = [f"NASA step_type values observed: {sorted(observed)}"]
    missing = expected - observed
    if missing:
        notes.append(f"NASA sample did not include these expected cycle types: {sorted(missing)}")
    else:
        notes.append("NASA charge/discharge/impedance cycle types are all represented in this sample.")

    discharge = df[df["step_type"].astype(str).str.lower() == "discharge"]
    non_null_capacity = discharge["capacity_ah"].notna().sum() if not discharge.empty else 0
    notes.append(f"NASA discharge rows: {len(discharge)}; non-null discharge capacity_ah rows: {non_null_capacity}")
    if len(discharge) and non_null_capacity:
        notes.append("NASA discharge Capacity field is retained in capacity_ah/discharge_capacity_ah.")
    elif len(discharge):
        notes.append("NASA discharge rows exist, but capacity_ah is missing; inspect NASA .mat field mapping.")
    return notes


def sample_for_output(df: pd.DataFrame, validation_case: str, max_rows: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["validation_case", *UNIFIED_COLUMNS])
    sample = ensure_unified_columns(df).head(max_rows).copy()
    sample.insert(0, "validation_case", validation_case)
    return sample


def validate_case(case: ValidationCase) -> tuple[str, pd.DataFrame]:
    lines = [f"## {case.name}", ""]
    LOGGER.info("Running validation case: %s", case.name)

    try:
        df = case.loader()
    except Exception as exc:
        LOGGER.exception("Validation case failed: %s", case.name)
        lines.extend(
            [
                f"Status: FAILED",
                "",
                f"Error: `{type(exc).__name__}: {exc}`",
                "",
                "Suggestion: check raw data path, dependencies, and loader logs.",
                "",
            ]
        )
        return "\n".join(lines), pd.DataFrame(columns=["validation_case", *UNIFIED_COLUMNS])

    if not df.empty:
        df = ensure_unified_columns(df)

    lines.extend(["### Required Columns", ""])
    lines.extend(f"- {item}" for item in check_required_columns(df))
    lines.append("")

    lines.extend(["### Basic Summary", ""])
    lines.extend(f"- {item}" for item in summarize_basic(df))
    lines.append("")

    lines.extend(["### Step Type Distribution", "", summarize_step_type(df), ""])
    lines.extend(["### Missing Rates", "", summarize_missing_rates(df), ""])
    lines.extend(["### Numeric Stats", "", summarize_numeric_stats(df), ""])

    cycle_summary, cycle_notes = cycle_quality_checks(df)
    lines.extend(["### Cycle Index Quality", "", cycle_summary, ""])
    lines.extend(["#### Notes", ""])
    lines.extend(f"- {note}" for note in cycle_notes)
    lines.append("")

    lines.extend(["### Capacity Anomaly Check", "", capacity_anomaly_check(df), ""])

    if case.special_check is not None:
        lines.extend(["### Dataset-Specific Checks", ""])
        lines.extend(f"- {note}" for note in case.special_check(df))
        lines.append("")

    lines.extend(["### Suggestions", ""])
    lines.extend(make_suggestions(df, case.dataset_label))
    lines.append("")

    return "\n".join(lines), df


def make_suggestions(df: pd.DataFrame, dataset_label: str) -> list[str]:
    if df.empty:
        return [f"- {dataset_label}: no rows loaded. Confirm raw data files and required Python dependencies."]

    suggestions = []
    missing_rates = df[UNIFIED_COLUMNS].isna().mean()
    high_missing = missing_rates[missing_rates > 0.95].index.tolist()
    if high_missing:
        suggestions.append(
            f"- Fields with >95% missing values should be checked before modeling: {', '.join(high_missing)}."
        )

    capacity = _safe_numeric(df["capacity_ah"])
    if (capacity <= 0).sum() > 0:
        suggestions.append(
            "- capacity_ah contains zero or negative values. This can be normal within raw curves, "
            "but cycle-level features should filter or aggregate these carefully."
        )

    cycle = _safe_numeric(df["cycle_index"])
    if cycle.isna().mean() > 0.1:
        suggestions.append("- More than 10% of cycle_index values are missing or non-numeric; inspect source mapping.")

    if not suggestions:
        suggestions.append("- No blocking issue detected in this validation sample.")
    return suggestions


def build_validation_cases() -> list[ValidationCase]:
    return [
        ValidationCase(
            name="MIT-Severson Summary-Only Sample",
            dataset_label="MIT-Severson summary-only",
            loader=lambda: load_mit_severson(
                max_files=1,
                max_batteries=1,
                summary_only=True,
                output_path=None,
            ),
            special_check=special_mit_summary_check,
        ),
        ValidationCase(
            name="MIT-Severson Detailed Cycle Sample",
            dataset_label="MIT-Severson detailed",
            loader=lambda: load_mit_severson(
                max_files=1,
                max_batteries=1,
                summary_only=False,
                output_path=None,
            ),
            special_check=special_mit_detailed_check,
        ),
        ValidationCase(
            name="CALCE Multi-Workbook Sample",
            dataset_label="CALCE",
            loader=lambda: load_calce(
                max_workbooks=3,
                output_path=None,
            ),
            special_check=special_calce_check,
        ),
        ValidationCase(
            name="NASA Mat/Zip Sample",
            dataset_label="NASA",
            loader=lambda: load_nasa(
                max_files=1,
                output_path=None,
            ),
            special_check=special_nasa_check,
        ),
    ]


def write_report(report_sections: list[str], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        "# Loader Output Validation Report",
        "",
        "This report validates small real samples from the source-domain loaders. "
        "It does not modify or delete raw data and does not perform modeling.",
        "",
        "Unified output fields:",
        "",
        "`" + ", ".join(UNIFIED_COLUMNS) + "`",
        "",
    ]
    content.extend(report_sections)
    report_path.write_text("\n".join(content), encoding="utf-8")
    LOGGER.info("Saved validation report: %s", report_path)


def write_sample(sample_frames: list[pd.DataFrame], sample_path: Path) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    valid = [frame for frame in sample_frames if frame is not None and not frame.empty]
    if valid:
        sample = pd.concat(valid, ignore_index=True)
    else:
        sample = pd.DataFrame(columns=["validation_case", *UNIFIED_COLUMNS])
    sample.to_parquet(sample_path, index=False)
    LOGGER.info("Saved validation sample parquet: %s rows -> %s", len(sample), sample_path)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Validate source-domain battery loader outputs.")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs" / "loader_validation_report.md",
        help="Markdown validation report path.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=PROJECT_ROOT / "data_interim" / "loader_validation_sample.parquet",
        help="Parquet file containing small row samples from each validation case.",
    )
    parser.add_argument(
        "--sample-rows-per-case",
        type=int,
        default=5000,
        help="Rows retained in the sample parquet for each validation case.",
    )
    args = parser.parse_args()

    report_sections = []
    sample_frames = []
    for case in build_validation_cases():
        section, df = validate_case(case)
        report_sections.append(section)
        sample_frames.append(sample_for_output(df, case.name, args.sample_rows_per_case))

    write_report(report_sections, args.report)
    write_sample(sample_frames, args.sample_output)


if __name__ == "__main__":
    main()
