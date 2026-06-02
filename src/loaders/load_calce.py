from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ._common import (
        concat_frames,
        ensure_unified_columns,
        get_logger,
        infer_step_type_from_current,
        project_root,
        require_existing_dir,
        to_numeric,
        write_table,
    )
except ImportError:
    from _common import (
        concat_frames,
        ensure_unified_columns,
        get_logger,
        infer_step_type_from_current,
        project_root,
        require_existing_dir,
        to_numeric,
        write_table,
    )


LOGGER = get_logger("load_calce")
DATASET_NAME = "CALCE"
DEFAULT_RAW_DIR = project_root() / "data_raw" / "CALCE"
DEFAULT_OUTPUT = project_root() / "data_processed" / "source_domain" / "calce_unified.parquet"


def _pick_data_sheets(excel_file: pd.ExcelFile) -> list[str]:
    data_sheets = []
    required_columns = {
        "Data_Point",
        "Test_Time(s)",
        "Step_Time(s)",
        "Cycle_Index",
        "Current(A)",
        "Voltage(V)",
    }
    for sheet in excel_file.sheet_names:
        if sheet.lower() == "info":
            continue
        try:
            header = pd.read_excel(excel_file, sheet_name=sheet, nrows=0)
        except Exception as exc:
            LOGGER.warning("Cannot inspect sheet %s: %s", sheet, exc)
            continue
        cols = set(map(str, header.columns))
        if required_columns.issubset(cols):
            data_sheets.append(sheet)
    return data_sheets


def _standardize_calce_sheet(
    df: pd.DataFrame,
    battery_id: str,
    protocol: str,
    source_file: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    current = to_numeric(df.get("Current(A)", np.nan))
    charge_capacity = to_numeric(df.get("Charge_Capacity(Ah)", np.nan))
    discharge_capacity = to_numeric(df.get("Discharge_Capacity(Ah)", np.nan))

    out["dataset"] = DATASET_NAME
    out["battery_id"] = battery_id
    out["chemistry"] = "Li-ion"
    out["cycle_index"] = to_numeric(df.get("Cycle_Index", np.nan))
    out["step_type"] = infer_step_type_from_current(current)
    out["time_s"] = to_numeric(df.get("Test_Time(s)", df.get("Step_Time(s)", np.nan)))
    out["voltage_v"] = to_numeric(df.get("Voltage(V)", np.nan))
    out["current_a"] = current
    out["temperature_c"] = np.nan
    out["charge_capacity_ah"] = charge_capacity
    out["discharge_capacity_ah"] = discharge_capacity
    out["capacity_ah"] = discharge_capacity.where(discharge_capacity.notna() & (discharge_capacity != 0), charge_capacity)
    out["protocol"] = protocol
    out["source_file"] = source_file
    return ensure_unified_columns(out)


def _read_workbook_bytes(
    workbook_bytes: bytes,
    battery_id: str,
    source_file: str,
) -> pd.DataFrame:
    frames = []
    bio = io.BytesIO(workbook_bytes)
    excel_file = pd.ExcelFile(bio)
    data_sheets = _pick_data_sheets(excel_file)
    if not data_sheets:
        LOGGER.warning("No data sheet found in %s", source_file)
        return pd.DataFrame()

    for sheet in data_sheets:
        try:
            bio.seek(0)
            df = pd.read_excel(bio, sheet_name=sheet)
            frames.append(
                _standardize_calce_sheet(
                    df=df,
                    battery_id=battery_id,
                    protocol=sheet,
                    source_file=f"{source_file}::{sheet}",
                )
            )
            LOGGER.info("Loaded CALCE sheet %s rows=%s", f"{source_file}::{sheet}", len(df))
        except Exception as exc:
            LOGGER.exception("Failed to read CALCE sheet %s::%s: %s", source_file, sheet, exc)
    return concat_frames(frames)


def _iter_zip_workbooks(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if not member.lower().endswith((".xlsx", ".xls")):
                continue
            yield member, zf.read(member)


def load_calce(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_path: str | Path | None = None,
    max_workbooks: int | None = None,
) -> pd.DataFrame:
    raw_dir = require_existing_dir(raw_dir, DATASET_NAME)
    frames = []
    workbook_count = 0

    zip_files = sorted(raw_dir.glob("*.zip"))
    loose_workbooks = sorted([*raw_dir.rglob("*.xlsx"), *raw_dir.rglob("*.xls")])

    if not zip_files and not loose_workbooks:
        LOGGER.warning(
            "No CALCE .zip/.xlsx/.xls files found in %s. "
            "Please place CS2/CX2 raw files in data_raw/CALCE/.",
            raw_dir,
        )
        return pd.DataFrame()

    for zip_path in zip_files:
        battery_id = zip_path.stem
        try:
            for member, workbook_bytes in _iter_zip_workbooks(zip_path):
                if max_workbooks is not None and workbook_count >= max_workbooks:
                    break
                source_file = f"{zip_path.name}::{member}"
                frames.append(_read_workbook_bytes(workbook_bytes, battery_id, source_file))
                workbook_count += 1
        except Exception as exc:
            LOGGER.exception("Failed to process CALCE zip %s: %s", zip_path, exc)

    for workbook_path in loose_workbooks:
        if max_workbooks is not None and workbook_count >= max_workbooks:
            break
        try:
            battery_id = workbook_path.parent.name if workbook_path.parent != raw_dir else workbook_path.stem.split("_")[0]
            frames.append(_read_workbook_bytes(workbook_path.read_bytes(), battery_id, str(workbook_path)))
            workbook_count += 1
        except Exception as exc:
            LOGGER.exception("Failed to process CALCE workbook %s: %s", workbook_path, exc)

    result = concat_frames(frames)
    write_table(result, output_path, LOGGER)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CALCE CS2/CX2 battery data into a unified long table.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workbooks", type=int, default=None, help="Optional small-run limit for testing.")
    args = parser.parse_args()

    df = load_calce(args.raw_dir, args.output, args.max_workbooks)
    LOGGER.info("CALCE unified shape: %s", df.shape)
    if not df.empty:
        LOGGER.info("Columns: %s", list(df.columns))


if __name__ == "__main__":
    main()
