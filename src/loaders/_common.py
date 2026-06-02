from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


UNIFIED_COLUMNS = [
    "dataset",
    "battery_id",
    "chemistry",
    "cycle_index",
    "step_type",
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "capacity_ah",
    "protocol",
    "source_file",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def empty_unified_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIFIED_COLUMNS)


def ensure_unified_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[UNIFIED_COLUMNS]


def concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid = [ensure_unified_columns(frame) for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return empty_unified_frame()
    return pd.concat(valid, ignore_index=True)


def to_numeric(series_or_value):
    return pd.to_numeric(series_or_value, errors="coerce")


def infer_step_type_from_current(current: pd.Series, eps: float = 1e-8) -> pd.Series:
    numeric = to_numeric(current)
    step = pd.Series(np.full(len(numeric), "rest", dtype=object), index=numeric.index)
    step[numeric > eps] = "charge"
    step[numeric < -eps] = "discharge"
    step[numeric.isna()] = np.nan
    return step


def write_table(df: pd.DataFrame, output_path: str | Path | None, logger: logging.Logger) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path}. Use .csv or .parquet.")
    logger.info("Saved unified table: %s rows -> %s", len(df), output_path)


def require_existing_dir(path: str | Path, dataset_name: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} raw data directory does not exist: {path}. "
            f"Please place the downloaded raw data in this folder."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"{dataset_name} raw data path is not a directory: {path}")
    return path

