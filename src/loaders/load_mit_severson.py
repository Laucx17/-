from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    h5py = None

try:
    from ._common import (
        concat_frames,
        ensure_unified_columns,
        get_logger,
        infer_step_type_from_current,
        project_root,
        require_existing_dir,
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
        write_table,
    )


LOGGER = get_logger("load_mit_severson")
DATASET_NAME = "MIT_Severson"
DEFAULT_RAW_DIR = project_root() / "data_raw" / "MIT_Severson"
DEFAULT_OUTPUT = project_root() / "data_processed" / "source_domain" / "mit_severson_unified.parquet"


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError(
            "h5py is required to read MIT-Severson MATLAB 7.3/HDF5 .mat files. "
            "Run: pip install -r requirements.txt"
        )


def _flatten_refs(dataset) -> list[Any]:
    refs = np.asarray(dataset[()]).reshape(-1, order="F")
    return [ref for ref in refs if ref]


def _decode_char_array(array: np.ndarray) -> str:
    values = np.asarray(array).squeeze()
    if values.size == 0:
        return ""
    if np.issubdtype(values.dtype, np.integer):
        return "".join(chr(int(v)) for v in values.reshape(-1) if int(v) != 0)
    if values.dtype.kind in {"S", "U"}:
        return "".join(values.astype(str).reshape(-1))
    return str(values)


def _read_dataset_value(dataset):
    data = dataset[()]
    arr = np.asarray(data)
    matlab_class = dataset.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode(errors="ignore")
    if matlab_class == "char" or arr.dtype.kind in {"S", "U"}:
        return _decode_char_array(arr)
    arr = np.squeeze(arr)
    if arr.shape == ():
        try:
            return arr.item()
        except Exception:
            return arr
    return arr.reshape(-1, order="F")


def _read_object(h5, obj):
    if h5py is None:
        return np.nan
    if isinstance(obj, h5py.Reference):
        if not obj:
            return np.nan
        return _read_object(h5, h5[obj])
    if isinstance(obj, h5py.Dataset):
        if h5py.check_dtype(ref=obj.dtype) is not None:
            refs = _flatten_refs(obj)
            if len(refs) == 1:
                return _read_object(h5, refs[0])
            return [_read_object(h5, ref) for ref in refs]
        return _read_dataset_value(obj)
    if isinstance(obj, h5py.Group):
        return {key: _read_object(h5, obj[key]) for key in obj.keys()}
    return obj


def _read_field(h5, group, field: str, index: int | None = None, default=np.nan):
    if h5py is None or field not in group:
        return default
    obj = group[field]
    try:
        if isinstance(obj, h5py.Dataset) and h5py.check_dtype(ref=obj.dtype) is not None:
            refs = _flatten_refs(obj)
            if index is not None:
                if index >= len(refs):
                    return default
                return _read_object(h5, refs[index])
            return [_read_object(h5, ref) for ref in refs]
        return _read_object(h5, obj)
    except Exception as exc:
        LOGGER.warning("Failed to read field %s: %s", field, exc)
        return default


def _as_array(value, length: int | None = None) -> np.ndarray:
    if isinstance(value, dict):
        arr = np.array([], dtype=float)
    elif value is None:
        arr = np.array([], dtype=float)
    else:
        arr = np.asarray(value).squeeze()
        if arr.shape == ():
            arr = np.array([arr.item()])
        else:
            arr = arr.reshape(-1, order="F")

    if length is None:
        return arr
    if arr.size == 0:
        return np.full(length, np.nan)
    if arr.size == 1 and length > 1:
        return np.full(length, arr.item())
    if arr.size < length:
        padded = np.full(length, np.nan, dtype=object)
        padded[: arr.size] = arr
        return padded
    return arr[:length]


def _battery_count(batch_group) -> int:
    for key in ["summary", "cycles", "policy_readable", "barcode", "cycle_life"]:
        if key in batch_group:
            obj = batch_group[key]
            if isinstance(obj, h5py.Dataset) and h5py.check_dtype(ref=obj.dtype) is not None:
                return len(_flatten_refs(obj))
            if isinstance(obj, h5py.Dataset):
                return int(np.asarray(obj).size)
    return 0


def _summary_rows(
    summary: dict,
    battery_id: str,
    protocol: str,
    source_file: str,
) -> pd.DataFrame:
    cycles = _as_array(summary.get("cycle", np.nan))
    length = max(len(cycles), len(_as_array(summary.get("QDischarge", np.nan))), 1)
    qd = _as_array(summary.get("QDischarge", np.nan), length)
    qc = _as_array(summary.get("QCharge", np.nan), length)
    temp = _as_array(summary.get("Tavg", np.nan), length)

    out = pd.DataFrame(
        {
            "dataset": DATASET_NAME,
            "battery_id": battery_id,
            "chemistry": "LFP/graphite",
            "cycle_index": _as_array(cycles, length),
            "step_type": "cycle_summary",
            "time_s": np.nan,
            "voltage_v": np.nan,
            "current_a": np.nan,
            "temperature_c": temp,
            "charge_capacity_ah": qc,
            "discharge_capacity_ah": qd,
            "capacity_ah": qd,
            "protocol": protocol,
            "source_file": source_file,
        }
    )
    return ensure_unified_columns(out)


def _cycle_rows(
    cycle: dict,
    cycle_index: int | float,
    battery_id: str,
    protocol: str,
    source_file: str,
) -> pd.DataFrame:
    time_s = _as_array(cycle.get("t", np.nan))
    voltage = _as_array(cycle.get("V", np.nan))
    current = _as_array(cycle.get("I", np.nan))
    temperature = _as_array(cycle.get("T", np.nan))
    charge_capacity = _as_array(cycle.get("Qc", np.nan))
    discharge_capacity = _as_array(cycle.get("Qd", np.nan))

    length = max(
        len(time_s),
        len(voltage),
        len(current),
        len(temperature),
        len(charge_capacity),
        len(discharge_capacity),
        1,
    )
    current_series = pd.Series(_as_array(current, length))
    step_type = infer_step_type_from_current(current_series)
    qd = pd.Series(_as_array(discharge_capacity, length))
    qc = pd.Series(_as_array(charge_capacity, length))

    out = pd.DataFrame(
        {
            "dataset": DATASET_NAME,
            "battery_id": battery_id,
            "chemistry": "LFP/graphite",
            "cycle_index": cycle_index,
            "step_type": step_type,
            "time_s": _as_array(time_s, length),
            "voltage_v": _as_array(voltage, length),
            "current_a": current_series,
            "temperature_c": _as_array(temperature, length),
            "charge_capacity_ah": qc,
            "discharge_capacity_ah": qd,
            "capacity_ah": qd.where(qd.notna() & (qd != 0), qc),
            "protocol": protocol,
            "source_file": source_file,
        }
    )
    return ensure_unified_columns(out)


def _cycles_field_dict_to_list(cycles: dict) -> list[dict]:
    lengths = [
        len(value)
        for value in cycles.values()
        if isinstance(value, list)
    ]
    if not lengths:
        return []

    cycle_count = max(lengths)
    cycle_list = []
    for idx in range(cycle_count):
        item = {}
        for key, value in cycles.items():
            if isinstance(value, list):
                item[key] = value[idx] if idx < len(value) else np.nan
            else:
                item[key] = value
        cycle_list.append(item)
    return cycle_list


def _standardize_battery(
    h5,
    batch_group,
    battery_index: int,
    mat_path: Path,
    summary_only: bool = False,
) -> pd.DataFrame:
    source_file = mat_path.name
    default_id = f"{mat_path.stem}_cell_{battery_index + 1:03d}"
    policy = _read_field(h5, batch_group, "policy_readable", battery_index, default="")
    if not isinstance(policy, str) or not policy:
        policy = _read_field(h5, batch_group, "policy", battery_index, default="")
    protocol = policy if isinstance(policy, str) and policy else np.nan

    barcode = _read_field(h5, batch_group, "barcode", battery_index, default="")
    if isinstance(barcode, str) and barcode.strip():
        battery_id = barcode.strip()
    else:
        battery_id = default_id

    summary = _read_field(h5, batch_group, "summary", battery_index, default={})
    if not isinstance(summary, dict):
        summary = {}

    if summary_only:
        return _summary_rows(summary, battery_id, protocol, source_file)

    cycles = _read_field(h5, batch_group, "cycles", battery_index, default=[])
    if isinstance(cycles, dict):
        cycles = _cycles_field_dict_to_list(cycles)
    if not isinstance(cycles, list) or not cycles:
        LOGGER.warning("No detailed cycles found for %s; using summary rows", battery_id)
        return _summary_rows(summary, battery_id, protocol, source_file)

    summary_cycle_index = _as_array(summary.get("cycle", np.nan))
    frames = []
    for cycle_position, cycle in enumerate(cycles, start=1):
        if not isinstance(cycle, dict):
            continue
        if len(summary_cycle_index) >= cycle_position:
            cycle_index = summary_cycle_index[cycle_position - 1]
        else:
            cycle_index = cycle_position
        try:
            frames.append(_cycle_rows(cycle, cycle_index, battery_id, protocol, source_file))
        except Exception as exc:
            LOGGER.exception("Failed MIT cycle %s battery %s: %s", cycle_position, battery_id, exc)

    if not frames:
        return _summary_rows(summary, battery_id, protocol, source_file)
    return concat_frames(frames)


def inspect_mit_file(mat_path: str | Path) -> dict[str, Any]:
    _require_h5py()
    mat_path = Path(mat_path)
    with h5py.File(mat_path, "r") as h5:
        return {
            "file": str(mat_path),
            "root_keys": list(h5.keys()),
            "batch_keys": list(h5["batch"].keys()) if "batch" in h5 and isinstance(h5["batch"], h5py.Group) else [],
        }


def load_mit_severson(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_path: str | Path | None = None,
    max_files: int | None = None,
    max_batteries: int | None = None,
    summary_only: bool = False,
) -> pd.DataFrame:
    _require_h5py()
    raw_dir = require_existing_dir(raw_dir, DATASET_NAME)
    mat_files = sorted(raw_dir.glob("*.mat"))
    if not mat_files:
        LOGGER.warning(
            "No MIT-Severson .mat files found in %s. "
            "Place batchdata_updated_struct_errorcorrect.mat files in data_raw/MIT_Severson/.",
            raw_dir,
        )
        return pd.DataFrame()

    frames = []
    for file_index, mat_path in enumerate(mat_files, start=1):
        if max_files is not None and file_index > max_files:
            break
        try:
            LOGGER.info("Opening MIT-Severson file %s", mat_path)
            with h5py.File(mat_path, "r") as h5:
                if "batch" not in h5 or not isinstance(h5["batch"], h5py.Group):
                    LOGGER.warning("File %s does not contain expected HDF5 group /batch", mat_path)
                    continue
                batch_group = h5["batch"]
                count = _battery_count(batch_group)
                if max_batteries is not None:
                    count = min(count, max_batteries)
                LOGGER.info("Detected %s batteries in %s", count, mat_path.name)
                for battery_index in range(count):
                    try:
                        frame = _standardize_battery(h5, batch_group, battery_index, mat_path, summary_only)
                        frames.append(frame)
                        LOGGER.info(
                            "Loaded MIT battery %s/%s from %s rows=%s",
                            battery_index + 1,
                            count,
                            mat_path.name,
                            len(frame),
                        )
                    except Exception as exc:
                        LOGGER.exception("Failed MIT battery index %s in %s: %s", battery_index, mat_path, exc)
        except Exception as exc:
            LOGGER.exception("Failed to process MIT-Severson file %s: %s", mat_path, exc)

    result = concat_frames(frames)
    write_table(result, output_path, LOGGER)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Load MIT-Severson battery cycle life data into a unified long table.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=None, help="Optional small-run limit for testing.")
    parser.add_argument("--max-batteries", type=int, default=None, help="Optional small-run limit for testing.")
    parser.add_argument("--summary-only", action="store_true", help="Use per-cycle summary rows instead of raw traces.")
    args = parser.parse_args()

    df = load_mit_severson(
        raw_dir=args.raw_dir,
        output_path=args.output,
        max_files=args.max_files,
        max_batteries=args.max_batteries,
        summary_only=args.summary_only,
    )
    LOGGER.info("MIT-Severson unified shape: %s", df.shape)
    if not df.empty:
        LOGGER.info("Columns: %s", list(df.columns))


if __name__ == "__main__":
    main()
