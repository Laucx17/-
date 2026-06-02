from __future__ import annotations

import argparse
import io
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

try:
    from ._common import (
        concat_frames,
        ensure_unified_columns,
        get_logger,
        project_root,
        require_existing_dir,
        write_table,
    )
except ImportError:
    from _common import (
        concat_frames,
        ensure_unified_columns,
        get_logger,
        project_root,
        require_existing_dir,
        write_table,
    )


LOGGER = get_logger("load_nasa")
DATASET_NAME = "NASA"
DEFAULT_RAW_DIR = project_root() / "data_raw" / "NASA"
DEFAULT_OUTPUT = project_root() / "data_processed" / "source_domain" / "nasa_unified.parquet"


def _mat_struct_get(obj, field: str, default=np.nan):
    if obj is None:
        return default
    if hasattr(obj, field):
        return getattr(obj, field)
    if isinstance(obj, dict):
        return obj.get(field, default)
    return default


def _as_1d(value, length: int | None = None) -> np.ndarray:
    if value is None:
        arr = np.array([], dtype=float)
    else:
        arr = np.asarray(value).squeeze()
        if arr.shape == ():
            arr = np.array([arr.item()])
        else:
            arr = arr.reshape(-1)
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


def _loadmat_from_bytes(data: bytes):
    if loadmat is None:
        raise ImportError("scipy is required to read NASA .mat files. Run: pip install -r requirements.txt")
    try:
        return loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            return loadmat(tmp.name, squeeze_me=True, struct_as_record=False)


def _loadmat_from_path(path: Path):
    if loadmat is None:
        raise ImportError("scipy is required to read NASA .mat files. Run: pip install -r requirements.txt")
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def _find_battery_key(mat_dict: dict, fallback: str) -> str:
    keys = [key for key in mat_dict if not key.startswith("__")]
    if not keys:
        return fallback
    exact = [key for key in keys if key.upper().startswith("B")]
    return exact[0] if exact else keys[0]


def _standardize_nasa_cycle(
    cycle,
    battery_id: str,
    cycle_index: int,
    source_file: str,
) -> pd.DataFrame:
    cycle_type = str(_mat_struct_get(cycle, "type", "unknown"))
    ambient_temperature = _mat_struct_get(cycle, "ambient_temperature", np.nan)
    data = _mat_struct_get(cycle, "data", None)

    time_s = _as_1d(_mat_struct_get(data, "Time", np.nan))
    voltage = _as_1d(
        _mat_struct_get(data, "Voltage_measured", _mat_struct_get(data, "Voltage_charge", np.nan))
    )
    current = _as_1d(
        _mat_struct_get(data, "Current_measured", _mat_struct_get(data, "Current_charge", np.nan))
    )
    temperature = _as_1d(_mat_struct_get(data, "Temperature_measured", ambient_temperature))

    length = max(time_s.size, voltage.size, current.size, temperature.size, 1)
    capacity = _as_1d(_mat_struct_get(data, "Capacity", np.nan), length)

    out = pd.DataFrame(
        {
            "dataset": DATASET_NAME,
            "battery_id": battery_id,
            "chemistry": "Li-ion",
            "cycle_index": cycle_index,
            "step_type": cycle_type.lower(),
            "time_s": _as_1d(time_s, length),
            "voltage_v": _as_1d(voltage, length),
            "current_a": _as_1d(current, length),
            "temperature_c": _as_1d(temperature, length),
            "charge_capacity_ah": np.nan,
            "discharge_capacity_ah": capacity if cycle_type.lower() == "discharge" else np.nan,
            "capacity_ah": capacity,
            "protocol": f"type={cycle_type}; ambient_temperature={ambient_temperature}",
            "source_file": source_file,
        }
    )
    return ensure_unified_columns(out)


def _standardize_nasa_mat(mat_dict: dict, source_file: str) -> pd.DataFrame:
    battery_key = _find_battery_key(mat_dict, Path(source_file).stem)
    battery = mat_dict[battery_key]
    cycles = _mat_struct_get(battery, "cycle", [])
    cycles = np.asarray(cycles).reshape(-1)

    frames = []
    for idx, cycle in enumerate(cycles, start=1):
        try:
            frames.append(_standardize_nasa_cycle(cycle, battery_key, idx, source_file))
        except Exception as exc:
            LOGGER.exception("Failed NASA cycle %s in %s: %s", idx, source_file, exc)
    return concat_frames(frames)


def _iter_nasa_mat_payloads(raw_dir: Path) -> Iterator[tuple[str, bytes | Path]]:
    for mat_path in sorted(raw_dir.rglob("*.mat")):
        yield str(mat_path), mat_path

    for zip_path in sorted(raw_dir.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    lower = member.lower()
                    if lower.endswith(".mat"):
                        yield f"{zip_path.name}::{member}", zf.read(member)
                    elif lower.endswith(".zip"):
                        nested_bytes = zf.read(member)
                        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
                            for inner_member in inner.namelist():
                                if inner_member.lower().endswith(".mat"):
                                    yield f"{zip_path.name}::{member}::{inner_member}", inner.read(inner_member)
        except Exception as exc:
            LOGGER.exception("Failed to inspect NASA zip %s: %s", zip_path, exc)


def load_nasa(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_path: str | Path | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    raw_dir = require_existing_dir(raw_dir, DATASET_NAME)
    frames = []
    count = 0

    for source_file, payload in _iter_nasa_mat_payloads(raw_dir):
        if max_files is not None and count >= max_files:
            break
        try:
            mat_dict = _loadmat_from_path(payload) if isinstance(payload, Path) else _loadmat_from_bytes(payload)
            frame = _standardize_nasa_mat(mat_dict, source_file)
            frames.append(frame)
            count += 1
            LOGGER.info("Loaded NASA file %s rows=%s", source_file, len(frame))
        except Exception as exc:
            LOGGER.exception("Failed to load NASA file %s: %s", source_file, exc)

    if count == 0:
        LOGGER.warning(
            "No NASA .mat files found in %s. Place B0005.mat/B0006.mat/... or NASA zip files in data_raw/NASA/.",
            raw_dir,
        )

    result = concat_frames(frames)
    write_table(result, output_path, LOGGER)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Load NASA Li-ion Battery Aging data into a unified long table.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-files", type=int, default=None, help="Optional small-run limit for testing.")
    args = parser.parse_args()

    df = load_nasa(args.raw_dir, args.output, args.max_files)
    LOGGER.info("NASA unified shape: %s", df.shape)
    if not df.empty:
        LOGGER.info("Columns: %s", list(df.columns))


if __name__ == "__main__":
    main()

