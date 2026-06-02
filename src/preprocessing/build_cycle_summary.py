from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loaders.load_calce import load_calce  # noqa: E402
from loaders.load_mit_severson import (  # noqa: E402
    _as_array as mit_as_array,
    _battery_count as mit_battery_count,
    _read_field as mit_read_field,
    h5py,
)
from loaders.load_nasa import (  # noqa: E402
    _as_1d as nasa_as_1d,
    _find_battery_key,
    _iter_nasa_mat_payloads,
    _loadmat_from_bytes,
    _loadmat_from_path,
    _mat_struct_get,
)


LOGGER = logging.getLogger("build_cycle_summary")

SUMMARY_COLUMNS = [
    "dataset",
    "battery_id",
    "chemistry",
    "cycle_index_raw",
    "global_cycle_index",
    "step_source",
    "capacity_ah",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "initial_capacity_ah",
    "soh",
    "cycle_life",
    "rul",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "voltage_mean_v",
    "voltage_min_v",
    "voltage_max_v",
    "current_mean_a",
    "current_abs_mean_a",
    "charge_time_s",
    "discharge_time_s",
    "ccct_s",
    "cvct_s",
    "adv_v",
    "icv_v",
    "source_file",
    "internal_resistance_ohm",
    "capacity_invalid_flag",
    "capacity_label_excluded_flag",
    "capacity_rebound_flag",
]

EIS_COLUMNS = [
    "dataset",
    "battery_id",
    "chemistry",
    "raw_cycle_index",
    "nearest_discharge_global_cycle",
    "step_source",
    "re_ohm",
    "rct_ohm",
    "battery_impedance_points",
    "rectified_impedance_points",
    "ambient_temperature_c",
    "source_file",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def as_numeric(series_or_value) -> pd.Series:
    return pd.to_numeric(series_or_value, errors="coerce")


def scalar_or_nan(value: Any) -> float:
    arr = np.asarray(value).squeeze()
    if arr.size == 0:
        return np.nan
    try:
        value = arr.reshape(-1)[0]
        if np.iscomplexobj(value):
            value = np.real(value)
        return float(value)
    except Exception:
        return np.nan


def duration_s(time_values) -> float:
    time = pd.to_numeric(pd.Series(np.asarray(time_values).reshape(-1)), errors="coerce").dropna()
    if len(time) < 2:
        return np.nan
    value = float(time.max() - time.min())
    return value if value >= 0 else np.nan


def first_valid(series_or_array) -> float:
    values = pd.to_numeric(pd.Series(np.asarray(series_or_array).reshape(-1)), errors="coerce").dropna()
    return float(values.iloc[0]) if len(values) else np.nan


def mean_or_nan(series_or_array) -> float:
    values = pd.to_numeric(pd.Series(np.asarray(series_or_array).reshape(-1)), errors="coerce")
    return float(values.mean()) if values.notna().any() else np.nan


def min_or_nan(series_or_array) -> float:
    values = pd.to_numeric(pd.Series(np.asarray(series_or_array).reshape(-1)), errors="coerce")
    return float(values.min()) if values.notna().any() else np.nan


def max_or_nan(series_or_array) -> float:
    values = pd.to_numeric(pd.Series(np.asarray(series_or_array).reshape(-1)), errors="coerce")
    return float(values.max()) if values.notna().any() else np.nan


def capacity_increment_ah(series_or_array) -> float:
    """Return within-cycle capacity increment.

    CALCE Arbin workbooks can store capacity as a running cumulative value
    within a file. Using max-min gives a per-cycle increment and remains
    equivalent to max when the raw capacity resets to zero each cycle.
    """
    values = pd.to_numeric(pd.Series(np.asarray(series_or_array).reshape(-1)), errors="coerce").dropna()
    if len(values) == 0:
        return np.nan
    increment = float(values.max() - values.min())
    return increment if increment > 0 else float(values.max())


def approx_cc_cv_times(charge_time, charge_voltage, charge_current) -> tuple[float, float]:
    """Approximate CC/CV durations from raw charge points.

    CCCT is estimated as time spent near the dominant positive current.
    CVCT is estimated as time spent near the maximum charge voltage while current is tapering.
    These are pragmatic first-pass features, not protocol-grade segmentation.
    """
    time = pd.to_numeric(pd.Series(np.asarray(charge_time).reshape(-1)), errors="coerce")
    voltage = pd.to_numeric(pd.Series(np.asarray(charge_voltage).reshape(-1)), errors="coerce")
    current = pd.to_numeric(pd.Series(np.asarray(charge_current).reshape(-1)), errors="coerce")
    mask = time.notna() & voltage.notna() & current.notna() & (current > 1e-8)
    if mask.sum() < 2:
        return np.nan, np.nan

    sub = pd.DataFrame({"time": time[mask], "voltage": voltage[mask], "current": current[mask]}).sort_values("time")
    current_abs = sub["current"].abs()
    dominant_current = current_abs.median()
    current_tol = max(0.05 * dominant_current, 0.02)
    cc_mask = (current_abs - dominant_current).abs() <= current_tol

    voltage_max = sub["voltage"].max()
    voltage_tol = max(0.01, 0.002 * abs(voltage_max))
    cv_mask = (voltage_max - sub["voltage"]).abs() <= voltage_tol

    return duration_s(sub.loc[cc_mask, "time"]), duration_s(sub.loc[cv_mask, "time"])


def ensure_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[SUMMARY_COLUMNS]


def ensure_eis_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in EIS_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[EIS_COLUMNS]


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return ensure_summary_columns(df)

    df = ensure_summary_columns(df.copy())
    df["capacity_ah"] = as_numeric(df["capacity_ah"])
    df["discharge_capacity_ah"] = as_numeric(df["discharge_capacity_ah"])
    df["charge_capacity_ah"] = as_numeric(df["charge_capacity_ah"])
    df["global_cycle_index"] = as_numeric(df["global_cycle_index"])

    df["capacity_invalid_flag"] = df["capacity_ah"].notna() & (df["capacity_ah"] <= 0)
    df.loc[df["capacity_invalid_flag"], "capacity_ah"] = np.nan
    df["capacity_label_excluded_flag"] = False

    result = []
    for battery_id, group in df.sort_values(["battery_id", "global_cycle_index"]).groupby("battery_id", sort=False):
        group = group.copy()
        positive_capacity = group["capacity_ah"].dropna()
        if len(positive_capacity):
            typical_capacity = float(positive_capacity.median())
        else:
            typical_capacity = np.nan

        if pd.notna(typical_capacity) and typical_capacity > 0:
            initial_candidates = positive_capacity[
                (positive_capacity >= 0.8 * typical_capacity)
                & (positive_capacity <= 1.2 * typical_capacity)
            ]
        else:
            initial_candidates = positive_capacity

        if len(initial_candidates) >= 3:
            initial_capacity = float(initial_candidates.iloc[:3].mean())
        elif len(initial_candidates) >= 1:
            initial_capacity = float(initial_candidates.iloc[0])
        elif len(positive_capacity) >= 3:
            initial_capacity = float(positive_capacity.iloc[:3].mean())
        elif len(positive_capacity) >= 1:
            initial_capacity = float(positive_capacity.iloc[0])
        else:
            initial_capacity = np.nan

        label_capacity = group["capacity_ah"].copy()
        if pd.notna(initial_capacity) and initial_capacity > 0:
            label_exclusion_mask = label_capacity.notna() & (
                (label_capacity < 0.75 * initial_capacity)
                | (label_capacity > 1.3 * initial_capacity)
            )
            group.loc[label_exclusion_mask, "capacity_label_excluded_flag"] = True
            label_capacity = label_capacity.mask(label_exclusion_mask)

        group["initial_capacity_ah"] = initial_capacity
        if pd.notna(initial_capacity) and initial_capacity > 0:
            group["soh"] = label_capacity / initial_capacity
        else:
            group["soh"] = np.nan

        eol_rows = group.loc[group["soh"].notna() & (group["soh"] <= 0.8), "global_cycle_index"]
        cycle_life = float(eol_rows.iloc[0]) if len(eol_rows) else np.nan
        group["cycle_life"] = cycle_life
        group["rul"] = group["cycle_life"] - group["global_cycle_index"] if pd.notna(cycle_life) else np.nan

        prev_capacity = label_capacity.ffill().shift(1)
        rebound_threshold = 0.05 * initial_capacity if pd.notna(initial_capacity) else np.nan
        group["capacity_rebound_flag"] = (
            label_capacity.notna()
            & prev_capacity.notna()
            & pd.notna(rebound_threshold)
            & ((label_capacity - prev_capacity) > rebound_threshold)
        )
        result.append(group)

    return ensure_summary_columns(pd.concat(result, ignore_index=True))


def load_mit_summary(raw_dir: Path) -> pd.DataFrame:
    if h5py is None:
        raise ImportError("h5py is required for MIT-Severson MATLAB 7.3/HDF5 .mat files.")
    frames = []
    for mat_path in sorted(raw_dir.glob("*.mat")):
        LOGGER.info("Building MIT summary from %s", mat_path)
        try:
            with h5py.File(mat_path, "r") as h5:
                if "batch" not in h5:
                    LOGGER.warning("Skipping MIT file without /batch: %s", mat_path)
                    continue
                batch = h5["batch"]
                for battery_idx in range(mit_battery_count(batch)):
                    summary = mit_read_field(h5, batch, "summary", battery_idx, default={})
                    if not isinstance(summary, dict):
                        continue
                    policy = mit_read_field(h5, batch, "policy_readable", battery_idx, default="")
                    barcode = mit_read_field(h5, batch, "barcode", battery_idx, default="")
                    battery_id = barcode.strip() if isinstance(barcode, str) and barcode.strip() else f"{mat_path.stem}_cell_{battery_idx + 1:03d}"
                    protocol = policy if isinstance(policy, str) and policy else np.nan

                    cycle = mit_as_array(summary.get("cycle", np.nan))
                    qd = mit_as_array(summary.get("QDischarge", np.nan), len(cycle))
                    qc = mit_as_array(summary.get("QCharge", np.nan), len(cycle))
                    tavg = mit_as_array(summary.get("Tavg", np.nan), len(cycle))
                    tmax = mit_as_array(summary.get("Tmax", np.nan), len(cycle))
                    tmin = mit_as_array(summary.get("Tmin", np.nan), len(cycle))
                    ir = mit_as_array(summary.get("IR", np.nan), len(cycle))
                    chargetime_min = mit_as_array(summary.get("chargetime", np.nan), len(cycle))

                    frame = pd.DataFrame(
                        {
                            "dataset": "MIT_Severson",
                            "battery_id": battery_id,
                            "chemistry": "LFP/graphite",
                            "cycle_index_raw": cycle,
                            "global_cycle_index": pd.to_numeric(pd.Series(cycle), errors="coerce").rank(method="first").astype("Int64"),
                            "step_source": "mit_summary",
                            "capacity_ah": qd,
                            "charge_capacity_ah": qc,
                            "discharge_capacity_ah": qd,
                            "temperature_mean_c": tavg,
                            "temperature_max_c": tmax,
                            "temperature_min_c": tmin,
                            "voltage_mean_v": np.nan,
                            "voltage_min_v": np.nan,
                            "voltage_max_v": np.nan,
                            "current_mean_a": np.nan,
                            "current_abs_mean_a": np.nan,
                            "charge_time_s": pd.to_numeric(pd.Series(chargetime_min), errors="coerce") * 60.0,
                            "discharge_time_s": np.nan,
                            "ccct_s": np.nan,
                            "cvct_s": np.nan,
                            "adv_v": np.nan,
                            "icv_v": np.nan,
                            "source_file": mat_path.name,
                            "internal_resistance_ohm": ir,
                            "protocol": protocol,
                        }
                    )
                    frames.append(frame)
        except Exception as exc:
            LOGGER.exception("Failed to build MIT summary from %s: %s", mat_path, exc)

    return assign_labels(pd.concat(frames, ignore_index=True)) if frames else ensure_summary_columns(pd.DataFrame())


def build_calce_summary(raw_dir: Path) -> pd.DataFrame:
    LOGGER.info("Loading CALCE long table for cycle summary")
    long_df = load_calce(raw_dir=raw_dir, output_path=None)
    if long_df.empty:
        return ensure_summary_columns(pd.DataFrame())

    long_df = long_df.copy()
    long_df["cycle_index"] = as_numeric(long_df["cycle_index"])
    long_df["time_s"] = as_numeric(long_df["time_s"])
    long_df["voltage_v"] = as_numeric(long_df["voltage_v"])
    long_df["current_a"] = as_numeric(long_df["current_a"])
    long_df["charge_capacity_ah"] = as_numeric(long_df["charge_capacity_ah"])
    long_df["discharge_capacity_ah"] = as_numeric(long_df["discharge_capacity_ah"])

    summary_rows = []
    for battery_id, battery_df in long_df.groupby("battery_id", sort=False):
        global_cycle = 0
        source_order = list(pd.unique(battery_df["source_file"]))
        for source_file in source_order:
            file_df = battery_df[battery_df["source_file"] == source_file]
            raw_cycles = sorted(file_df["cycle_index"].dropna().unique())
            for raw_cycle in raw_cycles:
                group = file_df[file_df["cycle_index"] == raw_cycle].copy()
                if group.empty:
                    continue
                global_cycle += 1
                charge_mask = group["current_a"] > 1e-8
                discharge_mask = group["current_a"] < -1e-8

                charge_capacity = capacity_increment_ah(group.loc[charge_mask, "charge_capacity_ah"])
                if pd.isna(charge_capacity):
                    charge_capacity = capacity_increment_ah(group["charge_capacity_ah"])
                discharge_capacity = capacity_increment_ah(group.loc[discharge_mask, "discharge_capacity_ah"])
                if pd.isna(discharge_capacity):
                    discharge_capacity = capacity_increment_ah(group["discharge_capacity_ah"])

                ccct, cvct = approx_cc_cv_times(
                    group.loc[charge_mask, "time_s"],
                    group.loc[charge_mask, "voltage_v"],
                    group.loc[charge_mask, "current_a"],
                )

                summary_rows.append(
                    {
                        "dataset": "CALCE",
                        "battery_id": battery_id,
                        "chemistry": "Li-ion",
                        "cycle_index_raw": raw_cycle,
                        "global_cycle_index": global_cycle,
                        "step_source": "calce_xlsx_cycle",
                        "capacity_ah": discharge_capacity,
                        "charge_capacity_ah": charge_capacity,
                        "discharge_capacity_ah": discharge_capacity,
                        "temperature_mean_c": mean_or_nan(group["temperature_c"]),
                        "temperature_max_c": max_or_nan(group["temperature_c"]),
                        "temperature_min_c": min_or_nan(group["temperature_c"]),
                        "voltage_mean_v": mean_or_nan(group["voltage_v"]),
                        "voltage_min_v": min_or_nan(group["voltage_v"]),
                        "voltage_max_v": max_or_nan(group["voltage_v"]),
                        "current_mean_a": mean_or_nan(group["current_a"]),
                        "current_abs_mean_a": mean_or_nan(group["current_a"].abs()),
                        "charge_time_s": duration_s(group.loc[charge_mask, "time_s"]),
                        "discharge_time_s": duration_s(group.loc[discharge_mask, "time_s"]),
                        "ccct_s": ccct,
                        "cvct_s": cvct,
                        "adv_v": mean_or_nan(group.loc[discharge_mask, "voltage_v"]),
                        "icv_v": first_valid(group.loc[charge_mask, "voltage_v"]),
                        "source_file": source_file,
                        "internal_resistance_ohm": np.nan,
                    }
                )

    return assign_labels(pd.DataFrame(summary_rows))


def _cycle_features(cycle, raw_index: int, source_file: str) -> dict[str, Any]:
    cycle_type = str(_mat_struct_get(cycle, "type", "unknown")).lower()
    data = _mat_struct_get(cycle, "data", None)
    time = nasa_as_1d(_mat_struct_get(data, "Time", np.nan))
    voltage = nasa_as_1d(_mat_struct_get(data, "Voltage_measured", _mat_struct_get(data, "Voltage_charge", np.nan)))
    current = nasa_as_1d(_mat_struct_get(data, "Current_measured", _mat_struct_get(data, "Current_charge", np.nan)))
    temperature = nasa_as_1d(_mat_struct_get(data, "Temperature_measured", _mat_struct_get(cycle, "ambient_temperature", np.nan)))
    ccct, cvct = approx_cc_cv_times(time, voltage, current)
    return {
        "cycle_type": cycle_type,
        "raw_index": raw_index,
        "time": time,
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "duration_s": duration_s(time),
        "voltage_mean_v": mean_or_nan(voltage),
        "voltage_min_v": min_or_nan(voltage),
        "voltage_max_v": max_or_nan(voltage),
        "current_mean_a": mean_or_nan(current),
        "current_abs_mean_a": mean_or_nan(np.abs(pd.to_numeric(pd.Series(current), errors="coerce"))),
        "temperature_mean_c": mean_or_nan(temperature),
        "temperature_max_c": max_or_nan(temperature),
        "temperature_min_c": min_or_nan(temperature),
        "ccct_s": ccct,
        "cvct_s": cvct,
        "icv_v": first_valid(voltage),
        "capacity_ah": scalar_or_nan(_mat_struct_get(data, "Capacity", np.nan)),
        "source_file": source_file,
    }


def _impedance_row(cycle, battery_id: str, raw_index: int, nearest_discharge_global: int | float, source_file: str) -> dict[str, Any]:
    data = _mat_struct_get(cycle, "data", None)
    battery_impedance = np.asarray(_mat_struct_get(data, "Battery_impedance", [])).reshape(-1)
    rectified = np.asarray(_mat_struct_get(data, "Rectified_Impedance", [])).reshape(-1)
    return {
        "dataset": "NASA",
        "battery_id": battery_id,
        "chemistry": "Li-ion",
        "raw_cycle_index": raw_index,
        "nearest_discharge_global_cycle": nearest_discharge_global,
        "step_source": "nasa_impedance",
        "re_ohm": scalar_or_nan(_mat_struct_get(data, "Re", np.nan)),
        "rct_ohm": scalar_or_nan(_mat_struct_get(data, "Rct", np.nan)),
        "battery_impedance_points": len(battery_impedance),
        "rectified_impedance_points": len(rectified),
        "ambient_temperature_c": scalar_or_nan(_mat_struct_get(cycle, "ambient_temperature", np.nan)),
        "source_file": source_file,
    }


def build_nasa_summary(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    eis_rows = []
    seen_batteries: set[str] = set()

    for source_file, payload in _iter_nasa_mat_payloads(raw_dir):
        try:
            mat_dict = _loadmat_from_path(payload) if isinstance(payload, Path) else _loadmat_from_bytes(payload)
            battery_key = _find_battery_key(mat_dict, Path(str(source_file)).stem)
            if battery_key in seen_batteries:
                LOGGER.warning("Skipping duplicate NASA battery_id %s from %s", battery_key, source_file)
                continue
            seen_batteries.add(battery_key)

            battery = mat_dict[battery_key]
            cycles = np.asarray(_mat_struct_get(battery, "cycle", [])).reshape(-1)
            latest_charge = None
            discharge_global = 0
            LOGGER.info("Building NASA summary for %s from %s cycles=%s", battery_key, source_file, len(cycles))

            for raw_idx, cycle in enumerate(cycles, start=1):
                cycle_type = str(_mat_struct_get(cycle, "type", "unknown")).lower()
                if cycle_type == "charge":
                    latest_charge = _cycle_features(cycle, raw_idx, source_file)
                elif cycle_type == "discharge":
                    discharge_global += 1
                    discharge = _cycle_features(cycle, raw_idx, source_file)
                    charge = latest_charge or {}
                    capacity = discharge["capacity_ah"]
                    summary_rows.append(
                        {
                            "dataset": "NASA",
                            "battery_id": battery_key,
                            "chemistry": "Li-ion",
                            "cycle_index_raw": raw_idx,
                            "global_cycle_index": discharge_global,
                            "step_source": "nasa_discharge_with_previous_charge",
                            "capacity_ah": capacity,
                            "charge_capacity_ah": np.nan,
                            "discharge_capacity_ah": capacity,
                            "temperature_mean_c": discharge["temperature_mean_c"],
                            "temperature_max_c": discharge["temperature_max_c"],
                            "temperature_min_c": discharge["temperature_min_c"],
                            "voltage_mean_v": discharge["voltage_mean_v"],
                            "voltage_min_v": discharge["voltage_min_v"],
                            "voltage_max_v": discharge["voltage_max_v"],
                            "current_mean_a": discharge["current_mean_a"],
                            "current_abs_mean_a": discharge["current_abs_mean_a"],
                            "charge_time_s": charge.get("duration_s", np.nan),
                            "discharge_time_s": discharge["duration_s"],
                            "ccct_s": charge.get("ccct_s", np.nan),
                            "cvct_s": charge.get("cvct_s", np.nan),
                            "adv_v": discharge["voltage_mean_v"],
                            "icv_v": charge.get("icv_v", np.nan),
                            "source_file": source_file,
                            "internal_resistance_ohm": np.nan,
                        }
                    )
                elif cycle_type == "impedance":
                    eis_rows.append(_impedance_row(cycle, battery_key, raw_idx, discharge_global, source_file))
        except Exception as exc:
            LOGGER.exception("Failed to build NASA summary from %s: %s", source_file, exc)

    return assign_labels(pd.DataFrame(summary_rows)), ensure_eis_columns(pd.DataFrame(eis_rows))


def missing_rates(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return df.isna().mean().sort_values(ascending=False)


def continuity_issues(df: pd.DataFrame) -> dict[str, list[str]]:
    issues = {}
    if df.empty:
        return issues
    for battery_id, group in df.groupby("battery_id"):
        cycles = pd.to_numeric(group["global_cycle_index"], errors="coerce").dropna().sort_values().astype(int).unique()
        if len(cycles) == 0:
            issues[battery_id] = ["no numeric global_cycle_index"]
            continue
        expected = np.arange(cycles.min(), cycles.max() + 1)
        missing = sorted(set(expected) - set(cycles))
        if missing:
            issues[battery_id] = [f"missing global_cycle_index values: {missing[:10]}"]
    return issues


def dataset_report_section(dataset_name: str, df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    if df.empty:
        return f"## {dataset_name}\n\n未处理到循环数据。\n", {
            "dataset": dataset_name,
            "battery_count": 0,
            "cycle_count": 0,
            "eol_reached_count": 0,
            "eol_not_reached_count": 0,
        }

    battery_count = df["battery_id"].nunique()
    cycle_count = len(df)
    eol_by_battery = df.groupby("battery_id")["cycle_life"].first()
    reached = eol_by_battery.notna()
    not_reached_ids = eol_by_battery[~reached].index.tolist()
    continuity = continuity_issues(df)

    miss = missing_rates(df)
    missing_rows = "\n".join(f"| {field} | {rate:.3f} |" for field, rate in miss.items())
    missing_table = f"| 字段 | 缺失率 |\n| --- | --- |\n{missing_rows}"

    available = [col for col in df.columns if df[col].notna().any()]
    unavailable = [col for col in df.columns if not df[col].notna().any()]
    cycle_life = pd.to_numeric(df["cycle_life"], errors="coerce")
    soh = pd.to_numeric(df["soh"], errors="coerce")
    invalid_capacity_count = int(df["capacity_invalid_flag"].fillna(False).sum())
    label_excluded_count = int(df["capacity_label_excluded_flag"].fillna(False).sum())
    rebound_count = int(df["capacity_rebound_flag"].fillna(False).sum())

    text = f"""## {dataset_name}

- 电池数量：{battery_count}
- 循环数量：{cycle_count}
- 达到 EOL 的电池数量：{int(reached.sum())}
- 未达到 EOL 的电池数量：{int((~reached).sum())}
- cycle_life 范围：{cycle_life.min(skipna=True)} 到 {cycle_life.max(skipna=True)}
- soh 范围：{soh.min(skipna=True)} 到 {soh.max(skipna=True)}
- capacity_ah <= 0 被标记为无效的循环数：{invalid_capacity_count}
- 容量过低或过高而未参与 SOH/RUL 标签计算的循环数：{label_excluded_count}
- 容量异常回升标记数：{rebound_count}
- global_cycle_index 连续性问题电池数：{len(continuity)}

可用字段：

`{", ".join(available)}`

完全缺失字段：

`{", ".join(unavailable) if unavailable else "无"}`

字段缺失率：

{missing_table}

未达到 EOL 的电池：

`{", ".join(not_reached_ids[:50]) if not_reached_ids else "无"}`

"""
    return text, {
        "dataset": dataset_name,
        "battery_count": battery_count,
        "cycle_count": cycle_count,
        "eol_reached_count": int(reached.sum()),
        "eol_not_reached_count": int((~reached).sum()),
        "not_reached_ids": not_reached_ids,
        "continuity_issues": continuity,
    }


def write_report(summary: pd.DataFrame, eis: pd.DataFrame, report_path: Path) -> None:
    sections = []
    stats = []
    for dataset_name, dataset_df in summary.groupby("dataset", sort=False):
        section, stat = dataset_report_section(dataset_name, dataset_df)
        sections.append(section)
        stats.append(stat)

    if summary.empty:
        sections.append("## 总览\n\n未生成任何循环级 summary 数据。\n")

    total_batteries = summary["battery_id"].nunique() if not summary.empty else 0
    total_cycles = len(summary)
    eol_missing = (
        summary.groupby(["dataset", "battery_id"])["cycle_life"].first().reset_index()
        if not summary.empty
        else pd.DataFrame(columns=["dataset", "battery_id", "cycle_life"])
    )
    not_reached = eol_missing[eol_missing["cycle_life"].isna()]

    content = f"""# 循环级统一汇总表构建报告

本报告由 `src/preprocessing/build_cycle_summary.py` 自动生成。脚本只读取源域数据并构建循环级汇总表，不进行任何模型训练，也不修改 `data_raw/` 原始文件。

## 总体结果

- 总电池数量：{total_batteries}
- 总循环数量：{total_cycles}
- 未达到 EOL 的电池数量：{len(not_reached)}
- NASA EIS/impedance 特征行数：{len(eis)}

标签定义：

- `initial_capacity_ah`：每个电池前 3 个有效容量的平均值；若有效容量不足 3 个，则取第一个有效容量。
- `soh`：当前循环容量 / 初始容量。
- `cycle_life`：第一次 `soh <= 0.8` 的 `global_cycle_index`。
- `rul`：`cycle_life - global_cycle_index`。若未达到 EOL，则为 NaN。

容量质量处理：

- `capacity_ah <= 0` 被标记为 `capacity_invalid_flag=True`，并在 SOH/RUL 标签计算中置为 NaN。
- `capacity_label_excluded_flag=True` 表示该循环容量明显偏离初始容量窗口，保留原值但不参与 SOH、cycle_life 和 RUL 计算；当前阈值为低于初始容量 75% 或高于 130%，初始容量优先从接近该电池典型容量的早期循环中选取。
- `capacity_rebound_flag=True` 表示当前容量较上一有效容量回升超过初始容量的 5%，仅作为异常标记，不自动删除。

## 数据集处理说明

### MIT-Severson

第一版循环级 summary 只使用 MATLAB HDF5 文件中的 `summary` 结构，不全量展开 detailed cycles。容量使用 `QDischarge`，充电容量使用 `QCharge`，温度使用 `Tavg/Tmax/Tmin`，内阻 `IR` 保留为 `internal_resistance_ohm`。`chargetime` 按官方 summary 数值近似理解为分钟，并转换为 `charge_time_s`。

### CALCE

CALCE 的多个 Excel 文件中 `Cycle_Index` 会在每个文件内重新从 1 开始。因此脚本保留原始 `Cycle_Index` 为 `cycle_index_raw`，并按每个 `battery_id` 的 `source_file` 出现顺序与文件内 `Cycle_Index` 重新生成连续的 `global_cycle_index`。`ccct_s/cvct_s` 使用电流平台和最高电压附近时长的近似估计；这是第一版工程特征，后续可根据协议精细切分。

CALCE 的 Arbin 容量列在部分文件中表现为文件内累计值，因此循环级 `charge_capacity_ah` 和 `discharge_capacity_ah` 使用同一循环内容量列的 `max-min` 增量，而不是直接使用最大值。

### NASA

NASA 的普通循环 summary 只为 `discharge` cycle 生成一行；相邻前序 `charge` cycle 的充电时间、ICV 和 CC/CV 近似特征会合并到该 discharge 行。`impedance` cycle 不混入普通循环表，而是单独提取 `Re/Rct` 等字段到 `nasa_eis_features.csv`。

## 各数据集结果

{''.join(sections)}

## 未达到 EOL 的电池汇总

`{', '.join((not_reached['dataset'] + ':' + not_reached['battery_id']).head(100).tolist()) if len(not_reached) else '无'}`

## 后续建模建议

1. 优先使用 MIT-Severson summary 表作为早期寿命预测和迁移学习源域基准，因为其 cycle-level 容量、温度和内阻字段较完整。
2. CALCE 可用于验证跨文件循环对齐和容量退化建模，但必须使用 `global_cycle_index`，不要直接把各 Excel 内部的 `Cycle_Index` 当作全局循环。
3. NASA 可用于 SOH/RUL 与 EIS 辅助特征研究，普通循环表与 `nasa_eis_features.csv` 建议在后续特征工程阶段按最近邻循环或时间顺序合并。
4. 温度、电压、电流曲线类模型优先从 CALCE 和 NASA 入手；MIT 第一版 summary 不包含完整电压/电流曲线。
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    LOGGER.info("Saved cycle summary report: %s", report_path)


def build_cycle_summary(
    output_dir: Path,
    report_path: Path,
    max_calce_workbooks: int | None = None,
    max_nasa_files: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_root = PROJECT_ROOT / "data_raw"
    mit_df = load_mit_summary(raw_root / "MIT_Severson")

    if max_calce_workbooks is not None:
        LOGGER.info("Building CALCE summary using temporary max_workbooks=%s", max_calce_workbooks)
        calce_long = load_calce(raw_dir=raw_root / "CALCE", output_path=None, max_workbooks=max_calce_workbooks)
        # Reuse a temporary directory-free path by summarizing the already-loaded frame.
        original_load_calce = globals()["load_calce"]
        try:
            globals()["load_calce"] = lambda raw_dir, output_path=None: calce_long
            calce_df = build_calce_summary(raw_root / "CALCE")
        finally:
            globals()["load_calce"] = original_load_calce
    else:
        calce_df = build_calce_summary(raw_root / "CALCE")

    if max_nasa_files is not None:
        # Keep this option for quick debugging without changing the default full-data behavior.
        LOGGER.warning("max_nasa_files is currently a debug hint; full NASA iterator is used in production mode.")
    nasa_df, eis_df = build_nasa_summary(raw_root / "NASA")

    summary = pd.concat([mit_df, calce_df, nasa_df], ignore_index=True)
    summary = ensure_summary_columns(summary)
    eis_df = ensure_eis_columns(eis_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "unified_cycle_summary.csv"
    parquet_path = output_dir / "unified_cycle_summary.parquet"
    eis_path = output_dir / "nasa_eis_features.csv"

    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary.to_parquet(parquet_path, index=False)
    eis_df.to_csv(eis_path, index=False, encoding="utf-8-sig")

    LOGGER.info("Saved unified cycle summary CSV: %s rows -> %s", len(summary), csv_path)
    LOGGER.info("Saved unified cycle summary parquet: %s rows -> %s", len(summary), parquet_path)
    LOGGER.info("Saved NASA EIS features: %s rows -> %s", len(eis_df), eis_path)

    write_report(summary, eis_df, report_path)
    return summary, eis_df


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Build unified cycle-level summary tables for source-domain datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_processed" / "unified_tables",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs" / "cycle_summary_report.md",
    )
    parser.add_argument("--max-calce-workbooks", type=int, default=None, help="Debug-only CALCE workbook limit.")
    parser.add_argument("--max-nasa-files", type=int, default=None, help="Reserved debug hint.")
    args = parser.parse_args()

    build_cycle_summary(
        output_dir=args.output_dir,
        report_path=args.report,
        max_calce_workbooks=args.max_calce_workbooks,
        max_nasa_files=args.max_nasa_files,
    )


if __name__ == "__main__":
    main()
