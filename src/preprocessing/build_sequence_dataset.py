from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELING_READY_PATH = PROJECT_ROOT / "data_processed" / "source_domain" / "modeling_ready_cycle_life_dataset.parquet"
UNIFIED_SUMMARY_PATH = PROJECT_ROOT / "data_processed" / "unified_tables" / "unified_cycle_summary.parquet"
NASA_EIS_PATH = PROJECT_ROOT / "data_processed" / "unified_tables" / "nasa_eis_features.csv"

SEQUENCE_DIR = PROJECT_ROOT / "data_processed" / "sequences"
FIGURE_DIR = PROJECT_ROOT / "figures" / "sequences"
REPORT_PATH = PROJECT_ROOT / "docs" / "sequence_dataset_report.md"

CALCE_SEQUENCE_PATH = SEQUENCE_DIR / "calce_soh_sequences.parquet"
NASA_SEQUENCE_PATH = SEQUENCE_DIR / "nasa_soh_sequences.parquet"
MIT_SEQUENCE_PATH = SEQUENCE_DIR / "mit_soh_sequences.parquet"
METADATA_PATH = SEQUENCE_DIR / "sequence_metadata.csv"

CALCE_FIGURE_PATH = FIGURE_DIR / "calce_soh_sequences.png"
NASA_CORE_FIGURE_PATH = FIGURE_DIR / "nasa_core_soh_sequences.png"
MIT_SAMPLE_FIGURE_PATH = FIGURE_DIR / "mit_sample_soh_sequences.png"
LENGTH_DISTRIBUTION_PATH = FIGURE_DIR / "sequence_length_distribution.png"
MISSING_RATE_FIGURE_PATH = FIGURE_DIR / "feature_missing_rate.png"

LOGGER = logging.getLogger("build_sequence_dataset")

NASA_CORE_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
CALCE_BATTERIES = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
CALCE_BATTERY_LEVEL_SPLIT = {
    "CS2_35": "battery_level_train",
    "CS2_36": "battery_level_train",
    "CS2_37": "battery_level_external_test",
    "CS2_38": "battery_level_external_test",
}

REQUIRED_SEQUENCE_COLUMNS = [
    "dataset",
    "battery_id",
    "time_idx",
    "global_cycle_index",
    "capacity_ah",
    "soh",
    "rul",
    "target_soh",
    "ccct_s",
    "cvct_s",
    "adv_v",
    "icv_v",
    "charge_time_s",
    "discharge_time_s",
    "internal_resistance_ohm",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "voltage_mean_v",
    "voltage_min_v",
    "voltage_max_v",
    "current_mean_a",
    "current_abs_mean_a",
    "has_eol",
    "split",
    "source_file",
]

QUALITY_FLAG_COLUMNS = [
    "capacity_invalid_flag",
    "capacity_label_excluded_flag",
    "capacity_rebound_flag",
]

MISSING_RATE_FEATURES = [
    "capacity_ah",
    "soh",
    "rul",
    "ccct_s",
    "cvct_s",
    "adv_v",
    "icv_v",
    "charge_time_s",
    "discharge_time_s",
    "internal_resistance_ohm",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "voltage_mean_v",
    "voltage_min_v",
    "voltage_max_v",
    "current_mean_a",
    "current_abs_mean_a",
    "eis_re_ohm",
    "eis_rct_ohm",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [MODELING_READY_PATH, UNIFIED_SUMMARY_PATH, NASA_EIS_PATH]:
        require_file(path)
    modeling = pd.read_parquet(MODELING_READY_PATH)
    unified = pd.read_parquet(UNIFIED_SUMMARY_PATH)
    eis = pd.read_csv(NASA_EIS_PATH)
    LOGGER.info("Loaded modeling-ready table: %s", modeling.shape)
    LOGGER.info("Loaded unified cycle summary: %s", unified.shape)
    LOGGER.info("Loaded NASA EIS features: %s", eis.shape)
    return modeling, unified, eis


def as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def split_within_battery(group: pd.DataFrame) -> pd.Series:
    length = len(group)
    cutoff = max(1, int(np.floor(length * 0.70)))
    return pd.Series(np.where(np.arange(length) < cutoff, "train", "test"), index=group.index)


def prepare_base_sequence(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    seq = df.copy()
    for col in REQUIRED_SEQUENCE_COLUMNS + QUALITY_FLAG_COLUMNS:
        if col not in seq.columns:
            seq[col] = np.nan

    numeric_cols = [
        "global_cycle_index",
        "capacity_ah",
        "soh",
        "target_cycle_life",
        "ccct_s",
        "cvct_s",
        "adv_v",
        "icv_v",
        "charge_time_s",
        "discharge_time_s",
        "internal_resistance_ohm",
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "voltage_mean_v",
        "voltage_min_v",
        "voltage_max_v",
        "current_mean_a",
        "current_abs_mean_a",
    ]
    for col in numeric_cols:
        if col in seq.columns:
            seq[col] = pd.to_numeric(seq[col], errors="coerce")

    seq = seq.sort_values(["battery_id", "global_cycle_index", "source_file"]).reset_index(drop=True)
    seq["time_idx"] = seq.groupby("battery_id").cumcount().astype(int)
    seq["target_soh"] = seq["soh"]
    seq["has_eol"] = as_bool(seq["has_eol"])
    seq["rul"] = np.where(
        seq["has_eol"] & seq["target_cycle_life"].notna(),
        seq["target_cycle_life"] - seq["global_cycle_index"],
        np.nan,
    )
    seq["split"] = seq.groupby("battery_id", group_keys=False).apply(split_within_battery)

    seq["sequence_dataset_role"] = dataset_name
    if "battery_level_split_suggestion" not in seq.columns:
        seq["battery_level_split_suggestion"] = ""
    if dataset_name == "CALCE":
        seq["battery_level_split_suggestion"] = seq["battery_id"].map(CALCE_BATTERY_LEVEL_SPLIT).fillna("")
    elif dataset_name == "NASA":
        seq["battery_level_split_suggestion"] = np.where(
            as_bool(seq.get("is_nasa_core", pd.Series(False, index=seq.index))),
            "nasa_core_validation",
            "nasa_all_clean_sequence",
        )
    elif dataset_name == "MIT_Severson":
        seq["battery_level_split_suggestion"] = "pretraining_candidate"

    for flag in QUALITY_FLAG_COLUMNS:
        seq[flag] = as_bool(seq[flag])

    return seq


def select_calce(modeling: pd.DataFrame) -> pd.DataFrame:
    flag = as_bool(modeling.get("is_calce_feature_validation", pd.Series(False, index=modeling.index)))
    mask = (modeling["dataset"] == "CALCE") | flag
    calce = modeling.loc[mask].copy()
    calce = calce[calce["battery_id"].isin(CALCE_BATTERIES)].copy()
    return prepare_base_sequence(calce, "CALCE")


def select_mit(modeling: pd.DataFrame) -> pd.DataFrame:
    flag = as_bool(modeling.get("is_mit_curated_124", pd.Series(False, index=modeling.index)))
    mit = modeling.loc[(modeling["dataset"] == "MIT_Severson") & flag].copy()
    return prepare_base_sequence(mit, "MIT_Severson")


def select_nasa(modeling: pd.DataFrame, eis: pd.DataFrame) -> pd.DataFrame:
    core = as_bool(modeling.get("is_nasa_core", pd.Series(False, index=modeling.index)))
    clean = as_bool(modeling.get("is_nasa_all_clean", pd.Series(False, index=modeling.index)))
    nasa = modeling.loc[(modeling["dataset"] == "NASA") & (core | clean)].copy()
    nasa = prepare_base_sequence(nasa, "NASA")
    return merge_nasa_eis(nasa, eis)


def merge_nasa_eis(nasa: pd.DataFrame, eis: pd.DataFrame) -> pd.DataFrame:
    if nasa.empty:
        for col in [
            "eis_re_ohm",
            "eis_rct_ohm",
            "eis_raw_cycle_index",
            "eis_nearest_discharge_global_cycle",
            "eis_ambient_temperature_c",
            "eis_measurement_count",
            "eis_cycles_since_measurement",
            "eis_alignment",
        ]:
            nasa[col] = np.nan
        return nasa

    if eis.empty:
        LOGGER.warning("NASA EIS table is empty; EIS fields will remain missing.")
        return nasa

    e = eis.copy()
    e = e[e["battery_id"].isin(nasa["battery_id"].unique())].copy()
    e = e.sort_values(["battery_id", "nearest_discharge_global_cycle", "raw_cycle_index"])
    grouped = (
        e.groupby(["battery_id", "nearest_discharge_global_cycle"], as_index=False)
        .agg(
            eis_re_ohm=("re_ohm", "last"),
            eis_rct_ohm=("rct_ohm", "last"),
            eis_raw_cycle_index=("raw_cycle_index", "last"),
            eis_ambient_temperature_c=("ambient_temperature_c", "last"),
            eis_battery_impedance_points=("battery_impedance_points", "last"),
            eis_rectified_impedance_points=("rectified_impedance_points", "last"),
            eis_measurement_count=("raw_cycle_index", "count"),
        )
        .rename(columns={"nearest_discharge_global_cycle": "global_cycle_index"})
    )

    out = nasa.merge(grouped, on=["battery_id", "global_cycle_index"], how="left")
    out["_eis_exact_match"] = out["eis_re_ohm"].notna()
    eis_cols = [
        "eis_re_ohm",
        "eis_rct_ohm",
        "eis_raw_cycle_index",
        "eis_ambient_temperature_c",
        "eis_battery_impedance_points",
        "eis_rectified_impedance_points",
        "eis_measurement_count",
    ]
    out = out.sort_values(["battery_id", "global_cycle_index"]).reset_index(drop=True)
    for col in eis_cols:
        out[col] = out.groupby("battery_id")[col].ffill()
    out["eis_nearest_discharge_global_cycle"] = out.groupby("battery_id")["global_cycle_index"].transform(
        lambda s: s.where(out.loc[s.index, "_eis_exact_match"]).ffill()
    )
    out["eis_cycles_since_measurement"] = out["global_cycle_index"] - out["eis_nearest_discharge_global_cycle"]
    out["eis_alignment"] = np.where(
        out["_eis_exact_match"],
        "exact_nearest_cycle",
        np.where(out["eis_re_ohm"].notna(), "forward_fill_from_previous_eis", "missing_before_first_eis"),
    )
    out = out.drop(columns=["_eis_exact_match"])
    return out


def final_column_order(seq: pd.DataFrame) -> pd.DataFrame:
    extra_cols = [
        "target_cycle_life",
        "capacity_invalid_flag",
        "capacity_label_excluded_flag",
        "capacity_rebound_flag",
        "is_calce_feature_validation",
        "is_nasa_core",
        "is_nasa_all_clean",
        "is_mit_curated_124",
        "battery_level_split_suggestion",
        "sequence_dataset_role",
        "eis_re_ohm",
        "eis_rct_ohm",
        "eis_raw_cycle_index",
        "eis_nearest_discharge_global_cycle",
        "eis_cycles_since_measurement",
        "eis_ambient_temperature_c",
        "eis_battery_impedance_points",
        "eis_rectified_impedance_points",
        "eis_measurement_count",
        "eis_alignment",
    ]
    ordered = REQUIRED_SEQUENCE_COLUMNS + [col for col in extra_cols if col in seq.columns]
    for col in ordered:
        if col not in seq.columns:
            seq[col] = np.nan
    remaining = [col for col in seq.columns if col not in ordered]
    return seq[ordered + remaining]


def build_metadata(sequence_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, seq in sequence_tables.items():
        if seq.empty:
            continue
        for battery_id, group in seq.groupby("battery_id"):
            group = group.sort_values("time_idx")
            time_diff = group["time_idx"].diff().dropna()
            split_test = group[group["split"] == "test"]
            rows.append(
                {
                    "dataset": dataset_name,
                    "battery_id": battery_id,
                    "sequence_length": len(group),
                    "global_cycle_min": group["global_cycle_index"].min(),
                    "global_cycle_max": group["global_cycle_index"].max(),
                    "time_idx_min": group["time_idx"].min(),
                    "time_idx_max": group["time_idx"].max(),
                    "time_idx_continuous": bool(
                        group["time_idx"].min() == 0
                        and group["time_idx"].max() == len(group) - 1
                        and (time_diff.eq(1).all() if len(time_diff) else True)
                    ),
                    "soh_min": group["soh"].min(skipna=True),
                    "soh_max": group["soh"].max(skipna=True),
                    "capacity_min_ah": group["capacity_ah"].min(skipna=True),
                    "capacity_max_ah": group["capacity_ah"].max(skipna=True),
                    "has_eol": bool(group["has_eol"].fillna(False).astype(bool).any()),
                    "target_cycle_life": group["target_cycle_life"].dropna().iloc[0]
                    if group["target_cycle_life"].notna().any()
                    else np.nan,
                    "train_steps": int((group["split"] == "train").sum()),
                    "test_steps": int((group["split"] == "test").sum()),
                    "split_first_test_time_idx": int(split_test["time_idx"].min()) if len(split_test) else np.nan,
                    "split_first_test_global_cycle": split_test["global_cycle_index"].min() if len(split_test) else np.nan,
                    "soh_anomaly_count": int(((group["soh"] <= 0) | (group["soh"] > 1.2)).fillna(False).sum()),
                    "capacity_nonpositive_count": int((group["capacity_ah"] <= 0).fillna(False).sum()),
                    "ccct_missing_rate": group["ccct_s"].isna().mean(),
                    "cvct_missing_rate": group["cvct_s"].isna().mean(),
                    "adv_missing_rate": group["adv_v"].isna().mean(),
                    "icv_missing_rate": group["icv_v"].isna().mean(),
                    "eis_re_missing_rate": group["eis_re_ohm"].isna().mean()
                    if "eis_re_ohm" in group.columns
                    else np.nan,
                    "eis_rct_missing_rate": group["eis_rct_ohm"].isna().mean()
                    if "eis_rct_ohm" in group.columns
                    else np.nan,
                    "is_nasa_core": bool(group.get("is_nasa_core", pd.Series(False, index=group.index)).fillna(False).astype(bool).any()),
                    "is_nasa_all_clean": bool(
                        group.get("is_nasa_all_clean", pd.Series(False, index=group.index)).fillna(False).astype(bool).any()
                    ),
                    "battery_level_split_suggestion": group["battery_level_split_suggestion"].dropna().iloc[0]
                    if "battery_level_split_suggestion" in group.columns and group["battery_level_split_suggestion"].notna().any()
                    else "",
                    "source_file_count": group["source_file"].nunique(dropna=True),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "battery_id"]).reset_index(drop=True)


def field_missing_rate_table(sequence_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_name, seq in sequence_tables.items():
        row: dict[str, Any] = {"dataset": dataset_name}
        for field in MISSING_RATE_FEATURES:
            row[field] = seq[field].isna().mean() if field in seq.columns and len(seq) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_stats(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby("dataset", as_index=False)
        .agg(
            battery_count=("battery_id", "nunique"),
            sequence_length_min=("sequence_length", "min"),
            sequence_length_max=("sequence_length", "max"),
            sequence_length_mean=("sequence_length", "mean"),
            soh_min=("soh_min", "min"),
            soh_max=("soh_max", "max"),
            time_idx_discontinuous_count=("time_idx_continuous", lambda s: int((~s.astype(bool)).sum())),
            soh_anomaly_count=("soh_anomaly_count", "sum"),
            capacity_nonpositive_count=("capacity_nonpositive_count", "sum"),
        )
        .sort_values("dataset")
    )


def plot_soh_sequences(seq: pd.DataFrame, batteries: list[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for battery_id in batteries:
        group = seq[seq["battery_id"] == battery_id].sort_values("time_idx")
        if group.empty:
            continue
        ax.plot(group["time_idx"], group["target_soh"], linewidth=1.6, label=battery_id)
    ax.set_title(title)
    ax.set_xlabel("time_idx")
    ax.set_ylabel("SOH")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sequence_length_distribution(metadata: pd.DataFrame, output_path: Path) -> None:
    labels = []
    values = []
    for dataset_name, group in metadata.groupby("dataset", sort=True):
        labels.append(dataset_name)
        values.append(group["sequence_length"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_title("Sequence length distribution")
    ax.set_ylabel("sequence length")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_missing_rate(missing_rates: pd.DataFrame, output_path: Path) -> None:
    plot_features = [
        "ccct_s",
        "cvct_s",
        "adv_v",
        "icv_v",
        "internal_resistance_ohm",
        "temperature_mean_c",
        "voltage_mean_v",
        "current_mean_a",
        "eis_re_ohm",
        "eis_rct_ohm",
    ]
    long = missing_rates.melt(id_vars="dataset", value_vars=[f for f in plot_features if f in missing_rates.columns])
    long["value"] = long["value"] * 100.0
    pivot = long.pivot(index="variable", columns="dataset", values="value").fillna(100.0)
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    x = np.arange(len(pivot.index))
    width = 0.22
    for idx, dataset_name in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[dataset_name], width=width, label=dataset_name)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("missing rate (%)")
    ax.set_title("Key sequence feature missing rates")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{float(value):.6g}"
        return str(value)
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_无_"
    shown = df.copy()
    if max_rows is not None:
        shown = shown.head(max_rows)
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = [
        "| " + " | ".join(format_value(row[col]) for col in shown.columns) + " |"
        for _, row in shown.iterrows()
    ]
    suffix = ""
    if max_rows is not None and len(df) > max_rows:
        suffix = f"\n\n_仅显示前 {max_rows} 行，共 {len(df)} 行。_"
    return "\n".join([header, sep, *rows]) + suffix


def generate_report(
    sequence_tables: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    missing_rates: pd.DataFrame,
    unified: pd.DataFrame,
    eis: pd.DataFrame,
) -> str:
    stats = dataset_stats(metadata)
    missing_view = missing_rates.copy()
    for col in missing_view.columns:
        if col != "dataset":
            missing_view[col] = missing_view[col] * 100.0

    ccct_view = missing_view[["dataset", "ccct_s", "cvct_s", "adv_v", "icv_v"]].copy()
    nasa_meta = metadata[metadata["dataset"] == "NASA"].copy()
    nasa_core_meta = nasa_meta[nasa_meta["is_nasa_core"]].copy()
    calce_meta = metadata[metadata["dataset"] == "CALCE"].copy()
    split_view = metadata[
        [
            "dataset",
            "battery_id",
            "sequence_length",
            "train_steps",
            "test_steps",
            "split_first_test_time_idx",
            "split_first_test_global_cycle",
            "battery_level_split_suggestion",
        ]
    ].copy()

    nasa_eis_alignment = (
        sequence_tables["NASA"].groupby("eis_alignment", as_index=False).size().rename(columns={"size": "row_count"})
        if "eis_alignment" in sequence_tables["NASA"].columns and len(sequence_tables["NASA"])
        else pd.DataFrame()
    )

    return f"""# DeepAR / TAM-DeepAR-EN 序列数据集报告

本报告由 `src/preprocessing/build_sequence_dataset.py` 自动生成。本阶段只构建序列数据集，不训练 DeepAR，不训练 TAM-DeepAR-EN，不修改原始数据，也不修改 modeling-ready 数据集。

## 输入文件

- `data_processed/source_domain/modeling_ready_cycle_life_dataset.parquet`
- `data_processed/unified_tables/unified_cycle_summary.parquet`
- `data_processed/unified_tables/nasa_eis_features.csv`

输入表规模：

- modeling-ready 表：`{sum(len(seq) for seq in sequence_tables.values())}` 行被选入序列输出；原始 unified summary 表：`{len(unified)}` 行；
- NASA EIS 表：`{len(eis)}` 行。

## 输出文件

- `data_processed/sequences/calce_soh_sequences.parquet`
- `data_processed/sequences/nasa_soh_sequences.parquet`
- `data_processed/sequences/mit_soh_sequences.parquet`
- `data_processed/sequences/sequence_metadata.csv`
- `figures/sequences/calce_soh_sequences.png`
- `figures/sequences/nasa_core_soh_sequences.png`
- `figures/sequences/mit_sample_soh_sequences.png`
- `figures/sequences/sequence_length_distribution.png`
- `figures/sequences/feature_missing_rate.png`

## 序列数据概况

每个 `battery_id` 是一条序列，每个 `global_cycle_index` 是一个时间步；脚本按电池内循环顺序生成从 0 开始的连续 `time_idx`。`target_soh = soh`。`rul` 只作为后续 RUL 评价标签，不应作为 SOH 预测输入。

{markdown_table(stats)}

## 每个数据集的处理方式

### CALCE

CALCE 保留 `CS2_35`、`CS2_36`、`CS2_37`、`CS2_38` 四条序列。每个电池前 70% 时间步标记为 `train`，后 30% 标记为 `test`。另外在 metadata 中给出 battery-level 外部测试建议：`CS2_35/CS2_36` 作为训练，`CS2_37/CS2_38` 作为外部测试。

CALCE 分割点：

{markdown_table(calce_meta[["battery_id", "sequence_length", "train_steps", "test_steps", "split_first_test_global_cycle", "battery_level_split_suggestion"]])}

### NASA

NASA 使用两类标记：`is_nasa_core=True` 和 `is_nasa_all_clean=True`，二者取并集后构建序列。NASA core 电池 `B0005/B0006/B0007/B0018` 单独统计如下：

{markdown_table(nasa_core_meta[["battery_id", "sequence_length", "train_steps", "test_steps", "soh_min", "soh_max", "eis_re_missing_rate", "eis_rct_missing_rate"]])}

### MIT-Severson

MIT 暂时只构建 SOH 序列表，不作为 DeepAR 第一版主训练。当前 MIT summary-only 表有容量、SOH、温度和内阻，但缺少 detailed cycles 中的电压/电流曲线信息，也缺少 CCCT、CVCT、ADV、ICV。因此更适合作为后续大规模预训练候选，或在补充 detailed-cycle 特征后再用于 DeepAR/TAM-DeepAR-EN 主训练。

## 字段缺失率

下表为关键时序特征缺失率，单位为百分比。

{markdown_table(missing_view, max_rows=20)}

CCCT、CVCT、ADV、ICV 可用性：

{markdown_table(ccct_view)}

## NASA EIS 合并方式

`nasa_eis_features.csv` 中的 Re、Rct 通过 `battery_id + nearest_discharge_global_cycle` 合并到 NASA 序列。若某个普通循环没有精确对应 EIS，则使用同一电池此前最近一次 EIS 特征向前填充；首次 EIS 之前的普通循环仍保留 NaN，不反向填充。

EIS 对齐情况：

{markdown_table(nasa_eis_alignment)}

## 每个 battery 的 train/test 分割点

完整分割点已保存到 `data_processed/sequences/sequence_metadata.csv`。报告中仅显示前 40 行：

{markdown_table(split_view, max_rows=40)}

## 数据质量检查

- `time_idx` 不连续的序列数量见概况表中的 `time_idx_discontinuous_count`。
- `SOH <= 0` 或 `SOH > 1.2` 的异常点数量见 `soh_anomaly_count`。
- `capacity_ah <= 0` 的异常点数量见 `capacity_nonpositive_count`。
- 异常点没有被删除，只在序列和 metadata 中保留质量标记，供后续训练前过滤或加权处理。

## DeepAR 第一版建议

第一版 DeepAR/SOH 序列预测建议优先使用 NASA core + NASA all-clean 的并集。原因是 NASA 序列数量多于 CALCE，且普通循环和 EIS 特征已经可以按时间步合并。CALCE 只有 4 条电池序列，更适合做 TAM-DeepAR-EN 的间接健康因子复现和小样本验证。

## TAM-DeepAR-EN 特征建议

优先使用以下特征：

- 容量与 SOH：`capacity_ah`、`soh`
- 间接健康因子：`ccct_s`、`cvct_s`、`adv_v`、`icv_v`
- 工况与环境：`temperature_mean_c`、`temperature_max_c`、`temperature_min_c`
- 充放电时间：`charge_time_s`、`discharge_time_s`
- NASA 阻抗特征：`eis_re_ohm`、`eis_rct_ohm`

注意：`rul` 不应作为 SOH 预测输入，只能作为后续 RUL 评价或联合预测标签。

## 下一步如何训练 DeepAR

下一步建议创建独立训练脚本，读取本阶段 parquet：

1. 选择数据集和特征列；
2. 按 `battery_id` 构造多条相关时间序列；
3. 使用 `split` 字段做每条电池内部 70/30 时间切分；
4. 先训练普通 DeepAR 预测 `target_soh`；
5. 再加入时间注意力 TAM 和 Elastic Net/稀疏正则思想，形成 TAM-DeepAR-EN；
6. 最后用 `rul` 和 EOL 标签做 RUL 评价。

## 当前风险和注意事项

- CALCE 电池数量太少，不适合单独训练复杂深度模型。
- MIT 当前缺少详细曲线特征，暂不作为第一版 DeepAR 主训练。
- NASA EIS 是最近邻合并并前向填充，非严格同一循环同步测量，论文中需要说明。
- 部分电池存在 SOH 或容量异常点，本阶段不删除，后续训练前需要决定过滤、截断或鲁棒损失。
- 不同数据集的 `rul` 标签来源和 EOL 定义不完全一致，SOH 预测和 RUL 评价应分开描述。
"""


def build_sequence_dataset() -> dict[str, pd.DataFrame]:
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    modeling, unified, eis = load_inputs()

    calce = final_column_order(select_calce(modeling))
    nasa = final_column_order(select_nasa(modeling, eis))
    mit = final_column_order(select_mit(modeling))
    sequence_tables = {"CALCE": calce, "NASA": nasa, "MIT_Severson": mit}

    calce.to_parquet(CALCE_SEQUENCE_PATH, index=False)
    nasa.to_parquet(NASA_SEQUENCE_PATH, index=False)
    mit.to_parquet(MIT_SEQUENCE_PATH, index=False)

    metadata = build_metadata(sequence_tables)
    metadata.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")
    missing_rates = field_missing_rate_table(sequence_tables)

    plot_soh_sequences(calce, CALCE_BATTERIES, CALCE_FIGURE_PATH, "CALCE SOH sequences")
    plot_soh_sequences(nasa, NASA_CORE_BATTERIES, NASA_CORE_FIGURE_PATH, "NASA core SOH sequences")
    mit_batteries = sorted(mit["battery_id"].dropna().unique())
    sample_count = min(10, len(mit_batteries))
    mit_sample = list(pd.Series(mit_batteries).sample(n=sample_count, random_state=42)) if sample_count else []
    plot_soh_sequences(mit, mit_sample, MIT_SAMPLE_FIGURE_PATH, "MIT sample SOH sequences")
    plot_sequence_length_distribution(metadata, LENGTH_DISTRIBUTION_PATH)
    plot_feature_missing_rate(missing_rates, MISSING_RATE_FIGURE_PATH)

    REPORT_PATH.write_text(generate_report(sequence_tables, metadata, missing_rates, unified, eis), encoding="utf-8")

    LOGGER.info("Saved CALCE sequences: %s", CALCE_SEQUENCE_PATH)
    LOGGER.info("Saved NASA sequences: %s", NASA_SEQUENCE_PATH)
    LOGGER.info("Saved MIT sequences: %s", MIT_SEQUENCE_PATH)
    LOGGER.info("Saved sequence metadata: %s", METADATA_PATH)
    LOGGER.info("Saved sequence figures under: %s", FIGURE_DIR)
    LOGGER.info("Saved sequence report: %s", REPORT_PATH)
    print(
        "Sequence datasets built: "
        f"CALCE={calce['battery_id'].nunique()} batteries/{len(calce)} rows, "
        f"NASA={nasa['battery_id'].nunique()} batteries/{len(nasa)} rows, "
        f"MIT={mit['battery_id'].nunique()} batteries/{len(mit)} rows"
    )
    return sequence_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SOH sequence datasets for DeepAR/TAM-DeepAR-EN.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    parse_args()
    build_sequence_dataset()


if __name__ == "__main__":
    main()
