from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_DIR = PROJECT_ROOT / "data_processed" / "sequences"
FIGURE_DIR = PROJECT_ROOT / "figures" / "sequences_clean"
REPORT_PATH = PROJECT_ROOT / "docs" / "clean_sequence_dataset_report.md"

CALCE_INPUT = SEQUENCE_DIR / "calce_soh_sequences.parquet"
NASA_INPUT = SEQUENCE_DIR / "nasa_soh_sequences.parquet"
MIT_INPUT = SEQUENCE_DIR / "mit_soh_sequences.parquet"
METADATA_INPUT = SEQUENCE_DIR / "sequence_metadata.csv"

CALCE_CLEAN_OUTPUT = SEQUENCE_DIR / "calce_soh_sequences_clean.parquet"
NASA_CLEAN_OUTPUT = SEQUENCE_DIR / "nasa_soh_sequences_clean.parquet"
MIT_CLEAN_OUTPUT = SEQUENCE_DIR / "mit_soh_sequences_clean.parquet"
METADATA_CLEAN_OUTPUT = SEQUENCE_DIR / "sequence_metadata_clean.csv"
CALCE_RESET_OUTPUT = SEQUENCE_DIR / "calce_soh_reset_detection.csv"

CALCE_CLEAN_FIGURE = FIGURE_DIR / "calce_clean_soh_sequences.png"
NASA_CLEAN_FIGURE = FIGURE_DIR / "nasa_clean_soh_sequences.png"
MIT_CLEAN_SAMPLE_FIGURE = FIGURE_DIR / "mit_clean_sample_soh_sequences.png"
LENGTH_FIGURE = FIGURE_DIR / "clean_sequence_length_distribution.png"
MISSING_FIGURE = FIGURE_DIR / "clean_feature_missing_rate.png"
CALCE_RESET_FIGURE = FIGURE_DIR / "calce_soh_reset_detection.png"

LOGGER = logging.getLogger("build_clean_sequence_dataset")

RESET_THRESHOLD = 0.05
NASA_HIGH_SOH_THRESHOLD = 1.15
NASA_CORE_BATTERIES = {"B0005", "B0006", "B0007", "B0018"}

KEY_FEATURES = [
    "capacity_ah",
    "target_soh",
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


def load_sequences() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [CALCE_INPUT, NASA_INPUT, MIT_INPUT, METADATA_INPUT]:
        require_file(path)
    calce = pd.read_parquet(CALCE_INPUT)
    nasa = pd.read_parquet(NASA_INPUT)
    mit = pd.read_parquet(MIT_INPUT)
    metadata = pd.read_csv(METADATA_INPUT)
    LOGGER.info("Loaded CALCE sequences: %s", calce.shape)
    LOGGER.info("Loaded NASA sequences: %s", nasa.shape)
    LOGGER.info("Loaded MIT sequences: %s", mit.shape)
    LOGGER.info("Loaded sequence metadata: %s", metadata.shape)
    return calce, nasa, mit, metadata


def as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def generic_clean(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    for flag in ["capacity_invalid_flag", "capacity_label_excluded_flag", "capacity_rebound_flag"]:
        if flag not in work.columns:
            work[flag] = False
        work[flag] = as_bool(work[flag])

    checks = {
        "target_soh_missing": work["target_soh"].isna(),
        "soh_missing": work["soh"].isna(),
        "capacity_missing": work["capacity_ah"].isna(),
        "capacity_invalid_flag": work["capacity_invalid_flag"],
        "capacity_label_excluded_flag": work["capacity_label_excluded_flag"],
        "target_soh_out_of_range": (work["target_soh"] <= 0) | (work["target_soh"] > 1.2),
    }
    rows = []
    for reason, mask in checks.items():
        rows.append({"dataset": dataset_name, "filter_reason": reason, "row_count": int(mask.fillna(False).sum())})
    filter_counts = pd.DataFrame(rows)

    keep = np.ones(len(work), dtype=bool)
    for mask in checks.values():
        keep &= ~mask.fillna(False).to_numpy()
    clean = work.loc[keep].copy()
    clean["split_original"] = clean.get("split", pd.Series(index=clean.index, dtype=object))
    return clean, filter_counts


def assign_clean_index_and_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["time_idx_clean"] = pd.Series(dtype=int)
        df["split"] = pd.Series(dtype=object)
        return df
    out = df.sort_values(["battery_id", "time_idx", "global_cycle_index"]).copy()
    out["time_idx_clean"] = out.groupby("battery_id").cumcount().astype(int)
    split_values = []
    for _, group in out.groupby("battery_id", sort=False):
        length = len(group)
        cutoff = max(1, int(np.floor(length * 0.70)))
        split_values.extend(np.where(np.arange(length) < cutoff, "train", "test"))
    out["split"] = split_values
    return out.reset_index(drop=True)


def clean_calce(calce: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base, filter_counts = generic_clean(calce, "CALCE")
    retained_groups = []
    reset_rows = []
    for battery_id, group in base.sort_values(["battery_id", "time_idx", "global_cycle_index"]).groupby("battery_id"):
        group = group.copy().reset_index(drop=True)
        group["target_soh_delta"] = group["target_soh"].diff()
        group["soh_reset_flag"] = group["target_soh_delta"] > RESET_THRESHOLD
        reset_positions = group.index[group["soh_reset_flag"]].tolist()
        if reset_positions:
            reset_pos = int(reset_positions[0])
            before = group.loc[reset_pos - 1] if reset_pos > 0 else None
            reset = group.loc[reset_pos]
            retained = group.loc[: reset_pos - 1].copy()
            reset_rows.append(
                {
                    "battery_id": battery_id,
                    "reset_detected": True,
                    "reset_threshold": RESET_THRESHOLD,
                    "first_reset_position_after_generic_clean": reset_pos,
                    "first_reset_original_time_idx": reset.get("time_idx", np.nan),
                    "first_reset_global_cycle_index": reset.get("global_cycle_index", np.nan),
                    "target_soh_before_reset": before.get("target_soh", np.nan) if before is not None else np.nan,
                    "target_soh_at_reset": reset.get("target_soh", np.nan),
                    "reset_delta": reset.get("target_soh_delta", np.nan),
                    "rows_after_generic_clean": len(group),
                    "rows_retained_before_reset": len(retained),
                    "rows_removed_by_reset_truncation": len(group) - len(retained),
                }
            )
        else:
            retained = group.copy()
            reset_rows.append(
                {
                    "battery_id": battery_id,
                    "reset_detected": False,
                    "reset_threshold": RESET_THRESHOLD,
                    "first_reset_position_after_generic_clean": np.nan,
                    "first_reset_original_time_idx": np.nan,
                    "first_reset_global_cycle_index": np.nan,
                    "target_soh_before_reset": np.nan,
                    "target_soh_at_reset": np.nan,
                    "reset_delta": np.nan,
                    "rows_after_generic_clean": len(group),
                    "rows_retained_before_reset": len(retained),
                    "rows_removed_by_reset_truncation": 0,
                }
            )
        retained["truncated_at_first_soh_reset"] = bool(reset_positions)
        retained["first_reset_global_cycle_index"] = reset_rows[-1]["first_reset_global_cycle_index"]
        retained_groups.append(retained)

    clean = pd.concat(retained_groups, ignore_index=True) if retained_groups else base.iloc[0:0].copy()
    clean["soh_reset_flag"] = clean.get("soh_reset_flag", False).fillna(False).astype(bool)
    clean = assign_clean_index_and_split(clean)
    return clean, pd.DataFrame(reset_rows), filter_counts


def clean_nasa(nasa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean, filter_counts = generic_clean(nasa, "NASA")
    clean = clean.sort_values(["battery_id", "time_idx", "global_cycle_index"]).copy()
    clean["target_soh_delta"] = clean.groupby("battery_id")["target_soh"].diff()
    clean["soh_rebound_flag"] = clean["target_soh_delta"] > 0
    clean["high_soh_flag"] = clean["target_soh"] > NASA_HIGH_SOH_THRESHOLD
    clean = assign_clean_index_and_split(clean)
    return clean, filter_counts


def clean_mit(mit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean, filter_counts = generic_clean(mit, "MIT_Severson")
    clean["pretraining_candidate"] = True
    clean["soh_rebound_flag"] = clean.sort_values(["battery_id", "time_idx"]).groupby("battery_id")["target_soh"].diff() > 0
    clean = assign_clean_index_and_split(clean)
    return clean, filter_counts


def build_clean_metadata(
    raw_tables: dict[str, pd.DataFrame],
    clean_tables: dict[str, pd.DataFrame],
    original_metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    original_lookup = {
        (row["dataset"], row["battery_id"]): row.to_dict() for _, row in original_metadata.iterrows()
    }
    for dataset, clean in clean_tables.items():
        raw = raw_tables[dataset]
        all_batteries = sorted(set(raw["battery_id"].dropna().astype(str)) | set(clean["battery_id"].dropna().astype(str)))
        for battery_id in all_batteries:
            raw_group = raw[raw["battery_id"].astype(str) == battery_id]
            group = clean[clean["battery_id"].astype(str) == battery_id].sort_values("time_idx_clean")
            original = original_lookup.get((dataset, battery_id), {})
            time_diff = group["time_idx_clean"].diff().dropna() if "time_idx_clean" in group.columns else pd.Series(dtype=float)
            split_test = group[group["split"] == "test"] if "split" in group.columns else pd.DataFrame()
            rows.append(
                {
                    "dataset": dataset,
                    "battery_id": battery_id,
                    "original_rows": len(raw_group),
                    "clean_rows": len(group),
                    "rows_removed": len(raw_group) - len(group),
                    "original_sequence_length": original.get("sequence_length", len(raw_group)),
                    "clean_sequence_length": len(group),
                    "time_idx_clean_continuous": bool(
                        len(group) == 0
                        or (
                            group["time_idx_clean"].min() == 0
                            and group["time_idx_clean"].max() == len(group) - 1
                            and (time_diff.eq(1).all() if len(time_diff) else True)
                        )
                    ),
                    "global_cycle_min": group["global_cycle_index"].min(skipna=True) if len(group) else np.nan,
                    "global_cycle_max": group["global_cycle_index"].max(skipna=True) if len(group) else np.nan,
                    "target_soh_min": group["target_soh"].min(skipna=True) if len(group) else np.nan,
                    "target_soh_max": group["target_soh"].max(skipna=True) if len(group) else np.nan,
                    "capacity_min_ah": group["capacity_ah"].min(skipna=True) if len(group) else np.nan,
                    "capacity_max_ah": group["capacity_ah"].max(skipna=True) if len(group) else np.nan,
                    "train_steps": int((group["split"] == "train").sum()) if len(group) else 0,
                    "test_steps": int((group["split"] == "test").sum()) if len(group) else 0,
                    "split_first_test_time_idx_clean": int(split_test["time_idx_clean"].min()) if len(split_test) else np.nan,
                    "split_first_test_global_cycle": split_test["global_cycle_index"].min() if len(split_test) else np.nan,
                    "soh_anomaly_count": int(((group["target_soh"] <= 0) | (group["target_soh"] > 1.2)).fillna(False).sum())
                    if len(group)
                    else 0,
                    "capacity_nonpositive_count": int((group["capacity_ah"] <= 0).fillna(False).sum()) if len(group) else 0,
                    "soh_reset_flag_count": int(group.get("soh_reset_flag", pd.Series(False, index=group.index)).fillna(False).sum())
                    if len(group)
                    else 0,
                    "soh_rebound_flag_count": int(group.get("soh_rebound_flag", pd.Series(False, index=group.index)).fillna(False).sum())
                    if len(group)
                    else 0,
                    "high_soh_flag_count": int(group.get("high_soh_flag", pd.Series(False, index=group.index)).fillna(False).sum())
                    if len(group)
                    else 0,
                    "ccct_missing_rate": group["ccct_s"].isna().mean() if len(group) and "ccct_s" in group.columns else np.nan,
                    "cvct_missing_rate": group["cvct_s"].isna().mean() if len(group) and "cvct_s" in group.columns else np.nan,
                    "adv_missing_rate": group["adv_v"].isna().mean() if len(group) and "adv_v" in group.columns else np.nan,
                    "icv_missing_rate": group["icv_v"].isna().mean() if len(group) and "icv_v" in group.columns else np.nan,
                    "eis_re_missing_rate": group["eis_re_ohm"].isna().mean() if len(group) and "eis_re_ohm" in group.columns else np.nan,
                    "eis_rct_missing_rate": group["eis_rct_ohm"].isna().mean() if len(group) and "eis_rct_ohm" in group.columns else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "battery_id"]).reset_index(drop=True)


def summarize_before_after(raw_tables: dict[str, pd.DataFrame], clean_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset in raw_tables:
        raw = raw_tables[dataset]
        clean = clean_tables[dataset]
        rows.append(
            {
                "dataset": dataset,
                "raw_batteries": raw["battery_id"].nunique(),
                "clean_batteries": clean["battery_id"].nunique(),
                "raw_rows": len(raw),
                "clean_rows": len(clean),
                "rows_removed": len(raw) - len(clean),
                "row_retention_rate": len(clean) / len(raw) if len(raw) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def missing_rate_table(clean_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, clean in clean_tables.items():
        row: dict[str, Any] = {"dataset": dataset}
        for field in KEY_FEATURES:
            row[field] = clean[field].isna().mean() if field in clean.columns and len(clean) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_calce_reset_detection(raw_calce: pd.DataFrame, reset_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for battery_id, group in raw_calce.sort_values(["battery_id", "time_idx"]).groupby("battery_id"):
        valid = group[group["target_soh"].notna()].copy()
        ax.plot(valid["time_idx"], valid["target_soh"], linewidth=1.3, label=battery_id)
        row = reset_table[reset_table["battery_id"] == battery_id]
        if not row.empty and bool(row.iloc[0]["reset_detected"]):
            ax.axvline(row.iloc[0]["first_reset_original_time_idx"], color="black", linestyle="--", alpha=0.25)
    ax.set_title("CALCE SOH reset detection")
    ax.set_xlabel("original time_idx")
    ax.set_ylabel("target_soh")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_soh_sequences(clean: pd.DataFrame, output_path: Path, title: str, sample_batteries: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    data = clean.copy()
    if sample_batteries is not None:
        data = data[data["battery_id"].isin(sample_batteries)]
    for battery_id, group in data.sort_values(["battery_id", "time_idx_clean"]).groupby("battery_id"):
        label = battery_id if sample_batteries is not None or len(data["battery_id"].unique()) <= 12 else None
        if title.startswith("NASA") and battery_id in NASA_CORE_BATTERIES:
            ax.plot(group["time_idx_clean"], group["target_soh"], linewidth=1.8, label=battery_id)
        elif title.startswith("NASA"):
            ax.plot(group["time_idx_clean"], group["target_soh"], color="#999999", alpha=0.35, linewidth=0.9)
        else:
            ax.plot(group["time_idx_clean"], group["target_soh"], linewidth=1.3, label=label)
    ax.set_title(title)
    ax.set_xlabel("time_idx_clean")
    ax.set_ylabel("target_soh")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_length_distribution(metadata_clean: pd.DataFrame, output_path: Path) -> None:
    labels = []
    values = []
    for dataset, group in metadata_clean.groupby("dataset", sort=True):
        labels.append(dataset)
        values.append(group["clean_sequence_length"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_title("Clean sequence length distribution")
    ax.set_ylabel("clean sequence length")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_missing_rates(missing_rates: pd.DataFrame, output_path: Path) -> None:
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
    for idx, dataset in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[dataset], width=width, label=dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("missing rate (%)")
    ax.set_title("Clean sequence feature missing rates")
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
    before_after: pd.DataFrame,
    reset_table: pd.DataFrame,
    metadata_clean: pd.DataFrame,
    missing_rates: pd.DataFrame,
    filter_counts: pd.DataFrame,
) -> str:
    missing_view = missing_rates.copy()
    for col in missing_view.columns:
        if col != "dataset":
            missing_view[col] = missing_view[col] * 100.0
    length_stats = (
        metadata_clean.groupby("dataset", as_index=False)
        .agg(
            clean_batteries=("battery_id", "nunique"),
            clean_length_min=("clean_sequence_length", "min"),
            clean_length_max=("clean_sequence_length", "max"),
            clean_length_mean=("clean_sequence_length", "mean"),
            discontinuous_count=("time_idx_clean_continuous", lambda s: int((~s.astype(bool)).sum())),
            soh_anomaly_count=("soh_anomaly_count", "sum"),
            capacity_nonpositive_count=("capacity_nonpositive_count", "sum"),
        )
        .sort_values("dataset")
    )
    calce_clean_lengths = metadata_clean[metadata_clean["dataset"] == "CALCE"][
        ["battery_id", "original_rows", "clean_rows", "rows_removed", "target_soh_min", "target_soh_max", "train_steps", "test_steps"]
    ]
    nasa_flags = metadata_clean[metadata_clean["dataset"] == "NASA"][
        ["battery_id", "clean_rows", "soh_rebound_flag_count", "high_soh_flag_count", "eis_re_missing_rate", "eis_rct_missing_rate"]
    ]

    return f"""# Clean sequence 数据集报告

本报告由 `src/preprocessing/build_clean_sequence_dataset.py` 自动生成。本阶段只生成 clean 版本，不覆盖原始序列 parquet，不修改 `data_raw/`，不训练模型。

## 输出文件

- `data_processed/sequences/calce_soh_sequences_clean.parquet`
- `data_processed/sequences/nasa_soh_sequences_clean.parquet`
- `data_processed/sequences/mit_soh_sequences_clean.parquet`
- `data_processed/sequences/sequence_metadata_clean.csv`
- `data_processed/sequences/calce_soh_reset_detection.csv`
- `figures/sequences_clean/calce_clean_soh_sequences.png`
- `figures/sequences_clean/nasa_clean_soh_sequences.png`
- `figures/sequences_clean/mit_clean_sample_soh_sequences.png`
- `figures/sequences_clean/clean_sequence_length_distribution.png`
- `figures/sequences_clean/clean_feature_missing_rate.png`
- `figures/sequences_clean/calce_soh_reset_detection.png`

## 清洗前后规模变化

{markdown_table(before_after)}

通用过滤原因统计：

{markdown_table(filter_counts)}

clean 后序列长度与质量概况：

{markdown_table(length_stats)}

## CALCE SOH reset 检测

CALCE reset 规则：在通用过滤后，若相邻有效 `target_soh` 增加超过 `{RESET_THRESHOLD}`，标记为 `soh_reset_flag`。第一版 clean 数据只保留第一次 reset 之前的连续退化段。

{markdown_table(reset_table)}

CALCE clean 后每条序列情况：

{markdown_table(calce_clean_lengths)}

严格按第一次 reset 截断后，CALCE 不再包含后半段跳回 1.1 附近的片段；但四条序列都被截到较短的早期退化段，因此更适合做 TAM-DeepAR-EN 小样本复现的流程验证和特征可用性验证，不适合直接训练复杂深度模型。

## NASA clean 结果

NASA 保留 NASA core 与 NASA all-clean 并集；本阶段允许轻微 SOH 回升，并新增 `soh_rebound_flag`。当 `target_soh > {NASA_HIGH_SOH_THRESHOLD}` 时新增 `high_soh_flag`，但不直接删除。EIS 的 `eis_re_ohm/eis_rct_ohm` 保留已有的前向填充结果，不做反向填充。

{markdown_table(nasa_flags, max_rows=80)}

NASA clean 后仍然是第一版 DeepAR 的优先数据集，因为序列数量多于 CALCE，IHF 和 EIS 特征可用性较好。

## MIT clean 结果

MIT clean 后仍作为 `pretraining_candidate`。本阶段过滤了无效容量和 SOH 缺失点，保留 `capacity_ah`、`soh/target_soh`、温度、`internal_resistance_ohm`、`charge_time_s` 等 summary 特征；不补造 CCCT、CVCT、ADV、ICV。

MIT 仍暂不建议作为第一版 DeepAR 主训练集，因为其当前序列来自 summary-only 数据，电压/电流曲线与 IHF 特征缺失较多。

## clean 特征缺失率

下表单位为百分比。

{markdown_table(missing_view, max_rows=20)}

## DeepAR 训练特征建议

第一版 DeepAR 建议使用 NASA clean：

- 输入候选：`capacity_ah`、`target_soh` 的历史值、`ccct_s`、`cvct_s`、`adv_v`、`icv_v`、`charge_time_s`、`discharge_time_s`、温度、`eis_re_ohm`、`eis_rct_ohm`
- 目标：`target_soh`
- 禁止输入：`rul`
- RUL：只作为后续评价或联合预测标签保留

CALCE clean 建议用于 TAM-DeepAR-EN 小样本复现和 IHF 消融验证；MIT clean 建议作为后续大规模预训练候选。

## 当前风险

- CALCE 的第一次 reset 发生较早，严格截断后可用长度明显变短，模型训练价值有限。
- NASA 仍存在 SOH 回升和少量高 SOH 点，本阶段只标记不删除。
- EIS 特征仍是前向填充结果，非逐循环同步测量。
- MIT summary-only 缺少 CCCT/CVCT/ADV/ICV 和详细电压电流曲线。
- 不同数据集的 RUL/EOL 标签来源不完全一致，训练 SOH 模型和评价 RUL 时应分开处理。
"""


def build_clean_sequences() -> dict[str, pd.DataFrame]:
    SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    calce_raw, nasa_raw, mit_raw, metadata = load_sequences()
    raw_tables = {"CALCE": calce_raw, "NASA": nasa_raw, "MIT_Severson": mit_raw}

    calce_clean, reset_table, calce_filter = clean_calce(calce_raw)
    nasa_clean, nasa_filter = clean_nasa(nasa_raw)
    mit_clean, mit_filter = clean_mit(mit_raw)
    clean_tables = {"CALCE": calce_clean, "NASA": nasa_clean, "MIT_Severson": mit_clean}
    filter_counts = pd.concat([calce_filter, nasa_filter, mit_filter], ignore_index=True)

    calce_clean.to_parquet(CALCE_CLEAN_OUTPUT, index=False)
    nasa_clean.to_parquet(NASA_CLEAN_OUTPUT, index=False)
    mit_clean.to_parquet(MIT_CLEAN_OUTPUT, index=False)
    reset_table.to_csv(CALCE_RESET_OUTPUT, index=False, encoding="utf-8-sig")

    metadata_clean = build_clean_metadata(raw_tables, clean_tables, metadata)
    metadata_clean.to_csv(METADATA_CLEAN_OUTPUT, index=False, encoding="utf-8-sig")

    before_after = summarize_before_after(raw_tables, clean_tables)
    missing_rates = missing_rate_table(clean_tables)

    plot_calce_reset_detection(calce_raw, reset_table, CALCE_RESET_FIGURE)
    plot_soh_sequences(calce_clean, CALCE_CLEAN_FIGURE, "CALCE clean SOH sequences")
    plot_soh_sequences(nasa_clean, NASA_CLEAN_FIGURE, "NASA clean SOH sequences")
    mit_batteries = sorted(mit_clean["battery_id"].dropna().astype(str).unique())
    sample_count = min(10, len(mit_batteries))
    mit_sample = list(pd.Series(mit_batteries).sample(n=sample_count, random_state=42)) if sample_count else []
    plot_soh_sequences(mit_clean, MIT_CLEAN_SAMPLE_FIGURE, "MIT clean sample SOH sequences", sample_batteries=mit_sample)
    plot_length_distribution(metadata_clean, LENGTH_FIGURE)
    plot_missing_rates(missing_rates, MISSING_FIGURE)

    REPORT_PATH.write_text(
        generate_report(before_after, reset_table, metadata_clean, missing_rates, filter_counts),
        encoding="utf-8",
    )

    LOGGER.info("Saved CALCE clean sequences: %s", CALCE_CLEAN_OUTPUT)
    LOGGER.info("Saved NASA clean sequences: %s", NASA_CLEAN_OUTPUT)
    LOGGER.info("Saved MIT clean sequences: %s", MIT_CLEAN_OUTPUT)
    LOGGER.info("Saved clean metadata: %s", METADATA_CLEAN_OUTPUT)
    LOGGER.info("Saved CALCE reset detection table: %s", CALCE_RESET_OUTPUT)
    LOGGER.info("Saved clean report: %s", REPORT_PATH)
    print(
        "Clean sequence datasets built: "
        f"CALCE={calce_clean['battery_id'].nunique()} batteries/{len(calce_clean)} rows, "
        f"NASA={nasa_clean['battery_id'].nunique()} batteries/{len(nasa_clean)} rows, "
        f"MIT={mit_clean['battery_id'].nunique()} batteries/{len(mit_clean)} rows"
    )
    return clean_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean SOH sequence datasets without overwriting originals.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    parse_args()
    build_clean_sequences()


if __name__ == "__main__":
    main()
