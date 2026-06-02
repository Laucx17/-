from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loaders.load_mit_severson import (  # noqa: E402
    _battery_count as mit_battery_count,
    _read_field as mit_read_field,
    h5py,
)


LOGGER = logging.getLogger("build_modeling_ready_dataset")

SUMMARY_PATH = PROJECT_ROOT / "data_processed" / "unified_tables" / "unified_cycle_summary.parquet"
NASA_EIS_PATH = PROJECT_ROOT / "data_processed" / "unified_tables" / "nasa_eis_features.csv"
QC_REPORT_PATH = PROJECT_ROOT / "docs" / "cycle_summary_visual_qc_report.md"
OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "source_domain"
OUTPUT_PARQUET = OUTPUT_DIR / "modeling_ready_cycle_life_dataset.parquet"
OUTPUT_CSV = OUTPUT_DIR / "modeling_ready_cycle_life_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "modeling_ready_dataset_report.md"

MIT_RAW_DIR = PROJECT_ROOT / "data_raw" / "MIT_Severson"
MIT_BATCH_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
NASA_CORE_BATTERIES = {"B0005", "B0006", "B0007", "B0018"}
NASA_CLEAN_BAD_RATE_THRESHOLD = 0.10

REQUESTED_OUTPUT_COLUMNS = [
    "dataset",
    "battery_id",
    "source_batch",
    "chemistry",
    "global_cycle_index",
    "capacity_ah",
    "soh",
    "target_cycle_life",
    "cycle_life_label_official",
    "cycle_life_label_recomputed",
    "rul",
    "has_eol",
    "is_mit_curated_124",
    "is_nasa_core",
    "is_nasa_all_clean",
    "is_calce_feature_validation",
    "capacity_invalid_flag",
    "capacity_label_excluded_flag",
    "capacity_rebound_flag",
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
    "internal_resistance_ohm",
]

EXTRA_OUTPUT_COLUMNS = [
    "cycle_index_raw",
    "step_source",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "initial_capacity_ah",
    "source_file",
    "target_label_source",
    "baseline_role",
    "is_mit_raw_140",
    "MIT_raw_140",
    "MIT_curated_124",
    "official_exclusion_candidate",
    "official_exclusion_reason",
    "nasa_bad_flag_rate",
    "nasa_valid_soh_count",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def parse_mit_cell_index(battery_id: str) -> int | None:
    match = re.search(r"_cell_(\d+)$", str(battery_id))
    return int(match.group(1)) if match else None


def mit_batch_name(source_file: str) -> str:
    for batch_name, filename in MIT_BATCH_FILES.items():
        if str(source_file) == filename:
            return batch_name
    return "unknown"


def source_batch_from_source_file(dataset: str, battery_id: str, source_file: str) -> str:
    if dataset == "MIT_Severson":
        return mit_batch_name(source_file)
    if dataset == "CALCE":
        return str(battery_id)
    if dataset == "NASA":
        parts = str(source_file).split("::")
        if len(parts) >= 2:
            inner_zip = parts[-2].replace("\\", "/").split("/")[-1]
            return Path(inner_zip).stem
        return "NASA"
    return "unknown"


def scalar_or_nan(value: Any) -> float:
    try:
        arr = np.asarray(value).squeeze()
        if arr.size == 0:
            return np.nan
        return float(np.real(arr.reshape(-1)[0]))
    except Exception:
        return np.nan


def load_mit_official_cycle_life(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if h5py is None:
        LOGGER.warning("h5py is unavailable; MIT official cycle_life labels cannot be read.")
        return pd.DataFrame(
            columns=["source_file", "source_batch", "cell_index", "battery_id", "cycle_life_label_official"]
        )

    for mat_path in sorted(raw_dir.glob("*.mat")):
        try:
            with h5py.File(mat_path, "r") as h5:
                if "batch" not in h5:
                    LOGGER.warning("MIT file has no batch group: %s", mat_path)
                    continue
                batch = h5["batch"]
                count = mit_battery_count(batch)
                for idx in range(count):
                    official_life = scalar_or_nan(mit_read_field(h5, batch, "cycle_life", idx, default=np.nan))
                    rows.append(
                        {
                            "source_file": mat_path.name,
                            "source_batch": mit_batch_name(mat_path.name),
                            "cell_index": idx + 1,
                            "battery_id": f"{mat_path.stem}_cell_{idx + 1:03d}",
                            "cycle_life_label_official": official_life,
                        }
                    )
        except Exception as exc:
            LOGGER.exception("Failed reading MIT official labels from %s: %s", mat_path, exc)

    return pd.DataFrame(rows)


def official_mit_exclusion_candidates(mit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(source_file: str, cell_index: int, reason: str) -> None:
        rows.append(
            {
                "source_file": source_file,
                "source_batch": mit_batch_name(source_file),
                "cell_index": cell_index,
                "battery_id": f"{Path(source_file).stem}_cell_{cell_index:03d}",
                "official_exclusion_reason": reason,
            }
        )

    batch1 = MIT_BATCH_FILES["batch1"]
    batch2 = MIT_BATCH_FILES["batch2"]
    batch3 = MIT_BATCH_FILES["batch3"]

    for idx in [9, 11, 13, 14, 23]:
        add(batch1, idx, "official optional removal: unfinished Batch 1 battery")
    for idx in [8, 9, 10, 16, 17]:
        add(batch2, idx, "official merge/removal: continuation appended to Batch 1 cells 1-5")
    add(batch3, 38, "official removal: problematic channel 46 in Batch 3")

    batch3_df = mit[mit["source_file"] == batch3].copy()
    batch3_df["cell_index"] = batch3_df["battery_id"].map(parse_mit_cell_index)
    final_capacity = (
        batch3_df.sort_values("global_cycle_index")
        .groupby(["cell_index", "battery_id"], as_index=False)["capacity_ah"]
        .last()
        .dropna(subset=["cell_index"])
    )
    high_end = final_capacity[final_capacity["capacity_ah"] > 0.885]
    for _, row in high_end.iterrows():
        add(batch3, int(row["cell_index"]), "official removal: Batch 3 final QDischarge > 0.885 Ah")

    raw_indices = sorted(final_capacity["cell_index"].dropna().astype(int).unique())
    high_end_indices = set(high_end["cell_index"].dropna().astype(int))
    filtered = [idx for idx in raw_indices if idx != 38 and idx not in high_end_indices]
    for position in [3, 40, 41]:
        if len(filtered) >= position:
            add(
                batch3,
                filtered[position - 1],
                f"official removal: noisy Batch 8 battery position {position} after prior removals",
            )

    if not rows:
        return pd.DataFrame(
            columns=["source_file", "source_batch", "cell_index", "battery_id", "official_exclusion_reason"]
        )

    return (
        pd.DataFrame(rows)
        .drop_duplicates(["battery_id", "official_exclusion_reason"])
        .sort_values(["source_file", "cell_index", "official_exclusion_reason"])
        .reset_index(drop=True)
    )


def nasa_clean_battery_table(summary: pd.DataFrame) -> pd.DataFrame:
    nasa = summary[summary["dataset"] == "NASA"].copy()
    if nasa.empty:
        return pd.DataFrame(
            columns=[
                "battery_id",
                "nasa_cycle_count",
                "nasa_invalid_count",
                "nasa_label_excluded_count",
                "nasa_bad_flag_rate",
                "nasa_valid_soh_count",
                "is_nasa_all_clean",
            ]
        )

    for col in ["capacity_invalid_flag", "capacity_label_excluded_flag"]:
        nasa[col] = nasa[col].fillna(False).astype(bool)

    stats = (
        nasa.groupby("battery_id")
        .agg(
            nasa_cycle_count=("global_cycle_index", "count"),
            nasa_invalid_count=("capacity_invalid_flag", "sum"),
            nasa_label_excluded_count=("capacity_label_excluded_flag", "sum"),
            nasa_valid_soh_count=("soh", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    stats["nasa_bad_flag_rate"] = (
        stats["nasa_invalid_count"] + stats["nasa_label_excluded_count"]
    ) / stats["nasa_cycle_count"].replace(0, np.nan)
    stats["is_nasa_all_clean"] = (
        (stats["nasa_bad_flag_rate"] <= NASA_CLEAN_BAD_RATE_THRESHOLD)
        & (stats["nasa_valid_soh_count"] >= 3)
    )
    return stats


def add_labels_and_flags(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df = summary.copy()

    for col in ["capacity_invalid_flag", "capacity_label_excluded_flag", "capacity_rebound_flag"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)

    df["source_batch"] = [
        source_batch_from_source_file(dataset, battery_id, source_file)
        for dataset, battery_id, source_file in zip(df["dataset"], df["battery_id"], df["source_file"])
    ]
    df["cycle_life_label_recomputed"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    df["cycle_life_label_official"] = np.nan

    mit = df[df["dataset"] == "MIT_Severson"].copy()
    official_labels = load_mit_official_cycle_life(MIT_RAW_DIR)
    if not official_labels.empty:
        df = df.merge(
            official_labels[["source_file", "battery_id", "cycle_life_label_official"]],
            on=["source_file", "battery_id"],
            how="left",
            suffixes=("", "_from_raw"),
        )
        df["cycle_life_label_official"] = df["cycle_life_label_official_from_raw"].combine_first(
            df["cycle_life_label_official"]
        )
        df = df.drop(columns=["cycle_life_label_official_from_raw"])

    exclusions = official_mit_exclusion_candidates(mit)
    exclusion_reason = exclusions.groupby("battery_id")["official_exclusion_reason"].apply("; ".join).reset_index()
    df = df.merge(exclusion_reason, on="battery_id", how="left")
    df["official_exclusion_candidate"] = df["official_exclusion_reason"].notna() & (df["dataset"] == "MIT_Severson")

    df["is_mit_raw_140"] = df["dataset"] == "MIT_Severson"
    df["is_mit_curated_124"] = df["is_mit_raw_140"] & ~df["official_exclusion_candidate"]
    df["MIT_raw_140"] = df["is_mit_raw_140"]
    df["MIT_curated_124"] = df["is_mit_curated_124"]

    df["is_nasa_core"] = (df["dataset"] == "NASA") & df["battery_id"].isin(NASA_CORE_BATTERIES)
    nasa_clean = nasa_clean_battery_table(df)
    df = df.merge(
        nasa_clean[["battery_id", "nasa_bad_flag_rate", "nasa_valid_soh_count", "is_nasa_all_clean"]],
        on="battery_id",
        how="left",
    )
    df["is_nasa_all_clean"] = df["is_nasa_all_clean"].fillna(False).astype(bool) & (df["dataset"] == "NASA")
    df["is_calce_feature_validation"] = df["dataset"] == "CALCE"

    df["target_cycle_life"] = np.nan
    df["target_label_source"] = "not_for_cycle_life_baseline"

    mit_mask = df["dataset"] == "MIT_Severson"
    nasa_mask = df["dataset"] == "NASA"
    official_available = mit_mask & df["cycle_life_label_official"].notna()
    official_missing = mit_mask & df["cycle_life_label_official"].isna()

    df.loc[official_available, "target_cycle_life"] = df.loc[official_available, "cycle_life_label_official"]
    df.loc[official_available, "target_label_source"] = "MIT_official_mat_cycle_life"

    recomputed_fallback = official_missing & df["cycle_life_label_recomputed"].notna()
    df.loc[recomputed_fallback, "target_cycle_life"] = df.loc[
        recomputed_fallback, "cycle_life_label_recomputed"
    ]
    df.loc[recomputed_fallback, "target_label_source"] = "MIT_recomputed_soh80_fallback"

    nasa_recomputed = nasa_mask & df["cycle_life_label_recomputed"].notna()
    df.loc[nasa_recomputed, "target_cycle_life"] = df.loc[nasa_recomputed, "cycle_life_label_recomputed"]
    df.loc[nasa_recomputed, "target_label_source"] = "NASA_recomputed_soh80"

    df["has_eol"] = df["cycle_life_label_official"].notna() | df["cycle_life_label_recomputed"].notna()
    df["rul"] = np.where(
        df["target_cycle_life"].notna(),
        df["target_cycle_life"] - pd.to_numeric(df["global_cycle_index"], errors="coerce"),
        np.nan,
    )

    df["baseline_role"] = "excluded_or_reference"
    df.loc[df["is_mit_curated_124"], "baseline_role"] = "main_train_mit_curated_124"
    df.loc[df["is_nasa_core"], "baseline_role"] = "supplement_validation_nasa_core"
    df.loc[df["is_calce_feature_validation"], "baseline_role"] = "feature_validation_calce"

    tables = {
        "official_labels": official_labels,
        "mit_exclusions": exclusions,
        "nasa_clean": nasa_clean,
    }
    return df, tables


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = REQUESTED_OUTPUT_COLUMNS + EXTRA_OUTPUT_COLUMNS
    for col in ordered:
        if col not in df.columns:
            df[col] = np.nan
    remaining = [col for col in df.columns if col not in ordered and col != "cycle_life"]
    return df[ordered + remaining]


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


def dataset_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("dataset")
        .agg(
            battery_count=("battery_id", "nunique"),
            cycle_count=("global_cycle_index", "count"),
            target_battery_count=("target_cycle_life", lambda s: int(s.notna().groupby(df.loc[s.index, "battery_id"]).any().sum())),
            target_cycle_rows=("target_cycle_life", lambda s: int(s.notna().sum())),
            has_eol_batteries=("has_eol", lambda s: int(s.groupby(df.loc[s.index, "battery_id"]).any().sum())),
        )
        .reset_index()
    )


def flag_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    flags = [
        "is_mit_raw_140",
        "is_mit_curated_124",
        "is_nasa_core",
        "is_nasa_all_clean",
        "is_calce_feature_validation",
        "capacity_invalid_flag",
        "capacity_label_excluded_flag",
        "capacity_rebound_flag",
    ]
    rows = []
    for dataset, group in df.groupby("dataset"):
        row: dict[str, Any] = {"dataset": dataset}
        for flag in flags:
            row[flag] = int(group[flag].fillna(False).astype(bool).sum()) if flag in group.columns else 0
        rows.append(row)
    return pd.DataFrame(rows)


def battery_level_label_table(df: pd.DataFrame) -> pd.DataFrame:
    per_battery = (
        df.groupby(["dataset", "battery_id"], as_index=False)
        .agg(
            source_batch=("source_batch", "first"),
            target_cycle_life=("target_cycle_life", "first"),
            official_label=("cycle_life_label_official", "first"),
            recomputed_label=("cycle_life_label_recomputed", "first"),
            target_label_source=("target_label_source", "first"),
            is_mit_curated_124=("is_mit_curated_124", "first"),
            is_nasa_core=("is_nasa_core", "first"),
            is_nasa_all_clean=("is_nasa_all_clean", "first"),
            is_calce_feature_validation=("is_calce_feature_validation", "first"),
            official_exclusion_candidate=("official_exclusion_candidate", "first"),
            nasa_bad_flag_rate=("nasa_bad_flag_rate", "first"),
        )
        .sort_values(["dataset", "source_batch", "battery_id"])
    )
    return per_battery


def generate_report(df: pd.DataFrame, tables: dict[str, pd.DataFrame], eis: pd.DataFrame) -> str:
    dataset_stats = dataset_summary_table(df)
    flag_stats = flag_summary_table(df)
    per_battery = battery_level_label_table(df)

    mit_batteries = per_battery[per_battery["dataset"] == "MIT_Severson"].copy()
    mit_label_sources = (
        mit_batteries.groupby("target_label_source", as_index=False)["battery_id"]
        .nunique()
        .rename(columns={"battery_id": "battery_count"})
        .sort_values("target_label_source")
    )
    mit_curated_count = int(mit_batteries["is_mit_curated_124"].fillna(False).sum())
    mit_raw_count = len(mit_batteries)

    nasa_batteries = per_battery[per_battery["dataset"] == "NASA"].copy()
    nasa_clean_list = nasa_batteries[nasa_batteries["is_nasa_all_clean"].fillna(False)]
    nasa_core_list = nasa_batteries[nasa_batteries["is_nasa_core"].fillna(False)]

    calce_batteries = per_battery[per_battery["dataset"] == "CALCE"].copy()

    official_label_count = int(tables["official_labels"]["cycle_life_label_official"].notna().sum())
    official_candidate_count = int(tables["mit_exclusions"]["battery_id"].nunique())
    qc_report_status = "存在" if QC_REPORT_PATH.exists() else "未找到"

    target_recommendation = df[
        (
            df["is_mit_curated_124"]
            | df["is_nasa_core"]
            | df["is_calce_feature_validation"]
        )
    ]
    recommendation_stats = (
        target_recommendation.groupby("baseline_role", as_index=False)
        .agg(
            battery_count=("battery_id", "nunique"),
            cycle_count=("global_cycle_index", "count"),
            target_available_rows=("target_cycle_life", lambda s: int(s.notna().sum())),
        )
        .sort_values("baseline_role")
    )

    eis_summary = pd.DataFrame(
        [
            {
                "eis_rows": len(eis),
                "eis_battery_count": eis["battery_id"].nunique() if "battery_id" in eis.columns else 0,
                "re_ohm_missing_rate": eis["re_ohm"].isna().mean() if "re_ohm" in eis.columns and len(eis) else np.nan,
                "rct_ohm_missing_rate": eis["rct_ohm"].isna().mean()
                if "rct_ohm" in eis.columns and len(eis)
                else np.nan,
            }
        ]
    )

    return f"""# Modeling-ready cycle life 数据集报告

本报告由 `src/preprocessing/build_modeling_ready_dataset.py` 自动生成。脚本读取：

- `data_processed/unified_tables/unified_cycle_summary.parquet`
- `data_processed/unified_tables/nasa_eis_features.csv`
- `docs/cycle_summary_visual_qc_report.md`：状态为 `{qc_report_status}`，其中的数据质量结论已转化为本脚本中的标记逻辑。

脚本不修改 `data_raw/`，不删除 `unified_cycle_summary`，也不进行任何模型训练。

## 输出文件

- `data_processed/source_domain/modeling_ready_cycle_life_dataset.parquet`
- `data_processed/source_domain/modeling_ready_cycle_life_dataset.csv`
- `docs/modeling_ready_dataset_report.md`

## 每个数据集保留情况

{markdown_table(dataset_stats)}

## 主要标记字段统计

下表统计的是循环行数量，不是电池数量。

{markdown_table(flag_stats)}

## MIT-Severson：从 raw 140 到 curated 124

当前 `unified_cycle_summary` 中 MIT-Severson 保留了 `{mit_raw_count}` 个原始通道/电池记录，因此 `MIT_raw_140=True` 与 `is_mit_raw_140=True` 用来标记这些原始 MIT 行。

第一版 baseline 默认使用 `MIT_curated_124=True` / `is_mit_curated_124=True` 的样本，共 `{mit_curated_count}` 个电池。该标记参考官方 `LoadData.m` 公开处理逻辑，对 `{official_candidate_count}` 个候选电池只做标记，不从本表物理删除。候选类型包括：

- Batch 1 未完成电池；
- Batch 2 continuation，应与 Batch 1 前 5 个电池合并或剔除；
- Batch 3 问题通道；
- Batch 3 末端容量过高样本；
- 官方 noisy Batch 8 位置对应样本。

官方剔除/合并候选如下：

{markdown_table(tables["mit_exclusions"], max_rows=80)}

## MIT target_cycle_life 标签来源

MIT 的 `cycle_life_label_official` 直接从原始 `.mat` 文件中的 `cycle_life` 字段读取，不伪造标签。本次读取到非空官方标签 `{official_label_count}` 个。

`cycle_life_label_recomputed` 保留 `unified_cycle_summary` 中按 `SOH <= 0.8` 重新计算得到的标签。

`target_cycle_life` 的优先级为：

1. MIT 优先使用 `cycle_life_label_official`；
2. 如果 MIT 官方标签缺失，再回退到 `cycle_life_label_recomputed`；
3. NASA 使用 `cycle_life_label_recomputed`；
4. CALCE 暂不作为第一版 cycle_life baseline 主训练集，因此 `target_cycle_life` 默认保持为空。

MIT 电池级标签来源统计：

{markdown_table(mit_label_sources)}

## NASA_core 与 NASA_all_clean

`is_nasa_core=True` 标记 NASA 经典核心电池：`B0005`、`B0006`、`B0007`、`B0018`，主要用于补充验证和与文献结果对齐。

`is_nasa_all_clean=True` 是数据质量导向标记：按电池统计 `capacity_invalid_flag` 与 `capacity_label_excluded_flag`，当坏标记比例不超过 `{NASA_CLEAN_BAD_RATE_THRESHOLD:.0%}` 且有效 SOH 行数不少于 3 时标记为 clean。它不等同于 NASA_core，部分核心电池因为早期容量标签排除比例较高，不一定属于 all_clean。

NASA_core 电池：

{markdown_table(nasa_core_list[["battery_id", "source_batch", "target_cycle_life", "target_label_source", "nasa_bad_flag_rate"]])}

NASA_all_clean 电池：

{markdown_table(nasa_clean_list[["battery_id", "source_batch", "target_cycle_life", "target_label_source", "nasa_bad_flag_rate"]], max_rows=80)}

NASA EIS 表未直接混入 cycle-level 主表。本阶段只检查其可用性，后续可按 `nearest_discharge_global_cycle` 合并 Re/Rct 等特征。

{markdown_table(eis_summary)}

## CALCE 的定位

CALCE 四个电池保留在 modeling-ready 表中，并标记为 `is_calce_feature_validation=True`：

{markdown_table(calce_batteries[["battery_id", "source_batch", "recomputed_label", "is_calce_feature_validation"]])}

CALCE 暂不作为第一版 cycle_life baseline 主训练集，原因是：

- 只有 4 个 CS2 电池，电池数量太少；
- 多个 Excel 文件的 `Cycle_Index` 会重新从 1 开始，虽然已生成 `global_cycle_index`，但仍更适合先做特征工程验证；
- 本阶段更适合用它验证 CCCT、CVCT、ADV、ICV 等过程特征，以及 SOH 曲线预测，而不是直接训练 cycle life baseline。

## 第一版 baseline 推荐样本

{markdown_table(recommendation_stats)}

推荐用法：

- 主训练：`is_mit_curated_124=True`，并优先过滤 `capacity_invalid_flag=False`、`capacity_label_excluded_flag=False` 的循环；
- 补充验证：`is_nasa_core=True`，NASA 的 `target_cycle_life` 使用 `SOH <= 0.8` 重新计算标签；
- 特征验证：`is_calce_feature_validation=True`，用于检查容量、电压、电流衍生特征，而不纳入第一版 cycle life 主训练集。

## 风险和注意事项

- MIT 当前没有执行官方 continuation 合并，只把相关电池标记为非 curated；如果要严格复现 Severson 官方 124 电池数据，应在后续单独实现合并策略。
- MIT 官方 `cycle_life` 与本项目 `SOH <= 0.8` 重算标签定义不同，建模时不能混用为同一个评价口径。
- NASA 的 `target_cycle_life` 是重算标签，不是原始文件中的官方 cycle life 字段；报告和论文中需要明确说明。
- CALCE 保留了较多 `capacity_label_excluded_flag=True` 循环，后续做 SOH 或特征实验时应先过滤或单独分析这些点。
- 本表是 modeling-ready 数据工程产物，不代表最终训练集；真正训练前仍应按任务目标、早期循环窗口和数据划分策略进一步生成实验样本。
"""


def build_modeling_ready_dataset() -> pd.DataFrame:
    require_file(SUMMARY_PATH)
    require_file(NASA_EIS_PATH)
    if not QC_REPORT_PATH.exists():
        LOGGER.warning("QC report not found: %s", QC_REPORT_PATH)

    summary = pd.read_parquet(SUMMARY_PATH)
    eis = pd.read_csv(NASA_EIS_PATH)
    LOGGER.info("Loaded unified summary with shape %s", summary.shape)
    LOGGER.info("Loaded NASA EIS table with shape %s", eis.shape)

    modeled, tables = add_labels_and_flags(summary)
    modeled = ensure_output_columns(modeled)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    modeled.to_parquet(OUTPUT_PARQUET, index=False)
    modeled.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(generate_report(modeled, tables, eis), encoding="utf-8")

    LOGGER.info("Saved modeling-ready parquet: %s", OUTPUT_PARQUET)
    LOGGER.info("Saved modeling-ready csv: %s", OUTPUT_CSV)
    LOGGER.info("Saved report: %s", REPORT_PATH)
    return modeled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build modeling-ready source-domain cycle life dataset.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    parse_args()
    build_modeling_ready_dataset()


if __name__ == "__main__":
    main()
