from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


LOGGER = logging.getLogger("visualize_cycle_summary")

SUMMARY_PATH = PROJECT_ROOT / "data_processed" / "unified_tables" / "unified_cycle_summary.parquet"
EIS_PATH = PROJECT_ROOT / "data_processed" / "unified_tables" / "nasa_eis_features.csv"
FIGURE_DIR = PROJECT_ROOT / "figures" / "data_quality"
REPORT_PATH = PROJECT_ROOT / "docs" / "cycle_summary_visual_qc_report.md"

MIT_BATCH_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "无"
    table_df = df.copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)
    headers = [str(col) for col in table_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table_df.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("NaN")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_line_plot(
    df: pd.DataFrame,
    dataset: str,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    legend: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    plot_df = df.dropna(subset=["global_cycle_index", y_col]).copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    if plot_df.empty:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        for battery_id, group in plot_df.groupby("battery_id", sort=False):
            group = group.sort_values("global_cycle_index")
            ax.plot(
                group["global_cycle_index"],
                group[y_col],
                linewidth=1.0,
                alpha=0.55 if group["battery_id"].nunique() <= 20 else 0.28,
                label=str(battery_id),
            )
    ax.set_title(title)
    ax.set_xlabel("global_cycle_index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend and not plot_df.empty:
        ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)


def plot_dataset_curves(summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    paths = []
    for dataset, dataset_df in summary.groupby("dataset", sort=False):
        cap_path = figure_dir / f"{safe_filename(dataset)}_capacity_curves.png"
        soh_path = figure_dir / f"{safe_filename(dataset)}_soh_curves.png"
        show_legend = dataset in {"CALCE", "NASA"}
        save_line_plot(
            dataset_df,
            dataset,
            "capacity_ah",
            "capacity_ah",
            f"{dataset}: capacity_ah vs global_cycle_index",
            cap_path,
            legend=show_legend,
        )
        save_line_plot(
            dataset_df,
            dataset,
            "soh",
            "SOH",
            f"{dataset}: SOH vs global_cycle_index",
            soh_path,
            legend=show_legend,
            ylim=(0.6, 1.25),
        )
        paths.extend([cap_path, soh_path])
    return paths


def plot_selected_batteries(summary: pd.DataFrame, figure_dir: Path) -> list[Path]:
    paths = []
    calce_ids = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
    nasa_ids = ["B0005", "B0006", "B0007", "B0018"]

    calce_df = summary[(summary["dataset"] == "CALCE") & (summary["battery_id"].isin(calce_ids))]
    calce_path = figure_dir / "CALCE_CS2_capacity_curves.png"
    save_line_plot(
        calce_df,
        "CALCE",
        "capacity_ah",
        "capacity_ah",
        "CALCE CS2 cells: capacity_ah vs global_cycle_index",
        calce_path,
        legend=True,
    )
    paths.append(calce_path)

    nasa_df = summary[(summary["dataset"] == "NASA") & (summary["battery_id"].isin(nasa_ids))]
    nasa_path = figure_dir / "NASA_core_capacity_curves.png"
    save_line_plot(
        nasa_df,
        "NASA",
        "capacity_ah",
        "capacity_ah",
        "NASA core cells: capacity_ah vs global_cycle_index",
        nasa_path,
        legend=True,
    )
    paths.append(nasa_path)
    return paths


def plot_mit_cycle_life_hist(summary: pd.DataFrame, figure_dir: Path) -> Path:
    mit = summary[summary["dataset"] == "MIT_Severson"]
    life = mit.groupby("battery_id")["cycle_life"].first().dropna()
    output_path = figure_dir / "MIT_Severson_cycle_life_histogram.png"
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    if life.empty:
        ax.text(0.5, 0.5, "No non-null MIT cycle_life", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(life, bins=30, color="#4477AA", edgecolor="white")
    ax.set_title("MIT-Severson cycle_life distribution")
    ax.set_xlabel("cycle_life")
    ax.set_ylabel("battery count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)
    return output_path


def plot_cycle_life_missing(summary: pd.DataFrame, figure_dir: Path) -> tuple[Path, pd.DataFrame]:
    per_battery = (
        summary.groupby(["dataset", "battery_id"], as_index=False)["cycle_life"]
        .first()
        .assign(cycle_life_missing=lambda d: d["cycle_life"].isna())
    )
    stats = (
        per_battery.groupby("dataset")["cycle_life_missing"]
        .agg(missing="sum", total="count")
        .reset_index()
    )
    stats["available"] = stats["total"] - stats["missing"]
    stats["missing_rate"] = stats["missing"] / stats["total"]

    output_path = figure_dir / "cycle_life_missing_by_dataset.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    x = np.arange(len(stats))
    ax.bar(x, stats["available"], label="cycle_life available", color="#66AA55")
    ax.bar(x, stats["missing"], bottom=stats["available"], label="cycle_life missing", color="#CC6677")
    ax.set_xticks(x)
    ax.set_xticklabels(stats["dataset"], rotation=15, ha="right")
    ax.set_ylabel("battery count")
    ax.set_title("cycle_life availability by dataset")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)
    return output_path, stats


def plot_flag_counts(summary: pd.DataFrame, figure_dir: Path) -> tuple[Path, pd.DataFrame]:
    flag_cols = ["capacity_invalid_flag", "capacity_label_excluded_flag", "capacity_rebound_flag"]
    flag_counts = summary.groupby("dataset")[flag_cols].sum().reset_index()

    output_path = figure_dir / "capacity_quality_flags_by_dataset.png"
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    x = np.arange(len(flag_counts))
    width = 0.24
    for offset, col in zip([-width, 0, width], flag_cols):
        ax.bar(x + offset, flag_counts[col], width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(flag_counts["dataset"], rotation=15, ha="right")
    ax.set_ylabel("cycle count")
    ax.set_title("Capacity quality flags by dataset")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved figure: %s", output_path)
    return output_path, flag_counts


def parse_mit_cell_index(battery_id: str) -> int | None:
    match = re.search(r"_cell_(\d+)$", str(battery_id))
    return int(match.group(1)) if match else None


def mit_batch_name(source_file: str) -> str:
    for batch_name, filename in MIT_BATCH_FILES.items():
        if source_file == filename:
            return batch_name
    return "unknown"


def scalar_or_nan(value: Any) -> float:
    try:
        arr = np.asarray(value).squeeze()
        if arr.size == 0:
            return np.nan
        return float(np.real(arr.reshape(-1)[0]))
    except Exception:
        return np.nan


def load_mit_raw_cycle_life(raw_dir: Path) -> pd.DataFrame:
    rows = []
    if h5py is None:
        LOGGER.warning("h5py unavailable; cannot compare MIT raw cycle_life labels.")
        return pd.DataFrame(columns=["source_file", "cell_index", "battery_id", "mit_raw_cycle_life"])

    for mat_path in sorted(raw_dir.glob("*.mat")):
        try:
            with h5py.File(mat_path, "r") as h5:
                if "batch" not in h5:
                    continue
                batch = h5["batch"]
                count = mit_battery_count(batch)
                for idx in range(count):
                    raw_life = scalar_or_nan(mit_read_field(h5, batch, "cycle_life", idx, default=np.nan))
                    battery_id = f"{mat_path.stem}_cell_{idx + 1:03d}"
                    rows.append(
                        {
                            "source_file": mat_path.name,
                            "batch": mit_batch_name(mat_path.name),
                            "cell_index": idx + 1,
                            "battery_id": battery_id,
                            "mit_raw_cycle_life": raw_life,
                        }
                    )
        except Exception as exc:
            LOGGER.exception("Failed reading MIT raw cycle_life from %s: %s", mat_path, exc)
    return pd.DataFrame(rows)


def official_mit_exclusion_candidates(mit: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(source_file: str, cell_index: int, reason: str) -> None:
        rows.append(
            {
                "source_file": source_file,
                "batch": mit_batch_name(source_file),
                "cell_index": cell_index,
                "battery_id": f"{Path(source_file).stem}_cell_{cell_index:03d}",
                "reason": reason,
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
    filtered = [idx for idx in raw_indices if idx != 38 and idx not in set(high_end["cell_index"].dropna().astype(int))]
    for position in [3, 40, 41]:
        if len(filtered) >= position:
            add(batch3, filtered[position - 1], f"official removal: noisy Batch 8 battery position {position} after prior removals")

    if not rows:
        return pd.DataFrame(columns=["source_file", "batch", "cell_index", "battery_id", "reason"])
    return pd.DataFrame(rows).drop_duplicates(["battery_id", "reason"]).sort_values(["source_file", "cell_index", "reason"])


def mit_qc(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mit = summary[summary["dataset"] == "MIT_Severson"].copy()
    mit["batch"] = mit["source_file"].map(mit_batch_name)
    mit["cell_index"] = mit["battery_id"].map(parse_mit_cell_index)

    battery_list = (
        mit[["batch", "source_file", "cell_index", "battery_id"]]
        .drop_duplicates()
        .sort_values(["source_file", "cell_index", "battery_id"])
        .reset_index(drop=True)
    )
    batch_counts = battery_list.groupby(["batch", "source_file"], as_index=False)["battery_id"].nunique()
    batch_counts = batch_counts.rename(columns={"battery_id": "battery_count"})

    duplicate_ids = (
        battery_list.groupby("battery_id", as_index=False)
        .size()
        .query("size > 1")
        .rename(columns={"size": "duplicate_count"})
    )

    exclusions = official_mit_exclusion_candidates(mit)

    raw_life = load_mit_raw_cycle_life(PROJECT_ROOT / "data_raw" / "MIT_Severson")
    recomputed = (
        mit.groupby(["source_file", "battery_id"], as_index=False)["cycle_life"]
        .first()
        .rename(columns={"cycle_life": "recomputed_soh80_cycle_life"})
    )
    comparison = raw_life.merge(recomputed, on=["source_file", "battery_id"], how="left")
    comparison["difference_recomputed_minus_raw"] = (
        comparison["recomputed_soh80_cycle_life"] - comparison["mit_raw_cycle_life"]
    )
    comparison["both_available"] = comparison["mit_raw_cycle_life"].notna() & comparison["recomputed_soh80_cycle_life"].notna()
    comparison["exact_match"] = comparison["difference_recomputed_minus_raw"].abs() < 1e-9

    return {
        "battery_list": battery_list,
        "batch_counts": batch_counts,
        "duplicate_ids": duplicate_ids,
        "official_exclusion_candidates": exclusions,
        "cycle_life_comparison": comparison,
    }


def build_report(
    summary: pd.DataFrame,
    eis: pd.DataFrame,
    figure_paths: list[Path],
    cycle_missing: pd.DataFrame,
    flag_counts: pd.DataFrame,
    mit_tables: dict[str, pd.DataFrame],
    report_path: Path,
) -> None:
    dataset_stats = (
        summary.groupby("dataset")
        .agg(
            battery_count=("battery_id", "nunique"),
            cycle_count=("battery_id", "size"),
            valid_capacity_rows=("capacity_ah", lambda s: s.notna().sum()),
            valid_soh_rows=("soh", lambda s: s.notna().sum()),
            soh_min=("soh", "min"),
            soh_max=("soh", "max"),
        )
        .reset_index()
    )

    mit_comp = mit_tables["cycle_life_comparison"]
    comp_available = mit_comp[mit_comp["both_available"]]
    diff_stats = pd.DataFrame(
        [
            {
                "both_available_count": len(comp_available),
                "exact_match_count": int(comp_available["exact_match"].sum()) if not comp_available.empty else 0,
                "mean_difference": comp_available["difference_recomputed_minus_raw"].mean() if not comp_available.empty else np.nan,
                "max_abs_difference": comp_available["difference_recomputed_minus_raw"].abs().max() if not comp_available.empty else np.nan,
            }
        ]
    )

    official_candidates = mit_tables["official_exclusion_candidates"]
    official_overlap = official_candidates[
        official_candidates["battery_id"].isin(set(mit_tables["battery_list"]["battery_id"]))
    ]

    not_eol = (
        summary.groupby(["dataset", "battery_id"], as_index=False)["cycle_life"]
        .first()
        .query("cycle_life.isna()")
    )

    figures_md = "\n".join(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in figure_paths)
    mit_battery_ids = mit_tables["battery_list"]["battery_id"].tolist()

    suitability = [
        "MIT-Severson：适合优先进入 baseline。其循环级容量、温度和内阻字段较完整，但当前表保留了 140 个原始 battery，未完全套用官方 124 个电池的合并/剔除流程；建模前建议根据任务选择是否使用官方过滤清单。",
        "CALCE：适合做容量曲线与电压/电流工程特征验证。由于多个 Excel 的 Cycle_Index 会重启，后续建模必须使用 global_cycle_index，并建议过滤 capacity_label_excluded_flag=True 的循环。",
        "NASA：适合做 SOH/RUL 与 EIS 辅助特征研究。普通循环和 impedance 已分表，后续可按 nearest_discharge_global_cycle 合并 Re/Rct。",
        "整体结论：可以进入 baseline 建模准备阶段，但第一版 baseline 应优先使用 MIT-Severson 和 NASA 中标签较完整、未标记容量异常的样本；CALCE 更适合作为跨文件对齐和特征工程验证集。",
    ]

    content = f"""# 循环级数据质量可视化检查报告

本报告由 `src/preprocessing/visualize_cycle_summary.py` 自动生成。脚本只读取：

- `data_processed/unified_tables/unified_cycle_summary.parquet`
- `data_processed/unified_tables/nasa_eis_features.csv`

不修改原始数据，不删除异常值，也不进行模型训练。

## 输出图表

{figures_md}

## 数据集总体情况

{markdown_table(dataset_stats)}

## cycle_life 缺失统计

{markdown_table(cycle_missing)}

## 容量质量标记统计

{markdown_table(flag_counts)}

## NASA EIS 表检查

- EIS 特征行数：{len(eis)}
- `re_ohm` 缺失率：{eis['re_ohm'].isna().mean() if 're_ohm' in eis else np.nan:.3f}
- `rct_ohm` 缺失率：{eis['rct_ohm'].isna().mean() if 'rct_ohm' in eis else np.nan:.3f}

## MIT-Severson 140 个 battery 检查

### battery_id 列表

`{', '.join(mit_battery_ids)}`

### 每个 batch 的 battery 数

{markdown_table(mit_tables['batch_counts'])}

### 重复 battery_id 检查

{markdown_table(mit_tables['duplicate_ids'])}

### 官方应剔除/合并的异常电池候选

说明：该表根据官方 `LoadData.m` 的公开逻辑推断，包括 Batch 1 未完成电池、Batch 2 continuation、Batch 3 problem channel、Batch 3 final capacity 过高以及 noisy Batch 8 位置。当前 summary 保留的是原始 140 个通道/电池记录，因此这些候选只做标记，不自动删除。

候选数量：{len(official_overlap)}

{markdown_table(official_overlap, max_rows=80)}

## MIT cycle_life 来源检查

当前 `unified_cycle_summary` 中的 MIT `cycle_life` 是由本项目脚本按 `SOH <= 0.8` 重新计算得到；MIT 原始 `.mat` 文件中也存在 `cycle_life` 字段。两者对比如下：

{markdown_table(diff_stats)}

差异样例：

{markdown_table(mit_comp.sort_values('difference_recomputed_minus_raw', key=lambda s: s.abs(), ascending=False).head(20))}

## 未达到 EOL 的电池

未达到 EOL 的电池数量：{len(not_eol)}

{markdown_table(not_eol, max_rows=120)}

## 是否适合进入 baseline 建模

""" + "\n".join(f"- {item}" for item in suitability) + "\n"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    LOGGER.info("Saved visual QC report: %s", report_path)


def visualize_cycle_summary(
    summary_path: Path,
    eis_path: Path,
    figure_dir: Path,
    report_path: Path,
) -> None:
    LOGGER.info("Reading cycle summary: %s", summary_path)
    summary = pd.read_parquet(summary_path)
    LOGGER.info("Reading NASA EIS features: %s", eis_path)
    eis = pd.read_csv(eis_path)

    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    figure_paths.extend(plot_dataset_curves(summary, figure_dir))
    figure_paths.extend(plot_selected_batteries(summary, figure_dir))
    figure_paths.append(plot_mit_cycle_life_hist(summary, figure_dir))
    missing_path, cycle_missing = plot_cycle_life_missing(summary, figure_dir)
    flag_path, flag_counts = plot_flag_counts(summary, figure_dir)
    figure_paths.extend([missing_path, flag_path])

    mit_tables = mit_qc(summary)
    build_report(summary, eis, figure_paths, cycle_missing, flag_counts, mit_tables, report_path)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Visual QC for unified cycle-level battery summary.")
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--eis", type=Path, default=EIS_PATH)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    visualize_cycle_summary(args.summary, args.eis, args.figure_dir, args.report)


if __name__ == "__main__":
    main()
