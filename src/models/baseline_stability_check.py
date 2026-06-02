from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "src" / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from baseline_cycle_life import (  # noqa: E402
    FIGURE_DIR,
    INPUT_PATH,
    RESULTS_DIR,
    build_feature_table,
    format_value,
    load_main_training_rows,
    markdown_table,
    metric_dict,
)


LOGGER = logging.getLogger("baseline_stability_check")

BEST_PARAMS_PATH = RESULTS_DIR / "baseline_cycle_life_best_params.csv"
STABILITY_RESULTS_PATH = RESULTS_DIR / "baseline_stability_results.csv"
STABILITY_SUMMARY_PATH = RESULTS_DIR / "baseline_stability_summary.csv"
STABILITY_PREDICTIONS_PATH = RESULTS_DIR / "baseline_stability_predictions.csv"
STABILITY_MAE_BOXPLOT_PATH = FIGURE_DIR / "stability_mae_boxplot.png"
STABILITY_R2_BOXPLOT_PATH = FIGURE_DIR / "stability_r2_boxplot.png"
REPORT_PATH = PROJECT_ROOT / "docs" / "baseline_stability_report.md"

RANDOM_STATES = [0, 1, 2, 3, 4, 5, 10, 21, 42, 100]
MODEL_CONFIGS = [
    {"model": "RandomForest", "N": 30},
    {"model": "XGBoost", "N": 10},
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def load_best_params(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    require_file(path)
    params = pd.read_csv(path)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in params.iterrows():
        model_name = str(row["model"])
        n_cycles = int(row["N"])
        try:
            parsed = json.loads(row.get("best_params_json", "{}"))
        except Exception:
            parsed = {}
        out[(model_name, n_cycles)] = parsed
    return out


def make_model(model_name: str, params: dict[str, Any], random_state: int) -> Pipeline:
    if model_name == "RandomForest":
        estimator = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError(
                "XGBoost is required for the XGBoost stability check. "
                "Install it with: pip install xgboost"
            ) from exc
        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )
    pipeline.set_params(**params)
    return pipeline


def split_batteries(samples: pd.DataFrame, random_state: int) -> tuple[set[str], set[str]]:
    train_ids, test_ids = train_test_split(
        sorted(samples["battery_id"].astype(str).unique()),
        train_size=0.80,
        random_state=random_state,
        shuffle=True,
    )
    return set(train_ids), set(test_ids)


def run_single_experiment(
    samples: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    n_cycles: int,
    params: dict[str, Any],
    random_state: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_batteries, test_batteries = split_batteries(samples, random_state)
    train = samples[samples["battery_id"].isin(train_batteries)].copy()
    test = samples[samples["battery_id"].isin(test_batteries)].copy()

    model = make_model(model_name, params, random_state)
    model.fit(train[feature_columns], train["target_cycle_life"])
    pred = model.predict(test[feature_columns])
    metrics = metric_dict(test["target_cycle_life"].to_numpy(), pred)

    result = {
        "model": model_name,
        "N": n_cycles,
        "random_state": random_state,
        "train_batteries": len(train_batteries),
        "test_batteries": len(test_batteries),
        "n_features": len(feature_columns),
        **metrics,
    }

    prediction_rows = []
    for battery_id, source_batch, y_true, y_pred in zip(
        test["battery_id"],
        test["source_batch"],
        test["target_cycle_life"],
        pred,
    ):
        error = float(y_pred - y_true)
        prediction_rows.append(
            {
                "model": model_name,
                "N": n_cycles,
                "random_state": random_state,
                "battery_id": battery_id,
                "source_batch": source_batch,
                "target_cycle_life": float(y_true),
                "predicted_cycle_life": float(y_pred),
                "error": error,
                "abs_error": abs(error),
                "percentage_error": abs(error) / float(y_true) * 100.0,
            }
        )
    return result, prediction_rows


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    rows = []
    for (model_name, n_cycles), group in results.groupby(["model", "N"], sort=True):
        row: dict[str, Any] = {
            "model": model_name,
            "N": n_cycles,
            "runs": len(group),
            "train_batteries": int(group["train_batteries"].median()),
            "test_batteries": int(group["test_batteries"].median()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        row["MAE_cv"] = row["MAE_std"] / row["MAE_mean"] if row["MAE_mean"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["MAE_mean", "RMSE_mean"]).reset_index(drop=True)


def plot_metric_boxplot(results: pd.DataFrame, metric: str, output_path: Path) -> None:
    labels = []
    values = []
    for (model_name, n_cycles), group in results.groupby(["model", "N"], sort=True):
        labels.append(f"{model_name}\nN={n_cycles}")
        values.append(group[metric].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title(f"Stability check: {metric} across random seeds")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def stability_text(summary: pd.DataFrame, model_name: str, n_cycles: int) -> str:
    row = summary[(summary["model"] == model_name) & (summary["N"] == n_cycles)]
    if row.empty:
        return f"`{model_name} + N={n_cycles}` 未成功运行。"
    item = row.iloc[0]
    mae_cv = float(item["MAE_cv"])
    if mae_cv <= 0.15:
        level = "较稳定"
    elif mae_cv <= 0.30:
        level = "中等稳定"
    else:
        level = "不够稳定"
    return (
        f"`{model_name} + N={n_cycles}` 的 MAE 均值为 `{item['MAE_mean']:.4f}`，"
        f"标准差为 `{item['MAE_std']:.4f}`，变异系数约 `{mae_cv:.3f}`，整体判断为：{level}。"
    )


def generate_report(
    filtered: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    best_params: dict[tuple[str, int], dict[str, Any]],
    xgboost_status: str,
) -> str:
    best = summary.sort_values(["MAE_mean", "RMSE_mean"]).iloc[0]
    rf_text = stability_text(summary, "RandomForest", 30)
    xgb_text = stability_text(summary, "XGBoost", 10)

    params_rows = []
    for config in MODEL_CONFIGS:
        model_name = config["model"]
        n_cycles = config["N"]
        params_rows.append(
            {
                "model": model_name,
                "N": n_cycles,
                "best_params_json": json.dumps(best_params.get((model_name, n_cycles), {}), ensure_ascii=False, sort_keys=True),
            }
        )
    params_table = pd.DataFrame(params_rows)

    return f"""# Baseline 稳定性检查报告

本报告由 `src/models/baseline_stability_check.py` 自动生成。本次不修改原始数据，不修改 modeling-ready 数据集。

## 实验设置

- 数据：仅使用 MIT-Severson curated 电池
- 筛选条件：`dataset == "MIT_Severson"`、`is_mit_curated_124 == True`、`target_cycle_life` 非空、容量异常标记为 False
- 每个 `battery_id` 在指定 N 下只形成一行样本
- 每次实验按 `battery_id` 做 80/20 train/test 划分
- 随机种子：`{RANDOM_STATES}`
- 候选模型：`RandomForest + N=30`，`XGBoost + N=10`
- XGBoost 状态：{xgboost_status}

过滤后进入稳定性检查的数据池：`{filtered["battery_id"].nunique()}` 个电池，`{len(filtered)}` 行循环记录。

使用的 tuned 参数：

{markdown_table(params_table)}

## 单次重复实验结果

{markdown_table(results, max_rows=80)}

## 均值和标准差

{markdown_table(summary)}

## 稳定性判断

- {rf_text}
- {xgb_text}

## 哪个模型更适合作为迁移学习源域 baseline

按多随机种子的平均 MAE 排序，本次更推荐 `{best["model"]} + N={int(best["N"])}` 作为后续迁移学习源域 baseline：

- MAE mean：`{best["MAE_mean"]:.4f}`
- MAE std：`{best["MAE_std"]:.4f}`
- RMSE mean：`{best["RMSE_mean"]:.4f}`
- MAPE mean：`{best["MAPE_mean"]:.4f}%`
- R2 mean：`{best["R2_mean"]:.4f}`

如果后续更看重 R2 或 RMSE，也可以保留另一个模型作为并行对照。由于样本只有 124 个电池，结论应以多随机种子均值和标准差为主，不宜只看某一次 train/test split。

## 当前 baseline 的局限性

- 仍然只使用 MIT summary-only 表，电压/电流详细曲线特征没有进入模型。
- 随机划分虽然避免了同一电池泄漏，但还没有做按 batch/protocol 的外推验证。
- tuned 参数来自轻量搜索，不是完整超参数优化。
- 目标标签使用 MIT 原始 `.mat` 中的官方 `cycle_life`，与 `SOH <= 0.8` 重算标签口径不同。
- 当前实验仍是传统机器学习的一行一电池寿命回归，尚未利用完整 SOH 时间序列建模。

## 是否可以进入 DeepAR / TAM-DeepAR-EN

可以进入下一阶段。建议保留本稳定性检查中表现更稳的模型作为传统 baseline，同时开始构建序列样本：

- 输入形式：`battery_id × cycle_index × IHFs`
- 初始特征：容量、SOH、温度、内阻、充电时间，后续补充 CCCT、CVCT、ADV、ICV
- 目标：SOH 序列预测、RUL 推断，并为固态/半固态小样本迁移做源域预训练

输出文件：

- `results/baseline_stability_results.csv`
- `results/baseline_stability_summary.csv`
- `results/baseline_stability_predictions.csv`
- `figures/baseline/stability_mae_boxplot.png`
- `figures/baseline/stability_r2_boxplot.png`
"""


def run_stability_check() -> dict[str, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    filtered = load_main_training_rows(INPUT_PATH)
    best_params = load_best_params(BEST_PARAMS_PATH)
    xgboost_status = "可用"
    try:
        import xgboost

        xgboost_status = f"可用，版本 {xgboost.__version__}"
    except Exception:
        xgboost_status = "不可用，请运行 pip install xgboost"

    samples_by_n: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    for n_cycles in sorted({config["N"] for config in MODEL_CONFIGS}):
        samples, feature_columns, _ = build_feature_table(filtered, n_cycles)
        samples_by_n[n_cycles] = (samples, feature_columns)
        LOGGER.info("Prepared N=%s feature table: %s samples, %s features", n_cycles, len(samples), len(feature_columns))

    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for config in MODEL_CONFIGS:
        model_name = config["model"]
        n_cycles = int(config["N"])
        params = best_params.get((model_name, n_cycles), {})
        samples, feature_columns = samples_by_n[n_cycles]
        for seed in RANDOM_STATES:
            LOGGER.info("Running stability experiment: model=%s, N=%s, random_state=%s", model_name, n_cycles, seed)
            result, predictions = run_single_experiment(
                samples=samples,
                feature_columns=feature_columns,
                model_name=model_name,
                n_cycles=n_cycles,
                params=params,
                random_state=seed,
            )
            result_rows.append(result)
            prediction_rows.extend(predictions)

    results = pd.DataFrame(result_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_results(results)

    results.to_csv(STABILITY_RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(STABILITY_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    predictions.to_csv(STABILITY_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    plot_metric_boxplot(results, "MAE", STABILITY_MAE_BOXPLOT_PATH)
    plot_metric_boxplot(results, "R2", STABILITY_R2_BOXPLOT_PATH)
    REPORT_PATH.write_text(
        generate_report(filtered, results, summary, best_params, xgboost_status),
        encoding="utf-8",
    )

    best = summary.sort_values(["MAE_mean", "RMSE_mean"]).iloc[0]
    print(
        "Best stable baseline: "
        f"model={best['model']}, N={int(best['N'])}, "
        f"MAE_mean={best['MAE_mean']:.4f}, MAE_std={best['MAE_std']:.4f}, "
        f"RMSE_mean={best['RMSE_mean']:.4f}, MAPE_mean={best['MAPE_mean']:.4f}%, "
        f"R2_mean={best['R2_mean']:.4f}"
    )
    LOGGER.info("Saved stability results: %s", STABILITY_RESULTS_PATH)
    LOGGER.info("Saved stability summary: %s", STABILITY_SUMMARY_PATH)
    LOGGER.info("Saved stability predictions: %s", STABILITY_PREDICTIONS_PATH)
    LOGGER.info("Saved stability figures: %s, %s", STABILITY_MAE_BOXPLOT_PATH, STABILITY_R2_BOXPLOT_PATH)
    LOGGER.info("Saved stability report: %s", REPORT_PATH)
    return {"results": results, "summary": summary, "predictions": predictions}


def main() -> None:
    configure_logging()
    run_stability_check()


if __name__ == "__main__":
    main()
