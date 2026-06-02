from __future__ import annotations

import argparse
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data_processed" / "source_domain" / "modeling_ready_cycle_life_dataset.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = PROJECT_ROOT / "figures" / "baseline"
REPORT_PATH = PROJECT_ROOT / "docs" / "baseline_cycle_life_report.md"

RESULTS_PATH = RESULTS_DIR / "baseline_cycle_life_results.csv"
CV_RESULTS_PATH = RESULTS_DIR / "baseline_cycle_life_cv_results.csv"
PREDICTIONS_PATH = RESULTS_DIR / "baseline_cycle_life_predictions.csv"
SCATTER_PATH = FIGURE_DIR / "cycle_life_prediction_scatter.png"
ERROR_BY_MODEL_PATH = FIGURE_DIR / "error_by_model.png"
ERROR_BY_N_PATH = FIGURE_DIR / "error_by_early_cycles.png"

TUNED_RESULTS_PATH = RESULTS_DIR / "baseline_cycle_life_results_tuned.csv"
TUNED_PREDICTIONS_PATH = RESULTS_DIR / "baseline_cycle_life_predictions_tuned.csv"
BEST_PARAMS_PATH = RESULTS_DIR / "baseline_cycle_life_best_params.csv"
TUNED_COMPARISON_PATH = FIGURE_DIR / "tuned_model_comparison.png"
TUNED_REPORT_PATH = PROJECT_ROOT / "docs" / "baseline_cycle_life_tuned_report.md"

RANDOM_STATE = 42
EARLY_CYCLE_WINDOWS = [10, 30, 50, 100]
MISSING_SKIP_THRESHOLD = 0.80
TUNING_CV_SPLITS = 3
RF_TUNING_ITERATIONS = 12
XGB_TUNING_ITERATIONS = 18

CANDIDATE_FEATURE_FIELDS = [
    "capacity_ah",
    "soh",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "internal_resistance_ohm",
    "charge_time_s",
    "voltage_mean_v",
    "voltage_min_v",
    "voltage_max_v",
    "current_mean_a",
    "current_abs_mean_a",
]

STATS = ["mean", "std", "min", "max", "median", "first_value", "last_value", "delta", "slope"]

FORBIDDEN_FEATURE_COLUMNS = {
    "target_cycle_life",
    "cycle_life_label_official",
    "cycle_life_label_recomputed",
    "rul",
    "has_eol",
    "battery_id",
    "dataset",
    "source_batch",
    "global_cycle_index",
    "is_mit_curated_124",
    "is_nasa_core",
    "is_nasa_all_clean",
    "is_calce_feature_validation",
}


LOGGER = logging.getLogger("baseline_cycle_life")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")


def load_main_training_rows(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_parquet(path)
    mask = (
        (df["dataset"] == "MIT_Severson")
        & (df["is_mit_curated_124"] == True)  # noqa: E712
        & df["target_cycle_life"].notna()
        & (df["capacity_invalid_flag"] != True)  # noqa: E712
        & (df["capacity_label_excluded_flag"] != True)  # noqa: E712
    )
    filtered = df.loc[mask].copy()
    filtered["global_cycle_index"] = pd.to_numeric(filtered["global_cycle_index"], errors="coerce")
    filtered["target_cycle_life"] = pd.to_numeric(filtered["target_cycle_life"], errors="coerce")
    filtered = filtered.dropna(subset=["battery_id", "global_cycle_index", "target_cycle_life"])
    LOGGER.info(
        "Filtered MIT curated training rows: %s rows, %s batteries",
        len(filtered),
        filtered["battery_id"].nunique(),
    )
    return filtered


def linear_slope(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 2 or np.unique(x_values[mask]).size < 2:
        return np.nan
    return float(np.polyfit(x_values[mask], y_values[mask], 1)[0])


def first_valid(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return float(valid.iloc[0]) if len(valid) else np.nan


def last_valid(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return float(valid.iloc[-1]) if len(valid) else np.nan


def summarize_field(window: pd.DataFrame, field: str) -> dict[str, float]:
    values = pd.to_numeric(window[field], errors="coerce")
    first = first_valid(values)
    last = last_valid(values)
    return {
        "mean": float(values.mean(skipna=True)) if values.notna().any() else np.nan,
        "std": float(values.std(skipna=True)) if values.notna().sum() >= 2 else np.nan,
        "min": float(values.min(skipna=True)) if values.notna().any() else np.nan,
        "max": float(values.max(skipna=True)) if values.notna().any() else np.nan,
        "median": float(values.median(skipna=True)) if values.notna().any() else np.nan,
        "first_value": first,
        "last_value": last,
        "delta": last - first if np.isfinite(first) and np.isfinite(last) else np.nan,
        "slope": linear_slope(window["global_cycle_index"], values),
    }


def fields_for_window(filtered: pd.DataFrame, n_cycles: int) -> tuple[list[str], pd.DataFrame]:
    early_rows = (
        filtered.sort_values(["battery_id", "global_cycle_index"])
        .groupby("battery_id", group_keys=False)
        .head(n_cycles)
    )
    rows = []
    selected = []
    for field in CANDIDATE_FEATURE_FIELDS:
        if field not in early_rows.columns:
            rows.append(
                {
                    "N": n_cycles,
                    "field": field,
                    "missing_rate": 1.0,
                    "selected": False,
                    "reason": "field_not_found",
                }
            )
            continue
        missing_rate = pd.to_numeric(early_rows[field], errors="coerce").isna().mean()
        selected_flag = missing_rate <= MISSING_SKIP_THRESHOLD
        rows.append(
            {
                "N": n_cycles,
                "field": field,
                "missing_rate": float(missing_rate),
                "selected": selected_flag,
                "reason": "selected" if selected_flag else f"missing_rate>{MISSING_SKIP_THRESHOLD:.0%}",
            }
        )
        if selected_flag:
            selected.append(field)
    return selected, pd.DataFrame(rows)


def build_feature_table(filtered: pd.DataFrame, n_cycles: int) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    selected_fields, field_qc = fields_for_window(filtered, n_cycles)
    if not selected_fields:
        raise ValueError(f"No usable fields remain after missing-rate filtering for N={n_cycles}.")

    rows: list[dict[str, Any]] = []
    for battery_id, group in filtered.sort_values(["battery_id", "global_cycle_index"]).groupby("battery_id"):
        window = group.head(n_cycles).copy()
        if len(window) < n_cycles:
            continue

        sample: dict[str, Any] = {
            "battery_id": battery_id,
            "dataset": "MIT_Severson",
            "source_batch": window["source_batch"].iloc[0],
            "target_cycle_life": float(window["target_cycle_life"].iloc[0]),
            "n_cycles_used": int(len(window)),
        }
        for field in selected_fields:
            stats = summarize_field(window, field)
            for stat_name, value in stats.items():
                sample[f"{field}__{stat_name}"] = value
        rows.append(sample)

    feature_table = pd.DataFrame(rows)
    feature_columns = [col for col in feature_table.columns if "__" in col]
    forbidden_overlap = sorted(set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS)
    if forbidden_overlap:
        raise ValueError(f"Forbidden columns leaked into features: {forbidden_overlap}")
    return feature_table, feature_columns, field_qc


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    return {"MAE": float(mae), "RMSE": rmse, "MAPE": mape, "R2": float(r2)}


def make_models() -> tuple[dict[str, Pipeline], str | None]:
    models: dict[str, Pipeline] = {
        "ElasticNet": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=50000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    xgboost_status = None
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=350,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except Exception as exc:  # pragma: no cover - depends on local environment
        xgboost_status = (
            "XGBoost is not available in the current Python environment; "
            f"ElasticNet and RandomForest will continue. Import error: {exc}"
        )
        LOGGER.warning(xgboost_status)
    return models, xgboost_status


def get_xgboost_version() -> str | None:
    try:
        import xgboost

        return str(xgboost.__version__)
    except Exception:
        return None


def tuning_param_distributions(model_name: str) -> dict[str, list[Any]]:
    if model_name == "RandomForest":
        return {
            "model__n_estimators": [200, 500, 800],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5, 1.0],
        }
    if model_name == "XGBoost":
        return {
            "model__n_estimators": [200, 500, 800],
            "model__max_depth": [2, 3, 4, 5],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__reg_alpha": [0, 0.1, 1.0],
            "model__reg_lambda": [1.0, 5.0, 10.0],
        }
    return {}


def train_test_battery_split(battery_ids: np.ndarray) -> tuple[set[str], set[str]]:
    train_ids, test_ids = train_test_split(
        sorted(map(str, battery_ids)),
        train_size=0.80,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return set(train_ids), set(test_ids)


def evaluate_train_test(
    model_template: Pipeline,
    model_name: str,
    samples: pd.DataFrame,
    feature_columns: list[str],
    train_batteries: set[str],
    test_batteries: set[str],
    n_cycles: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Pipeline]:
    train_mask = samples["battery_id"].isin(train_batteries)
    test_mask = samples["battery_id"].isin(test_batteries)
    train = samples.loc[train_mask].copy()
    test = samples.loc[test_mask].copy()

    model = clone(model_template)
    model.fit(train[feature_columns], train["target_cycle_life"])

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for split_name, split_df in [("train", train), ("test", test)]:
        pred = model.predict(split_df[feature_columns])
        metrics = metric_dict(split_df["target_cycle_life"].to_numpy(), pred)
        metric_rows.append(
            {
                "N": n_cycles,
                "model": model_name,
                "split": split_name,
                "n_samples": len(split_df),
                "n_features": len(feature_columns),
                **metrics,
            }
        )
        for battery_id, source_batch, y_true, y_pred in zip(
            split_df["battery_id"],
            split_df["source_batch"],
            split_df["target_cycle_life"],
            pred,
        ):
            error = float(y_pred - y_true)
            prediction_rows.append(
                {
                    "battery_id": battery_id,
                    "source_batch": source_batch,
                    "N": n_cycles,
                    "model": model_name,
                    "split": split_name,
                    "target_cycle_life": float(y_true),
                    "predicted_cycle_life": float(y_pred),
                    "error": error,
                    "abs_error": abs(error),
                    "percentage_error": abs(error) / float(y_true) * 100.0,
                }
            )
    return metric_rows, prediction_rows, model


def evaluate_group_cv(
    model_template: Pipeline,
    model_name: str,
    samples: pd.DataFrame,
    feature_columns: list[str],
    n_cycles: int,
) -> list[dict[str, Any]]:
    groups = samples["battery_id"].astype(str).to_numpy()
    unique_group_count = len(np.unique(groups))
    n_splits = min(5, unique_group_count)
    if n_splits < 2:
        LOGGER.warning("N=%s has fewer than 2 batteries; skipping GroupKFold.", n_cycles)
        return []
    if n_splits < 5:
        LOGGER.info("N=%s only has %s batteries; GroupKFold reduced to %s folds.", n_cycles, unique_group_count, n_splits)

    cv = GroupKFold(n_splits=n_splits)
    rows: list[dict[str, Any]] = []
    X = samples[feature_columns]
    y = samples["target_cycle_life"].to_numpy(dtype=float)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        model = clone(model_template)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[val_idx])
        metrics = metric_dict(y[val_idx], pred)
        rows.append(
            {
                "N": n_cycles,
                "model": model_name,
                "fold": fold_idx,
                "n_splits": n_splits,
                "n_train": int(len(train_idx)),
                "n_validation": int(len(val_idx)),
                "n_features": int(len(feature_columns)),
                **metrics,
            }
        )
    return rows


def evaluate_fitted_model(
    fitted_model: Pipeline,
    model_name: str,
    samples: pd.DataFrame,
    feature_columns: list[str],
    train_batteries: set[str],
    test_batteries: set[str],
    n_cycles: int,
    run_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for split_name, split_df in [
        ("train", samples[samples["battery_id"].isin(train_batteries)].copy()),
        ("test", samples[samples["battery_id"].isin(test_batteries)].copy()),
    ]:
        pred = fitted_model.predict(split_df[feature_columns])
        metrics = metric_dict(split_df["target_cycle_life"].to_numpy(), pred)
        metric_rows.append(
            {
                "N": n_cycles,
                "model": model_name,
                "run_type": run_type,
                "split": split_name,
                "n_samples": len(split_df),
                "n_features": len(feature_columns),
                **metrics,
            }
        )
        for battery_id, source_batch, y_true, y_pred in zip(
            split_df["battery_id"],
            split_df["source_batch"],
            split_df["target_cycle_life"],
            pred,
        ):
            error = float(y_pred - y_true)
            prediction_rows.append(
                {
                    "battery_id": battery_id,
                    "source_batch": source_batch,
                    "N": n_cycles,
                    "model": model_name,
                    "run_type": run_type,
                    "split": split_name,
                    "target_cycle_life": float(y_true),
                    "predicted_cycle_life": float(y_pred),
                    "error": error,
                    "abs_error": abs(error),
                    "percentage_error": abs(error) / float(y_true) * 100.0,
                }
            )
    return metric_rows, prediction_rows


def tune_and_evaluate_model(
    model_template: Pipeline,
    model_name: str,
    samples: pd.DataFrame,
    feature_columns: list[str],
    train_batteries: set[str],
    test_batteries: set[str],
    n_cycles: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], Pipeline]:
    train = samples[samples["battery_id"].isin(train_batteries)].copy()
    X_train = train[feature_columns]
    y_train = train["target_cycle_life"].to_numpy(dtype=float)
    groups_train = train["battery_id"].astype(str).to_numpy()

    param_distributions = tuning_param_distributions(model_name)
    if not param_distributions:
        fitted = clone(model_template)
        fitted.fit(X_train, y_train)
        metrics, predictions = evaluate_fitted_model(
            fitted,
            model_name,
            samples,
            feature_columns,
            train_batteries,
            test_batteries,
            n_cycles,
            run_type="default_no_search",
        )
        best_params = {
            "N": n_cycles,
            "model": model_name,
            "search_status": "not_tuned",
            "n_iter": 0,
            "cv_splits": 0,
            "best_cv_MAE": np.nan,
            "best_params_json": "{}",
        }
        return metrics, predictions, best_params, fitted

    unique_train_groups = len(np.unique(groups_train))
    cv_splits = min(TUNING_CV_SPLITS, unique_train_groups)
    if cv_splits < 2:
        raise ValueError(f"Not enough train groups for tuning {model_name} at N={n_cycles}.")

    n_iter = RF_TUNING_ITERATIONS if model_name == "RandomForest" else XGB_TUNING_ITERATIONS
    LOGGER.info("Tuning %s at N=%s with %s random candidates and %s-fold GroupKFold.", model_name, n_cycles, n_iter, cv_splits)
    search = RandomizedSearchCV(
        estimator=clone(model_template),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(n_splits=cv_splits),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train, groups=groups_train)
    fitted = search.best_estimator_

    metrics, predictions = evaluate_fitted_model(
        fitted,
        model_name,
        samples,
        feature_columns,
        train_batteries,
        test_batteries,
        n_cycles,
        run_type="tuned_random_search",
    )
    best_params = {
        "N": n_cycles,
        "model": model_name,
        "search_status": "tuned_random_search",
        "n_iter": n_iter,
        "cv_splits": cv_splits,
        "best_cv_MAE": float(-search.best_score_),
        "best_params_json": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
    }
    return metrics, predictions, best_params, fitted


def plot_prediction_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    test_predictions = predictions[predictions["split"] == "test"].copy()
    models = list(test_predictions["model"].drop_duplicates())
    n_values = sorted(test_predictions["N"].drop_duplicates())
    color_map = {n: color for n, color in zip(n_values, ["#4477AA", "#66AA55", "#EE7733", "#CC6677"])}

    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.8), squeeze=False)
    min_value = min(test_predictions["target_cycle_life"].min(), test_predictions["predicted_cycle_life"].min())
    max_value = max(test_predictions["target_cycle_life"].max(), test_predictions["predicted_cycle_life"].max())
    for ax, model_name in zip(axes[0], models):
        subset = test_predictions[test_predictions["model"] == model_name]
        for n_cycles in n_values:
            part = subset[subset["N"] == n_cycles]
            if part.empty:
                continue
            ax.scatter(
                part["target_cycle_life"],
                part["predicted_cycle_life"],
                s=28,
                alpha=0.75,
                label=f"N={n_cycles}",
                color=color_map[n_cycles],
            )
        ax.plot([min_value, max_value], [min_value, max_value], color="black", linewidth=1, linestyle="--")
        ax.set_title(model_name)
        ax.set_xlabel("True cycle life")
        ax.set_ylabel("Predicted cycle life")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("MIT curated test predictions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_by_model(results: pd.DataFrame, output_path: Path) -> None:
    test_results = results[results["split"] == "test"].copy()
    pivot = test_results.pivot(index="model", columns="N", values="MAE").sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.18
    for idx, n_cycles in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[n_cycles], width=width, label=f"N={n_cycles}")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylabel("Test MAE")
    ax.set_title("Test MAE by model")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_by_n(results: pd.DataFrame, output_path: Path) -> None:
    test_results = results[results["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for model_name, group in test_results.groupby("model"):
        ordered = group.sort_values("N")
        ax.plot(ordered["N"], ordered["MAE"], marker="o", linewidth=2, label=model_name)
    ax.set_xlabel("Early cycles used (N)")
    ax.set_ylabel("Test MAE")
    ax.set_title("Effect of early cycle window")
    ax.set_xticks(EARLY_CYCLE_WINDOWS)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_tuned_model_comparison(tuned_results: pd.DataFrame, output_path: Path) -> None:
    test_results = tuned_results[tuned_results["split"] == "test"].copy()
    pivot = test_results.pivot(index="N", columns="model", values="MAE").sort_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model_name in pivot.columns:
        ax.plot(pivot.index, pivot[model_name], marker="o", linewidth=2, label=model_name)
    ax.set_xlabel("Early cycles used (N)")
    ax.set_ylabel("Test MAE")
    ax.set_title("Tuned baseline comparison")
    ax.set_xticks(EARLY_CYCLE_WINDOWS)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def extract_feature_importance(model: Pipeline, feature_columns: list[str], top_k: int = 15) -> pd.DataFrame:
    estimator = model.named_steps.get("model")
    if estimator is None:
        return pd.DataFrame(columns=["feature", "importance"])

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_, dtype=float))
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    out = pd.DataFrame({"feature": feature_columns, "importance": importance})
    return out.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


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
    filtered: pd.DataFrame,
    sample_counts: pd.DataFrame,
    field_qc: pd.DataFrame,
    results: pd.DataFrame,
    cv_results: pd.DataFrame,
    predictions: pd.DataFrame,
    best_row: pd.Series,
    top_features: pd.DataFrame,
    xgboost_status: str | None,
) -> str:
    test_results = results[results["split"] == "test"].copy().sort_values("MAE")
    train_results = results[results["split"] == "train"].copy()
    comparison = test_results[["N", "model", "n_samples", "n_features", "MAE", "RMSE", "MAPE", "R2"]]

    cv_summary = (
        cv_results.groupby(["N", "model"], as_index=False)
        .agg(
            folds=("fold", "count"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            MAPE_mean=("MAPE", "mean"),
            R2_mean=("R2", "mean"),
        )
        .sort_values("MAE_mean")
        if not cv_results.empty
        else pd.DataFrame()
    )

    early_effect = (
        test_results.groupby("N", as_index=False)
        .agg(best_MAE=("MAE", "min"), best_RMSE=("RMSE", "min"), best_MAPE=("MAPE", "min"))
        .sort_values("N")
    )

    overfit = train_results.merge(
        test_results,
        on=["N", "model"],
        suffixes=("_train", "_test"),
    )
    overfit["MAE_gap_test_minus_train"] = overfit["MAE_test"] - overfit["MAE_train"]
    overfit["RMSE_gap_test_minus_train"] = overfit["RMSE_test"] - overfit["RMSE_train"]
    overfit_view = overfit[
        ["N", "model", "MAE_train", "MAE_test", "MAE_gap_test_minus_train", "R2_train", "R2_test"]
    ].sort_values("MAE_gap_test_minus_train", ascending=False)

    skipped_fields = field_qc[~field_qc["selected"]].copy()
    selected_fields = field_qc[field_qc["selected"]].copy()

    model_best_by_name = (
        test_results.sort_values("MAE")
        .groupby("model", as_index=False)
        .first()[["model", "N", "MAE", "RMSE", "MAPE", "R2"]]
        .sort_values("MAE")
    )

    xgb_note = (
        f"\n\nXGBoost 状态：{xgboost_status}\n"
        if xgboost_status
        else "\n\nXGBoost 状态：当前环境可用，已纳入本次对比。\n"
    )

    return f"""# 第一版 cycle life baseline 建模报告

本报告由 `src/models/baseline_cycle_life.py` 自动生成。本次只使用 MIT-Severson curated 电池，不使用 NASA 和 CALCE 参与主训练。

输入文件：

- `data_processed/source_domain/modeling_ready_cycle_life_dataset.parquet`

输出文件：

- `results/baseline_cycle_life_results.csv`
- `results/baseline_cycle_life_cv_results.csv`
- `results/baseline_cycle_life_predictions.csv`
- `figures/baseline/cycle_life_prediction_scatter.png`
- `figures/baseline/error_by_model.png`
- `figures/baseline/error_by_early_cycles.png`

## 数据筛选

筛选条件：

- `dataset == "MIT_Severson"`
- `is_mit_curated_124 == True`
- `target_cycle_life` 非空
- `capacity_invalid_flag != True`
- `capacity_label_excluded_flag != True`

本次进入候选建模池的 MIT curated 电池数：`{filtered["battery_id"].nunique()}`；循环行数：`{len(filtered)}`。

每个 `battery_id` 在每个 N 下只生成一行样本，避免把循环级行错误当作独立样本。

## 每个 N 的样本数

{markdown_table(sample_counts)}

## 特征字段可用性

字段缺失率超过 `{MISSING_SKIP_THRESHOLD:.0%}` 会自动跳过。电压和电流类字段在当前 MIT summary-only 表中大多缺失，因此第一版 baseline 主要由容量、SOH、温度、内阻和充电时间特征驱动。

已选字段：

{markdown_table(selected_fields[["N", "field", "missing_rate", "reason"]], max_rows=80)}

跳过字段：

{markdown_table(skipped_fields[["N", "field", "missing_rate", "reason"]], max_rows=80)}

## 模型效果对比

测试集指标按 MAE 从低到高排序：

{markdown_table(comparison, max_rows=80)}

各模型自身最佳测试结果：

{markdown_table(model_best_by_name)}
{xgb_note}

## 最佳模型

按测试集 MAE 选择，本次最佳组合为：

- 模型：`{best_row["model"]}`
- 早期循环窗口：`N={int(best_row["N"])}`
- 测试集 MAE：`{best_row["MAE"]:.4f}`
- 测试集 RMSE：`{best_row["RMSE"]:.4f}`
- 测试集 MAPE：`{best_row["MAPE"]:.4f}%`
- 测试集 R2：`{best_row["R2"]:.4f}`

## 前 10/30/50/100 圈影响

每个 N 下取最佳模型后的测试误差：

{markdown_table(early_effect)}

总体上，N 越大通常提供更完整的早期退化轨迹，但也更接近寿命过程本身。后续论文实验应同时报告 N=10、30、50、100，避免只展示最有利窗口。

## GroupKFold 交叉验证

按 `battery_id` 分组做 GroupKFold，默认 5 折。当前每个 N 的样本数足够 5 折。

{markdown_table(cv_summary, max_rows=80)}

## 过拟合风险

训练集和测试集误差差距如下，若训练误差显著低于测试误差，说明树模型可能存在过拟合风险。

{markdown_table(overfit_view, max_rows=80)}

## 可能重要的特征

下表给出最佳模型对应的前若干特征重要性。树模型使用 `feature_importances_`，Elastic Net 使用标准化后系数绝对值。

{markdown_table(top_features)}

从特征命名看，容量/SOH 的早期水平、变化量、斜率，以及内阻和充电时间相关统计量，是后续特征工程最值得优先关注的方向。

## 推荐源域 baseline

第一版建议把 `{best_row["model"]}` 在 `N={int(best_row["N"])}` 下的结果作为源域 cycle life baseline，同时保留 Elastic Net 作为可解释线性基线，Random Forest / XGBoost 作为非线性树模型基线。

后续迁移到固态或半固态小样本时，建议优先迁移“早期循环特征工程 + 源域训练得到的非线性回归器”这一路线，再和深度时序模型比较。

## 下一步扩展到 DeepAR / TAM-DeepAR-EN

下一步可以从两条线推进：

- 将本脚本生成的一行一电池寿命样本保留下来，作为传统机器学习 baseline；
- 另行构造按循环展开的序列样本：`battery_id × cycle_index × IHFs`，输入 DeepAR / TAM-DeepAR-EN，用 SOH 序列预测和 RUL 推断替代单点 cycle life 回归。

TAM-DeepAR-EN 阶段可重点加入 CCCT、CVCT、ADV、ICV、容量、温度、内阻等时序特征，并在源域 MIT/CALCE/NASA 上预训练，再迁移到固态小样本。

## 当前局限性

- 本 baseline 只使用 MIT curated 样本，没有纳入 NASA/CALCE 训练，因此只能作为第一版源域基准。
- 当前 MIT 使用 summary-only 数据，电压/电流曲线特征缺失较多，尚未使用 detailed cycles 的曲线信息。
- 模型使用固定默认超参数，没有做系统调参。
- `target_cycle_life` 采用 MIT 原始 `.mat` 官方标签，和 `SOH<=0.8` 重新计算标签不是同一口径。
- 数据划分是随机 battery-level split，后续应增加按 batch 或协议划分的外推验证。
"""


def generate_tuned_report(
    filtered: pd.DataFrame,
    sample_counts: pd.DataFrame,
    baseline_results: pd.DataFrame,
    tuned_results: pd.DataFrame,
    best_params: pd.DataFrame,
    best_row: pd.Series,
    top_features: pd.DataFrame,
    xgboost_status: str | None,
    xgboost_version: str | None,
) -> str:
    tuned_test = tuned_results[tuned_results["split"] == "test"].copy().sort_values("MAE")
    tuned_train = tuned_results[tuned_results["split"] == "train"].copy()
    tuned_comparison = tuned_test[["N", "model", "run_type", "n_samples", "n_features", "MAE", "RMSE", "MAPE", "R2"]]

    baseline_test = baseline_results[baseline_results["split"] == "test"].copy()
    rf_before = baseline_test[baseline_test["model"] == "RandomForest"][
        ["N", "MAE", "RMSE", "MAPE", "R2"]
    ].rename(columns={"MAE": "MAE_before", "RMSE": "RMSE_before", "MAPE": "MAPE_before", "R2": "R2_before"})
    rf_after = tuned_test[tuned_test["model"] == "RandomForest"][
        ["N", "MAE", "RMSE", "MAPE", "R2"]
    ].rename(columns={"MAE": "MAE_after", "RMSE": "RMSE_after", "MAPE": "MAPE_after", "R2": "R2_after"})
    rf_delta = rf_before.merge(rf_after, on="N", how="outer").sort_values("N")
    if not rf_delta.empty:
        rf_delta["MAE_delta_after_minus_before"] = rf_delta["MAE_after"] - rf_delta["MAE_before"]
        rf_delta["improved_by_MAE"] = rf_delta["MAE_delta_after_minus_before"] < 0

    best_by_model = (
        tuned_test.sort_values("MAE")
        .groupby("model", as_index=False)
        .first()[["model", "N", "run_type", "MAE", "RMSE", "MAPE", "R2"]]
        .sort_values("MAE")
    )

    overfit = tuned_train.merge(
        tuned_test,
        on=["N", "model"],
        suffixes=("_train", "_test"),
    )
    overfit["MAE_gap_test_minus_train"] = overfit["MAE_test"] - overfit["MAE_train"]
    overfit_view = overfit[
        ["N", "model", "run_type_train", "MAE_train", "MAE_test", "MAE_gap_test_minus_train", "R2_train", "R2_test"]
    ].sort_values("MAE_gap_test_minus_train", ascending=False)

    xgb_ran = "XGBoost" in set(tuned_results["model"])
    if xgb_ran:
        xgb_status_text = f"成功运行，当前环境版本为 `{xgboost_version}`。"
    else:
        xgb_status_text = (
            f"未成功运行。{xgboost_status or '当前环境未检测到 xgboost。'} "
            "请在当前 Python 环境中运行：`pip install xgboost`。"
        )

    rf_improved_count = int(rf_delta["improved_by_MAE"].sum()) if "improved_by_MAE" in rf_delta else 0
    rf_total_count = len(rf_delta)
    if rf_total_count:
        rf_summary = f"Random Forest 调参后在 {rf_improved_count}/{rf_total_count} 个 N 窗口上按测试集 MAE 获得提升。"
    else:
        rf_summary = "未找到 Random Forest 调参前后可比结果。"

    return f"""# 调参版 cycle life baseline 报告

本报告由 `src/models/baseline_cycle_life.py` 自动生成。本轮继续只使用 MIT-Severson curated 124 电池，不修改 `data_raw/`，不修改 `modeling_ready_cycle_life_dataset.parquet`，也不使用 NASA/CALCE 参与主训练。

## 输出文件

- `results/baseline_cycle_life_results_tuned.csv`
- `results/baseline_cycle_life_predictions_tuned.csv`
- `results/baseline_cycle_life_best_params.csv`
- `figures/baseline/tuned_model_comparison.png`
- `docs/baseline_cycle_life_tuned_report.md`

## 数据和划分

- MIT curated 电池数：`{filtered["battery_id"].nunique()}`
- 过滤后循环行数：`{len(filtered)}`
- 每个 N 下每个 battery 只生成一行样本
- 划分方式：按 `battery_id` 进行 80%/20% train/test 随机划分，`random_state=42`

每个 N 的样本数：

{markdown_table(sample_counts)}

## XGBoost 运行状态

{xgb_status_text}

如果迁移到新的 Python 环境后 XGBoost 不可用，请运行：

```bash
pip install xgboost
```

## 调参设置

Elastic Net 作为线性基准保留默认设置；Random Forest 和 XGBoost 使用轻量 `RandomizedSearchCV`，只在训练集上用 GroupKFold 搜索，测试集只用于最终评估。

- Random Forest：`n_iter={RF_TUNING_ITERATIONS}`，搜索 `n_estimators/max_depth/min_samples_leaf/max_features`
- XGBoost：`n_iter={XGB_TUNING_ITERATIONS}`，搜索 `n_estimators/max_depth/learning_rate/subsample/colsample_bytree/reg_alpha/reg_lambda`
- 调参 CV：`{TUNING_CV_SPLITS}` 折 GroupKFold，按 `battery_id` 分组

最佳参数：

{markdown_table(best_params, max_rows=80)}

## 调参前后 Random Forest 对比

{rf_summary}

{markdown_table(rf_delta)}

## 最终模型对比

测试集指标按 MAE 从低到高排序：

{markdown_table(tuned_comparison, max_rows=80)}

各模型自身最佳结果：

{markdown_table(best_by_model)}

## 最佳模型

按测试集 MAE 选择，本轮最佳组合为：

- 模型：`{best_row["model"]}`
- 早期循环窗口：`N={int(best_row["N"])}`
- 测试集 MAE：`{best_row["MAE"]:.4f}`
- 测试集 RMSE：`{best_row["RMSE"]:.4f}`
- 测试集 MAPE：`{best_row["MAPE"]:.4f}%`
- 测试集 R2：`{best_row["R2"]:.4f}`

## 过拟合风险

训练误差显著低于测试误差时，需要警惕过拟合。调参后的 train/test 差距如下：

{markdown_table(overfit_view, max_rows=80)}

由于样本只有 124 个电池，调参结果对 train/test 随机划分较敏感。当前轻量搜索的目的不是追求极限分数，而是建立一个更稳妥、可复现的源域 baseline。

## 可能重要的特征

最佳模型的前若干特征重要性如下：

{markdown_table(top_features)}

## 推荐源域 baseline

建议将 `{best_row["model"]}` 在 `N={int(best_row["N"])}` 下的调参结果作为后续迁移学习的源域 baseline。Elastic Net 继续作为可解释线性基准；Random Forest 和 XGBoost 作为非线性基准。

若后续要迁移到固态/半固态小样本，建议保留本次的一行一电池早期循环特征 baseline，同时另起 DeepAR / TAM-DeepAR-EN 序列分支，用 `battery_id × cycle_index × IHFs` 的时序输入做 SOH/RUL 预测。

## 局限性

- 调参搜索规模刻意较小，避免在 124 个电池上过度调参。
- 仍使用 MIT summary-only 特征，电压/电流曲线特征暂时缺失。
- 当前测试集是随机 battery-level split，后续应增加按 batch/protocol 的外推测试。
- MIT 官方 `cycle_life` 与 `SOH<=0.8` 重算标签口径不同，后续论文中要明确标签来源。
"""


def run_baseline() -> dict[str, Any]:
    filtered = load_main_training_rows(INPUT_PATH)
    if filtered["battery_id"].nunique() < 10:
        raise ValueError("Too few MIT curated batteries for baseline modeling.")

    train_batteries, test_batteries = train_test_battery_split(filtered["battery_id"].unique())
    LOGGER.info("Battery split: %s train batteries, %s test batteries", len(train_batteries), len(test_batteries))

    models, xgboost_status = make_models()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_cv_results: list[dict[str, Any]] = []
    all_tuned_results: list[dict[str, Any]] = []
    all_tuned_predictions: list[dict[str, Any]] = []
    best_param_rows: list[dict[str, Any]] = []
    field_qc_frames: list[pd.DataFrame] = []
    sample_count_rows: list[dict[str, Any]] = []
    fitted_models: dict[tuple[int, str], tuple[Pipeline, list[str]]] = {}
    tuned_fitted_models: dict[tuple[int, str], tuple[Pipeline, list[str]]] = {}

    for n_cycles in EARLY_CYCLE_WINDOWS:
        samples, feature_columns, field_qc = build_feature_table(filtered, n_cycles)
        field_qc_frames.append(field_qc)
        sample_count_rows.append(
            {
                "N": n_cycles,
                "battery_samples": samples["battery_id"].nunique(),
                "train_batteries": int(samples["battery_id"].isin(train_batteries).sum()),
                "test_batteries": int(samples["battery_id"].isin(test_batteries).sum()),
                "feature_count": len(feature_columns),
            }
        )
        LOGGER.info("N=%s: %s samples, %s features", n_cycles, len(samples), len(feature_columns))

        for model_name, model_template in models.items():
            metric_rows, prediction_rows, fitted = evaluate_train_test(
                model_template=model_template,
                model_name=model_name,
                samples=samples,
                feature_columns=feature_columns,
                train_batteries=train_batteries,
                test_batteries=test_batteries,
                n_cycles=n_cycles,
            )
            all_results.extend(metric_rows)
            all_predictions.extend(prediction_rows)
            all_cv_results.extend(evaluate_group_cv(model_template, model_name, samples, feature_columns, n_cycles))
            fitted_models[(n_cycles, model_name)] = (fitted, feature_columns)

            tuned_metric_rows, tuned_prediction_rows, best_params, tuned_fitted = tune_and_evaluate_model(
                model_template=model_template,
                model_name=model_name,
                samples=samples,
                feature_columns=feature_columns,
                train_batteries=train_batteries,
                test_batteries=test_batteries,
                n_cycles=n_cycles,
            )
            all_tuned_results.extend(tuned_metric_rows)
            all_tuned_predictions.extend(tuned_prediction_rows)
            best_param_rows.append(best_params)
            tuned_fitted_models[(n_cycles, model_name)] = (tuned_fitted, feature_columns)

    results = pd.DataFrame(all_results)
    predictions = pd.DataFrame(all_predictions)
    cv_results = pd.DataFrame(all_cv_results)
    tuned_results = pd.DataFrame(all_tuned_results)
    tuned_predictions = pd.DataFrame(all_tuned_predictions)
    best_params_df = pd.DataFrame(best_param_rows)
    field_qc = pd.concat(field_qc_frames, ignore_index=True)
    sample_counts = pd.DataFrame(sample_count_rows)

    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    cv_results.to_csv(CV_RESULTS_PATH, index=False, encoding="utf-8-sig")
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    tuned_results.to_csv(TUNED_RESULTS_PATH, index=False, encoding="utf-8-sig")
    tuned_predictions.to_csv(TUNED_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    best_params_df.to_csv(BEST_PARAMS_PATH, index=False, encoding="utf-8-sig")

    plot_prediction_scatter(predictions, SCATTER_PATH)
    plot_error_by_model(results, ERROR_BY_MODEL_PATH)
    plot_error_by_n(results, ERROR_BY_N_PATH)
    plot_tuned_model_comparison(tuned_results, TUNED_COMPARISON_PATH)

    best_row = results[results["split"] == "test"].sort_values("MAE").iloc[0]
    best_key = (int(best_row["N"]), str(best_row["model"]))
    best_model, best_features = fitted_models[best_key]
    top_features = extract_feature_importance(best_model, best_features)

    REPORT_PATH.write_text(
        generate_report(
            filtered=filtered,
            sample_counts=sample_counts,
            field_qc=field_qc,
            results=results,
            cv_results=cv_results,
            predictions=predictions,
            best_row=best_row,
            top_features=top_features,
            xgboost_status=xgboost_status,
        ),
        encoding="utf-8",
    )

    tuned_best_row = tuned_results[tuned_results["split"] == "test"].sort_values("MAE").iloc[0]
    tuned_best_key = (int(tuned_best_row["N"]), str(tuned_best_row["model"]))
    tuned_best_model, tuned_best_features = tuned_fitted_models[tuned_best_key]
    tuned_top_features = extract_feature_importance(tuned_best_model, tuned_best_features)
    TUNED_REPORT_PATH.write_text(
        generate_tuned_report(
            filtered=filtered,
            sample_counts=sample_counts,
            baseline_results=results,
            tuned_results=tuned_results,
            best_params=best_params_df,
            best_row=tuned_best_row,
            top_features=tuned_top_features,
            xgboost_status=xgboost_status,
            xgboost_version=get_xgboost_version(),
        ),
        encoding="utf-8",
    )

    print(
        "Best baseline: "
        f"model={best_row['model']}, N={int(best_row['N'])}, "
        f"test_MAE={best_row['MAE']:.4f}, test_RMSE={best_row['RMSE']:.4f}, "
        f"test_MAPE={best_row['MAPE']:.4f}%, test_R2={best_row['R2']:.4f}"
    )
    LOGGER.info("Saved results: %s", RESULTS_PATH)
    LOGGER.info("Saved CV results: %s", CV_RESULTS_PATH)
    LOGGER.info("Saved predictions: %s", PREDICTIONS_PATH)
    LOGGER.info("Saved tuned results: %s", TUNED_RESULTS_PATH)
    LOGGER.info("Saved tuned predictions: %s", TUNED_PREDICTIONS_PATH)
    LOGGER.info("Saved best params: %s", BEST_PARAMS_PATH)
    LOGGER.info("Saved figures under: %s", FIGURE_DIR)
    LOGGER.info("Saved report: %s", REPORT_PATH)
    LOGGER.info("Saved tuned report: %s", TUNED_REPORT_PATH)

    print(
        "Best tuned baseline: "
        f"model={tuned_best_row['model']}, N={int(tuned_best_row['N'])}, "
        f"test_MAE={tuned_best_row['MAE']:.4f}, test_RMSE={tuned_best_row['RMSE']:.4f}, "
        f"test_MAPE={tuned_best_row['MAPE']:.4f}%, test_R2={tuned_best_row['R2']:.4f}"
    )

    return {
        "results": results,
        "cv_results": cv_results,
        "predictions": predictions,
        "tuned_results": tuned_results,
        "tuned_predictions": tuned_predictions,
        "best_params": best_params_df,
        "sample_counts": sample_counts,
        "field_qc": field_qc,
        "best_row": best_row,
        "tuned_best_row": tuned_best_row,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train first MIT-Severson cycle life baseline models.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    parse_args()
    run_baseline()


if __name__ == "__main__":
    main()
