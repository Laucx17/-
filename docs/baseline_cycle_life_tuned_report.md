# 调参版 cycle life baseline 报告

本报告由 `src/models/baseline_cycle_life.py` 自动生成。本轮继续只使用 MIT-Severson curated 124 电池，不修改 `data_raw/`，不修改 `modeling_ready_cycle_life_dataset.parquet`，也不使用 NASA/CALCE 参与主训练。

## 输出文件

- `results/baseline_cycle_life_results_tuned.csv`
- `results/baseline_cycle_life_predictions_tuned.csv`
- `results/baseline_cycle_life_best_params.csv`
- `figures/baseline/tuned_model_comparison.png`
- `docs/baseline_cycle_life_tuned_report.md`

## 数据和划分

- MIT curated 电池数：`124`
- 过滤后循环行数：`97063`
- 每个 N 下每个 battery 只生成一行样本
- 划分方式：按 `battery_id` 进行 80%/20% train/test 随机划分，`random_state=42`

每个 N 的样本数：

| N | battery_samples | train_batteries | test_batteries | feature_count |
| --- | --- | --- | --- | --- |
| 10 | 124 | 99 | 25 | 63 |
| 30 | 124 | 99 | 25 | 63 |
| 50 | 124 | 99 | 25 | 63 |
| 100 | 124 | 99 | 25 | 63 |

## XGBoost 运行状态

成功运行，当前环境版本为 `3.2.0`。

如果迁移到新的 Python 环境后 XGBoost 不可用，请运行：

```bash
pip install xgboost
```

## 调参设置

Elastic Net 作为线性基准保留默认设置；Random Forest 和 XGBoost 使用轻量 `RandomizedSearchCV`，只在训练集上用 GroupKFold 搜索，测试集只用于最终评估。

- Random Forest：`n_iter=12`，搜索 `n_estimators/max_depth/min_samples_leaf/max_features`
- XGBoost：`n_iter=18`，搜索 `n_estimators/max_depth/learning_rate/subsample/colsample_bytree/reg_alpha/reg_lambda`
- 调参 CV：`3` 折 GroupKFold，按 `battery_id` 分组

最佳参数：

| N | model | search_status | n_iter | cv_splits | best_cv_MAE | best_params_json |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | ElasticNet | not_tuned | 0 | 0 | NaN | {} |
| 10 | RandomForest | tuned_random_search | 12 | 3 | 131.756 | {"model__max_depth": 20, "model__max_features": "sqrt", "model__min_samples_leaf": 1, "model__n_estimators": 800} |
| 10 | XGBoost | tuned_random_search | 18 | 3 | 129.551 | {"model__colsample_bytree": 1.0, "model__learning_rate": 0.03, "model__max_depth": 2, "model__n_estimators": 200, "model__reg_alpha": 0, "model__reg_lambda": 5.0, "model__subsample": 0.7} |
| 30 | ElasticNet | not_tuned | 0 | 0 | NaN | {} |
| 30 | RandomForest | tuned_random_search | 12 | 3 | 122.855 | {"model__max_depth": 20, "model__max_features": 0.5, "model__min_samples_leaf": 2, "model__n_estimators": 800} |
| 30 | XGBoost | tuned_random_search | 18 | 3 | 115.287 | {"model__colsample_bytree": 0.7, "model__learning_rate": 0.03, "model__max_depth": 2, "model__n_estimators": 200, "model__reg_alpha": 0, "model__reg_lambda": 10.0, "model__subsample": 0.7} |
| 50 | ElasticNet | not_tuned | 0 | 0 | NaN | {} |
| 50 | RandomForest | tuned_random_search | 12 | 3 | 121.743 | {"model__max_depth": 10, "model__max_features": 0.5, "model__min_samples_leaf": 2, "model__n_estimators": 500} |
| 50 | XGBoost | tuned_random_search | 18 | 3 | 109.842 | {"model__colsample_bytree": 1.0, "model__learning_rate": 0.03, "model__max_depth": 2, "model__n_estimators": 200, "model__reg_alpha": 0, "model__reg_lambda": 5.0, "model__subsample": 0.7} |
| 100 | ElasticNet | not_tuned | 0 | 0 | NaN | {} |
| 100 | RandomForest | tuned_random_search | 12 | 3 | 117.08 | {"model__max_depth": 20, "model__max_features": "sqrt", "model__min_samples_leaf": 1, "model__n_estimators": 800} |
| 100 | XGBoost | tuned_random_search | 18 | 3 | 112.151 | {"model__colsample_bytree": 0.7, "model__learning_rate": 0.03, "model__max_depth": 2, "model__n_estimators": 200, "model__reg_alpha": 0, "model__reg_lambda": 10.0, "model__subsample": 0.7} |

## 调参前后 Random Forest 对比

Random Forest 调参后在 2/4 个 N 窗口上按测试集 MAE 获得提升。

| N | MAE_before | RMSE_before | MAPE_before | R2_before | MAE_after | RMSE_after | MAPE_after | R2_after | MAE_delta_after_minus_before | improved_by_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 161.096 | 227.109 | 24.2472 | 0.604599 | 168.035 | 232.824 | 29.0954 | 0.58445 | 6.9391 | False |
| 30 | 148.07 | 256.012 | 21.7277 | 0.497555 | 138.717 | 240.58 | 21.4514 | 0.556301 | -9.35302 | True |
| 50 | 177.874 | 252.512 | 26.7308 | 0.511201 | 159.17 | 239.232 | 24.7209 | 0.561261 | -18.7044 | True |
| 100 | 156.919 | 238.566 | 24.3514 | 0.5637 | 158.807 | 222.743 | 25.5116 | 0.619657 | 1.88755 | False |

## 最终模型对比

测试集指标按 MAE 从低到高排序：

| N | model | run_type | n_samples | n_features | MAE | RMSE | MAPE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | RandomForest | tuned_random_search | 25 | 63 | 138.717 | 240.58 | 21.4514 | 0.556301 |
| 10 | XGBoost | tuned_random_search | 25 | 63 | 143.665 | 213.259 | 23.1677 | 0.651355 |
| 30 | XGBoost | tuned_random_search | 25 | 63 | 158.694 | 262.645 | 24.9167 | 0.471182 |
| 100 | RandomForest | tuned_random_search | 25 | 63 | 158.807 | 222.743 | 25.5116 | 0.619657 |
| 50 | RandomForest | tuned_random_search | 25 | 63 | 159.17 | 239.232 | 24.7209 | 0.561261 |
| 100 | XGBoost | tuned_random_search | 25 | 63 | 162.654 | 248.238 | 26.7808 | 0.527607 |
| 50 | XGBoost | tuned_random_search | 25 | 63 | 165.998 | 242.254 | 26.2154 | 0.550107 |
| 10 | RandomForest | tuned_random_search | 25 | 63 | 168.035 | 232.824 | 29.0954 | 0.58445 |
| 30 | ElasticNet | default_no_search | 25 | 63 | 188.557 | 291.072 | 49.5338 | 0.350514 |
| 100 | ElasticNet | default_no_search | 25 | 63 | 210.919 | 299.931 | 50.1746 | 0.310379 |
| 10 | ElasticNet | default_no_search | 25 | 63 | 225.65 | 307.356 | 45.1258 | 0.27581 |
| 50 | ElasticNet | default_no_search | 25 | 63 | 243.449 | 346.641 | 57.9698 | 0.0788566 |

各模型自身最佳结果：

| model | N | run_type | MAE | RMSE | MAPE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 30 | tuned_random_search | 138.717 | 240.58 | 21.4514 | 0.556301 |
| XGBoost | 10 | tuned_random_search | 143.665 | 213.259 | 23.1677 | 0.651355 |
| ElasticNet | 30 | default_no_search | 188.557 | 291.072 | 49.5338 | 0.350514 |

## 最佳模型

按测试集 MAE 选择，本轮最佳组合为：

- 模型：`RandomForest`
- 早期循环窗口：`N=30`
- 测试集 MAE：`138.7173`
- 测试集 RMSE：`240.5804`
- 测试集 MAPE：`21.4514%`
- 测试集 R2：`0.5563`

## 过拟合风险

训练误差显著低于测试误差时，需要警惕过拟合。调参后的 train/test 差距如下：

| N | model | run_type_train | MAE_train | MAE_test | MAE_gap_test_minus_train | R2_train | R2_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | ElasticNet | default_no_search | 91.4861 | 243.449 | 151.963 | 0.781759 | 0.0788566 |
| 10 | ElasticNet | default_no_search | 97.5011 | 225.65 | 128.149 | 0.76636 | 0.27581 |
| 100 | ElasticNet | default_no_search | 85.1337 | 210.919 | 125.785 | 0.814871 | 0.310379 |
| 10 | RandomForest | tuned_random_search | 46.8159 | 168.035 | 121.219 | 0.941135 | 0.58445 |
| 100 | RandomForest | tuned_random_search | 40.3648 | 158.807 | 118.442 | 0.959733 | 0.619657 |
| 50 | XGBoost | tuned_random_search | 47.7999 | 165.998 | 118.198 | 0.933291 | 0.550107 |
| 50 | RandomForest | tuned_random_search | 47.6021 | 159.17 | 111.568 | 0.929192 | 0.561261 |
| 100 | XGBoost | tuned_random_search | 53.8938 | 162.654 | 108.76 | 0.893701 | 0.527607 |
| 30 | XGBoost | tuned_random_search | 55.0084 | 158.694 | 103.686 | 0.885568 | 0.471182 |
| 30 | ElasticNet | default_no_search | 94.1846 | 188.557 | 94.3724 | 0.760569 | 0.350514 |
| 10 | XGBoost | tuned_random_search | 53.1612 | 143.665 | 90.5033 | 0.913428 | 0.651355 |
| 30 | RandomForest | tuned_random_search | 49.3252 | 138.717 | 89.3921 | 0.922393 | 0.556301 |

由于样本只有 124 个电池，调参结果对 train/test 随机划分较敏感。当前轻量搜索的目的不是追求极限分数，而是建立一个更稳妥、可复现的源域 baseline。

## 可能重要的特征

最佳模型的前若干特征重要性如下：

| feature | importance |
| --- | --- |
| soh__min | 0.256549 |
| soh__first_value | 0.0861686 |
| internal_resistance_ohm__max | 0.0687901 |
| internal_resistance_ohm__first_value | 0.0678577 |
| internal_resistance_ohm__delta | 0.0441799 |
| internal_resistance_ohm__min | 0.0323591 |
| temperature_max_c__delta | 0.0320574 |
| internal_resistance_ohm__mean | 0.0292033 |
| charge_time_s__std | 0.0204515 |
| capacity_ah__std | 0.0198552 |
| soh__std | 0.0195672 |
| temperature_max_c__std | 0.0176362 |
| temperature_mean_c__delta | 0.0159684 |
| internal_resistance_ohm__slope | 0.0148171 |
| temperature_min_c__min | 0.0125052 |

## 推荐源域 baseline

建议将 `RandomForest` 在 `N=30` 下的调参结果作为后续迁移学习的源域 baseline。Elastic Net 继续作为可解释线性基准；Random Forest 和 XGBoost 作为非线性基准。

若后续要迁移到固态/半固态小样本，建议保留本次的一行一电池早期循环特征 baseline，同时另起 DeepAR / TAM-DeepAR-EN 序列分支，用 `battery_id × cycle_index × IHFs` 的时序输入做 SOH/RUL 预测。

## 局限性

- 调参搜索规模刻意较小，避免在 124 个电池上过度调参。
- 仍使用 MIT summary-only 特征，电压/电流曲线特征暂时缺失。
- 当前测试集是随机 battery-level split，后续应增加按 batch/protocol 的外推测试。
- MIT 官方 `cycle_life` 与 `SOH<=0.8` 重算标签口径不同，后续论文中要明确标签来源。
