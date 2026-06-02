# Baseline 稳定性检查报告

本报告由 `src/models/baseline_stability_check.py` 自动生成。本次不修改原始数据，不修改 modeling-ready 数据集。

## 实验设置

- 数据：仅使用 MIT-Severson curated 电池
- 筛选条件：`dataset == "MIT_Severson"`、`is_mit_curated_124 == True`、`target_cycle_life` 非空、容量异常标记为 False
- 每个 `battery_id` 在指定 N 下只形成一行样本
- 每次实验按 `battery_id` 做 80/20 train/test 划分
- 随机种子：`[0, 1, 2, 3, 4, 5, 10, 21, 42, 100]`
- 候选模型：`RandomForest + N=30`，`XGBoost + N=10`
- XGBoost 状态：可用，版本 3.2.0

过滤后进入稳定性检查的数据池：`124` 个电池，`97063` 行循环记录。

使用的 tuned 参数：

| model | N | best_params_json |
| --- | --- | --- |
| RandomForest | 30 | {"model__max_depth": 20, "model__max_features": 0.5, "model__min_samples_leaf": 2, "model__n_estimators": 800} |
| XGBoost | 10 | {"model__colsample_bytree": 1.0, "model__learning_rate": 0.03, "model__max_depth": 2, "model__n_estimators": 200, "model__reg_alpha": 0, "model__reg_lambda": 5.0, "model__subsample": 0.7} |

## 单次重复实验结果

| model | N | random_state | train_batteries | test_batteries | n_features | MAE | RMSE | MAPE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 30 | 0 | 99 | 25 | 63 | 124.888 | 230.535 | 12.867 | 0.591625 |
| RandomForest | 30 | 1 | 99 | 25 | 63 | 130.783 | 179.065 | 17.1691 | 0.603203 |
| RandomForest | 30 | 2 | 99 | 25 | 63 | 141.497 | 195.374 | 14.7349 | 0.5239 |
| RandomForest | 30 | 3 | 99 | 25 | 63 | 140.468 | 219.07 | 20.8582 | 0.634472 |
| RandomForest | 30 | 4 | 99 | 25 | 63 | 103.997 | 148.356 | 12.2869 | 0.715509 |
| RandomForest | 30 | 5 | 99 | 25 | 63 | 120.603 | 194.215 | 20.2839 | 0.620806 |
| RandomForest | 30 | 10 | 99 | 25 | 63 | 96.3412 | 130.366 | 12.9374 | 0.762552 |
| RandomForest | 30 | 21 | 99 | 25 | 63 | 81.8995 | 109.96 | 15.7661 | 0.828368 |
| RandomForest | 30 | 42 | 99 | 25 | 63 | 138.717 | 240.58 | 21.4514 | 0.556301 |
| RandomForest | 30 | 100 | 99 | 25 | 63 | 85.9017 | 109.206 | 10.501 | 0.783569 |
| XGBoost | 10 | 0 | 99 | 25 | 63 | 142.98 | 239.424 | 14.5696 | 0.559523 |
| XGBoost | 10 | 1 | 99 | 25 | 63 | 156.38 | 213.417 | 20.8213 | 0.436354 |
| XGBoost | 10 | 2 | 99 | 25 | 63 | 148.701 | 200.196 | 14.9479 | 0.500105 |
| XGBoost | 10 | 3 | 99 | 25 | 63 | 142.002 | 238.114 | 21.1033 | 0.568159 |
| XGBoost | 10 | 4 | 99 | 25 | 63 | 119.084 | 169.826 | 13.6475 | 0.627209 |
| XGBoost | 10 | 5 | 99 | 25 | 63 | 139.177 | 223.49 | 21.9846 | 0.497872 |
| XGBoost | 10 | 10 | 99 | 25 | 63 | 106.387 | 149.395 | 14.3802 | 0.688174 |
| XGBoost | 10 | 21 | 99 | 25 | 63 | 121.121 | 157.582 | 21.377 | 0.647517 |
| XGBoost | 10 | 42 | 99 | 25 | 63 | 143.665 | 213.259 | 23.1677 | 0.651355 |
| XGBoost | 10 | 100 | 99 | 25 | 63 | 99.8073 | 123.559 | 12.2833 | 0.722938 |

## 均值和标准差

| model | N | runs | train_batteries | test_batteries | MAE_mean | MAE_std | MAE_min | MAE_max | RMSE_mean | RMSE_std | RMSE_min | RMSE_max | MAPE_mean | MAPE_std | MAPE_min | MAPE_max | R2_mean | R2_std | R2_min | R2_max | MAE_cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 30 | 10 | 99 | 25 | 116.51 | 22.8043 | 81.8995 | 141.497 | 175.673 | 48.7448 | 109.206 | 240.58 | 15.8856 | 3.91271 | 10.501 | 21.4514 | 0.662031 | 0.10357 | 0.5239 | 0.828368 | 0.195728 |
| XGBoost | 10 | 10 | 99 | 25 | 131.931 | 19.0291 | 99.8073 | 156.38 | 192.826 | 40.1579 | 123.559 | 239.424 | 17.8282 | 4.17786 | 12.2833 | 23.1677 | 0.589921 | 0.0927058 | 0.436354 | 0.722938 | 0.144236 |

## 稳定性判断

- `RandomForest + N=30` 的 MAE 均值为 `116.5098`，标准差为 `22.8043`，变异系数约 `0.196`，整体判断为：中等稳定。
- `XGBoost + N=10` 的 MAE 均值为 `131.9305`，标准差为 `19.0291`，变异系数约 `0.144`，整体判断为：较稳定。

## 哪个模型更适合作为迁移学习源域 baseline

按多随机种子的平均 MAE 排序，本次更推荐 `RandomForest + N=30` 作为后续迁移学习源域 baseline：

- MAE mean：`116.5098`
- MAE std：`22.8043`
- RMSE mean：`175.6727`
- MAPE mean：`15.8856%`
- R2 mean：`0.6620`

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
