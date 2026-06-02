# 第一版 cycle life baseline 建模报告

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

本次进入候选建模池的 MIT curated 电池数：`124`；循环行数：`97063`。

每个 `battery_id` 在每个 N 下只生成一行样本，避免把循环级行错误当作独立样本。

## 每个 N 的样本数

| N | battery_samples | train_batteries | test_batteries | feature_count |
| --- | --- | --- | --- | --- |
| 10 | 124 | 99 | 25 | 63 |
| 30 | 124 | 99 | 25 | 63 |
| 50 | 124 | 99 | 25 | 63 |
| 100 | 124 | 99 | 25 | 63 |

## 特征字段可用性

字段缺失率超过 `80%` 会自动跳过。电压和电流类字段在当前 MIT summary-only 表中大多缺失，因此第一版 baseline 主要由容量、SOH、温度、内阻和充电时间特征驱动。

已选字段：

| N | field | missing_rate | reason |
| --- | --- | --- | --- |
| 10 | capacity_ah | 0 | selected |
| 10 | soh | 0 | selected |
| 10 | temperature_mean_c | 0 | selected |
| 10 | temperature_max_c | 0 | selected |
| 10 | temperature_min_c | 0 | selected |
| 10 | internal_resistance_ohm | 0 | selected |
| 10 | charge_time_s | 0 | selected |
| 30 | capacity_ah | 0 | selected |
| 30 | soh | 0 | selected |
| 30 | temperature_mean_c | 0 | selected |
| 30 | temperature_max_c | 0 | selected |
| 30 | temperature_min_c | 0 | selected |
| 30 | internal_resistance_ohm | 0 | selected |
| 30 | charge_time_s | 0 | selected |
| 50 | capacity_ah | 0 | selected |
| 50 | soh | 0 | selected |
| 50 | temperature_mean_c | 0 | selected |
| 50 | temperature_max_c | 0 | selected |
| 50 | temperature_min_c | 0 | selected |
| 50 | internal_resistance_ohm | 0 | selected |
| 50 | charge_time_s | 0 | selected |
| 100 | capacity_ah | 0 | selected |
| 100 | soh | 0 | selected |
| 100 | temperature_mean_c | 0 | selected |
| 100 | temperature_max_c | 0 | selected |
| 100 | temperature_min_c | 0 | selected |
| 100 | internal_resistance_ohm | 0 | selected |
| 100 | charge_time_s | 0 | selected |

跳过字段：

| N | field | missing_rate | reason |
| --- | --- | --- | --- |
| 10 | voltage_mean_v | 1 | missing_rate>80% |
| 10 | voltage_min_v | 1 | missing_rate>80% |
| 10 | voltage_max_v | 1 | missing_rate>80% |
| 10 | current_mean_a | 1 | missing_rate>80% |
| 10 | current_abs_mean_a | 1 | missing_rate>80% |
| 30 | voltage_mean_v | 1 | missing_rate>80% |
| 30 | voltage_min_v | 1 | missing_rate>80% |
| 30 | voltage_max_v | 1 | missing_rate>80% |
| 30 | current_mean_a | 1 | missing_rate>80% |
| 30 | current_abs_mean_a | 1 | missing_rate>80% |
| 50 | voltage_mean_v | 1 | missing_rate>80% |
| 50 | voltage_min_v | 1 | missing_rate>80% |
| 50 | voltage_max_v | 1 | missing_rate>80% |
| 50 | current_mean_a | 1 | missing_rate>80% |
| 50 | current_abs_mean_a | 1 | missing_rate>80% |
| 100 | voltage_mean_v | 1 | missing_rate>80% |
| 100 | voltage_min_v | 1 | missing_rate>80% |
| 100 | voltage_max_v | 1 | missing_rate>80% |
| 100 | current_mean_a | 1 | missing_rate>80% |
| 100 | current_abs_mean_a | 1 | missing_rate>80% |

## 模型效果对比

测试集指标按 MAE 从低到高排序：

| N | model | n_samples | n_features | MAE | RMSE | MAPE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | XGBoost | 25 | 63 | 145.05 | 226.198 | 23.1839 | 0.607766 |
| 30 | RandomForest | 25 | 63 | 148.07 | 256.012 | 21.7277 | 0.497555 |
| 30 | XGBoost | 25 | 63 | 153.07 | 258.909 | 22.6435 | 0.486119 |
| 10 | XGBoost | 25 | 63 | 155.267 | 228.615 | 22.7099 | 0.599338 |
| 100 | RandomForest | 25 | 63 | 156.919 | 238.566 | 24.3514 | 0.5637 |
| 10 | RandomForest | 25 | 63 | 161.096 | 227.109 | 24.2472 | 0.604599 |
| 50 | XGBoost | 25 | 63 | 164.703 | 238.23 | 25.1496 | 0.564927 |
| 50 | RandomForest | 25 | 63 | 177.874 | 252.512 | 26.7308 | 0.511201 |
| 30 | ElasticNet | 25 | 63 | 188.557 | 291.072 | 49.5338 | 0.350514 |
| 100 | ElasticNet | 25 | 63 | 210.919 | 299.931 | 50.1746 | 0.310379 |
| 10 | ElasticNet | 25 | 63 | 225.65 | 307.356 | 45.1258 | 0.27581 |
| 50 | ElasticNet | 25 | 63 | 243.449 | 346.641 | 57.9698 | 0.0788566 |

各模型自身最佳测试结果：

| model | N | MAE | RMSE | MAPE | R2 |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 100 | 145.05 | 226.198 | 23.1839 | 0.607766 |
| RandomForest | 30 | 148.07 | 256.012 | 21.7277 | 0.497555 |
| ElasticNet | 30 | 188.557 | 291.072 | 49.5338 | 0.350514 |


XGBoost 状态：当前环境可用，已纳入本次对比。


## 最佳模型

按测试集 MAE 选择，本次最佳组合为：

- 模型：`XGBoost`
- 早期循环窗口：`N=100`
- 测试集 MAE：`145.0499`
- 测试集 RMSE：`226.1979`
- 测试集 MAPE：`23.1839%`
- 测试集 R2：`0.6078`

## 前 10/30/50/100 圈影响

每个 N 下取最佳模型后的测试误差：

| N | best_MAE | best_RMSE | best_MAPE |
| --- | --- | --- | --- |
| 10 | 155.267 | 227.109 | 22.7099 |
| 30 | 148.07 | 256.012 | 21.7277 |
| 50 | 164.703 | 238.23 | 25.1496 |
| 100 | 145.05 | 226.198 | 23.1839 |

总体上，N 越大通常提供更完整的早期退化轨迹，但也更接近寿命过程本身。后续论文实验应同时报告 N=10、30、50、100，避免只展示最有利窗口。

## GroupKFold 交叉验证

按 `battery_id` 分组做 GroupKFold，默认 5 折。当前每个 N 的样本数足够 5 折。

| N | model | folds | MAE_mean | MAE_std | RMSE_mean | MAPE_mean | R2_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | XGBoost | 5 | 112.81 | 29.8141 | 182.955 | 14.34 | 0.637438 |
| 100 | RandomForest | 5 | 115.275 | 23.521 | 180.138 | 14.7414 | 0.659454 |
| 100 | XGBoost | 5 | 115.342 | 20.6857 | 181.29 | 14.6186 | 0.654286 |
| 30 | RandomForest | 5 | 118.378 | 29.7707 | 185.212 | 14.7766 | 0.638698 |
| 50 | XGBoost | 5 | 118.573 | 31.2579 | 183.155 | 14.896 | 0.64842 |
| 50 | RandomForest | 5 | 121.882 | 31.596 | 183.232 | 15.5574 | 0.64744 |
| 10 | XGBoost | 5 | 126.482 | 35.337 | 195.457 | 16.4133 | 0.596588 |
| 10 | RandomForest | 5 | 135.902 | 32.8535 | 200.054 | 17.6562 | 0.580646 |
| 30 | ElasticNet | 5 | 143.278 | 23.6031 | 223.123 | 24.112 | 0.457032 |
| 10 | ElasticNet | 5 | 147.271 | 20.4529 | 212.212 | 20.5175 | 0.506383 |
| 100 | ElasticNet | 5 | 151.819 | 35.4954 | 259.635 | 24.4382 | 0.175922 |
| 50 | ElasticNet | 5 | 172.644 | 11.9474 | 306.096 | 30.2138 | -0.0889029 |

## 过拟合风险

训练集和测试集误差差距如下，若训练误差显著低于测试误差，说明树模型可能存在过拟合风险。

| N | model | MAE_train | MAE_test | MAE_gap_test_minus_train | R2_train | R2_test |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | XGBoost | 2.24241 | 164.703 | 162.46 | 0.999911 | 0.564927 |
| 10 | XGBoost | 2.31508 | 155.267 | 152.952 | 0.999899 | 0.599338 |
| 50 | ElasticNet | 91.4861 | 243.449 | 151.963 | 0.781759 | 0.0788566 |
| 30 | XGBoost | 1.81354 | 153.07 | 151.256 | 0.99994 | 0.486119 |
| 100 | XGBoost | 1.87033 | 145.05 | 143.18 | 0.999941 | 0.607766 |
| 50 | RandomForest | 48.8059 | 177.874 | 129.068 | 0.926784 | 0.511201 |
| 10 | ElasticNet | 97.5011 | 225.65 | 128.149 | 0.76636 | 0.27581 |
| 100 | ElasticNet | 85.1337 | 210.919 | 125.785 | 0.814871 | 0.310379 |
| 100 | RandomForest | 44.6386 | 156.919 | 112.281 | 0.939228 | 0.5637 |
| 10 | RandomForest | 54.3427 | 161.096 | 106.753 | 0.910701 | 0.604599 |
| 30 | RandomForest | 50.3939 | 148.07 | 97.6765 | 0.920913 | 0.497555 |
| 30 | ElasticNet | 94.1846 | 188.557 | 94.3724 | 0.760569 | 0.350514 |

## 可能重要的特征

下表给出最佳模型对应的前若干特征重要性。树模型使用 `feature_importances_`，Elastic Net 使用标准化后系数绝对值。

| feature | importance |
| --- | --- |
| soh__min | 0.284038 |
| temperature_max_c__delta | 0.0741675 |
| capacity_ah__std | 0.0657989 |
| internal_resistance_ohm__std | 0.0649073 |
| soh__std | 0.0587242 |
| temperature_max_c__std | 0.0481319 |
| soh__first_value | 0.0385168 |
| charge_time_s__mean | 0.0334098 |
| internal_resistance_ohm__first_value | 0.0310663 |
| temperature_min_c__last_value | 0.0253966 |
| charge_time_s__min | 0.0218132 |
| internal_resistance_ohm__slope | 0.0207176 |
| temperature_max_c__slope | 0.0195963 |
| internal_resistance_ohm__delta | 0.0155726 |
| charge_time_s__first_value | 0.0148223 |

从特征命名看，容量/SOH 的早期水平、变化量、斜率，以及内阻和充电时间相关统计量，是后续特征工程最值得优先关注的方向。

## 推荐源域 baseline

第一版建议把 `XGBoost` 在 `N=100` 下的结果作为源域 cycle life baseline，同时保留 Elastic Net 作为可解释线性基线，Random Forest / XGBoost 作为非线性树模型基线。

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
