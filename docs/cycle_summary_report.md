# 循环级统一汇总表构建报告

本报告由 `src/preprocessing/build_cycle_summary.py` 自动生成。脚本只读取源域数据并构建循环级汇总表，不进行任何模型训练，也不修改 `data_raw/` 原始文件。

## 总体结果

- 总电池数量：178
- 总循环数量：121569
- 未达到 EOL 的电池数量：114
- NASA EIS/impedance 特征行数：1956

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

## MIT_Severson

- 电池数量：140
- 循环数量：114738
- 达到 EOL 的电池数量：43
- 未达到 EOL 的电池数量：97
- cycle_life 范围：158.0 到 742.0
- soh 范围：0.7639924519751652 到 1.0496122617466712
- capacity_ah <= 0 被标记为无效的循环数：46
- 容量过低或过高而未参与 SOH/RUL 标签计算的循环数：4
- 容量异常回升标记数：12
- global_cycle_index 连续性问题电池数：0

可用字段：

`dataset, battery_id, chemistry, cycle_index_raw, global_cycle_index, step_source, capacity_ah, charge_capacity_ah, discharge_capacity_ah, initial_capacity_ah, soh, cycle_life, rul, temperature_mean_c, temperature_max_c, temperature_min_c, charge_time_s, source_file, internal_resistance_ohm, capacity_invalid_flag, capacity_label_excluded_flag, capacity_rebound_flag`

完全缺失字段：

`voltage_mean_v, voltage_min_v, voltage_max_v, current_mean_a, current_abs_mean_a, discharge_time_s, ccct_s, cvct_s, adv_v, icv_v`

字段缺失率：

| 字段 | 缺失率 |
| --- | --- |
| cvct_s | 1.000 |
| ccct_s | 1.000 |
| discharge_time_s | 1.000 |
| current_abs_mean_a | 1.000 |
| current_mean_a | 1.000 |
| voltage_max_v | 1.000 |
| adv_v | 1.000 |
| icv_v | 1.000 |
| voltage_min_v | 1.000 |
| voltage_mean_v | 1.000 |
| rul | 0.812 |
| cycle_life | 0.812 |
| soh | 0.000 |
| capacity_ah | 0.000 |
| cycle_index_raw | 0.000 |
| dataset | 0.000 |
| temperature_min_c | 0.000 |
| temperature_max_c | 0.000 |
| temperature_mean_c | 0.000 |
| discharge_capacity_ah | 0.000 |
| initial_capacity_ah | 0.000 |
| global_cycle_index | 0.000 |
| step_source | 0.000 |
| charge_capacity_ah | 0.000 |
| battery_id | 0.000 |
| chemistry | 0.000 |
| charge_time_s | 0.000 |
| source_file | 0.000 |
| internal_resistance_ohm | 0.000 |
| capacity_invalid_flag | 0.000 |
| capacity_label_excluded_flag | 0.000 |
| capacity_rebound_flag | 0.000 |

未达到 EOL 的电池：

`2017-05-12_batchdata_updated_struct_errorcorrect_cell_001, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_002, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_003, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_004, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_005, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_006, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_007, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_008, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_009, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_010, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_011, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_012, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_013, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_014, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_015, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_016, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_017, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_018, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_019, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_020, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_021, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_022, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_023, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_024, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_025, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_026, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_027, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_028, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_029, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_030, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_031, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_032, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_033, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_034, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_035, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_036, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_037, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_038, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_039, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_040, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_041, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_042, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_043, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_044, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_045, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_046, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_008, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_009, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_010, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_016`

## CALCE

- 电池数量：4
- 循环数量：4037
- 达到 EOL 的电池数量：4
- 未达到 EOL 的电池数量：0
- cycle_life 范围：274.0 到 358.0
- soh 范围：0.7512813046635347 到 1.1342391935553255
- capacity_ah <= 0 被标记为无效的循环数：0
- 容量过低或过高而未参与 SOH/RUL 标签计算的循环数：1150
- 容量异常回升标记数：71
- global_cycle_index 连续性问题电池数：0

可用字段：

`dataset, battery_id, chemistry, cycle_index_raw, global_cycle_index, step_source, capacity_ah, charge_capacity_ah, discharge_capacity_ah, initial_capacity_ah, soh, cycle_life, rul, voltage_mean_v, voltage_min_v, voltage_max_v, current_mean_a, current_abs_mean_a, charge_time_s, discharge_time_s, ccct_s, cvct_s, adv_v, icv_v, source_file, capacity_invalid_flag, capacity_label_excluded_flag, capacity_rebound_flag`

完全缺失字段：

`temperature_mean_c, temperature_max_c, temperature_min_c, internal_resistance_ohm`

字段缺失率：

| 字段 | 缺失率 |
| --- | --- |
| temperature_min_c | 1.000 |
| temperature_mean_c | 1.000 |
| temperature_max_c | 1.000 |
| internal_resistance_ohm | 1.000 |
| soh | 0.285 |
| discharge_time_s | 0.004 |
| adv_v | 0.004 |
| ccct_s | 0.003 |
| cvct_s | 0.001 |
| charge_capacity_ah | 0.000 |
| step_source | 0.000 |
| capacity_ah | 0.000 |
| dataset | 0.000 |
| battery_id | 0.000 |
| chemistry | 0.000 |
| cycle_index_raw | 0.000 |
| rul | 0.000 |
| discharge_capacity_ah | 0.000 |
| initial_capacity_ah | 0.000 |
| cycle_life | 0.000 |
| global_cycle_index | 0.000 |
| current_mean_a | 0.000 |
| voltage_mean_v | 0.000 |
| voltage_min_v | 0.000 |
| current_abs_mean_a | 0.000 |
| charge_time_s | 0.000 |
| voltage_max_v | 0.000 |
| icv_v | 0.000 |
| source_file | 0.000 |
| capacity_invalid_flag | 0.000 |
| capacity_label_excluded_flag | 0.000 |
| capacity_rebound_flag | 0.000 |

未达到 EOL 的电池：

`无`

## NASA

- 电池数量：34
- 循环数量：2794
- 达到 EOL 的电池数量：17
- 未达到 EOL 的电池数量：17
- cycle_life 范围：6.0 到 125.0
- soh 范围：0.7502025333656073 到 1.2994053689779341
- capacity_ah <= 0 被标记为无效的循环数：19
- 容量过低或过高而未参与 SOH/RUL 标签计算的循环数：371
- 容量异常回升标记数：48
- global_cycle_index 连续性问题电池数：0

可用字段：

`dataset, battery_id, chemistry, cycle_index_raw, global_cycle_index, step_source, capacity_ah, discharge_capacity_ah, initial_capacity_ah, soh, cycle_life, rul, temperature_mean_c, temperature_max_c, temperature_min_c, voltage_mean_v, voltage_min_v, voltage_max_v, current_mean_a, current_abs_mean_a, charge_time_s, discharge_time_s, ccct_s, cvct_s, adv_v, icv_v, source_file, capacity_invalid_flag, capacity_label_excluded_flag, capacity_rebound_flag`

完全缺失字段：

`charge_capacity_ah, internal_resistance_ohm`

字段缺失率：

| 字段 | 缺失率 |
| --- | --- |
| charge_capacity_ah | 1.000 |
| internal_resistance_ohm | 1.000 |
| rul | 0.443 |
| cycle_life | 0.443 |
| soh | 0.149 |
| cvct_s | 0.053 |
| capacity_ah | 0.016 |
| discharge_capacity_ah | 0.009 |
| charge_time_s | 0.008 |
| ccct_s | 0.008 |
| icv_v | 0.008 |
| chemistry | 0.000 |
| global_cycle_index | 0.000 |
| step_source | 0.000 |
| dataset | 0.000 |
| battery_id | 0.000 |
| temperature_min_c | 0.000 |
| temperature_max_c | 0.000 |
| temperature_mean_c | 0.000 |
| initial_capacity_ah | 0.000 |
| cycle_index_raw | 0.000 |
| current_mean_a | 0.000 |
| voltage_mean_v | 0.000 |
| voltage_min_v | 0.000 |
| discharge_time_s | 0.000 |
| current_abs_mean_a | 0.000 |
| voltage_max_v | 0.000 |
| adv_v | 0.000 |
| source_file | 0.000 |
| capacity_invalid_flag | 0.000 |
| capacity_label_excluded_flag | 0.000 |
| capacity_rebound_flag | 0.000 |

未达到 EOL 的电池：

`B0025, B0027, B0028, B0029, B0030, B0031, B0032, B0033, B0034, B0036, B0038, B0040, B0050, B0052, B0053, B0055, B0056`



## 未达到 EOL 的电池汇总

`MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_001, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_002, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_003, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_004, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_005, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_006, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_007, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_008, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_009, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_010, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_011, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_012, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_013, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_014, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_015, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_016, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_017, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_018, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_019, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_020, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_021, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_022, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_023, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_024, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_025, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_026, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_027, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_028, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_029, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_030, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_031, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_032, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_033, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_034, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_035, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_036, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_037, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_038, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_039, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_040, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_041, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_042, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_043, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_044, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_045, MIT_Severson:2017-05-12_batchdata_updated_struct_errorcorrect_cell_046, MIT_Severson:2017-06-30_batchdata_updated_struct_errorcorrect_cell_008, MIT_Severson:2017-06-30_batchdata_updated_struct_errorcorrect_cell_009, MIT_Severson:2017-06-30_batchdata_updated_struct_errorcorrect_cell_010, MIT_Severson:2017-06-30_batchdata_updated_struct_errorcorrect_cell_016, MIT_Severson:2017-06-30_batchdata_updated_struct_errorcorrect_cell_017, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_001, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_002, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_003, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_004, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_005, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_006, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_007, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_008, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_009, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_010, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_011, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_012, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_013, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_014, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_015, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_016, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_017, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_018, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_019, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_020, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_021, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_022, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_023, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_024, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_025, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_026, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_027, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_028, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_029, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_030, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_031, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_032, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_033, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_034, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_035, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_036, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_037, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_038, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_039, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_040, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_041, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_042, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_043, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_044, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_045, MIT_Severson:2018-04-12_batchdata_updated_struct_errorcorrect_cell_046, NASA:B0025, NASA:B0027, NASA:B0028`

## 后续建模建议

1. 优先使用 MIT-Severson summary 表作为早期寿命预测和迁移学习源域基准，因为其 cycle-level 容量、温度和内阻字段较完整。
2. CALCE 可用于验证跨文件循环对齐和容量退化建模，但必须使用 `global_cycle_index`，不要直接把各 Excel 内部的 `Cycle_Index` 当作全局循环。
3. NASA 可用于 SOH/RUL 与 EIS 辅助特征研究，普通循环表与 `nasa_eis_features.csv` 建议在后续特征工程阶段按最近邻循环或时间顺序合并。
4. 温度、电压、电流曲线类模型优先从 CALCE 和 NASA 入手；MIT 第一版 summary 不包含完整电压/电流曲线。
