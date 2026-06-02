# 循环级数据质量可视化检查报告

本报告由 `src/preprocessing/visualize_cycle_summary.py` 自动生成。脚本只读取：

- `data_processed/unified_tables/unified_cycle_summary.parquet`
- `data_processed/unified_tables/nasa_eis_features.csv`

不修改原始数据，不删除异常值，也不进行模型训练。

## 输出图表

- `figures\data_quality\MIT_Severson_capacity_curves.png`
- `figures\data_quality\MIT_Severson_soh_curves.png`
- `figures\data_quality\CALCE_capacity_curves.png`
- `figures\data_quality\CALCE_soh_curves.png`
- `figures\data_quality\NASA_capacity_curves.png`
- `figures\data_quality\NASA_soh_curves.png`
- `figures\data_quality\CALCE_CS2_capacity_curves.png`
- `figures\data_quality\NASA_core_capacity_curves.png`
- `figures\data_quality\MIT_Severson_cycle_life_histogram.png`
- `figures\data_quality\cycle_life_missing_by_dataset.png`
- `figures\data_quality\capacity_quality_flags_by_dataset.png`

## 数据集总体情况

| dataset | battery_count | cycle_count | valid_capacity_rows | valid_soh_rows | soh_min | soh_max |
| --- | --- | --- | --- | --- | --- | --- |
| CALCE | 4 | 4037 | 4037 | 2887 | 0.751281 | 1.13424 |
| MIT_Severson | 140 | 114738 | 114692 | 114688 | 0.763992 | 1.04961 |
| NASA | 34 | 2794 | 2750 | 2379 | 0.750203 | 1.29941 |

## cycle_life 缺失统计

| dataset | missing | total | available | missing_rate |
| --- | --- | --- | --- | --- |
| CALCE | 0 | 4 | 4 | 0 |
| MIT_Severson | 97 | 140 | 43 | 0.692857 |
| NASA | 17 | 34 | 17 | 0.5 |

## 容量质量标记统计

| dataset | capacity_invalid_flag | capacity_label_excluded_flag | capacity_rebound_flag |
| --- | --- | --- | --- |
| CALCE | 0 | 1150 | 71 |
| MIT_Severson | 46 | 4 | 12 |
| NASA | 19 | 371 | 48 |

## NASA EIS 表检查

- EIS 特征行数：1956
- `re_ohm` 缺失率：0.000
- `rct_ohm` 缺失率：0.000

## MIT-Severson 140 个 battery 检查

### battery_id 列表

`2017-05-12_batchdata_updated_struct_errorcorrect_cell_001, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_002, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_003, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_004, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_005, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_006, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_007, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_008, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_009, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_010, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_011, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_012, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_013, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_014, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_015, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_016, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_017, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_018, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_019, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_020, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_021, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_022, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_023, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_024, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_025, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_026, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_027, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_028, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_029, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_030, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_031, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_032, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_033, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_034, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_035, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_036, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_037, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_038, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_039, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_040, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_041, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_042, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_043, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_044, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_045, 2017-05-12_batchdata_updated_struct_errorcorrect_cell_046, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_001, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_002, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_003, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_004, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_005, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_006, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_007, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_008, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_009, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_010, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_011, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_012, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_013, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_014, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_015, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_016, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_017, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_018, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_019, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_020, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_021, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_022, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_023, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_024, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_025, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_026, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_027, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_028, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_029, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_030, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_031, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_032, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_033, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_034, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_035, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_036, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_037, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_038, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_039, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_040, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_041, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_042, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_043, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_044, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_045, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_046, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_047, 2017-06-30_batchdata_updated_struct_errorcorrect_cell_048, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_001, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_002, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_003, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_004, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_005, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_006, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_007, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_008, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_009, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_010, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_011, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_012, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_013, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_014, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_015, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_016, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_017, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_018, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_019, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_020, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_021, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_022, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_023, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_024, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_025, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_026, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_027, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_028, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_029, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_030, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_031, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_032, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_033, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_034, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_035, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_036, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_037, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_038, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_039, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_040, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_041, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_042, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_043, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_044, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_045, 2018-04-12_batchdata_updated_struct_errorcorrect_cell_046`

### 每个 batch 的 battery 数

| batch | source_file | battery_count |
| --- | --- | --- |
| batch1 | 2017-05-12_batchdata_updated_struct_errorcorrect.mat | 46 |
| batch2 | 2017-06-30_batchdata_updated_struct_errorcorrect.mat | 48 |
| batch3 | 2018-04-12_batchdata_updated_struct_errorcorrect.mat | 46 |

### 重复 battery_id 检查

无

### 官方应剔除/合并的异常电池候选

说明：该表根据官方 `LoadData.m` 的公开逻辑推断，包括 Batch 1 未完成电池、Batch 2 continuation、Batch 3 problem channel、Batch 3 final capacity 过高以及 noisy Batch 8 位置。当前 summary 保留的是原始 140 个通道/电池记录，因此这些候选只做标记，不自动删除。

候选数量：16

| source_file | batch | cell_index | battery_id | reason |
| --- | --- | --- | --- | --- |
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat | batch1 | 9 | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_009 | official optional removal: unfinished Batch 1 battery |
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat | batch1 | 11 | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_011 | official optional removal: unfinished Batch 1 battery |
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat | batch1 | 13 | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_013 | official optional removal: unfinished Batch 1 battery |
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat | batch1 | 14 | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_014 | official optional removal: unfinished Batch 1 battery |
| 2017-05-12_batchdata_updated_struct_errorcorrect.mat | batch1 | 23 | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_023 | official optional removal: unfinished Batch 1 battery |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 8 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_008 | official merge/removal: continuation appended to Batch 1 cells 1-5 |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 9 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_009 | official merge/removal: continuation appended to Batch 1 cells 1-5 |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 10 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_010 | official merge/removal: continuation appended to Batch 1 cells 1-5 |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 16 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_016 | official merge/removal: continuation appended to Batch 1 cells 1-5 |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 17 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_017 | official merge/removal: continuation appended to Batch 1 cells 1-5 |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | batch3 | 3 | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_003 | official removal: noisy Batch 8 battery position 3 after prior removals |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | batch3 | 24 | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_024 | official removal: Batch 3 final QDischarge > 0.885 Ah |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | batch3 | 33 | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_033 | official removal: Batch 3 final QDischarge > 0.885 Ah |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | batch3 | 38 | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_038 | official removal: problematic channel 46 in Batch 3 |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | batch3 | 43 | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_043 | official removal: noisy Batch 8 battery position 40 after prior removals |
| 2018-04-12_batchdata_updated_struct_errorcorrect.mat | batch3 | 44 | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_044 | official removal: noisy Batch 8 battery position 41 after prior removals |

## MIT cycle_life 来源检查

当前 `unified_cycle_summary` 中的 MIT `cycle_life` 是由本项目脚本按 `SOH <= 0.8` 重新计算得到；MIT 原始 `.mat` 文件中也存在 `cycle_life` 字段。两者对比如下：

| both_available_count | exact_match_count | mean_difference | max_abs_difference |
| --- | --- | --- | --- |
| 43 | 0 | 12.6279 | 29 |

差异样例：

| source_file | batch | cell_index | battery_id | mit_raw_cycle_life | recomputed_soh80_cycle_life | difference_recomputed_minus_raw | both_available | exact_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 48 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_048 | 713 | 742 | 29 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 20 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_020 | 461 | 483 | 22 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 19 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_019 | 487 | 507 | 20 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 7 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_007 | 511 | 529 | 18 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 18 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_018 | 494 | 510 | 16 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 11 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_011 | 561 | 577 | 16 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 29 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_029 | 509 | 525 | 16 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 25 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_025 | 495 | 510 | 15 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 28 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_028 | 468 | 483 | 15 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 13 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_013 | 458 | 473 | 15 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 31 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_031 | 481 | 495 | 14 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 35 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_035 | 499 | 513 | 14 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 24 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_024 | 527 | 541 | 14 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 26 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_026 | 461 | 474 | 13 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 46 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_046 | 487 | 500 | 13 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 4 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_004 | 335 | 348 | 13 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 6 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_006 | 480 | 493 | 13 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 12 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_012 | 477 | 489 | 12 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 1 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_001 | 300 | 312 | 12 | True | False |
| 2017-06-30_batchdata_updated_struct_errorcorrect.mat | batch2 | 36 | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_036 | 463 | 475 | 12 | True | False |

## 未达到 EOL 的电池

未达到 EOL 的电池数量：114

| dataset | battery_id | cycle_life |
| --- | --- | --- |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_001 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_002 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_003 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_004 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_005 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_006 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_007 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_008 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_009 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_010 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_011 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_012 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_013 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_014 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_015 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_016 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_017 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_018 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_019 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_020 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_021 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_022 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_023 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_024 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_025 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_026 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_027 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_028 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_029 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_030 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_031 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_032 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_033 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_034 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_035 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_036 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_037 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_038 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_039 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_040 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_041 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_042 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_043 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_044 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_045 | NaN |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_046 | NaN |
| MIT_Severson | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_008 | NaN |
| MIT_Severson | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_009 | NaN |
| MIT_Severson | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_010 | NaN |
| MIT_Severson | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_016 | NaN |
| MIT_Severson | 2017-06-30_batchdata_updated_struct_errorcorrect_cell_017 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_001 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_002 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_003 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_004 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_005 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_006 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_007 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_008 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_009 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_010 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_011 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_012 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_013 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_014 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_015 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_016 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_017 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_018 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_019 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_020 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_021 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_022 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_023 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_024 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_025 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_026 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_027 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_028 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_029 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_030 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_031 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_032 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_033 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_034 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_035 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_036 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_037 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_038 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_039 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_040 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_041 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_042 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_043 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_044 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_045 | NaN |
| MIT_Severson | 2018-04-12_batchdata_updated_struct_errorcorrect_cell_046 | NaN |
| NASA | B0025 | NaN |
| NASA | B0027 | NaN |
| NASA | B0028 | NaN |
| NASA | B0029 | NaN |
| NASA | B0030 | NaN |
| NASA | B0031 | NaN |
| NASA | B0032 | NaN |
| NASA | B0033 | NaN |
| NASA | B0034 | NaN |
| NASA | B0036 | NaN |
| NASA | B0038 | NaN |
| NASA | B0040 | NaN |
| NASA | B0050 | NaN |
| NASA | B0052 | NaN |
| NASA | B0053 | NaN |
| NASA | B0055 | NaN |
| NASA | B0056 | NaN |

## 是否适合进入 baseline 建模

- MIT-Severson：适合优先进入 baseline。其循环级容量、温度和内阻字段较完整，但当前表保留了 140 个原始 battery，未完全套用官方 124 个电池的合并/剔除流程；建模前建议根据任务选择是否使用官方过滤清单。
- CALCE：适合做容量曲线与电压/电流工程特征验证。由于多个 Excel 的 Cycle_Index 会重启，后续建模必须使用 global_cycle_index，并建议过滤 capacity_label_excluded_flag=True 的循环。
- NASA：适合做 SOH/RUL 与 EIS 辅助特征研究。普通循环和 impedance 已分表，后续可按 nearest_discharge_global_cycle 合并 Re/Rct。
- 整体结论：可以进入 baseline 建模准备阶段，但第一版 baseline 应优先使用 MIT-Severson 和 NASA 中标签较完整、未标记容量异常的样本；CALCE 更适合作为跨文件对齐和特征工程验证集。
