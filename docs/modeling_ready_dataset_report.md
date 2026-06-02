# Modeling-ready cycle life 数据集报告

本报告由 `src/preprocessing/build_modeling_ready_dataset.py` 自动生成。脚本读取：

- `data_processed/unified_tables/unified_cycle_summary.parquet`
- `data_processed/unified_tables/nasa_eis_features.csv`
- `docs/cycle_summary_visual_qc_report.md`：状态为 `存在`，其中的数据质量结论已转化为本脚本中的标记逻辑。

脚本不修改 `data_raw/`，不删除 `unified_cycle_summary`，也不进行任何模型训练。

## 输出文件

- `data_processed/source_domain/modeling_ready_cycle_life_dataset.parquet`
- `data_processed/source_domain/modeling_ready_cycle_life_dataset.csv`
- `docs/modeling_ready_dataset_report.md`

## 每个数据集保留情况

| dataset | battery_count | cycle_count | target_battery_count | target_cycle_rows | has_eol_batteries |
| --- | --- | --- | --- | --- | --- |
| CALCE | 4 | 4037 | 0 | 0 | 4 |
| MIT_Severson | 140 | 114738 | 138 | 110312 | 138 |
| NASA | 34 | 2794 | 17 | 1555 | 17 |

## 主要标记字段统计

下表统计的是循环行数量，不是电池数量。

| dataset | is_mit_raw_140 | is_mit_curated_124 | is_nasa_core | is_nasa_all_clean | is_calce_feature_validation | capacity_invalid_flag | capacity_label_excluded_flag | capacity_rebound_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CALCE | 0 | 0 | 0 | 0 | 4037 | 0 | 1150 | 71 |
| MIT_Severson | 114738 | 97108 | 0 | 0 | 0 | 46 | 4 | 12 |
| NASA | 0 | 0 | 636 | 1642 | 0 | 19 | 371 | 48 |

## MIT-Severson：从 raw 140 到 curated 124

当前 `unified_cycle_summary` 中 MIT-Severson 保留了 `140` 个原始通道/电池记录，因此 `MIT_raw_140=True` 与 `is_mit_raw_140=True` 用来标记这些原始 MIT 行。

第一版 baseline 默认使用 `MIT_curated_124=True` / `is_mit_curated_124=True` 的样本，共 `124` 个电池。该标记参考官方 `LoadData.m` 公开处理逻辑，对 `16` 个候选电池只做标记，不从本表物理删除。候选类型包括：

- Batch 1 未完成电池；
- Batch 2 continuation，应与 Batch 1 前 5 个电池合并或剔除；
- Batch 3 问题通道；
- Batch 3 末端容量过高样本；
- 官方 noisy Batch 8 位置对应样本。

官方剔除/合并候选如下：

| source_file | source_batch | cell_index | battery_id | official_exclusion_reason |
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

## MIT target_cycle_life 标签来源

MIT 的 `cycle_life_label_official` 直接从原始 `.mat` 文件中的 `cycle_life` 字段读取，不伪造标签。本次读取到非空官方标签 `138` 个。

`cycle_life_label_recomputed` 保留 `unified_cycle_summary` 中按 `SOH <= 0.8` 重新计算得到的标签。

`target_cycle_life` 的优先级为：

1. MIT 优先使用 `cycle_life_label_official`；
2. 如果 MIT 官方标签缺失，再回退到 `cycle_life_label_recomputed`；
3. NASA 使用 `cycle_life_label_recomputed`；
4. CALCE 暂不作为第一版 cycle_life baseline 主训练集，因此 `target_cycle_life` 默认保持为空。

MIT 电池级标签来源统计：

| target_label_source | battery_count |
| --- | --- |
| MIT_official_mat_cycle_life | 138 |
| not_for_cycle_life_baseline | 2 |

## NASA_core 与 NASA_all_clean

`is_nasa_core=True` 标记 NASA 经典核心电池：`B0005`、`B0006`、`B0007`、`B0018`，主要用于补充验证和与文献结果对齐。

`is_nasa_all_clean=True` 是数据质量导向标记：按电池统计 `capacity_invalid_flag` 与 `capacity_label_excluded_flag`，当坏标记比例不超过 `10%` 且有效 SOH 行数不少于 3 时标记为 clean。它不等同于 NASA_core，部分核心电池因为早期容量标签排除比例较高，不一定属于 all_clean。

NASA_core 电池：

| battery_id | source_batch | target_cycle_life | target_label_source | nasa_bad_flag_rate |
| --- | --- | --- | --- | --- |
| B0005 | 1. BatteryAgingARC-FY08Q4 | 102 | NASA_recomputed_soh80 | 0.238095 |
| B0006 | 1. BatteryAgingARC-FY08Q4 | 106 | NASA_recomputed_soh80 | 0.232143 |
| B0007 | 1. BatteryAgingARC-FY08Q4 | 125 | NASA_recomputed_soh80 | 0.0297619 |
| B0018 | 1. BatteryAgingARC-FY08Q4 | 79 | NASA_recomputed_soh80 | 0.0984848 |

NASA_all_clean 电池：

| battery_id | source_batch | target_cycle_life | target_label_source | nasa_bad_flag_rate |
| --- | --- | --- | --- | --- |
| B0007 | 1. BatteryAgingARC-FY08Q4 | 125 | NASA_recomputed_soh80 | 0.0297619 |
| B0018 | 1. BatteryAgingARC-FY08Q4 | 79 | NASA_recomputed_soh80 | 0.0984848 |
| B0025 | 2. BatteryAgingARC_25_26_27_28_P1 | NaN | not_for_cycle_life_baseline | 0 |
| B0026 | 2. BatteryAgingARC_25_26_27_28_P1 | 6 | NASA_recomputed_soh80 | 0 |
| B0027 | 2. BatteryAgingARC_25_26_27_28_P1 | NaN | not_for_cycle_life_baseline | 0 |
| B0028 | 2. BatteryAgingARC_25_26_27_28_P1 | NaN | not_for_cycle_life_baseline | 0 |
| B0029 | 3. BatteryAgingARC_25-44 | NaN | not_for_cycle_life_baseline | 0 |
| B0030 | 3. BatteryAgingARC_25-44 | NaN | not_for_cycle_life_baseline | 0 |
| B0031 | 3. BatteryAgingARC_25-44 | NaN | not_for_cycle_life_baseline | 0 |
| B0032 | 3. BatteryAgingARC_25-44 | NaN | not_for_cycle_life_baseline | 0 |
| B0034 | 3. BatteryAgingARC_25-44 | NaN | not_for_cycle_life_baseline | 0.00507614 |
| B0036 | 3. BatteryAgingARC_25-44 | NaN | not_for_cycle_life_baseline | 0.0101523 |
| B0045 | 4. BatteryAgingARC_45_46_47_48 | 47 | NASA_recomputed_soh80 | 0.0555556 |
| B0046 | 4. BatteryAgingARC_45_46_47_48 | 45 | NASA_recomputed_soh80 | 0.0416667 |
| B0047 | 4. BatteryAgingARC_45_46_47_48 | 45 | NASA_recomputed_soh80 | 0.0416667 |
| B0048 | 4. BatteryAgingARC_45_46_47_48 | 46 | NASA_recomputed_soh80 | 0.0416667 |
| B0052 | 5. BatteryAgingARC_49_50_51_52 | NaN | not_for_cycle_life_baseline | 0.04 |
| B0053 | 6. BatteryAgingARC_53_54_55_56 | NaN | not_for_cycle_life_baseline | 0.0178571 |
| B0054 | 6. BatteryAgingARC_53_54_55_56 | 78 | NASA_recomputed_soh80 | 0.0194175 |
| B0055 | 6. BatteryAgingARC_53_54_55_56 | NaN | not_for_cycle_life_baseline | 0.00980392 |
| B0056 | 6. BatteryAgingARC_53_54_55_56 | NaN | not_for_cycle_life_baseline | 0.00980392 |

NASA EIS 表未直接混入 cycle-level 主表。本阶段只检查其可用性，后续可按 `nearest_discharge_global_cycle` 合并 Re/Rct 等特征。

| eis_rows | eis_battery_count | re_ohm_missing_rate | rct_ohm_missing_rate |
| --- | --- | --- | --- |
| 1956 | 34 | 0 | 0 |

## CALCE 的定位

CALCE 四个电池保留在 modeling-ready 表中，并标记为 `is_calce_feature_validation=True`：

| battery_id | source_batch | recomputed_label | is_calce_feature_validation |
| --- | --- | --- | --- |
| CS2_35 | CS2_35 | 311 | True |
| CS2_36 | CS2_36 | 321 | True |
| CS2_37 | CS2_37 | 274 | True |
| CS2_38 | CS2_38 | 358 | True |

CALCE 暂不作为第一版 cycle_life baseline 主训练集，原因是：

- 只有 4 个 CS2 电池，电池数量太少；
- 多个 Excel 文件的 `Cycle_Index` 会重新从 1 开始，虽然已生成 `global_cycle_index`，但仍更适合先做特征工程验证；
- 本阶段更适合用它验证 CCCT、CVCT、ADV、ICV 等过程特征，以及 SOH 曲线预测，而不是直接训练 cycle life baseline。

## 第一版 baseline 推荐样本

| baseline_role | battery_count | cycle_count | target_available_rows |
| --- | --- | --- | --- |
| feature_validation_calce | 4 | 4037 | 0 |
| main_train_mit_curated_124 | 124 | 97108 | 97108 |
| supplement_validation_nasa_core | 4 | 636 | 636 |

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
