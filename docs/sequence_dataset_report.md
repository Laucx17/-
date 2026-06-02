# DeepAR / TAM-DeepAR-EN 序列数据集报告

本报告由 `src/preprocessing/build_sequence_dataset.py` 自动生成。本阶段只构建序列数据集，不训练 DeepAR，不训练 TAM-DeepAR-EN，不修改原始数据，也不修改 modeling-ready 数据集。

## 输入文件

- `data_processed/source_domain/modeling_ready_cycle_life_dataset.parquet`
- `data_processed/unified_tables/unified_cycle_summary.parquet`
- `data_processed/unified_tables/nasa_eis_features.csv`

输入表规模：

- modeling-ready 表：`103123` 行被选入序列输出；原始 unified summary 表：`121569` 行；
- NASA EIS 表：`1956` 行。

## 输出文件

- `data_processed/sequences/calce_soh_sequences.parquet`
- `data_processed/sequences/nasa_soh_sequences.parquet`
- `data_processed/sequences/mit_soh_sequences.parquet`
- `data_processed/sequences/sequence_metadata.csv`
- `figures/sequences/calce_soh_sequences.png`
- `figures/sequences/nasa_core_soh_sequences.png`
- `figures/sequences/mit_sample_soh_sequences.png`
- `figures/sequences/sequence_length_distribution.png`
- `figures/sequences/feature_missing_rate.png`

## 序列数据概况

每个 `battery_id` 是一条序列，每个 `global_cycle_index` 是一个时间步；脚本按电池内循环顺序生成从 0 开始的连续 `time_idx`。`target_soh = soh`。`rul` 只作为后续 RUL 评价标签，不应作为 SOH 预测输入。

| dataset | battery_count | sequence_length_min | sequence_length_max | sequence_length_mean | soh_min | soh_max | time_idx_discontinuous_count | soh_anomaly_count | capacity_nonpositive_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CALCE | 4 | 936 | 1082 | 1009.25 | 0.751281 | 1.13424 | 0 | 0 | 0 |
| MIT_Severson | 124 | 170 | 1934 | 783.129 | 0.763992 | 1.00938 | 0 | 0 | 0 |
| NASA | 23 | 25 | 197 | 86 | 0.750203 | 1.1629 | 0 | 0 | 0 |

## 每个数据集的处理方式

### CALCE

CALCE 保留 `CS2_35`、`CS2_36`、`CS2_37`、`CS2_38` 四条序列。每个电池前 70% 时间步标记为 `train`，后 30% 标记为 `test`。另外在 metadata 中给出 battery-level 外部测试建议：`CS2_35/CS2_36` 作为训练，`CS2_37/CS2_38` 作为外部测试。

CALCE 分割点：

| battery_id | sequence_length | train_steps | test_steps | split_first_test_global_cycle | battery_level_split_suggestion |
| --- | --- | --- | --- | --- | --- |
| CS2_35 | 936 | 655 | 281 | 656 | battery_level_train |
| CS2_36 | 976 | 683 | 293 | 684 | battery_level_train |
| CS2_37 | 1043 | 730 | 313 | 731 | battery_level_external_test |
| CS2_38 | 1082 | 757 | 325 | 758 | battery_level_external_test |

### NASA

NASA 使用两类标记：`is_nasa_core=True` 和 `is_nasa_all_clean=True`，二者取并集后构建序列。NASA core 电池 `B0005/B0006/B0007/B0018` 单独统计如下：

| battery_id | sequence_length | train_steps | test_steps | soh_min | soh_max | eis_re_missing_rate | eis_rct_missing_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0005 | 168 | 117 | 51 | 0.750851 | 1.00565 | 0.107143 | 0.107143 |
| B0006 | 168 | 117 | 51 | 0.751758 | 1.14937 | 0.107143 | 0.107143 |
| B0007 | 168 | 117 | 51 | 0.751719 | 1.00368 | 0.107143 | 0.107143 |
| B0018 | 132 | 92 | 40 | 0.750203 | 1.01331 | 0 | 0 |

### MIT-Severson

MIT 暂时只构建 SOH 序列表，不作为 DeepAR 第一版主训练。当前 MIT summary-only 表有容量、SOH、温度和内阻，但缺少 detailed cycles 中的电压/电流曲线信息，也缺少 CCCT、CVCT、ADV、ICV。因此更适合作为后续大规模预训练候选，或在补充 detailed-cycle 特征后再用于 DeepAR/TAM-DeepAR-EN 主训练。

## 字段缺失率

下表为关键时序特征缺失率，单位为百分比。

| dataset | capacity_ah | soh | rul | ccct_s | cvct_s | adv_v | icv_v | charge_time_s | discharge_time_s | internal_resistance_ohm | temperature_mean_c | temperature_max_c | temperature_min_c | voltage_mean_v | voltage_min_v | voltage_max_v | current_mean_a | current_abs_mean_a | eis_re_ohm | eis_rct_ohm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CALCE | 0 | 28.4865 | 100 | 0.27248 | 0.0743126 | 0.396334 | 0 | 0 | 0.396334 | 100 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | NaN | NaN |
| NASA | 1.71891 | 7.07786 | 46.6633 | 0.758342 | 1.56724 | 0 | 0.758342 | 0.758342 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4.34783 | 4.34783 |
| MIT_Severson | 0.042221 | 0.0463402 | 0 | 100 | 100 | 100 | 100 | 0 | 100 | 0 | 0 | 0 | 0 | 100 | 100 | 100 | 100 | 100 | NaN | NaN |

CCCT、CVCT、ADV、ICV 可用性：

| dataset | ccct_s | cvct_s | adv_v | icv_v |
| --- | --- | --- | --- | --- |
| CALCE | 0.27248 | 0.0743126 | 0.396334 | 0 |
| NASA | 0.758342 | 1.56724 | 0 | 0.758342 |
| MIT_Severson | 100 | 100 | 100 | 100 |

## NASA EIS 合并方式

`nasa_eis_features.csv` 中的 Re、Rct 通过 `battery_id + nearest_discharge_global_cycle` 合并到 NASA 序列。若某个普通循环没有精确对应 EIS，则使用同一电池此前最近一次 EIS 特征向前填充；首次 EIS 之前的普通循环仍保留 NaN，不反向填充。

EIS 对齐情况：

| eis_alignment | row_count |
| --- | --- |
| exact_nearest_cycle | 805 |
| forward_fill_from_previous_eis | 1087 |
| missing_before_first_eis | 86 |

## 每个 battery 的 train/test 分割点

完整分割点已保存到 `data_processed/sequences/sequence_metadata.csv`。报告中仅显示前 40 行：

| dataset | battery_id | sequence_length | train_steps | test_steps | split_first_test_time_idx | split_first_test_global_cycle | battery_level_split_suggestion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CALCE | CS2_35 | 936 | 655 | 281 | 655 | 656 | battery_level_train |
| CALCE | CS2_36 | 976 | 683 | 293 | 683 | 684 | battery_level_train |
| CALCE | CS2_37 | 1043 | 730 | 313 | 730 | 731 | battery_level_external_test |
| CALCE | CS2_38 | 1082 | 757 | 325 | 757 | 758 | battery_level_external_test |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_001 | 1189 | 832 | 357 | 832 | 833 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_002 | 1178 | 824 | 354 | 824 | 825 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_003 | 1176 | 823 | 353 | 823 | 824 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_004 | 1225 | 857 | 368 | 857 | 858 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_005 | 1226 | 858 | 368 | 858 | 859 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_006 | 1073 | 751 | 322 | 751 | 752 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_007 | 635 | 444 | 191 | 444 | 445 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_008 | 869 | 608 | 261 | 608 | 609 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_010 | 1053 | 737 | 316 | 737 | 738 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_012 | 787 | 550 | 237 | 550 | 551 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_015 | 879 | 615 | 264 | 615 | 616 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_016 | 718 | 502 | 216 | 502 | 503 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_017 | 861 | 602 | 259 | 602 | 603 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_018 | 856 | 599 | 257 | 599 | 600 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_019 | 690 | 482 | 208 | 482 | 483 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_020 | 787 | 550 | 237 | 550 | 551 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_021 | 533 | 373 | 160 | 373 | 374 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_022 | 558 | 390 | 168 | 390 | 391 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_024 | 1013 | 709 | 304 | 709 | 710 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_025 | 1016 | 711 | 305 | 711 | 712 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_026 | 853 | 597 | 256 | 597 | 598 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_027 | 869 | 608 | 261 | 608 | 609 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_028 | 841 | 588 | 253 | 588 | 589 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_029 | 859 | 601 | 258 | 601 | 602 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_030 | 916 | 641 | 275 | 641 | 642 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_031 | 708 | 495 | 213 | 495 | 496 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_032 | 875 | 612 | 263 | 612 | 613 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_033 | 730 | 510 | 220 | 510 | 511 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_034 | 756 | 529 | 227 | 529 | 530 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_035 | 741 | 518 | 223 | 518 | 519 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_036 | 702 | 491 | 211 | 491 | 492 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_037 | 703 | 492 | 211 | 492 | 493 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_038 | 647 | 452 | 195 | 452 | 453 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_039 | 616 | 431 | 185 | 431 | 432 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_040 | 624 | 436 | 188 | 436 | 437 | pretraining_candidate |
| MIT_Severson | 2017-05-12_batchdata_updated_struct_errorcorrect_cell_041 | 965 | 675 | 290 | 675 | 676 | pretraining_candidate |

_仅显示前 40 行，共 151 行。_

## 数据质量检查

- `time_idx` 不连续的序列数量见概况表中的 `time_idx_discontinuous_count`。
- `SOH <= 0` 或 `SOH > 1.2` 的异常点数量见 `soh_anomaly_count`。
- `capacity_ah <= 0` 的异常点数量见 `capacity_nonpositive_count`。
- 异常点没有被删除，只在序列和 metadata 中保留质量标记，供后续训练前过滤或加权处理。

## DeepAR 第一版建议

第一版 DeepAR/SOH 序列预测建议优先使用 NASA core + NASA all-clean 的并集。原因是 NASA 序列数量多于 CALCE，且普通循环和 EIS 特征已经可以按时间步合并。CALCE 只有 4 条电池序列，更适合做 TAM-DeepAR-EN 的间接健康因子复现和小样本验证。

## TAM-DeepAR-EN 特征建议

优先使用以下特征：

- 容量与 SOH：`capacity_ah`、`soh`
- 间接健康因子：`ccct_s`、`cvct_s`、`adv_v`、`icv_v`
- 工况与环境：`temperature_mean_c`、`temperature_max_c`、`temperature_min_c`
- 充放电时间：`charge_time_s`、`discharge_time_s`
- NASA 阻抗特征：`eis_re_ohm`、`eis_rct_ohm`

注意：`rul` 不应作为 SOH 预测输入，只能作为后续 RUL 评价或联合预测标签。

## 下一步如何训练 DeepAR

下一步建议创建独立训练脚本，读取本阶段 parquet：

1. 选择数据集和特征列；
2. 按 `battery_id` 构造多条相关时间序列；
3. 使用 `split` 字段做每条电池内部 70/30 时间切分；
4. 先训练普通 DeepAR 预测 `target_soh`；
5. 再加入时间注意力 TAM 和 Elastic Net/稀疏正则思想，形成 TAM-DeepAR-EN；
6. 最后用 `rul` 和 EOL 标签做 RUL 评价。

## 当前风险和注意事项

- CALCE 电池数量太少，不适合单独训练复杂深度模型。
- MIT 当前缺少详细曲线特征，暂不作为第一版 DeepAR 主训练。
- NASA EIS 是最近邻合并并前向填充，非严格同一循环同步测量，论文中需要说明。
- 部分电池存在 SOH 或容量异常点，本阶段不删除，后续训练前需要决定过滤、截断或鲁棒损失。
- 不同数据集的 `rul` 标签来源和 EOL 定义不完全一致，SOH 预测和 RUL 评价应分开描述。
