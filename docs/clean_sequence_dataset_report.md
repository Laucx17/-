# Clean sequence 数据集报告

本报告由 `src/preprocessing/build_clean_sequence_dataset.py` 自动生成。本阶段只生成 clean 版本，不覆盖原始序列 parquet，不修改 `data_raw/`，不训练模型。

## 输出文件

- `data_processed/sequences/calce_soh_sequences_clean.parquet`
- `data_processed/sequences/nasa_soh_sequences_clean.parquet`
- `data_processed/sequences/mit_soh_sequences_clean.parquet`
- `data_processed/sequences/sequence_metadata_clean.csv`
- `data_processed/sequences/calce_soh_reset_detection.csv`
- `figures/sequences_clean/calce_clean_soh_sequences.png`
- `figures/sequences_clean/nasa_clean_soh_sequences.png`
- `figures/sequences_clean/mit_clean_sample_soh_sequences.png`
- `figures/sequences_clean/clean_sequence_length_distribution.png`
- `figures/sequences_clean/clean_feature_missing_rate.png`
- `figures/sequences_clean/calce_soh_reset_detection.png`

## 清洗前后规模变化

| dataset | raw_batteries | clean_batteries | raw_rows | clean_rows | rows_removed | row_retention_rate |
| --- | --- | --- | --- | --- | --- | --- |
| CALCE | 4 | 4 | 4037 | 142 | 3895 | 0.0351746 |
| NASA | 23 | 23 | 1978 | 1838 | 140 | 0.929221 |
| MIT_Severson | 124 | 124 | 97108 | 97063 | 45 | 0.999537 |

通用过滤原因统计：

| dataset | filter_reason | row_count |
| --- | --- | --- |
| CALCE | target_soh_missing | 1150 |
| CALCE | soh_missing | 1150 |
| CALCE | capacity_missing | 0 |
| CALCE | capacity_invalid_flag | 0 |
| CALCE | capacity_label_excluded_flag | 1150 |
| CALCE | target_soh_out_of_range | 0 |
| NASA | target_soh_missing | 140 |
| NASA | soh_missing | 140 |
| NASA | capacity_missing | 34 |
| NASA | capacity_invalid_flag | 13 |
| NASA | capacity_label_excluded_flag | 106 |
| NASA | target_soh_out_of_range | 0 |
| MIT_Severson | target_soh_missing | 45 |
| MIT_Severson | soh_missing | 45 |
| MIT_Severson | capacity_missing | 41 |
| MIT_Severson | capacity_invalid_flag | 41 |
| MIT_Severson | capacity_label_excluded_flag | 4 |
| MIT_Severson | target_soh_out_of_range | 0 |

clean 后序列长度与质量概况：

| dataset | clean_batteries | clean_length_min | clean_length_max | clean_length_mean | discontinuous_count | soh_anomaly_count | capacity_nonpositive_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CALCE | 4 | 17 | 65 | 35.5 | 0 | 0 | 0 |
| MIT_Severson | 124 | 170 | 1934 | 782.766 | 0 | 0 | 0 |
| NASA | 23 | 3 | 196 | 79.913 | 0 | 0 | 0 |

## CALCE SOH reset 检测

CALCE reset 规则：在通用过滤后，若相邻有效 `target_soh` 增加超过 `0.05`，标记为 `soh_reset_flag`。第一版 clean 数据只保留第一次 reset 之前的连续退化段。

| battery_id | reset_detected | reset_threshold | first_reset_position_after_generic_clean | first_reset_original_time_idx | first_reset_global_cycle_index | target_soh_before_reset | target_soh_at_reset | reset_delta | rows_after_generic_clean | rows_retained_before_reset | rows_removed_by_reset_truncation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CS2_35 | True | 0.05 | 17 | 17 | 18 | 0.892309 | 1.00056 | 0.108255 | 662 | 17 | 645 |
| CS2_36 | True | 0.05 | 65 | 67 | 68 | 0.852724 | 0.977364 | 0.12464 | 657 | 65 | 592 |
| CS2_37 | True | 0.05 | 33 | 34 | 35 | 0.833834 | 0.991644 | 0.15781 | 778 | 33 | 745 |
| CS2_38 | True | 0.05 | 27 | 28 | 29 | 0.880078 | 1.00369 | 0.123612 | 790 | 27 | 763 |

CALCE clean 后每条序列情况：

| battery_id | original_rows | clean_rows | rows_removed | target_soh_min | target_soh_max | train_steps | test_steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CS2_35 | 936 | 17 | 919 | 0.892309 | 1.00641 | 11 | 6 |
| CS2_36 | 976 | 65 | 911 | 0.852724 | 1.00614 | 45 | 20 |
| CS2_37 | 1043 | 33 | 1010 | 0.833834 | 1.00759 | 23 | 10 |
| CS2_38 | 1082 | 27 | 1055 | 0.880078 | 1.00948 | 18 | 9 |

严格按第一次 reset 截断后，CALCE 不再包含后半段跳回 1.1 附近的片段；但四条序列都被截到较短的早期退化段，因此更适合做 TAM-DeepAR-EN 小样本复现的流程验证和特征可用性验证，不适合直接训练复杂深度模型。

## NASA clean 结果

NASA 保留 NASA core 与 NASA all-clean 并集；本阶段允许轻微 SOH 回升，并新增 `soh_rebound_flag`。当 `target_soh > 1.15` 时新增 `high_soh_flag`，但不直接删除。EIS 的 `eis_re_ohm/eis_rct_ohm` 保留已有的前向填充结果，不做反向填充。

| battery_id | clean_rows | soh_rebound_flag_count | high_soh_flag_count | eis_re_missing_rate | eis_rct_missing_rate |
| --- | --- | --- | --- | --- | --- |
| B0005 | 128 | 24 | 0 | 0.140625 | 0.140625 |
| B0006 | 129 | 20 | 0 | 0.139535 | 0.139535 |
| B0007 | 163 | 45 | 0 | 0.110429 | 0.110429 |
| B0018 | 119 | 17 | 0 | 0 | 0 |
| B0025 | 28 | 8 | 0 | 0.142857 | 0.142857 |
| B0026 | 28 | 9 | 0 | 0.142857 | 0.142857 |
| B0027 | 28 | 9 | 0 | 0.142857 | 0.142857 |
| B0028 | 28 | 8 | 0 | 0.142857 | 0.142857 |
| B0029 | 40 | 8 | 0 | 0.1 | 0.1 |
| B0030 | 40 | 9 | 0 | 0.1 | 0.1 |
| B0031 | 40 | 9 | 0 | 0.1 | 0.1 |
| B0032 | 40 | 9 | 0 | 0.1 | 0.1 |
| B0034 | 196 | 84 | 1 | 0 | 0 |
| B0036 | 195 | 64 | 0 | 0 | 0 |
| B0045 | 68 | 22 | 0 | 0 | 0 |
| B0046 | 69 | 12 | 1 | 0 | 0 |
| B0047 | 69 | 13 | 1 | 0 | 0 |
| B0048 | 69 | 16 | 0 | 0 | 0 |
| B0052 | 3 | 0 | 0 | 0 | 0 |
| B0053 | 55 | 27 | 0 | 0 | 0 |
| B0054 | 101 | 40 | 0 | 0 | 0 |
| B0055 | 101 | 39 | 0 | 0 | 0 |
| B0056 | 101 | 40 | 0 | 0 | 0 |

NASA clean 后仍然是第一版 DeepAR 的优先数据集，因为序列数量多于 CALCE，IHF 和 EIS 特征可用性较好。

## MIT clean 结果

MIT clean 后仍作为 `pretraining_candidate`。本阶段过滤了无效容量和 SOH 缺失点，保留 `capacity_ah`、`soh/target_soh`、温度、`internal_resistance_ohm`、`charge_time_s` 等 summary 特征；不补造 CCCT、CVCT、ADV、ICV。

MIT 仍暂不建议作为第一版 DeepAR 主训练集，因为其当前序列来自 summary-only 数据，电压/电流曲线与 IHF 特征缺失较多。

## clean 特征缺失率

下表单位为百分比。

| dataset | capacity_ah | target_soh | rul | ccct_s | cvct_s | adv_v | icv_v | charge_time_s | discharge_time_s | internal_resistance_ohm | temperature_mean_c | temperature_max_c | temperature_min_c | voltage_mean_v | voltage_min_v | voltage_max_v | current_mean_a | current_abs_mean_a | eis_re_ohm | eis_rct_ohm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CALCE | 0 | 0 | 100 | 1.40845 | 0.704225 | 0 | 0 | 0 | 0 | 100 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | 0 | NaN | NaN |
| NASA | 0 | 0 | 48.6942 | 0.435256 | 0.489663 | 0 | 0.435256 | 0.435256 | 0 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4.679 | 4.679 |
| MIT_Severson | 0 | 0 | 0 | 100 | 100 | 100 | 100 | 0 | 100 | 0 | 0 | 0 | 0 | 100 | 100 | 100 | 100 | 100 | NaN | NaN |

## DeepAR 训练特征建议

第一版 DeepAR 建议使用 NASA clean：

- 输入候选：`capacity_ah`、`target_soh` 的历史值、`ccct_s`、`cvct_s`、`adv_v`、`icv_v`、`charge_time_s`、`discharge_time_s`、温度、`eis_re_ohm`、`eis_rct_ohm`
- 目标：`target_soh`
- 禁止输入：`rul`
- RUL：只作为后续评价或联合预测标签保留

CALCE clean 建议用于 TAM-DeepAR-EN 小样本复现和 IHF 消融验证；MIT clean 建议作为后续大规模预训练候选。

## 当前风险

- CALCE 的第一次 reset 发生较早，严格截断后可用长度明显变短，模型训练价值有限。
- NASA 仍存在 SOH 回升和少量高 SOH 点，本阶段只标记不删除。
- EIS 特征仍是前向填充结果，非逐循环同步测量。
- MIT summary-only 缺少 CCCT/CVCT/ADV/ICV 和详细电压电流曲线。
- 不同数据集的 RUL/EOL 标签来源不完全一致，训练 SOH 模型和评价 RUL 时应分开处理。
