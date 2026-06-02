# Loader Output Validation Report

This report validates small real samples from the source-domain loaders. It does not modify or delete raw data and does not perform modeling.

Unified output fields:

`dataset, battery_id, chemistry, cycle_index, step_type, time_s, voltage_v, current_a, temperature_c, charge_capacity_ah, discharge_capacity_ah, capacity_ah, protocol, source_file`

## MIT-Severson Summary-Only Sample

### Required Columns

- All unified columns are present.

### Basic Summary

- Rows: 1189
- Battery count: 1
- Cycle index range: 1 to 1189

### Step Type Distribution

| step_type | rows | ratio |
| --- | --- | --- |
| cycle_summary | 1189 | 1 |

### Missing Rates

| field | missing_rate |
| --- | --- |
| dataset | 0 |
| battery_id | 0 |
| chemistry | 0 |
| cycle_index | 0 |
| step_type | 0 |
| time_s | 1 |
| voltage_v | 1 |
| current_a | 1 |
| temperature_c | 0 |
| charge_capacity_ah | 0 |
| discharge_capacity_ah | 0 |
| capacity_ah | 0 |
| protocol | 0 |
| source_file | 0 |

### Numeric Stats

| field | min | max | mean | non_null_rows |
| --- | --- | --- | --- | --- |
| voltage_v | NaN | NaN | NaN | 0 |
| current_a | NaN | NaN | NaN | 0 |
| temperature_c | 0 | 34.937 | 31.9553 | 1189 |
| capacity_ah | 0 | 1.53905 | 1.05327 | 1189 |

### Cycle Index Quality

| check | count |
| --- | --- |
| row-level duplicate (battery_id, cycle_index) | 0 |
| cross-source_file repeated cycle ids | 0 |
| battery ids with non-continuous integer cycles | 0 |

#### Notes

- No cross-file cycle repeats or integer cycle discontinuities were detected.

### Capacity Anomaly Check

| check | count | ratio_within_non_null_capacity |
| --- | --- | --- |
| capacity_ah <= 0 | 1 | 0.000841043 |

### Dataset-Specific Checks

- MIT summary-only output uses step_type=cycle_summary as expected.
- MIT summary-only non-null capacity_ah rows: 1189

### Suggestions

- Fields with >95% missing values should be checked before modeling: time_s, voltage_v, current_a.
- capacity_ah contains zero or negative values. This can be normal within raw curves, but cycle-level features should filter or aggregate these carefully.

## MIT-Severson Detailed Cycle Sample

### Required Columns

- All unified columns are present.

### Basic Summary

- Rows: 1295876
- Battery count: 1
- Cycle index range: 1 to 1189

### Step Type Distribution

| step_type | rows | ratio |
| --- | --- | --- |
| charge | 756003 | 0.583391 |
| discharge | 517037 | 0.398986 |
| rest | 22836 | 0.0176221 |

### Missing Rates

| field | missing_rate |
| --- | --- |
| dataset | 0 |
| battery_id | 0 |
| chemistry | 0 |
| cycle_index | 0 |
| step_type | 0 |
| time_s | 0 |
| voltage_v | 0 |
| current_a | 0 |
| temperature_c | 0 |
| charge_capacity_ah | 0 |
| discharge_capacity_ah | 0 |
| capacity_ah | 0 |
| protocol | 0 |
| source_file | 0 |

### Numeric Stats

| field | min | max | mean | non_null_rows |
| --- | --- | --- | --- | --- |
| voltage_v | 0 | 3.60047 | 3.11887 | 1295876 |
| current_a | -4.01229 | 3.60753 | -0.122214 | 1295876 |
| temperature_c | 0 | 38.9286 | 31.9801 | 1295876 |
| capacity_ah | 0 | 1.53905 | 0.635008 | 1295876 |

### Cycle Index Quality

| check | count |
| --- | --- |
| row-level duplicate (battery_id, cycle_index) | 1294687 |
| cross-source_file repeated cycle ids | 0 |
| battery ids with non-continuous integer cycles | 0 |

#### Notes

- Row-level duplicate cycle_index is expected for detailed time-series data, because many time points belong to the same battery cycle.

### Capacity Anomaly Check

| check | count | ratio_within_non_null_capacity |
| --- | --- | --- |
| capacity_ah <= 0 | 11262 | 0.00869065 |

### Dataset-Specific Checks

- MIT detailed step_type values observed: ['charge', 'discharge', 'rest']
- MIT detailed curves include current-derived charge/discharge labels.
- MIT detailed output can be very large; validation samples only one file and one battery by default.

### Suggestions

- capacity_ah contains zero or negative values. This can be normal within raw curves, but cycle-level features should filter or aggregate these carefully.

## CALCE Multi-Workbook Sample

### Required Columns

- All unified columns are present.

### Basic Summary

- Rows: 50102
- Battery count: 1
- Cycle index range: 1 to 50

### Step Type Distribution

| step_type | rows | ratio |
| --- | --- | --- |
| charge | 31725 | 0.633208 |
| discharge | 16726 | 0.333839 |
| rest | 1651 | 0.0329528 |

### Missing Rates

| field | missing_rate |
| --- | --- |
| dataset | 0 |
| battery_id | 0 |
| chemistry | 0 |
| cycle_index | 0 |
| step_type | 0 |
| time_s | 0 |
| voltage_v | 0 |
| current_a | 0 |
| temperature_c | 1 |
| charge_capacity_ah | 0 |
| discharge_capacity_ah | 0 |
| capacity_ah | 0 |
| protocol | 0 |
| source_file | 0 |

### Numeric Stats

| field | min | max | mean | non_null_rows |
| --- | --- | --- | --- | --- |
| voltage_v | 2.6993 | 4.2003 | 3.8613 | 50102 |
| current_a | -1.10029 | 1.01446 | -0.0179819 | 50102 |
| temperature_c | NaN | NaN | NaN | 0 |
| capacity_ah | 0 | 50.9359 | 24.5538 | 50102 |

### Cycle Index Quality

| check | count |
| --- | --- |
| row-level duplicate (battery_id, cycle_index) | 50052 |
| cross-source_file repeated cycle ids | 50 |
| battery ids with non-continuous integer cycles | 0 |

#### Notes

- Row-level duplicate cycle_index is expected for detailed time-series data, because many time points belong to the same battery cycle.
- Some battery_id/cycle_index pairs appear in multiple source files. For CALCE this can indicate Excel files restarting Cycle_Index from 1. Sample: [{'battery_id': 'CS2_35', 'cycle_index_num': 1, 'source_file_count': 3}, {'battery_id': 'CS2_35', 'cycle_index_num': 2, 'source_file_count': 3}, {'battery_id': 'CS2_35', 'cycle_index_num': 3, 'source_file_count': 3}, {'battery_id': 'CS2_35', 'cycle_index_num': 4, 'source_file_count': 3}, {'battery_id': 'CS2_35', 'cycle_index_num': 5, 'source_file_count': 3}]

### Capacity Anomaly Check

| check | count | ratio_within_non_null_capacity |
| --- | --- | --- |
| capacity_ah <= 0 | 13 | 0.000259471 |

### Dataset-Specific Checks

- CALCE sample contains 3 source_file entries.
- CALCE cycle_index reuse across Excel files was detected. This likely means each workbook restarts Cycle_Index; later preprocessing should build a global cycle index.
- CALCE cross-file repeat sample: [{'battery_id': 'CS2_35', 'cycle_index_num': 1, 'source_file_count': 3}, {'battery_id': 'CS2_35', 'cycle_index_num': 2, 'source_file_count': 3}, {'battery_id': 'CS2_35', 'cycle_index_num': 3, 'source_file_count': 3}, {'battery_id': 'CS2_35', 'cycle_index_num': 4, 'source_file_count': 3}, {'battery_id': 'CS2_35', 'cycle_index_num': 5, 'source_file_count': 3}]

### Suggestions

- Fields with >95% missing values should be checked before modeling: temperature_c.
- capacity_ah contains zero or negative values. This can be normal within raw curves, but cycle-level features should filter or aggregate these carefully.

## NASA Mat/Zip Sample

### Required Columns

- All unified columns are present.

### Basic Summary

- Rows: 591736
- Battery count: 1
- Cycle index range: 1 to 616

### Step Type Distribution

| step_type | rows | ratio |
| --- | --- | --- |
| charge | 541173 | 0.914551 |
| discharge | 50285 | 0.0849788 |
| impedance | 278 | 0.000469804 |

### Missing Rates

| field | missing_rate |
| --- | --- |
| dataset | 0 |
| battery_id | 0 |
| chemistry | 0 |
| cycle_index | 0 |
| step_type | 0 |
| time_s | 0.000469804 |
| voltage_v | 0.000469804 |
| current_a | 0.000469804 |
| temperature_c | 0 |
| charge_capacity_ah | 1 |
| discharge_capacity_ah | 0.915021 |
| capacity_ah | 0.915021 |
| protocol | 0 |
| source_file | 0 |

### Numeric Stats

| field | min | max | mean | non_null_rows |
| --- | --- | --- | --- | --- |
| voltage_v | 0.00293171 | 8.33291 | 4.09974 | 591458 |
| current_a | -2.26187 | 1.50717 | 0.386975 | 591458 |
| temperature_c | 22.9699 | 42.3325 | 26.1184 | 591736 |
| capacity_ah | 1.40046 | 1.89105 | 1.63291 | 50285 |

### Cycle Index Quality

| check | count |
| --- | --- |
| row-level duplicate (battery_id, cycle_index) | 591120 |
| cross-source_file repeated cycle ids | 0 |
| battery ids with non-continuous integer cycles | 0 |

#### Notes

- Row-level duplicate cycle_index is expected for detailed time-series data, because many time points belong to the same battery cycle.

### Capacity Anomaly Check

| check | count | ratio_within_non_null_capacity |
| --- | --- | --- |
| capacity_ah <= 0 | 0 | 0 |

### Dataset-Specific Checks

- NASA step_type values observed: ['charge', 'discharge', 'impedance']
- NASA charge/discharge/impedance cycle types are all represented in this sample.
- NASA discharge rows: 50285; non-null discharge capacity_ah rows: 50285
- NASA discharge Capacity field is retained in capacity_ah/discharge_capacity_ah.

### Suggestions

- Fields with >95% missing values should be checked before modeling: charge_capacity_ah.
