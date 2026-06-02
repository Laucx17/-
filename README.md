# 电池寿命预测项目

## 项目目标

本项目面向新型电化学储能电池寿命预测研究。第一阶段聚焦普通锂离子电池公开老化数据的收集、整理和统一化，构建可作为源域的数据工程基础，为后续迁移到固态电池、半固态电池等小样本目标域数据做准备。

当前阶段不直接建模，重点完成：

- 规范化保存公开锂离子电池老化数据。
- 建立源域数据集台账。
- 统一容量、循环、电压、电流、温度、EIS 和实验协议等关键信息。
- 为后续 SOH、RUL 和 cycle life 预测提供干净的数据入口。

## 数据处理流程

建议的数据流如下：

```text
data_raw/
    原始下载数据，保持数据来源的原始文件结构和命名
        ↓
data_interim/
    解压、格式转换、初步清洗后的中间数据
        ↓
data_processed/source_domain/
    普通锂离子电池源域数据
        ↓
data_processed/target_domain/
    固态电池、半固态电池等目标域数据
        ↓
data_processed/unified_tables/
    统一字段后的建模输入表
```

推荐统一表字段包括：

- `dataset_name`
- `battery_id`
- `chemistry`
- `cell_format`
- `cycle_index`
- `charge_capacity`
- `discharge_capacity`
- `voltage`
- `current`
- `temperature`
- `EIS_features`
- `protocol`
- `SOH`
- `RUL`
- `cycle_life`

## 目录结构

```text
data_raw/                         原始数据：保存从公开网站下载的原始文件
data_interim/                     中间数据：保存解压、格式转换、初步清洗后的文件
data_processed/                   处理后数据：保存清洗完成、可用于分析的数据
data_processed/source_domain/     源域数据：普通锂离子电池老化数据
data_processed/target_domain/     目标域数据：固态电池、半固态电池等小样本数据
data_processed/unified_tables/    统一表格：统一字段后的建模输入表
notebooks/                        笔记本：数据探索、可视化和临时分析
src/loaders/                      数据加载：不同数据集的读取脚本
src/preprocessing/                数据预处理：清洗、循环对齐、缺失值处理
src/features/                     特征工程：容量、电压、温度、EIS 等特征提取
src/models/                       模型代码：后续 baseline 和预测模型
src/transfer/                     迁移学习：后续源域到目标域迁移方法
docs/                             文档：数据台账、技术文档和实验说明
figures/                          图片：论文图、流程图和结果图输出
```

## 源域数据台账

源域数据集信息记录在：

```text
docs/source_dataset_inventory.csv
```

该文件用于记录每个公开数据集的来源、化学体系、电池数量、循环范围、是否包含容量/电压/电流/温度/EIS/协议信息、数据格式、许可和优先级。

## 第一批源域数据

请将已下载的源域数据保持在以下位置：

```text
data_raw/MIT_Severson/
    2017-05-12_batchdata_updated_struct_errorcorrect.mat
    2017-06-30_batchdata_updated_struct_errorcorrect.mat
    2018-04-12_batchdata_updated_struct_errorcorrect.mat

data_raw/CALCE/
    CS2_35.zip
    CS2_36.zip
    CS2_37.zip
    CS2_38.zip

data_raw/NASA/
    5.+Battery+Data+Set.zip
    B0005.mat
    B0006.mat
    B0007.mat
    B0018.mat

references/mit_severson_code/
    MIT-Severson 官方数据处理代码
```

## 数据读取脚本

当前 loader 只负责读取和统一字段，不进行模型训练。

统一输出字段为：

```text
dataset, battery_id, chemistry, cycle_index, step_type, time_s,
voltage_v, current_a, temperature_c, charge_capacity_ah,
discharge_capacity_ah, capacity_ah, protocol, source_file
```

运行示例：

```bash
python src/loaders/load_mit_severson.py
python src/loaders/load_calce.py
python src/loaders/load_nasa.py
```

小规模测试示例：

```bash
python src/loaders/load_mit_severson.py --summary-only --max-files 1 --max-batteries 2
python src/loaders/load_calce.py --max-workbooks 1
python src/loaders/load_nasa.py --max-files 1
```

MIT-Severson 详细循环数据非常大，单个电池也可能超过百万行。调试时建议先使用 `--summary-only` 或 `--max-batteries`。

默认输出位置：

```text
data_processed/source_domain/mit_severson_unified.parquet
data_processed/source_domain/calce_unified.parquet
data_processed/source_domain/nasa_unified.parquet
```

MIT-Severson 的 `.mat` 文件是 MATLAB 7.3/HDF5 格式，需要 `h5py`。NASA 的 `.mat` 文件需要 `scipy`。运行前建议先安装依赖：

```bash
pip install -r requirements.txt
```

## 后续建模目标

完成数据工程后，后续建模将按以下顺序推进：

1. 构建源域锂离子电池统一数据表。
2. 提取早期循环退化特征、容量特征、电压曲线特征、温度特征和 EIS 特征。
3. 建立 XGBoost、Random Forest、Elastic Net 等 baseline 模型。
4. 实现 SOH、RUL 和 cycle life 预测任务。
5. 在源域预训练基础上，迁移到固态电池或半固态电池小样本目标域。
6. 对比无迁移、微调迁移、特征对齐和物理约束迁移方法。
