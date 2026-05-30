# 电池寿命预测技术检索笔记

## 1. 当前建议主线

第一阶段建议聚焦：

普通锂电池公开大数据预训练，再迁移到固态电池小样本寿命预测。

推荐技术路线：

```text
源域数据：MIT / NASA / CALCE / BatteryLife
目标域数据：Zenodo Solid-State Lithium Battery features
        ↓
统一字段与标签：SOH / RUL / cycle life
        ↓
Baseline：Elastic Net / Random Forest / XGBoost
        ↓
主模型：TAM-DeepAR-EN
        ↓
增强模型：PINN / MMD / CORAL
```

## 2. 优先数据源

| 数据源 | 用途 | 备注 |
| --- | --- | --- |
| MIT/Severson | 早期循环特征预测 cycle life | Nature Energy 经典高引路线 |
| NASA Li-ion Battery Aging | SOH/RUL、EIS、不同温度老化 | 含充电、放电和阻抗测试 |
| CALCE Battery | 容量衰减、不同工况老化 | 常用于 SOH/RUL 论文验证 |
| BatteryLife Benchmark | 大规模预训练和基准比较 | 集成多数据集，适合源域 |
| Zenodo 固态锂电池特征数据 | 固态电池小样本迁移目标域 | CSV 特征数据，样本规模小 |
| PNNL VRFB | 后续液流电池扩展 | 多电流密度、多循环性能数据 |

## 3. 模型优先级

### 3.1 Baseline

- Elastic Net
- Random Forest
- XGBoost
- LightGBM
- SVR

目标：先快速跑通前 N 圈特征到 cycle life / SOH / RUL 的预测。

### 3.2 主模型

TAM-DeepAR-EN：

- DeepAR 用于多电池退化序列概率预测。
- Temporal Attention 用于捕捉关键循环阶段。
- Elastic Net 正则化用于小样本泛化。
- IHFs 可采用 CCCT、CVCT、ADV、ICV、容量、电压均值、温度等。

### 3.3 增强模型

PINN 物理约束：

- SOH 总体随循环衰减。
- 退化曲线应平滑。
- 内阻或阻抗相关特征总体随老化增大。
- 可加入源域与目标域迁移正则项。

建议损失函数：

```text
Loss = L_data + λ1 L_mono + λ2 L_smooth + λ3 L_physics + λ4 L_transfer
```

## 4. 第一版目录用途

```text
configs/              实验配置
data/raw/             原始数据
data/processed/       清洗后的统一数据
docs/                 技术路线、论文笔记、实验说明
notebooks/            探索性分析
references/           文献、数据说明、外部资料
results/figures/      图
results/tables/       表
src/data/             数据下载与加载
src/features/         特征提取
src/models/           模型结构
src/training/         训练流程
src/evaluation/       评估指标与画图
```

## 5. 下一步执行建议

1. 先下载 Zenodo 固态锂电池 CSV，确认字段含义。
2. 再选择一个源域数据，优先 MIT/Severson 或 NASA。
3. 写统一数据结构：battery_id、cycle_index、capacity、SOH、RUL、feature_*。
4. 跑 XGBoost baseline，输出 MAE、RMSE、MAPE、R2。
5. 再实现 DeepAR/TAM-DeepAR-EN。

## 6. 关键来源

- MIT/Severson Nature Energy: https://www.nature.com/articles/s41560-019-0356-8
- Severson 代码仓库: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
- BatteryLife Benchmark: https://github.com/Ruifeng-Tan/BatteryLife
- DeepAR: https://arxiv.org/abs/1704.04110
- PINN battery degradation: https://www.nature.com/articles/s41467-024-48779-z
- NASA Li-ion Battery Aging Datasets: https://catalog.data.gov/dataset/li-ion-battery-aging-datasets
- Zenodo Solid-State Lithium Battery features: https://zenodo.org/records/4743315
- PNNL VRFB dataset: https://www.osti.gov/dataexplorer/biblio/dataset/1862881-experimental-database-cell-performance-vanadium-redox-flow-battery
