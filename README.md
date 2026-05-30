# 电池寿命预测

面向锂电池、固态电池等储能电池的寿命预测研究项目。当前技术路线聚焦公开锂电池数据预训练，并迁移到固态电池小样本寿命预测。

## 项目结构

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

## 初始技术路线

- 源域数据：MIT / NASA / CALCE / BatteryLife
- 目标域数据：Zenodo Solid-State Lithium Battery features
- 统一标签：SOH / RUL / cycle life
- Baseline：Elastic Net / Random Forest / XGBoost
- 主模型：TAM-DeepAR-EN
- 增强方向：PINN / MMD / CORAL

详细调研记录见 [docs/tech_research_notes.md](docs/tech_research_notes.md)。

## 下一步

1. 下载并整理目标域固态电池 CSV 数据。
2. 设计统一数据结构：`battery_id`、`cycle_index`、`capacity`、`SOH`、`RUL`、`feature_*`。
3. 先实现 XGBoost baseline，输出 MAE、RMSE、MAPE、R2。
4. 再实现 DeepAR / TAM-DeepAR-EN 和迁移学习增强模块。

