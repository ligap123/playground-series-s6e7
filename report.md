# 分析报告: Predicting Student Health Risk (Playground Series S6E7)

## 竞赛概述

- 题目: 预测大学生健康状况 (三种类别: at-risk / unhealthy / fit)
- 数据: 合成数据 (基于 College Student Health Behavior Dataset), 690,088 训练 + 295,753 测试
- 指标: Balanced Accuracy (平衡准确率)
- 验证: LightGBM Balanced Accuracy **0.9497** (80/20 holdout)

## EDA 关键发现

### 1. 严重类别不平衡

| 类别 | 数量 | 占比 |
|------|------:|------|
| at-risk | 592,561 | 85.87% |
| unhealthy | 57,724 | 8.36% |
| fit | 39,803 | 5.77% |

Dummy 基线 (全猜 at-risk): Accuracy=85.87%, Balanced Accuracy=0.333

### 2. 缺失值分布 (train/test 完全相同→合成数据特征)

缺失率最高的特征: `stress_level` (12%), `sleep_duration` (11%), `sleep_quality` (8.5%)

关键发现: **各目标类别的缺失率几乎完全相同**, 缺失本身不提供分类信号。

### 3. 强预测特征

按类别区分度排列:
- `sleep_duration`: fit (7.95) >> unhealthy (5.37) — 最强信号
- `step_count`: fit (11,651) >> at-risk (8,407)
- `exercise_duration`: fit (50.0) >> at-risk (38.0)
- `bmi`: unhealthy (24.1) >> fit (21.8)
- 弱特征: `heart_rate`, `water_intake` (各类别几乎无差异)

### 4. 数据质量

- ID 无重叠、无重复
- 无完全重复行
- train/test 分布无显著漂移
- 类别特征均含 3 个值 + NaN (合成数据特征)

## 特征工程

| 工程 | 描述 | 效果 |
|------|------|------|
| **缺失指示** | 每列创建 is_missing 二值标记 | ⭐⭐⭐ 最关键 |
| **缺失计数** | 每行缺失特征数 | ⭐⭐ |
| **中位数填充** | 数值 NA 填充中位数 | 基础必需 |
| **标签编码** | 类别变量 0/1/2 编码 | 适合树模型 |

## 模型表现

| 模型 | 验证 BA | 备注 |
|------|:---:|------|
| Logistic Regression | 0.7700 | 线性基线 |
| Random Forest | 0.9481 | 简单但有效 |
| XGBoost | 0.8797 | 欠拟合, 需更多调参 |
| CatBoost | 0.9494 | 接近最佳 |
| **LightGBM** | **0.9497** | 最佳单模型 |
| LGBM+CatBoost 融合 | — | 已提交 |

## 简历条目建议

**中文**:
> 参加 Kaggle Playground Series S6E7 "Predicting Student Health Risk" 竞赛 (2,881 支队伍), 独立完成数据探索、特征工程和模型开发全流程。针对严重类别不平衡问题 (85.87% vs 5.77%), 设计缺失指示特征和 balanced class_weight 策略, 最终 LightGBM 模型 Balanced Accuracy 达到 0.9497 (验证集)。

**English**:
> Participated in Kaggle Playground Series S6E7 "Predicting Student Health Risk" competition (2,881 teams). Independently completed the full pipeline including EDA, feature engineering, and model development. Addressed severe class imbalance (85.87% vs 5.77%) with missing indicator features and balanced class weights. Achieved LightGBM validation Balanced Accuracy of 0.9497.

## 经验总结

1. **缺失指示特征**是合成数据竞赛的通用技巧 — 合成数据生成器往往在缺失模式中留下结构信息
2. **Balanced Accuracy** 要求不能忽视少数类, class_weight 是关键
3. **保持特征工程简单** — 参考仓库的消融实验证明复杂交互无稳定收益
4. **资源受限时**: 采样验证 + 单次分割能有效替代全量交叉验证
