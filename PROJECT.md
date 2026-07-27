# Predicting Student Health Risk — 项目技术文档

> **竞赛**: Kaggle Playground Series S6E7  
> **题型**: 三分类（at-risk / unhealthy / fit）  
> **评估指标**: Balanced Accuracy  
> **最终成绩**: Public Score 0.94918（LightGBM + CatBoost 等权融合）

---

## 1. 项目概述

### 1.1 竞赛背景

本竞赛是 Kaggle Playground Series 2026 年第 7 期赛事。Playground Series 面向社区提供轻量级、可快速迭代的机器学习练习数据集。本赛题的数据基于 [College Student Health Behavior Dataset](https://www.kaggle.com/datasets/ziya07/college-student-health-behavior-dataset) 合成生成，目标是预测大学生的健康状况为三类之一：**at-risk（高风险）、unhealthy（不健康）、fit（健康）**。

### 1.2 数据规模

| 数据集 | 样本数 | 特征数 | 大小 |
|--------|--------|--------|------|
| train.csv | 690,088 | 15（含 id 和 target） | 60 MB |
| test.csv | 295,753 | 14（含 id，不含 target） | 24 MB |
| sample_submission.csv | 295,753 | 2（id + health_condition） | 4.3 MB |

### 1.3 技术栈

| 类别 | 工具/库 | 版本 |
|------|---------|------|
| 数据处理 | pandas, numpy | 2.3.3 / 2.2.6 |
| 机器学习 | scikit-learn | 1.7.2 |
| 梯度提升 | LightGBM, XGBoost, CatBoost | 4.7.0 / 3.2.0 / 1.2.10 |
| 可视化 | matplotlib, seaborn | 3.10.8 / 0.13.2 |
| 交互报告 | HTML + CSS + Chart.js | — |

---

## 2. 项目流程

项目按照五阶段结构化流程推进：

```
阶段1: 环境搭建 & 数据获取
  └─ 目录结构 → 下载数据 → 克隆参考仓库 → 验证环境

阶段2: 探索性数据分析 (EDA)
  └─ 数据加载 → 缺失值分析 → 特征分布 → 可视化

阶段3: 特征工程
  └─ Missing Indicators → Missing Count → Imputation → 27维输出

阶段4: 建模与调参
  └─ 5个Baseline对比 → LightGBM最佳 → 融合预测

阶段5: 文档 & 提交
  └─ 生成submission.csv → Kaggle提交 → 文档编写 → Git
```

---

## 3. 探索性数据分析

### 3.1 目标变量分布

目标变量 `health_condition` 呈现严重类别不平衡：

| 类别 | 数量 | 占比 |
|------|------|------|
| at-risk | 592,561 | 85.87% |
| unhealthy | 57,724 | 8.36% |
| fit | 39,803 | 5.77% |

**影响**: 如果简单使用多数类投票（始终预测 at-risk），普通 Accuracy 可达 85.87%，但 Balanced Accuracy 仅为 0.333。这意味着评估指标刻意惩罚忽视少数类的行为，必须使用 `class_weight='balanced'` 或等效策略。

### 3.2 缺失值分析

| 特征 | 缺失率 (train) | 缺失率 (test) |
|------|:---:|:---:|
| stress_level | 12.00% | 12.00% |
| sleep_duration | 11.01% | 11.01% |
| sleep_quality | 8.45% | 8.45% |
| calorie_expenditure | 7.66% | 7.66% |
| water_intake | 6.30% | 6.30% |
| physical_activity_level | 5.31% | 5.31% |
| smoking_alcohol | 4.14% | 4.14% |
| gender | 3.10% | 3.10% |
| bmi | 2.01% | 2.01% |
| step_count | 2.02% | 2.02% |
| heart_rate | 1.14% | 1.14% |
| exercise_duration | 1.00% | 1.00% |
| diet_type | 1.00% | 1.00% |

**关键发现**: 训练集和测试集的缺失率精确到小数点后两位完全一致。这是合成数据的典型特征——缺失模式由生成器统一控制，而非来自真实世界的随机缺失机制。这意味着：

- 缺失模式 **不能** 用于检测 train/test 分布漂移
- 缺失率在三个目标类别间也几乎相同，缺失本身不提供分类信号
- 但缺失**指示特征**仍然有效——它告诉模型某个值是被填充的还是原始的

### 3.3 特征区分度分析

通过计算各数值特征在三类别下的均值差异，识别出预测信号强弱：

| 特征 | fit 均值 | unhealthy 均值 | at-risk 均值 | 区分度 |
|------|:---:|:---:|:---:|:---:|
| **sleep_duration** | 7.95 | **5.37** | 7.09 | ⭐⭐⭐ 最强 |
| **step_count** | 11,651 | 8,670 | 8,407 | ⭐⭐⭐ |
| **exercise_duration** | 50.0 | 39.0 | 38.0 | ⭐⭐⭐ |
| **bmi** | 21.8 | **24.1** | 23.0 | ⭐⭐ |
| calorie_expenditure | 2,364 | 2,245 | 2,215 | ⭐⭐ |
| heart_rate | 74.8 | 75.3 | 75.1 | ❌ 无区分 |
| water_intake | 2.18 | 2.19 | 2.19 | ❌ 无区分 |

**解释**:
- **sleep_duration**: fit 类（7.95）远高于 unhealthy 类（5.37），差值达 2.58 个标准差，是最强单特征预测器
- **step_count**: fit 类几乎是不健康类的 1.4 倍
- **heart_rate / water_intake**: 三类间几乎无差异，可作为噪声特征考虑剔除

### 3.4 类别特征结构

6 个类别特征均恰好包含 **3 个非空值加 NaN**，且各值分布高度均匀：

| 特征 | 唯一值数 | 值分布 |
|------|:---:|------|
| diet_type | 3 + NaN | veg 231k / balanced 227k / non-veg 225k |
| stress_level | 3 + NaN | medium 262k / high 178k / low 168k |
| sleep_quality | 3 + NaN | average 214k / poor 212k / good 206k |
| physical_activity_level | 3 + NaN | moderate 221k / sedentary 220k / active 213k |
| smoking_alcohol | 3 + NaN | yes 224k / no 220k / occasional 218k |
| gender | 3 + NaN | male 238k / female 224k / other 207k |

这是合成数据生成器的典型特征——每个类别变量被设计为三元变量。

### 3.5 数据质量审计

| 检查项 | 结果 |
|--------|------|
| train ID 重复 | 0 |
| test ID 重复 | 0 |
| train/test ID 重叠 | 0 |
| 完全重复行（不含 ID） | 0 |
| 特征重复行（不含 ID 和 target） | 0 |
| train/test 数值分布漂移 | 无（mean/std 几乎一致） |

---

## 4. 特征工程

### 4.1 设计原则

参考社区两个高质量方案（Stellan-04、stevenleehans）的消融实验结论：

- **Missing Indicators 是收益最大的特征工程**（Stellan-04 验证）
- 复杂交互特征（分箱、特征交叉、目标编码）**无稳定净收益**（stevenleehans 12 次实验验证）
- 合成数据的生成机制使复杂特征容易过拟合

### 4.2 实施的特征工程

| 操作 | 产出维度 | 说明 |
|------|:---:|------|
| 保留原始数值 | 7 | sleep_duration, heart_rate, bmi, calorie_expenditure, step_count, exercise_duration, water_intake |
| Missing Indicators | 13 | 每个原始特征对应一个二值列（1=缺失, 0=存在） |
| Missing Count | 1 | 每行缺失特征的总数 |
| Label Encoding | 6 | 类别特征编码为 0/1/2，NaN 保留为独立类别 |
| **合计** | **27** | |

### 4.3 预处理细节

```python
# 核心特征工程伪代码
for col in all_features:
    # 1. 缺失指示
    X[f"{col}_missing"] = X[col].isnull().astype(int)

# 2. 缺失计数
X["n_missing"] = X[all_features].isnull().sum(axis=1)

# 3. 中位数填充
X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])

# 4. 标签编码（NaN 作为单独类别）
X[cat_cols] = LabelEncoder().fit_transform(X[cat_cols].fillna("__MISSING__"))
```

所有填充和编码操作在实际使用时必须在每个交叉验证 fold 的训练集上单独拟合，在验证集上仅做 transform，以防止数据泄露。

---

## 5. 模型训练

### 5.1 训练策略

由于全量数据（690K 行 × 27 列）交叉验证计算量较大，采用分步策略：

1. **采样阶段**: 用 20% 数据（138K）快速对比 5 个模型，筛选候选
2. **全量阶段**: 对最佳候选在 80/20 分割上训练并评估
3. **融合阶段**: 概率等权平均生成最终预测

### 5.2 模型对比结果

| 模型 | 20% 采样 BA | 全量 80/20 BA | 训练时间 | 备注 |
|------|:---:|:---:|:---:|------|
| Logistic Regression | 0.7700 | — | ~11s | 线性基线，作为下限参考 |
| XGBoost | 0.8797 | — | ~17s | 默认参数欠拟合 |
| Random Forest | 0.9481 | — | ~10s | 简单但有效的树模型 |
| CatBoost | 0.9496 | 0.9494 | ~271s | 类别特征原生支持 |
| **LightGBM** | 0.9444 | **0.9497** | ~247s | 最终选择的最佳模型 |

### 5.3 LightGBM 配置

```python
LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=6,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',
    random_state=42,
    n_jobs=1,
)
```

关键参数说明：
- **class_weight='balanced'**: 自动为少数类 (fit: 5.77%, unhealthy: 8.36%) 分配更高权重
- **n_estimators=1000 + early_stopping=50**: 充分训练但避免过拟合
- **num_leaves=63, max_depth=6**: 中等复杂度，平衡偏差与方差
- **reg_alpha/reg_lambda**: L1/L2 正则化防止过拟合

### 5.4 最终预测

采用 LightGBM 与 CatBoost 的**概率等权融合**：

```python
test_probs = (lgbm.predict_proba(X_test) + cb_model.predict_proba(X_test)) / 2
predictions = label_encoder.inverse_transform(np.argmax(test_probs, axis=1))
```

预测分布（test 集 295,753 条）：

| 类别 | 预测数量 | 占比 |
|------|------:|------|
| at-risk | 239,161 | 80.87% |
| unhealthy | 34,973 | 11.82% |
| fit | 21,619 | 7.31% |

---

## 6. 结果分析

### 6.1 与社区方案对比

| 方案 | 方法 | Public Score | 与我们差距 |
|------|------|:---:|:---:|
| stevenleehans | HGBC + RealMLP 融合 | 0.95045 | +0.00127 |
| Stellan-04 | LightGBM 三种子集成 | 0.95010 | +0.00092 |
| **本项目** | **LGBM + CatBoost 等权融合** | **0.94918** | — |

差距仅 0.0009~0.0013，在 2,881 支参赛队伍中具有竞争力。

### 6.2 误差分析

验证集混淆矩阵显示：
- **unhealthy** 召回率最高（~0.963），说明模型对这一类的识别最好
- **at-risk** 召回率约 0.938，多数错误是将 at-risk 误判为 unhealthy 或 fit
- **fit** 召回率约 0.949

主要误差集中在 at-risk 类——因为它样本量最大、方差也最大。

### 6.3 可改进方向

| 改进 | 预期收益 | 实施难度 |
|------|:---:|:---:|
| 三种子集成（seed 42/2026/3407） | +0.0005 | 中 |
| 网格搜索 LGBM+CB 融合权重 | +0.0002 | 低 |
| 全量 5-fold CV OOF 训练 | +0.0003 | 高（资源） |
| HGBC + RealMLP（参考社区） | +0.0010 | 高 |

三项基础改进叠加预计可达 **0.950+**。

---

## 7. 关键决策与经验

### 7.1 为什么选 Balanced Accuracy

普通 Accuracy 在类别不平衡时具有欺骗性——全猜多数类即可得 85.87%。Balanced Accuracy 是各类别召回率的算术平均，强制模型公平对待每个类别。这是竞赛设计的核心考验。

### 7.2 为什么 Missing Indicators 有效

合成数据中，缺失值本身不携带类别信号（三类别间缺失率相同）。但缺失指示特征告诉模型**某个值是被观察到的还是被填充的**——填充值本质上带有不确定性，模型应该对它和真实值做出不同的判断。这类似于"我不知道"的信息传递。

### 7.3 为什么不做复杂特征

社区消融实验反复验证：特征交互、目标编码、分箱等操作在合成数据上**没有稳定收益**。原因是合成数据的生成过程本身可能没有这些交互结构，强行引入只会增加过拟合风险。保持特征简单 = 模型稳定 = 结果可复现。

### 7.4 资源受限下的训练策略

在计算资源有限（内存/CPU 不足以运行全量 5-fold CV）的情况下，采用的替代方案是：
1. 用 20% 采样快速筛选模型
2. 对候选模型在 80/20 单次分割上评估
3. 直接对测试集生成预测

这种策略牺牲了 CV 带来的方差估计，但保留了足够的信号用于模型选择和融合。

---

## 8. 结论

本项目在 Kaggle Playground S6E7 竞赛中完成了从数据获取到最终提交的完整数据科学流程。核心要点：

1. **数据理解优先于建模**：85.87% 的类别不平衡和合成数据的特殊结构决定了后续所有技术选择
2. **特征工程保持克制**：Missing Indicators 是唯一被验证有效的工程，其他复杂操作无益
3. **多模型对比 > 单模型调参**：在 5 个 baseline 中筛选最佳，比在单一模型上大量调参更高效
4. **融合带来稳定提升**：LGBM + CatBoost 概率等权平均，兼顾两个模型的互补优势
5. **最终成绩 0.94918 接近社区 SOTA**（0.95045），验证了方案有效性

---

*项目仓库: https://github.com/ligap123/playground-series-s6e7*  
*竞赛页面: https://www.kaggle.com/competitions/playground-series-s6e7*
