# 项目规划：Kaggle Playground Series S6E7 — Predicting Student Health Risk

## 1. 竞赛基本信息

| 项目 | 详情 |
|------|------|
| **竞赛名称** | Predicting Student Health Risk |
| **竞赛系列** | Kaggle Playground Series — Season 6, Episode 7 |
| **竞赛链接** | https://www.kaggle.com/competitions/playground-series-s6e7 |
| **题型** | 多分类（Multi-class Classification） |
| **目标变量** | `health_condition`（3 类：at-risk, unhealthy, fit） |
| **评估指标** | Balanced Accuracy（平衡准确率） |
| **提交格式** | `id,health_condition` |
| **数据来源** | 基于 [College Student Health Behavior Dataset](https://www.kaggle.com/datasets/ziya07/college-student-health-behavior-dataset) 的合成数据 |
| **开始时间** | 2026-07-01 |
| **截止时间** | 2026-07-31（还剩约 5 天） |
| **参赛队伍数** | 2,881 支队伍 / 6,105 名报名者 |
| **提交次数** | 27,887 次 |
| **标签** | Beginner, Tabular |
| **数据量** | 91.71 MB，3 个文件（train.csv, test.csv, sample_submission.csv），31 列 |

## 2. 项目目录结构规划

```
kaggle/
├── PLAN.md                    # 本文件：项目规划文档
├── README.md                  # 项目说明和复现步骤
├── report.md                  # 分析报告（含简历条目建议）
├── .gitignore                 # Git 忽略规则
│
├── data/                      # 原始数据（不提交到 Git）
│   ├── train.csv              # 训练集
│   ├── test.csv               # 测试集
│   └── sample_submission.csv  # 提交样例
│
├── references/                # 参考 Notebook（从 GitHub 下载）
│   └── ...
│
├── notebooks/                 # Jupyter Notebook
│   ├── 01_eda.ipynb           # 探索性数据分析
│   ├── 02_feature_engineering.ipynb  # 特征工程
│   ├── 03_modeling.ipynb      # 建模与调参
│   └── 04_final_prediction.ipynb     # 最终预测
│
├── src/                       # Python 源代码
│   ├── __init__.py
│   ├── config.py              # 配置文件
│   ├── data_loader.py         # 数据加载
│   ├── eda.py                 # EDA 工具函数
│   ├── preprocessing.py       # 数据预处理 & 特征工程
│   ├── models.py              # 模型定义
│   ├── train.py               # 训练流程
│   ├── evaluate.py            # 评估函数
│   └── submit.py              # 生成提交文件
│
├── outputs/                   # 输出文件
│   ├── models/                # 保存的模型
│   ├── predictions/           # 预测结果
│   └── figures/               # 可视化图表
│
└── tests/                     # 单元测试
    └── ...
```

## 3. 五个阶段划分

### 阶段 1：环境搭建与数据获取（Day 1）
- **目标**：搭建 Python 开发环境，下载竞赛数据，验证数据完整性
- **产出物**：
  - `data/` 目录中的 train.csv、test.csv、sample_submission.csv
  - `references/` 目录中的参考 Notebook
  - 确认 Python 环境可用（pandas, numpy, sklearn, lightgbm, xgboost, catboost）
  - `data/` 文件行数和列数验证报告

### 阶段 2：探索性数据分析（EDA）（Day 2）
- **目标**：深入理解数据特征，发现模式、缺失值、异常值、类别不平衡等问题
- **产出物**：
  - `notebooks/01_eda.ipynb`：完整 EDA Notebook
  - `outputs/figures/`：可视化图表
  - `src/eda.py`：可复用的 EDA 工具函数
  - 数据类型分布、缺失值统计、特征相关性分析
  - 目标变量分布分析

### 阶段 3：特征工程与数据预处理（Day 3）
- **目标**：清洗数据、编码类别特征、创建新特征、处理缺失值
- **产出物**：
  - `notebooks/02_feature_engineering.ipynb`
  - `src/preprocessing.py`：预处理管道
  - 特征重要性初步分析
  - 训练/验证集划分

### 阶段 4：建模与调参（Day 4）
- **目标**：训练多个 Baseline 模型，交叉验证评估，超参数调优
- **产出物**：
  - `notebooks/03_modeling.ipynb`
  - `src/models.py`、`src/train.py`、`src/evaluate.py`
  - 至少 3 个 Baseline：LightGBM、XGBoost、CatBoost（或 RandomForest/Logistic Regression）
  - 交叉验证分数对比表
  - 最佳模型超参数
  - 特征重要性排名
  - `outputs/models/`：保存最佳模型

### 阶段 5：最终预测与文档（Day 5）
- **目标**：生成最终预测、验证提交格式、编写文档、初始化 Git
- **产出物**：
  - `notebooks/04_final_prediction.ipynb`
  - `outputs/predictions/submission.csv`
  - `README.md`：项目说明和复现步骤
  - `report.md`：分析报告（含简历条目建议）
  - `.gitignore`
  - Git 仓库初始化

## 4. 技术栈清单

| 类别 | 工具/库 | 用途 |
|------|---------|------|
| **语言** | Python 3.10+ | 主要编程语言 |
| **数据处理** | pandas, numpy | 数据加载、清洗、转换 |
| **可视化** | matplotlib, seaborn, plotly | EDA 可视化 |
| **机器学习** | scikit-learn | 预处理、交叉验证、基线模型 |
| **梯度提升** | LightGBM, XGBoost, CatBoost | 主要建模算法 |
| **深度学习（可选）** | PyTorch / TensorFlow | TabNet 或简单 NN |
| **实验管理** | joblib, pickle | 模型持久化 |
| **开发环境** | Jupyter Notebook / VS Code | 交互式开发 |
| **版本控制** | Git + GitHub | 代码管理 |

## 5. 时间安排（5 天）

| 日期 | 阶段 | 主要任务 |
|------|------|----------|
| **Day 1** | 环境搭建 & 数据获取 | 创建目录结构、下载数据、克隆参考仓库 |
| **Day 2** | EDA | 数据探索、可视化、发现数据问题 |
| **Day 3** | 特征工程 | 数据清洗、特征构建、预处理管道 |
| **Day 4** | 建模 & 调参 | Baseline 模型、交叉验证、超参数优化 |
| **Day 5** | 最终预测 & 文档 | 生成提交文件、编写报告、Git 初始化 |

## 6. 关键参考资源

| 仓库 | 描述 |
|------|------|
| **stevenleehans/predicting-student-health-risk-s6e7** | EDA, baselines, reproducible experiments |
| **Stellan-04/predicting-student-health-risk** | Leakage-aware LightGBM ensemble |
| **oivler/kaggle-student-health-risk** | 持续更新（截至 7/25） |
| **Choki05/playground-series-s6e7** | 完整的提交方案 |

## 7. 注意事项

1. **数据泄露风险**：由于数据是合成生成的，要注意 train/test 分布差异，避免过拟合
2. **类别不平衡**：检查 target 分布，必要时使用 class_weight 或 SMOTE
3. **提交格式**：严格遵循 `id,health_condition` 格式
4. **中间结果保存**：每个阶段的结果保存到 `outputs/` 目录
5. **代码注释**：所有函数和关键代码段添加中英文注释
