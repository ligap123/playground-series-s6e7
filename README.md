# Predicting Student Health Risk — Playground Series S6E7

> 🎨 **[点击查看交互式可视化分析报告](https://ligap123.github.io/playground-series-s6e7/report.html)** — 滚动动画 · Chart.js 图表 · 完整数据故事

Kaggle Playground Series Season 6 Episode 7 竞赛方案。Public Score: **0.94918**。

## 竞赛信息

- **竞赛**: [Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7)
- **题型**: 多分类 (3 类: `at-risk`, `unhealthy`, `fit`)
- **评估指标**: Balanced Accuracy
- **数据来源**: 基于 College Student Health Behavior Dataset 的合成数据
- **截至时间**: 2026-07-31

## 项目结构

```
├── PLAN.md          # 项目规划文档
├── README.md        # 本文件
├── report.md        # 分析报告
├── .gitignore
├── data/            # 竞赛数据 (需自行下载, 不提交 Git)
├── references/      # 参考仓库 (从 GitHub 克隆)
├── notebooks/       # Jupyter Notebook
├── src/             # Python 源代码
│   ├── config.py    # 配置
│   ├── eda.py       # EDA 工具
│   └── preprocessing.py  # 特征工程
├── outputs/         # 输出文件
│   ├── figures/     # 可视化
│   └── predictions/ # 提交文件
└── tests/
```

## 复现步骤

### 1. 环境准备

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

### 2. 下载数据

从 Kaggle Data 页面下载 `train.csv`、`test.csv`、`sample_submission.csv` 放入 `data/` 目录。

### 3. 运行分析

```bash
# EDA 和特征工程
python src/preprocessing.py

# 训练模型并生成预测
python -c "
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb, catboost as cb

# 加载预处理数据
X = pd.read_parquet('outputs/X_train_processed.parquet')
X_test = pd.read_parquet('outputs/X_test_processed.parquet')
y_all = pd.read_parquet('outputs/y_train.parquet')['health_condition']
le = LabelEncoder(); y = le.fit_transform(y_all)

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# LightGBM
lgbm = lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=1000,
    learning_rate=0.03, num_leaves=63, class_weight='balanced', random_state=42)
lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50)])

# CatBoost
cb_model = cb.CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
    loss_function='MultiClass', random_seed=42, class_weights=[1, 10.28, 14.88])
cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)

# 等权融合
probs = (lgbm.predict_proba(X_test) + cb_model.predict_proba(X_test)) / 2
pred = le.inverse_transform(np.argmax(probs, axis=1))
pd.DataFrame({'id': pd.read_csv('data/test.csv')['id'], 'health_condition': pred}
    ).to_csv('outputs/predictions/submission.csv', index=False)
```

### 4. 验证提交

```bash
python -c "
import pandas as pd
sub = pd.read_csv('outputs/predictions/submission.csv')
assert sub.shape == (295753, 2)
assert list(sub.columns) == ['id', 'health_condition']
assert sub['id'].min() == 690088
print('Format OK')
"
```

## 结果

| 模型 | 验证 Balanced Accuracy |
|------|:---:|
| Logistic Regression | 0.7700 |
| Random Forest | 0.9481 |
| XGBoost | 0.8797 |
| CatBoost | 0.9494 |
| **LightGBM** | **0.9497** |

## 关键技术

1. **缺失指示特征** (missing indicators): 为每个特征创建二值缺失标记 — 最关键特征工程
2. **行级缺失计数**: 统计每行缺失特征总数
3. **中位数填充**: 数值特征用中位数填充
4. **balanced class_weight**: 处理严重类别不平衡 (85.87% vs 5.77%)
5. **多模型等权融合**: LightGBM + CatBoost 概率平均

## 参考

- [Stellan-04/predicting-student-health-risk](https://github.com/Stellan-04/predicting-student-health-risk) — Leakage-aware LightGBM ensemble
- [stevenleehans/predicting-student-health-risk-s6e7](https://github.com/stevenleehans/predicting-student-health-risk-s6e7) — EDA & reproducible experiments
