# Predicting Student Health Risk — Playground Series S6E7

> 🎨 **[点击查看交互式可视化分析报告](https://ligap123.github.io/playground-series-s6e7/report.html)** — 滚动动画 · Chart.js 图表 · 完整数据故事

Kaggle Playground Series Season 6 Episode 7 竞赛方案。**Public Score: 0.94990**。

## 竞赛信息

- **竞赛**: [Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7)
- **题型**: 多分类 (3 类: `at-risk`, `unhealthy`, `fit`)
- **评估指标**: Balanced Accuracy
- **数据来源**: 基于 College Student Health Behavior Dataset 的合成数据
- **截至时间**: 2026-07-31

## v2 最终方案

| 模型 | OOF BA | 权重 |
|------|--------|------|
| LGBM (seed=42) | 0.948446 | 25% |
| LGBM (seed=2026) | 0.948913 | 25% |
| LGBM (seed=3407) | 0.948381 | 25% |
| CatBoost (seed=42) | 0.948623 | 25% |
| **等权融合** | **0.949194** | — |
| **Kaggle Public** | **0.94990** | — |

## 版本历史

| 版本 | Public Score | 方法 |
|------|-------------|------|
| v1 | 0.94918 | LGBM + CatBoost 简单融合 |
| **v2** | **0.94990** | LGBM(3-seed) + CatBoost + 5-fold OOF |

## 项目结构

```
├── PLAN_v2.md           # v2 项目规划
├── EXPERIMENTS.md       # 完整实验报告
├── report.html          # 交互式可视化报告 (GitHub Pages)
├── README.md            # 本文件
├── src/                 # Python 源代码
│   ├── config.py        # 配置
│   ├── cv.py            # 5-fold OOF 验证框架
│   ├── preprocessing.py # fold-local 防泄漏预处理
│   ├── tune.py          # Optuna 调参（实验性）
│   └── hgbc_te.py       # HGBC 尝试（已弃用）
├── notebooks/           # 执行脚本
│   ├── stage4_final.py   # 最终融合 + 生成 submission
│   └── exp_b_*.py        # Optuna 实验脚本
└── outputs/
    └── predictions/
        └── submission.csv
```

## 快速复现

```bash
pip install pandas numpy scikit-learn lightgbm catboost optuna
python notebooks/stage4_final.py
```

## 关键技术

1. **缺失指示特征** (Missing Indicators): 为每个特征创建二值缺失标记 — 最关键特征工程
2. **fold-local 预处理**: 防止交叉验证中的数据泄漏
3. **多种子集成**: 3 个 LGBM 种子降低方差 (+0.0006 vs 单种子)
4. **balanced class_weight**: 处理严重类别不平衡 (85.87% vs 5.77%)
5. **等权融合**: 简单平均优于复杂权重搜索

## 实验记录

详见 [EXPERIMENTS.md](EXPERIMENTS.md) — 包含 6 个实验的完整数据、决策依据和最终融合策略。

## 参考

- [Stellan-04/predicting-student-health-risk](https://github.com/Stellan-04/predicting-student-health-risk)
- [stevenleehans/predicting-student-health-risk-s6e7](https://github.com/stevenleehans/predicting-student-health-risk-s6e7)
