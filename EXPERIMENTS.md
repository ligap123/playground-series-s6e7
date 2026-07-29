# v2 实验报告 — Predicting Student Health Risk (S6E7)

> 目标: 以 v1 为基座，逐项优化，目标本地 OOF >= 0.9505
> 验证框架: 5-fold StratifiedKFold OOF（220K 采样数据）
> 评估指标: Balanced Accuracy

---

## 实验汇总

| 实验 | 描述 | OOF BA | vs 基线 | 决策 |
|------|------|--------|---------|------|
| 基线 | LGBM seed=42, default params | 0.948446 | — | 基准 |
| A | 多种子 LGBM (3-seed avg: 42/2026/3407) | 0.949065 | +0.000619 | ✅ 保留 |
| B | Optuna 调参 LGBM (采样搜索) | 0.947462† | -0.000984 | ❌ 弃用 |
| C | CatBoost (seed=42) | 0.948623 | +0.000177 | ✅ 保留 |
| D | 融合权重优化 (LGBM3+CB, w=0.75~0.95) | 0.949194 | +0.000748 | ✅ 保留 |
| E | TargetEncoder + HGBC | 0.928295 | -0.020151 | ❌ 弃用 |

> † 实验B的 0.947462 是在 220K 采样 80/20 单次分割上的结果，低于同条件下基线。

---

## 最终方案

**策略**: 等权融合 4 个模型（LGBM×3 seeds + CatBoost）
- `OOF BA = 0.949194`
- 权重: LGBM(s42) 25% + LGBM(s2026) 25% + LGBM(s3407) 25% + CB(s42) 25%

**单模型 OOF BA**:
| 模型 | OOF BA |
|------|--------|
| LGBM seed=42 | 0.948446 |
| LGBM seed=2026 | 0.948913 |
| LGBM seed=3407 | 0.948381 |
| LGBM 3-seed avg | 0.949065 |
| CatBoost seed=42 | 0.948623 |

**权重搜索**:
| w_lgbm3 | w_cb | OOF BA |
|---------|------|--------|
| 0.50 | 0.50 | 0.948911 |
| 0.55 | 0.45 | 0.949000 |
| 0.65 | 0.35 | 0.949091 |
| **0.75** | **0.25** | **0.949194** |
| 0.80 | 0.20 | 0.949135 |
| 0.85 | 0.15 | 0.949082 |

等权 4-way 和 w=0.75 的 2-stage 融合结果相同（0.949194），选择更简单的等权方案。

---

## 技术栈

- pandas, numpy, scikit-learn
- LightGBM (base + multi-seed)
- CatBoost
- 5-fold StratifiedKFold (fold-local preprocessing)
- Missing indicator features

---

## 关键决策

1. **多种子集成有效**: 3-seed 平均从 0.948446 -> 0.949065 (+0.0006)
2. **CatBoost 有微弱增益**: 单模型 0.948623，融合后提升到 0.949194
3. **Optuna 调参无效**: 在采样数据上搜索到的参数（lr=0.01, leaves=53）反而不如默认参数，且在 690K 全量数据上训练太慢无法验证
4. **HGBC 不适用**: TargetEncoder + HGBC 严重低于基线，弃用
5. **简单融合优于复杂**: 等权平均和最优权重结果一致

---

## 最终产物

- `outputs/predictions/submission.csv` — 295,753 行，格式验证通过
- 预测分布: at-risk=240608, fit=21611, unhealthy=33534
- `outputs/final_result.json` — 完整实验数据

---

## 未达成的目标

- 目标 OOF >= 0.9505，实际 0.949194，差距 -0.0013
- 可能改进方向（受资源限制未实施）:
  - 更多种子（5-10 seeds）
  - XGBoost 加入融合
  - 特征工程（交互特征、多项式特征）
  - 在 690K 全量数据上跑完整 OOF（当前用 220K 采样）
