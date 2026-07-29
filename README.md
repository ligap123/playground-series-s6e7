# Predicting Student Health Risk — Playground Series S6E7

**Kaggle Public Score: 0.94990** | LGBM(3-seed) + CatBoost Ensemble

## Overview

Multi-class classification (at-risk / fit / unhealthy) on synthetic student health data. 5-fold StratifiedKFold OOF validation with fold-local preprocessing and missing indicator features.

## Results

| Strategy | OOF BA | Public Score |
|----------|--------|-------------|
| LGBM (seed=42) | 0.948446 | — |
| LGBM 3-seed avg | 0.949065 | — |
| CatBoost (seed=42) | 0.948623 | — |
| **Equal 4-way ensemble** | **0.949194** | **0.94990** |

## Structure

```
v2/
├── src/
│   ├── config.py          # Config & paths
│   ├── preprocessing.py   # Fold-local preprocessing
│   ├── cv.py              # 5-fold OOF framework
│   ├── tune.py            # Optuna tuning (experimental)
│   └── hgbc_te.py         # HGBC attempt (abandoned)
├── notebooks/
│   ├── stage4_final.py    # Final ensemble & submission
│   ├── exp_b_optuna.py    # Optuna experiment
│   └── exp_b_fast.py      # Optuna validation
├── outputs/
│   └── predictions/
│       └── submission.csv # Final submission (295,753 rows)
├── EXPERIMENTS.md         # Full experiment log
├── PLAN_v2.md             # Project plan
└── README.md
```

## Quick Start

```bash
pip install pandas numpy scikit-learn lightgbm catboost optuna
cd v2
python notebooks/stage4_final.py
```

## Key Decisions

- **Multi-seed LGBM** (+0.0006) — 3 seeds (42/2026/3407) reduce variance
- **CatBoost** (+0.0002) — marginal but consistent gain
- **Simple average beats weighted** — equal 4-way = optimal w=0.75
- **Optuna tuning discarded** — overfit on sample, no gain on full OOF
- **HGBC discarded** — severe underperformance (-0.02)

## License

MIT
