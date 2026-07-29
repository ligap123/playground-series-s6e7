"""实验B: Optuna 调参 LGBM（快速单分割搜索版）

用 80/20 单次分割做快速评估（每个 trial ~40s），搜索超参空间。
找到最优参数后，再用 5-fold OOF 完整验证（在后续步骤单独跑）。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src import config
from src.preprocessing import fit_fold_preprocess, transform_fold


def quick_eval(params, Xtr, ytr, Xva, yva):
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=1000,
        learning_rate=params["lr"], num_leaves=params["leaves"],
        max_depth=params["depth"], min_child_samples=params["mcs"],
        subsample=params["sub"], subsample_freq=1, colsample_bytree=params["col"],
        reg_alpha=params["ra"], reg_lambda=params["rl"],
        class_weight="balanced", random_state=config.RANDOM_STATE,
        verbose=-1, n_jobs=1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return balanced_accuracy_score(yva, m.predict(Xva)), m.best_iteration_


def tune(train_df, n_trials=15):
    y_le = LabelEncoder().fit_transform(train_df[config.TARGET_COL])
    tr_idx, va_idx = train_test_split(np.arange(len(train_df)), test_size=0.2,
                                      stratify=y_le, random_state=config.RANDOM_STATE)
    tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    params = fit_fold_preprocess(tr_df, config.NUM_COLS, config.CAT_COLS)
    Xtr, _ = transform_fold(tr_df, params, config.NUM_COLS, config.CAT_COLS)
    Xva, _ = transform_fold(va_df, params, config.NUM_COLS, config.CAT_COLS)
    ytr, yva = y_le[tr_idx], y_le[va_idx]

    def objective(trial):
        p = {
            "lr": trial.suggest_float("lr", 0.01, 0.1, log=True),
            "leaves": trial.suggest_int("leaves", 31, 255),
            "depth": trial.suggest_int("depth", 4, 10),
            "mcs": trial.suggest_int("mcs", 20, 100),
            "sub": trial.suggest_float("sub", 0.6, 1.0),
            "col": trial.suggest_float("col", 0.6, 1.0),
            "ra": trial.suggest_float("ra", 1e-3, 1.0, log=True),
            "rl": trial.suggest_float("rl", 1e-3, 1.0, log=True),
        }
        ba, iters = quick_eval(p, Xtr, ytr, Xva, yva)
        trial.set_user_attr("best_iter", int(iters) if iters else 0)
        return ba

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study
