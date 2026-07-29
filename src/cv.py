"""v2 5-fold OOF 训练框架（资源受限版）

要点：
- fold-local 预处理（防泄漏）
- 逐 fold 顺序训练，n_jobs=1 控制 CPU
- 保存每 fold 的 OOF 概率 + test 概率（取平均）
- 返回 OOF Balanced Accuracy 供实验对比
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from src import config
from src.preprocessing import fit_fold_preprocess, transform_fold


def run_oof(build_model_fn, seed, train_df, test_df, n_folds=None,
            verbose=True, save_models=False):
    """运行一次 5-fold OOF。

    build_model_fn(seed) -> 一个可 fit(X, y) / predict_proba(X) 的模型。
    返回 dict: oof_ba, oof_prob (n_train,3), test_prob (n_test,3), fold_bas
    """
    n_folds = n_folds or config.N_FOLDS
    y_le = LabelEncoder().fit_transform(train_df[config.TARGET_COL])
    n_train, n_test = len(train_df), len(test_df)
    oof_prob = np.zeros((n_train, 3), dtype=np.float32)
    test_prob = np.zeros((n_test, 3), dtype=np.float32)
    fold_bas = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_le)):
        tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        # fold-local 预处理
        params = fit_fold_preprocess(tr_df, config.NUM_COLS, config.CAT_COLS)
        Xtr, feats = transform_fold(tr_df, params, config.NUM_COLS, config.CAT_COLS)
        Xva, _ = transform_fold(va_df, params, config.NUM_COLS, config.CAT_COLS)
        Xte, _ = transform_fold(test_df, params, config.NUM_COLS, config.CAT_COLS)

        model = build_model_fn(seed + fold)  # 每 fold 不同种子提升多样性
        model.fit(Xtr, y_le[tr_idx])

        oof_prob[va_idx] = model.predict_proba(Xva)
        test_prob += model.predict_proba(Xte) / n_folds

        ba = balanced_accuracy_score(y_le[va_idx], oof_prob[va_idx].argmax(1))
        fold_bas.append(ba)
        if verbose:
            print(f"  fold{fold+1} BA={ba:.6f}", flush=True)

    oof_pred = oof_prob.argmax(1)
    oof_ba = balanced_accuracy_score(y_le, oof_pred)
    return {
        "oof_ba": oof_ba,
        "oof_prob": oof_prob,
        "test_prob": test_prob,
        "fold_bas": fold_bas,
        "y_le": y_le,
        "feature_names": feats,
    }
