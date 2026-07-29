"""实验E 专用: TargetEncoder + HGBC 的 fold-local 预处理

与主框架的差异：
1. TargetEncoder 需要 y 才能拟合（fit_transform 内部做 OOF 编码防泄漏）
2. HGBC 原生处理 NaN，数值特征不填充（stevenleehans exp003 验证 native NaN 更优）
3. 保留缺失指示特征
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder, LabelEncoder
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from src import config


def transform_hgbc(df, num_cols, cat_cols, enc=None, fit=False, y=None):
    """构建 HGBC 特征矩阵。

    - 数值特征: 保留 NaN（HGBC 原生处理）
    - 类别特征: TargetEncoder（fit 时用 y，内部 OOF 编码防泄漏）
    - 缺失指示: 每个原始特征一列 + 行级缺失计数
    """
    X = pd.DataFrame(index=df.index)

    # 缺失指示（在编码前计算）
    for c in num_cols + cat_cols:
        X[f"{c}_miss"] = df[c].isnull().astype(np.int8)
    X["n_missing"] = df[num_cols + cat_cols].isnull().sum(axis=1).astype(np.int8)

    # 数值: 保留原始（含 NaN），转 float32
    for c in num_cols:
        X[c] = df[c].astype(np.float32)

    # 类别: TargetEncoder
    cat_str = df[cat_cols].fillna("__MISSING__").astype(str)
    if fit:
        enc = TargetEncoder(target_type="multiclass", random_state=config.RANDOM_STATE)
        cat_enc = enc.fit_transform(cat_str, y)
    else:
        cat_enc = enc.transform(cat_str)
    cat_enc = np.asarray(cat_enc)
    for i, c in enumerate(cat_cols):
        X[f"{c}_te"] = cat_enc[:, i].astype(np.float32)

    return X, enc


def run_hgbc_oof(train_df, test_df, seed, n_folds=None, verbose=True):
    """运行 HGBC + TargetEncoder 的 5-fold OOF。"""
    n_folds = n_folds or config.N_FOLDS
    y_le = LabelEncoder().fit_transform(train_df[config.TARGET_COL])
    n_train, n_test = len(train_df), len(test_df)
    oof_prob = np.zeros((n_train, 3), dtype=np.float32)
    test_prob = np.zeros((n_test, 3), dtype=np.float32)
    fold_bas = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_le)):
        tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        ytr = y_le[tr_idx]

        Xtr, enc = transform_hgbc(tr_df, config.NUM_COLS, config.CAT_COLS, fit=True, y=ytr)
        Xva, _ = transform_hgbc(va_df, config.NUM_COLS, config.CAT_COLS, enc=enc)
        Xte, _ = transform_hgbc(test_df, config.NUM_COLS, config.CAT_COLS, enc=enc)

        model = HistGradientBoostingClassifier(
            loss="log_loss", max_iter=400, learning_rate=0.05,
            max_depth=6, min_samples_leaf=50, l2_regularization=0.1,
            class_weight="balanced", random_state=seed + fold, early_stopping=False)
        model.fit(Xtr, ytr)

        oof_prob[va_idx] = model.predict_proba(Xva)
        test_prob += model.predict_proba(Xte) / n_folds
        ba = balanced_accuracy_score(y_le[va_idx], oof_prob[va_idx].argmax(1))
        fold_bas.append(ba)
        if verbose:
            print(f"  fold{fold+1} BA={ba:.6f}", flush=True)

    oof_ba = balanced_accuracy_score(y_le, oof_prob.argmax(1))
    return {"oof_ba": oof_ba, "oof_prob": oof_prob, "test_prob": test_prob,
            "fold_bas": fold_bas, "y_le": y_le}
