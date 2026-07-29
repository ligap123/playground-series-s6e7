"""v2 防泄漏预处理（fold-local）

核心原则：所有填充/编码在每个 fold 的训练集上拟合，在验证集上仅 transform。
这与 v1 的全局填充不同 —— 避免预处理统计量泄漏。
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def fit_fold_preprocess(train_df, num_cols, cat_cols):
    """在训练 fold 上拟合预处理参数"""
    params = {}
    params["num_median"] = train_df[num_cols].median()
    # 类别众数（用于填充 NaN 后编码）
    params["cat_mode"] = train_df[cat_cols].mode().iloc[0]
    # OrdinalEncoder：NaN 已先填众数
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train_df[cat_cols].fillna(params["cat_mode"]).astype(str))
    params["encoder"] = enc
    return params


def transform_fold(df, params, num_cols, cat_cols, add_missing_feats=True):
    """应用 fold 预处理（含缺失指示特征）

    返回 (X, feature_names)。缺失指示在填充前计算，避免泄漏填充状态。
    """
    X = pd.DataFrame(index=df.index)

    # 1) 缺失指示（在填充前计算！）—— v1 验证有效
    if add_missing_feats:
        for c in num_cols + cat_cols:
            X[f"{c}_miss"] = df[c].isnull().astype(np.int8)
        X["n_missing"] = df[num_cols + cat_cols].isnull().sum(axis=1).astype(np.int8)

    # 2) 数值：fold 中位数填充
    for c in num_cols:
        X[c] = df[c].fillna(params["num_median"][c]).astype(np.float32)

    # 3) 类别：先填众数再 Ordinal 编码
    cat_filled = df[cat_cols].fillna(params["cat_mode"]).astype(str)
    cat_encoded = params["encoder"].transform(cat_filled)
    for i, c in enumerate(cat_cols):
        X[c] = cat_encoded[:, i].astype(np.int8)

    return X, X.columns.tolist()
