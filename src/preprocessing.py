"""
数据预处理模块 — Predicting Student Health Risk (S6E7)

核心策略（参考 Stellan-04 / stevenleehans 消融实验）:
1. 缺失指示特征 (missing indicators) — 最有效的特征工程
2. 行级缺失计数 (missing count per row)
3. Fold-local imputation — 防止数据泄露
4. 保持简单 — 复杂交互特征无稳定收益
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    为每列创建缺失指示特征 (1=缺失, 0=存在)
    参考: Stellan-04 实验显示这是最有效的特征工程
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        missing = pd.DataFrame(index=X.index)
        for col in self.columns:
            missing[f"{col}_missing"] = X[col].isnull().astype(np.int8)
        return missing


class RowMissingCounter(BaseEstimator, TransformerMixin):
    """计算每行的缺失特征总数"""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        missing_count = X[self.columns].isnull().sum(axis=1).values.reshape(-1, 1)
        return pd.DataFrame(missing_count, index=X.index, columns=["n_missing"])


class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """
    安全标签编码：NaN 单独编码为 -1，其余 0,1,2...
    相比 OneHot 更轻量，适合树模型
    """

    def __init__(self):
        self.encoders = {}
        self.columns = None

    def fit(self, X, y=None):
        # X 可能是 numpy array，需转换
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.columns = X.columns.tolist() if hasattr(X, "columns") else list(range(X.shape[1]))
        for col in self.columns:
            le = LabelEncoder()
            series = X[col].fillna("__MISSING__").astype(str)
            le.fit(series)
            self.encoders[col] = le
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.copy()
        for col in self.columns:
            le = self.encoders[col]
            series = X[col].fillna("__MISSING__").astype(str)
            # 处理未见过的类别
            known_classes = set(le.classes_)
            series = series.apply(lambda x: x if x in known_classes else "__MISSING__")
            X[col] = le.transform(series)
        return X


def create_preprocessing_pipeline(num_cols, cat_cols):
    """
    创建完整的预处理管道

    策略:
    - 数值特征: 中位数填充
    - 类别特征: 保留原始值 (树模型原生处理) + 标签编码备选
    - 缺失指示: 每列一个二进制特征
    - 缺失计数: 每行一个特征

    注意: 实际使用时需在 fold 内 fit，fold 外 transform
    """
    return {
        "num_imputer": SimpleImputer(strategy="median"),
        "cat_fill": lambda X: X.fillna("__MISSING__"),
        "missing_indicators": MissingIndicatorTransformer(),
        "missing_counter": RowMissingCounter(),
    }


def build_feature_matrix(train_df, test_df, num_cols, cat_cols,
                         fit_imputer=True, imputer=None):
    """
    构建最终特征矩阵

    返回: X_train, X_test, feature_names, fitted_imputer
    """
    all_features = []

    # 1. 数值特征填充
    if fit_imputer:
        imputer = SimpleImputer(strategy="median")
        num_train = imputer.fit_transform(train_df[num_cols])
        num_test = imputer.transform(test_df[num_cols])
    else:
        num_train = imputer.transform(train_df[num_cols])
        num_test = imputer.transform(test_df[num_cols])

    num_train_df = pd.DataFrame(num_train, columns=num_cols, index=train_df.index)
    num_test_df = pd.DataFrame(num_test, columns=num_cols, index=test_df.index)
    all_features.append(num_train_df)
    test_features = [num_test_df]

    # 2. 缺失指示特征 (每列一个)
    mi = MissingIndicatorTransformer(columns=num_cols + cat_cols)
    mi_train = mi.fit_transform(train_df[num_cols + cat_cols])
    mi_test = mi.transform(test_df[num_cols + cat_cols])
    all_features.append(mi_train)
    test_features.append(mi_test)

    # 3. 缺失计数
    mc = RowMissingCounter(columns=num_cols + cat_cols)
    mc_train = mc.fit_transform(train_df[num_cols + cat_cols])
    mc_test = mc.transform(test_df[num_cols + cat_cols])
    all_features.append(mc_train)
    test_features.append(mc_test)

    # 4. 类别特征 — 简单标签编码（保留 NaN 为单独类别）
    sle = SafeLabelEncoder()
    cat_train = sle.fit_transform(train_df[cat_cols].fillna("__MISSING__"))
    cat_test = sle.transform(test_df[cat_cols].fillna("__MISSING__"))
    # 重命名避免冲突
    cat_train = cat_train.add_prefix("cat_")
    cat_test = cat_test.add_prefix("cat_")
    all_features.append(cat_train)
    test_features.append(cat_test)

    # 合并
    X_train = pd.concat(all_features, axis=1)
    X_test = pd.concat(test_features, axis=1)

    # 确保所有列是数值类型
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            X_train[col] = X_train[col].astype("category")
    for col in X_test.columns:
        if X_test[col].dtype == "object":
            X_test[col] = X_test[col].astype("category")

    feature_names = X_train.columns.tolist()

    return X_train, X_test, feature_names, imputer


def get_feature_importance_df(model, feature_names):
    """从模型提取特征重要性 DataFrame"""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not have feature_importances_ or coef_")

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["cumulative"] = df["importance"].cumsum() / df["importance"].sum()
    return df
