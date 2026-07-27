"""
EDA 工具函数 — Predicting Student Health Risk (S6E7)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(data_dir="data"):
    """加载竞赛数据"""
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    sub = pd.read_csv(f"{data_dir}/sample_submission.csv")
    return train, test, sub


def get_feature_cols(train):
    """获取特征列名（排除 id 和 target）"""
    return [c for c in train.columns if c not in ["id", "health_condition"]]


def get_num_cols(train):
    """获取数值列名"""
    return train.select_dtypes(include=["float64", "int64"]).columns.drop("id").tolist()


def get_cat_cols(train):
    """获取类别列名"""
    return [c for c in train.select_dtypes(include=["object"]).columns
            if c != "health_condition"]


def missing_summary(train, test=None):
    """生成缺失值汇总表"""
    feat_cols = get_feature_cols(train)
    result = pd.DataFrame({
        "train_missing": train[feat_cols].isnull().sum(),
        "train_missing_pct": (train[feat_cols].isnull().mean() * 100).round(2),
    })
    if test is not None:
        result["test_missing"] = test[feat_cols].isnull().sum()
        result["test_missing_pct"] = (test[feat_cols].isnull().mean() * 100).round(2)
    return result.sort_values("train_missing_pct", ascending=False)


def target_summary(train):
    """目标变量分布摘要"""
    counts = train["health_condition"].value_counts()
    props = train["health_condition"].value_counts(normalize=True)
    return pd.DataFrame({"count": counts, "proportion": props.round(4)})


def plot_target_distribution(train, save_path="outputs/figures/target_distribution.png"):
    """绘制目标分布图"""
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = train["health_condition"].value_counts()

    ax = axes[0]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Target Distribution (Count)", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3000,
                f"{val:,}\n({val/len(train)*100:.1f}%)", ha="center",
                fontsize=11, fontweight="bold")

    ax = axes[1]
    ax.pie(counts.values, labels=counts.index, autopct="%1.2f%%", colors=colors,
           explode=(0.02, 0.02, 0.02), shadow=True, startangle=90,
           textprops={"fontsize": 12})
    ax.set_title("Target Distribution (Pie)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_missing_values(train, test, save_path="outputs/figures/missing_values.png"):
    """绘制缺失值对比图"""
    feat_cols = get_feature_cols(train)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    tm = train[feat_cols].isnull().mean().sort_values(ascending=False) * 100
    ax = axes[0]
    bars = ax.barh(range(len(tm)), tm.values, color="#3498db", edgecolor="white")
    ax.set_yticks(range(len(tm)))
    ax.set_yticklabels(tm.index, fontsize=10)
    ax.set_xlabel("Missing Rate (%)", fontsize=12)
    ax.set_title("Train Missing Rate", fontsize=14, fontweight="bold")
    for i, (bar, val) in enumerate(zip(bars, tm.values)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    te = test[feat_cols].isnull().mean().sort_values(ascending=False) * 100
    mc = pd.DataFrame({"train": tm, "test": te})
    ax = axes[1]
    x = np.arange(len(mc))
    w = 0.35
    ax.barh(x + w / 2, mc["train"].values, w, label="Train", color="#3498db", edgecolor="white")
    ax.barh(x - w / 2, mc["test"].values, w, label="Test", color="#e74c3c", edgecolor="white")
    ax.set_yticks(x)
    ax.set_yticklabels(mc.index, fontsize=10)
    ax.set_xlabel("Missing Rate (%)", fontsize=12)
    ax.set_title("Train vs Test Missing Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def audit_data(train, test):
    """数据质量审计"""
    results = {}
    results["train_shape"] = train.shape
    results["test_shape"] = test.shape
    results["train_id_duplicates"] = train["id"].duplicated().sum()
    results["test_id_duplicates"] = test["id"].duplicated().sum()
    results["id_overlap"] = len(set(train["id"]) & set(test["id"]))
    results["train_exact_duplicates"] = train.drop(columns=["id"]).duplicated().sum()
    results["train_feature_duplicates"] = train.drop(
        columns=["id", "health_condition"]
    ).duplicated().sum()
    results["train_id_range"] = (train["id"].min(), train["id"].max())
    results["test_id_range"] = (test["id"].min(), test["id"].max())
    return results
