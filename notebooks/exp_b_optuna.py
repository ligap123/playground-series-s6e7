"""实验B: Optuna 调参 LGBM — 采样加速版

策略：用 220K 采样数据做 Optuna 快速搜索，再用全量数据 5-fold OOF 验证。
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src import config
from src.cv import run_oof
from src.preprocessing import fit_fold_preprocess, transform_fold

# ── 0. 加载数据 ──────────────────────────────────────────────
print("=" * 60)
print("实验B: Optuna 调参 LGBM (采样加速版)")
print("=" * 60)

t0 = time.time()

# 用采样数据做 Optuna 搜索
sample_path = os.path.join(config.DATA_DIR, "train_sample220k.parquet")
train_sample = pd.read_parquet(sample_path)
print(f"采样数据: {train_sample.shape} (用于Optuna搜索)")

# 全量数据用于最终验证
train_full = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)
print(f"全量数据: train={train_full.shape}, test={test_df.shape} [{time.time()-t0:.1f}s]")

# ── 1. Optuna 快速搜索 (采样数据, 80/20) ─────────────────────
print("\n[步骤1] Optuna 搜索 (220K 采样, 80/20 split, 30 trials)")
print("-" * 40)

y_le_sample = LabelEncoder().fit_transform(train_sample[config.TARGET_COL])
tr_idx, va_idx = train_test_split(np.arange(len(train_sample)), test_size=0.2,
                                  stratify=y_le_sample, random_state=config.RANDOM_STATE)
tr_df, va_df = train_sample.iloc[tr_idx], train_sample.iloc[va_idx]
params = fit_fold_preprocess(tr_df, config.NUM_COLS, config.CAT_COLS)
Xtr, _ = transform_fold(tr_df, params, config.NUM_COLS, config.CAT_COLS)
Xva, _ = transform_fold(va_df, params, config.NUM_COLS, config.CAT_COLS)
ytr, yva = y_le_sample[tr_idx], y_le_sample[va_idx]
print(f"  训练集: {Xtr.shape}, 验证集: {Xva.shape}")


def quick_eval(p):
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=1000,
        learning_rate=p["lr"], num_leaves=p["leaves"],
        max_depth=p["depth"], min_child_samples=p["mcs"],
        subsample=p["sub"], subsample_freq=1, colsample_bytree=p["col"],
        reg_alpha=p["ra"], reg_lambda=p["rl"],
        class_weight="balanced", random_state=config.RANDOM_STATE,
        verbose=-1, n_jobs=1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return balanced_accuracy_score(yva, m.predict(Xva)), m.best_iteration_


def objective(trial):
    p = {
        "lr": trial.suggest_float("lr", 0.01, 0.12, log=True),
        "leaves": trial.suggest_int("leaves", 31, 511),
        "depth": trial.suggest_int("depth", 3, 12),
        "mcs": trial.suggest_int("mcs", 10, 200),
        "sub": trial.suggest_float("sub", 0.5, 1.0),
        "col": trial.suggest_float("col", 0.5, 1.0),
        "ra": trial.suggest_float("ra", 1e-4, 10.0, log=True),
        "rl": trial.suggest_float("rl", 1e-4, 10.0, log=True),
    }
    ba, iters = quick_eval(p)
    trial.set_user_attr("best_iter", int(iters) if iters else 0)
    return ba


t1 = time.time()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)
print(f"搜索完成 [{time.time()-t1:.0f}s]")
print(f"Best trial #{study.best_trial.number}: BA={study.best_value:.6f}")
best_params = study.best_params
best_iters = study.best_trial.user_attrs.get("best_iter", 0)
print(f"Best params: {json.dumps(best_params, indent=2)}")
print(f"Best iters: {best_iters}")

# ── 2. 5-fold OOF 验证 (全量数据) ────────────────────────────
print("\n[步骤2] 5-fold OOF 验证（全量数据 + 最优参数）")
print("-" * 40)


def build_lgbm_optuna(seed):
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=max(best_iters * 2, 500) if best_iters else 1000,
        learning_rate=best_params["lr"],
        num_leaves=best_params["leaves"],
        max_depth=best_params["depth"],
        min_child_samples=best_params["mcs"],
        subsample=best_params["sub"],
        subsample_freq=1,
        colsample_bytree=best_params["col"],
        reg_alpha=best_params["ra"],
        reg_lambda=best_params["rl"],
        class_weight="balanced",
        random_state=seed,
        verbose=-1, n_jobs=1,
    )


t2 = time.time()
result = run_oof(build_lgbm_optuna, config.RANDOM_STATE, train_full, test_df, verbose=True)
print(f"OOF 完成 [{time.time()-t2:.0f}s]")
print(f"各 fold BA: {[f'{b:.6f}' for b in result['fold_bas']]}")
print(f"OOF Balanced Accuracy: {result['oof_ba']:.6f}")

# ── 3. 对比 ─────────────────────────────────────────────────
print("\n[步骤3] 基线对比")
print("-" * 40)
print(f"│ 实验        │ OOF BA    │ vs 基线   │")
print(f"│ 基线 LGBM   │ 0.948446  │ —         │")
print(f"│ Optuna LGBM │ {result['oof_ba']:.6f} │ {result['oof_ba']-0.948446:+.6f} │")

# ── 4. 保存 ─────────────────────────────────────────────────
optuna_result = {
    "experiment": "B_optuna_lgbm",
    "best_params": best_params,
    "best_iters_from_sample": best_iters,
    "best_trial_value": study.best_value,
    "n_trials": len(study.trials),
    "oof_ba": float(result["oof_ba"]),
    "fold_bas": [float(b) for b in result["fold_bas"]],
    "vs_baseline": float(result["oof_ba"] - 0.948446),
}

os.makedirs(config.OUT_DIR, exist_ok=True)
with open(os.path.join(config.OUT_DIR, "exp_b_optuna_result.json"), "w") as f:
    json.dump(optuna_result, f, indent=2, ensure_ascii=False)

np.save(os.path.join(config.OUT_DIR, "exp_b_oof_prob.npy"), result["oof_prob"])
np.save(os.path.join(config.OUT_DIR, "exp_b_test_prob.npy"), result["test_prob"])

print(f"\n结果已保存到 outputs/exp_b_optuna_result.json")
print(f"总耗时: {time.time()-t0:.0f}s")
print("实验B 完成 ✓")
