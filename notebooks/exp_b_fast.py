"""实验B: Optuna调参 LGBM 验证（early_stopping加速版）

用 cv.py 的框架，但模型内加 early_stopping（每 fold 用 10% 训练数据做验证）
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from src import config
from src.preprocessing import fit_fold_preprocess, transform_fold

# ── 加载参数 ─────────────────────────────────────────────────
with open(os.path.join(config.OUT_DIR, "lgbm_best_params.json")) as f:
    prev = json.load(f)
bp = prev["params"]
print(f"加载参数: BA(sample)={prev['best_value']:.6f}")

# ── 加载数据 ─────────────────────────────────────────────────
t0 = time.time()
train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)
print(f"数据: train={train_df.shape}, test={test_df.shape} [{time.time()-t0:.0f}s]")

# ── 5-fold OOF (with early_stopping) ────────────────────────
y_le = LabelEncoder().fit_transform(train_df[config.TARGET_COL])
n_train, n_test = len(train_df), len(test_df)
oof_prob = np.zeros((n_train, 3), dtype=np.float32)
test_prob = np.zeros((n_test, 3), dtype=np.float32)
fold_bas = []

skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_le)):
    tfold = time.time()
    tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]

    # fold-local 预处理
    params = fit_fold_preprocess(tr_df, config.NUM_COLS, config.CAT_COLS)
    Xtr, feats = transform_fold(tr_df, params, config.NUM_COLS, config.CAT_COLS)
    Xva, _ = transform_fold(va_df, params, config.NUM_COLS, config.CAT_COLS)
    Xte, _ = transform_fold(test_df, params, config.NUM_COLS, config.CAT_COLS)

    # 从训练集中再分出 15% 做 early_stopping 验证
    Xtr_train, Xtr_es, ytr_train, ytr_es = train_test_split(
        Xtr, y_le[tr_idx], test_size=0.15, stratify=y_le[tr_idx],
        random_state=config.RANDOM_STATE + fold
    )

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=2000,
        learning_rate=bp["lr"],
        num_leaves=bp["leaves"],
        max_depth=bp["depth"],
        min_child_samples=bp["mcs"],
        subsample=bp["sub"],
        subsample_freq=1,
        colsample_bytree=bp["col"],
        reg_alpha=bp["ra"],
        reg_lambda=bp["rl"],
        class_weight="balanced",
        random_state=config.RANDOM_STATE + fold,
        verbose=-1, n_jobs=1,
    )

    model.fit(
        Xtr_train, ytr_train,
        eval_set=[(Xtr_es, ytr_es)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(0)],
    )

    best_iter = model.best_iteration_ or 2000
    oof_prob[va_idx] = model.predict_proba(Xva)
    test_prob += model.predict_proba(Xte) / config.N_FOLDS

    ba = balanced_accuracy_score(y_le[va_idx], oof_prob[va_idx].argmax(1))
    fold_bas.append(ba)
    elapsed = time.time() - tfold
    print(f"  fold{fold+1} BA={ba:.6f}  iters={best_iter}  [{elapsed:.0f}s]", flush=True)

oof_pred = oof_prob.argmax(1)
oof_ba = balanced_accuracy_score(y_le, oof_pred)

print(f"\n各 fold BA: {[f'{b:.6f}' for b in fold_bas]}")
print(f"OOF Balanced Accuracy: {oof_ba:.6f}")
print(f"总耗时: {time.time()-t0:.0f}s")

# ── 对比与决策 ──────────────────────────────────────────────
delta = oof_ba - 0.948446
decision = "✅ 保留" if delta >= 0.0002 else "⚠️ 收益微小"

print(f"\n基线对比:")
print(f"  基线 LGBM:     0.948446")
print(f"  Optuna LGBM:  {oof_ba:.6f}  ({delta:+.6f})")
print(f"  决策: {decision}")

# ── 保存 ────────────────────────────────────────────────────
result = {
    "experiment": "B_optuna_lgbm",
    "best_params": bp,
    "sample_ba": prev["best_value"],
    "oof_ba": float(oof_ba),
    "fold_bas": [float(b) for b in fold_bas],
    "vs_baseline": float(delta),
    "decision": decision,
}
with open(os.path.join(config.OUT_DIR, "exp_b_optuna_result.json"), "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

np.save(os.path.join(config.OUT_DIR, "exp_b_oof_prob.npy"), oof_prob)
np.save(os.path.join(config.OUT_DIR, "exp_b_test_prob.npy"), test_prob)

print(f"\n实验B 完成 ✓")
