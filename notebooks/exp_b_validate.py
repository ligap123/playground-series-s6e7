"""实验B: 用已有最优参数跑 5-fold OOF 完整验证"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import lightgbm as lgb

from src import config
from src.cv import run_oof

# ── 加载已有最优参数 ─────────────────────────────────────────
with open(os.path.join(config.OUT_DIR, "lgbm_best_params.json")) as f:
    prev = json.load(f)
best_params = prev["params"]
print(f"加载已有最优参数: BA(sample)={prev['best_value']:.6f}")
print(f"params: {json.dumps(best_params, indent=2)}")

# ── 加载全量数据 ─────────────────────────────────────────────
t0 = time.time()
train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)
print(f"数据: train={train_df.shape}, test={test_df.shape} [{time.time()-t0:.0f}s]")

# ── 5-fold OOF ──────────────────────────────────────────────
print("\n[5-fold OOF] 全量数据验证")
print("-" * 40)

def build_lgbm_tuned(seed):
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=1500,
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

t1 = time.time()
result = run_oof(build_lgbm_tuned, config.RANDOM_STATE, train_df, test_df, verbose=True)
oof_time = time.time() - t1

print(f"\n各 fold BA: {[f'{b:.6f}' for b in result['fold_bas']]}")
print(f"OOF Balanced Accuracy: {result['oof_ba']:.6f}")
print(f"耗时: {oof_time:.0f}s")

# ── 对比 ────────────────────────────────────────────────────
print("\n基线对比:")
print(f"  基线 LGBM (v1):     0.948446")
print(f"  Optuna LGBM (实验B): {result['oof_ba']:.6f}  ({result['oof_ba']-0.948446:+.6f})")
print(f"  3-seed LGBM (实验A):  0.949065")
print(f"  CatBoost (实验C):     0.948623")

delta = result['oof_ba'] - 0.948446
decision = "✅ 保留" if delta >= 0.0002 else "⚠️ 收益不足，可选保留"

# ── 保存 ────────────────────────────────────────────────────
optuna_result = {
    "experiment": "B_optuna_lgbm",
    "best_params": best_params,
    "sample_ba": prev["best_value"],
    "oof_ba": float(result["oof_ba"]),
    "fold_bas": [float(b) for b in result["fold_bas"]],
    "vs_baseline": float(delta),
    "decision": decision,
}

with open(os.path.join(config.OUT_DIR, "exp_b_optuna_result.json"), "w") as f:
    json.dump(optuna_result, f, indent=2, ensure_ascii=False)

np.save(os.path.join(config.OUT_DIR, "exp_b_oof_prob.npy"), result["oof_prob"])
np.save(os.path.join(config.OUT_DIR, "exp_b_test_prob.npy"), result["test_prob"])

print(f"\n决策: {decision}")
print(f"结果已保存: outputs/exp_b_optuna_result.json")
print(f"实验B 完成 ✓ [{time.time()-t0:.0f}s]")
