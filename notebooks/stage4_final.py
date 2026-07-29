"""阶段4: 最终融合 + 生成 submission.csv + 实验报告

OOF在220K采样数据上，test预测在295K全量test上。
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from src import config

# ── 0. 加载数据 ──────────────────────────────────────────────
print("=" * 60)
print("阶段4: 最终融合 + 生成 submission")
print("=" * 60)

t0 = time.time()
# OOF是在220K采样上生成的
sample_df = pd.read_parquet(os.path.join(config.DATA_DIR, "train_sample220k.parquet"))
sub_df = pd.read_csv(os.path.join(config.DATA_DIR, "sample_submission.csv"))

y_le = LabelEncoder()
y_true = y_le.fit_transform(sample_df[config.TARGET_COL])
classes = y_le.classes_
print(f"类别: {classes.tolist()}")
print(f"OOF评估样本: {len(sample_df)}")

# ── 1. 加载所有 OOF 概率 ────────────────────────────────────
pred_dir = os.path.join(config.OUT_DIR, "predictions")
oof = {}
test = {}

for s in [42, 2026, 3407]:
    oof[f"lgbm_s{s}"] = np.load(os.path.join(pred_dir, f"lgbm_oof_s{s}.npy"))
    test[f"lgbm_s{s}"] = np.load(os.path.join(pred_dir, f"lgbm_test_s{s}.npy"))
oof["cb_s42"] = np.load(os.path.join(pred_dir, "cb_oof_s42.npy"))
test["cb_s42"] = np.load(os.path.join(pred_dir, "cb_test_s42.npy"))

n = len(oof["lgbm_s42"])
print(f"OOF 样本数: {n}, 模型数: {len(oof)}")
print(f"Test 行数: {len(test['lgbm_s42'])}")

# ── 2. 单模型验证 ───────────────────────────────────────────
print("\n单模型 OOF BA:")
baselines = {}
for name, prob in oof.items():
    ba = balanced_accuracy_score(y_true, prob.argmax(1))
    baselines[name] = ba
    print(f"  {name:12s}: {ba:.6f}")

lgbm_avg = (oof["lgbm_s42"] + oof["lgbm_s2026"] + oof["lgbm_s3407"]) / 3
lgbm_avg_ba = balanced_accuracy_score(y_true, lgbm_avg.argmax(1))
print(f"  lgbm_3avg   : {lgbm_avg_ba:.6f}")

# ── 3. 融合策略 ─────────────────────────────────────────────
print("\n" + "-" * 40)
print("融合策略评估:")
print("-" * 40)

results = []

# (A) 等权平均
all_avg = (oof["lgbm_s42"] + oof["lgbm_s2026"] + oof["lgbm_s3407"] + oof["cb_s42"]) / 4
ba_a = balanced_accuracy_score(y_true, all_avg.argmax(1))
results.append(("A) equal_4way", ba_a))

# (B) LGBM 3-seed avg + CB 等权
mix_eq = (lgbm_avg + oof["cb_s42"]) / 2
ba_b = balanced_accuracy_score(y_true, mix_eq.argmax(1))
results.append(("B) lgbm3+cb_eq", ba_b))

# (C) LGBM 3-seed only
results.append(("C) lgbm3_only", lgbm_avg_ba))

# (D) 权重网格搜索
print("\n权重网格搜索 (w_lgbm3, w_cb):")
best_ba, best_w = 0, 0.5
for w in np.arange(0.50, 0.96, 0.05):
    prob = w * lgbm_avg + (1 - w) * oof["cb_s42"]
    ba = balanced_accuracy_score(y_true, prob.argmax(1))
    marker = " ←" if ba > best_ba else ""
    if ba > best_ba:
        best_ba, best_w = ba, w
    print(f"  w_lgbm={w:.2f} w_cb={1-w:.2f}  BA={ba:.6f}{marker}")
results.append((f"D) best_w={best_w:.2f}", best_ba))

# (E) Rank average
from scipy.stats import rankdata
rank_avg = np.zeros_like(lgbm_avg)
for prob in [oof["lgbm_s42"], oof["lgbm_s2026"], oof["lgbm_s3407"], oof["cb_s42"]]:
    for cls in range(3):
        rank_avg[:, cls] += rankdata(prob[:, cls]) / 4
ba_e = balanced_accuracy_score(y_true, rank_avg.argmax(1))
results.append(("E) rank_avg", ba_e))

# 打印汇总
print(f"\n{'='*50}")
print(f"│ 策略                    │ OOF BA    │")
print(f"│ 基线参考 (lgbm_s42)     │ {baselines.get('lgbm_s42', 0):.6f} │")
for name, ba in results:
    print(f"│ {name:24s} │ {ba:.6f} │")

# ── 4. 选择最优 ─────────────────────────────────────────────
best_name, best_ba = max(results, key=lambda x: x[1])
print(f"\n最优: {best_name}  OOF BA = {best_ba:.6f}")

# 生成 final_test
lgbm_test_avg = (test["lgbm_s42"] + test["lgbm_s2026"] + test["lgbm_s3407"]) / 3

if best_name.startswith("D)"):
    final_oof = best_w * lgbm_avg + (1 - best_w) * oof["cb_s42"]
    final_test = best_w * lgbm_test_avg + (1 - best_w) * test["cb_s42"]
elif best_name == "A) equal_4way":
    final_oof = all_avg
    final_test = (test["lgbm_s42"] + test["lgbm_s2026"] + test["lgbm_s3407"] + test["cb_s42"]) / 4
elif best_name == "B) lgbm3+cb_eq":
    final_oof = mix_eq
    final_test = (lgbm_test_avg + test["cb_s42"]) / 2
elif best_name == "C) lgbm3_only":
    final_oof = lgbm_avg
    final_test = lgbm_test_avg
else:  # E) rank_avg
    final_oof = rank_avg
    rank_test = np.zeros_like(test["lgbm_s42"])
    for prob in [test["lgbm_s42"], test["lgbm_s2026"], test["lgbm_s3407"], test["cb_s42"]]:
        for cls in range(3):
            rank_test[:, cls] += rankdata(prob[:, cls]) / 4
    final_test = rank_test

# ── 5. 生成 submission ─────────────────────────────────────
print(f"\n生成 submission.csv")
pred_labels = classes[final_test.argmax(1)]
sub_df["health_condition"] = pred_labels

sub_path = os.path.join(config.PRED_DIR, "submission.csv")
sub_df.to_csv(sub_path, index=False)
print(f"  已保存: {sub_path}")
print(f"  行数: {len(sub_df)}, 分布: {dict(zip(*np.unique(pred_labels, return_counts=True)))}")

# 验证
assert sub_df.columns.tolist() == ["id", "health_condition"]
assert len(sub_df) == 295753
assert set(sub_df["health_condition"].unique()).issubset(set(classes))
print("  [OK] 格式验证通过")

# ── 6. 保存 ────────────────────────────────────────────────
final_result = {
    "best_strategy": best_name,
    "oof_ba": float(best_ba),
    "all_results": [(n, float(b)) for n, b in results],
    "single_model_bas": {k: float(v) for k, v in baselines.items()},
    "n_models": len(oof),
    "classes": classes.tolist(),
    "eval_on": "220K sample",
    "test_on": "295K full test",
}
with open(os.path.join(config.OUT_DIR, "final_result.json"), "w") as f:
    json.dump(final_result, f, indent=2, ensure_ascii=False)

np.save(os.path.join(config.OUT_DIR, "final_oof_prob.npy"), final_oof)
np.save(os.path.join(config.OUT_DIR, "final_test_prob.npy"), final_test)

print(f"\n阶段4 融合完成 [OK] [{time.time()-t0:.0f}s]")
