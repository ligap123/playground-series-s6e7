"""
项目配置文件 — Predicting Student Health Risk (S6E7)
"""
import os

# ============================================================
# 路径配置（使用相对路径）
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
REF_DIR = os.path.join(BASE_DIR, "references")

# 数据文件
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(PRED_DIR, "submission.csv")

# ============================================================
# 竞赛配置
# ============================================================
TARGET_COL = "health_condition"
ID_COL = "id"
TARGET_CLASSES = ["at-risk", "unhealthy", "fit"]  # 按字母顺序

# ============================================================
# 训练配置
# ============================================================
RANDOM_STATE = 42
N_FOLDS = 5  # 交叉验证折数
TEST_SIZE = 0.2  # 验证集比例
N_TRIALS_OPTUNA = 50  # Optuna 超参数搜索次数（如果使用）

# ============================================================
# 模型保存配置
# ============================================================
BEST_MODEL_PREFIX = "best_model"

# 确保输出目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
