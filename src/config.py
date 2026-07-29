"""v2 配置"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
MODEL_DIR = os.path.join(OUT_DIR, "models")
FIG_DIR = os.path.join(OUT_DIR, "figures")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

TARGET_COL = "health_condition"
ID_COL = "id"

NUM_COLS = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
            "step_count", "exercise_duration", "water_intake"]
CAT_COLS = ["diet_type", "stress_level", "sleep_quality",
            "physical_activity_level", "smoking_alcohol", "gender"]
FEAT_COLS = NUM_COLS + CAT_COLS

N_FOLDS = 5
SEEDS = [42, 2026, 3407]
RANDOM_STATE = 42

for d in [PRED_DIR, MODEL_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)
