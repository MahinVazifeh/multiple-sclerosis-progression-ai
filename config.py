from pathlib import Path

# =========================
# Paths
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_INPUT_FILE = DATA_DIR / "Clinical_Demographic_Treatment_AllFeatures.csv"
REFERENCE_PICKLE_FILE = DATA_DIR / "Measurement_Group_3_Classify_AllTestDataClinical_Env.pickle"

RESULTS_PICKLE_FILE = OUTPUT_DIR / "sequence_model_results.pickle"
DISTRIBUTION_FILE = OUTPUT_DIR / "msss_decile_distribution.csv"
TRAINING_LOSS_PLOT_FILE = OUTPUT_DIR / "training_loss_comparison.jpg"

FEATURE_IMPORTANCE_OUTPUT_FILE = OUTPUT_DIR / "feature_importance.csv"
SELECTED_FEATURES_OUTPUT_FILE = OUTPUT_DIR / "selected_features.txt"
AUTOGLUON_MODEL_DIR = OUTPUT_DIR / "autogluon_feature_selection_model"

# =========================
# Column Names
# =========================

# Patient identifiers
PATIENT_ID_COLUMN = "Patient_ID"
PATIENT_NUMBER_COLUMN = "Patient_number"

# Time-related columns
OBSERVATION_TIME_COLUMN = "observation_time"
NUM_OBS_COLUMN = "num_obs"

# Target-related columns
MSSS_COLUMN = "MSSS"
MSSS_CLASSIFIED_COLUMN = "MSSS_classified"
TARGET_COLUMN = "MSSS_Classify"

# Clinical features
EDSS_COLUMN = "EDSS"
DISEASE_DURATION_YEARS_COLUMN = "DiseaseDuration_Years"

# =========================
# Feature Selection Parameters
# =========================

TEST_SIZE = 0.10
RANDOM_STATE = 1
TIME_LIMIT = 160
EVAL_METRIC = "balanced_accuracy"
PROBLEM_TYPE = "multiclass"
PRESETS = "best_quality"
TOP_K_FEATURES = 21

# =========================
# Preprocessing Parameters
# =========================

MAX_OBSERVATIONS_PER_PATIENT = 40

# Original notebook logic for dropping columns by position
DROP_COLUMN_INDEX_RANGES = [
    (1, 6),
    (7, 14),
]

# =========================
# Training Parameters
# =========================

N_SPLITS = 5
MODEL_NAMES = ["RNN", "LSTM", "GRU"]
SEQUENCE_LENGTHS = [2, 3, 4]
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001