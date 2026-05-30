from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DATASET_PATH = BASE_DIR / "data" / "raw" / "dataset_practica_final.csv"
COMPARATIVE_ROC_CURVE_PATH = OUTPUTS_DIR / "comparative_roc_curve.png"
BEST_MODEL_CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "best_model_confusion_matrix.png"
RANDOM_FOREST_IMPORTANCE_PLOT_PATH = (
    OUTPUTS_DIR / "random_forest_feature_importances.png"
)
BEST_MODEL_PIPELINE_PATH = OUTPUTS_DIR / "best_model_pipeline.joblib"
BEST_NEURAL_NETWORK_MODEL_PATH = OUTPUTS_DIR / "best_model.keras"
BEST_NEURAL_PREPROCESSOR_PATH = OUTPUTS_DIR / "best_model_preprocessor.joblib"
BEST_PREPROCESSING_RULES_PATH = (
    OUTPUTS_DIR / "best_model_preprocessing_rules.joblib"
)
TRAINING_REPORT_PATH = OUTPUTS_DIR / "training_report.md"

TARGET_COLUMN = "is_canceled"
MAIN_METRIC = "roc_auc"
CLASS_LABELS = ["Not canceled", "Canceled"]

RANDOM_FOREST_IMPORTANCE_TOP_N = 20
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

COUNTRY_TOP_N = 11
ADR_UPPER_QUANTILE = 0.99
AGENT_MISSING_VALUE = "no agent"
COMPANY_MISSING_VALUE = "no company"
COUNTRY_OTHER_LABEL = "OTHER"
UNDEFINED_MARKET_SEGMENT_VALUE = "Undefined"
CHILDREN_IMPUTATION_VALUE = 0

LOGISTIC_C = 0.1
LOGISTIC_MAX_ITER = 500
LOGISTIC_SOLVER = "liblinear"
DECISION_TREE_MAX_DEPTH = 8
RANDOM_FOREST_N_ESTIMATORS = 200
CATBOOST_ITERATIONS = 300
CATBOOST_DEPTH = 6
CATBOOST_LEARNING_RATE = 0.05
NEURAL_NETWORK_HIDDEN_UNITS = [64, 32]
NEURAL_NETWORK_EPOCHS = 30
NEURAL_NETWORK_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5

LEAKAGE_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
]
REDUNDANT_COLUMNS = [
    "arrival_date_week_number",
]

MONTH_MAPPING = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

BASE_TYPE_CASTS = {
    "hotel": "str",
    "arrival_date_month": "str",
    "meal": "str",
    "market_segment": "str",
    "distribution_channel": "str",
    "is_repeated_guest": "str",
    "reserved_room_type": "str",
    "assigned_room_type": "str",
    "deposit_type": "str",
    "customer_type": "str",
}

FINAL_TYPE_CASTS = {
    "hotel": "str",
    "meal": "str",
    "country": "str",
    "market_segment": "str",
    "distribution_channel": "str",
    "is_repeated_guest": "str",
    "reserved_room_type": "str",
    "assigned_room_type": "str",
    "deposit_type": "str",
    "customer_type": "str",
}

FIXED_FILL_VALUES = {
    "children": CHILDREN_IMPUTATION_VALUE,
    "agent": AGENT_MISSING_VALUE,
    "company": COMPANY_MISSING_VALUE,
}

RAW_DATASET_COLUMNS = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]

NUMERICAL_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "has_agent",
    "has_company",
]

FINAL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
FINAL_DATAFRAME_COLUMNS = FINAL_FEATURE_COLUMNS + [TARGET_COLUMN]
