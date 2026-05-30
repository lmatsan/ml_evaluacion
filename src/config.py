from pathlib import Path

# =====================================================================
# 1. Rutas del Sistema (Estructura de Carpetas)
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Rutas de datos
RAW_DATA_FILENAME = "dataset_practica_final.csv"
RAW_DATA_PATH = DATA_RAW_DIR / RAW_DATA_FILENAME
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "clean_dataset.csv"

# Rutas de Gráficas e Informes
COMPARATIVE_ROC_CURVE_PATH = OUTPUTS_DIR / "comparative_roc_curve.png"
BEST_MODEL_CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "best_model_confusion_matrix.png"
RANDOM_FOREST_IMPORTANCE_PLOT_PATH = OUTPUTS_DIR / "random_forest_feature_importances.png"
TRAINING_REPORT_PATH = OUTPUTS_DIR / "training_report.md"

# Rutas de Guardado de Modelos y Serialización
BEST_MODEL_PIPELINE_PATH = OUTPUTS_DIR / "best_model_pipeline.joblib"
BEST_NEURAL_NETWORK_MODEL_PATH = OUTPUTS_DIR / "best_model.keras"
BEST_NEURAL_PREPROCESSOR_PATH = OUTPUTS_DIR / "best_model_preprocessor.joblib"
BEST_PREPROCESSING_RULES_PATH = OUTPUTS_DIR / "best_model_preprocessing_rules.joblib"

# =====================================================================
# 2. Definición del Dataset y Tipados (Machine Learning)
# =====================================================================
TARGET_COL = "is_canceled"  
MAIN_METRIC = "roc_auc"
CLASS_LABELS = ["Not canceled", "Canceled"]

# Columnas esperadas
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

# Columnas a eliminar — decisiones tomadas en el EDA
COLS_TO_DROP = [
    "reservation_status",       # Info no disponible en la reserva.
    "reservation_status_date",  # Info no disponible en la reserva.
    "arrival_date_week_number", # Correlación perfecta con arrival_date_month 
    ]

# Columnas numéricas y categóricas (tras limpieza, antes de encoding)
NUMERIC_COLS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "has_agent",       
    "has_company",   
    "room_count",    
    ]

CATEGORICAL_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    ]

# Eliminamos percentiles muy altos de ADR para evitar outliers.
ADR_COLUMN = "adr"

# Países que cubren el 86.2% de las reservas
COUNTRY_COLUMN = "country"

# Variables compuestas para los modelos
FINAL_FEATURE_COLUMNS = CATEGORICAL_COLS + NUMERIC_COLS
FINAL_DATAFRAME_COLUMNS = FINAL_FEATURE_COLUMNS + [TARGET_COL]

# =====================================================================
# 3. Parámetros del Pipeline (Limpieza y Modelado)
# =====================================================================
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

# Percentil para mitigar outliers mediante Winsorización estricta en Train
ADR_PERCENTILE = 0.99

# Valor a limpiar en las filas de mercado
MARKET_SEGMENT_UNDEFINED = "Undefined"

# Configuración del tratamiento dinámico de países en preprocessor.py
COUNTRY_TOP_N = 11  # Número de países con más volumen a retener en Train (~86.2%)
COUNTRY_OTHER_LABEL = "OTHER"

# Parámetros globales para la división de conjuntos (train_test_split)
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

# Valores por defecto para la carga estática inicial (data_loader)
AGENT_MISSING_VALUE = "no agent"
COMPANY_MISSING_VALUE = "no company"
CHILDREN_IMPUTATION_VALUE = 0

FIXED_FILL_VALUES = {
    "children": CHILDREN_IMPUTATION_VALUE,
    "agent": AGENT_MISSING_VALUE,
    "company": COMPANY_MISSING_VALUE,
}

# =====================================================================
# 4. Hiperparámetros de los Modelos (Aportación de tu compañero)
# =====================================================================
RANDOM_FOREST_IMPORTANCE_TOP_N = 20

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

# =====================================================================
# 5. Tracking y Despliegue (MLflow / Registro de Modelos)
# =====================================================================
MLFLOW_EXPERIMENT_NAME = "hotel_cancellation_prediction"
MLFLOW_MODEL_NAME = "best_model"