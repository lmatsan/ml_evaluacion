"""
config.py
---------
Constantes globales del proyecto. Fuente única de verdad para nombres de columnas,
rutas, parámetros de limpieza y configuración de MLFlow.
 
Principio: cualquier valor que aparezca en más de un módulo vive aquí.
"""

from pathlib import Path

# Rutas
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
 
RAW_DATA_FILENAME = "dataset_practica_final.csv"
RAW_DATA_PATH = DATA_RAW_DIR / RAW_DATA_FILENAME

# Variable objetivo
TARGET_COL = "is_canceled"

# Columnas a eliminar — decisiones tomadas en el EDA
COLS_TO_DROP = [
    "reservation_status",       # Info no disponible en la reserva.
    "reservation_status_date",  # Info no disponible en la reserva.
    "arrival_date_week_number", # Correlación perfecta con arrival_date_month
    "company",                  # 94% nulos, varianza casi nula
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
    "has_agent",                # Binarizada desde agent
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

# Parámetros de limpieza

# Eliminamos percentiles muy altos de ADR para evitar outliers.
ADR_PERCENTILE = 0.99

# Valor de market_segment a eliminar (filas, no columna)
MARKET_SEGMENT_UNDEFINED = "Undefined"

# Países que cubren el 86.2% de las reservas
TOP_COUNTRIES = [
    'PRT', 'GBR', 'FRA', 'ESP', 'DEU',
    'ITA', 'IRL', 'BEL', 'BRA', 'NLD',
    'USA'] # Representan segun el 86.2% de las reservas

COUNTRY_OTHER_LABEL = "OTHER"

# Valor de market_segment a eliminar (filas, no columna)
MARKET_SEGMENT_UNDEFINED = "Undefined"

# MLFlow
MLFLOW_EXPERIMENT_NAME = "hotel_cancellation_prediction"
MLFLOW_MODEL_NAME = "best_model"
MLFLOW_STAGE = "Production"