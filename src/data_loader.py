import logging
from pathlib import Path
from typing import Union
import pandas as pd
 
from src.config import (
    COLS_TO_DROP,
    MARKET_SEGMENT_UNDEFINED,
    RAW_DATA_PATH,
    TARGET_COL,
    RAW_DATASET_COLUMNS,
    COUNTRY_COLUMN,
    )

logger = logging.getLogger(__name__)

def load_features_and_target(path: Union[str, Path] = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Reúne la carga, la revisión y la separación de los datos en un solo paso."""
    path = Path(path)
    dataframe_raw = _read_csv(path)
    validate_dataset_structure(dataframe_raw)
    dataframe = load_clean_data(dataframe_raw)
    
    return split_features_and_target(dataframe, TARGET_COL)

def validate_dataset_structure(dataframe: pd.DataFrame) -> None:
    """Revisa que el archivo original tenga la estructura mínima requerida."""
    if dataframe.empty:
        raise ValueError("dataset empty")

    if TARGET_COL not in dataframe.columns:
        raise ValueError(
            f"column '{TARGET_COL}' missing"
        )

    missing_columns = _find_missing_columns(dataframe)
    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        raise ValueError(
            f"missing columns: {missing_columns_text}"
        )

    duplicated_columns = _find_duplicated_columns(dataframe)
    if duplicated_columns:
        duplicated_columns_text = ", ".join(duplicated_columns)
        raise ValueError(
            "duplicated column: "
            f"{duplicated_columns_text}"
        )

def load_clean_data(
    df_or_path: Union[pd.DataFrame, str, Path] = RAW_DATA_PATH
) -> pd.DataFrame:
    """Carga el CSV y aplica las decisiones de limpieza tomadas durante el EDA."""
    if isinstance(df_or_path, (str, Path)):
        path = Path(df_or_path)
        logger.info(f"Cargando datos desde: {path}")
        df = _read_csv(path)
    else:
        df = df_or_path.copy()
    logger.info(f"Registros cargados: {len(df):,}")

    # Transformaciones secuenciales
    df = _drop_leakage_columns(df)
    df = _impute_country_mode(df)
    df = _impute_children(df)
    df = _binarize_agent_company(df)
    df = _remove_invalid_rows(df)
    df = _collapse_duplicates(df)
    # --- Feature engineering ---
    # df = _add_cancellation_ratio(df)
    df = _add_total_nights(df)
    # df = _add_weekend_flag(df)
    # ---------------------------

    _validate_output(df)
    logger.info(f"Registros tras limpieza: {len(df):,}")
    return df


def split_features_and_target(
    dataframe: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    '''Separa la columna que queremos predecir del resto de variables.'''
    if target_column not in dataframe.columns:
        raise ValueError(
            f"column '{target_column}' missing"
        )

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]
    return features, target


# =====================================================================
# Funciones auxiliares / privadas
# =====================================================================

def _find_missing_columns(dataframe: pd.DataFrame) -> list[str]:
    """Busca qué columnas de la definición teórica no están físicamente."""
    return [
        column_name
        for column_name in RAW_DATASET_COLUMNS
        if column_name not in dataframe.columns
    ]


def _find_duplicated_columns(dataframe: pd.DataFrame) -> list[str]:
    """Detecta si el archivo trae nombres de columnas repetidas."""
    duplicated_mask = dataframe.columns.duplicated()
    return dataframe.columns[duplicated_mask].tolist()


def _read_csv(path: Path) -> pd.DataFrame:
    """Lee el fichero CSV controlando su existencia."""
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en: {path}")
    return pd.read_csv(path)

def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas con data leakage o baja varianza según diseño en el EDA."""
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    dropped = [c for c in COLS_TO_DROP if c not in df.columns]
    if dropped:
        logger.warning(f"Columnas a eliminar no encontradas (ya ausentes): {dropped}")
    return df.drop(columns=cols_present)

def _impute_country_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa los valores nulos de la columna 'country' usando su moda."""
    df = df.copy()
    
    # Calculamos la moda y rellenamos en una única operación segura
    country_mode = df[COUNTRY_COLUMN].mode()[0]
    df[COUNTRY_COLUMN] = df[COUNTRY_COLUMN].fillna(country_mode)
    
    logger.info(f"Imputación en '{COUNTRY_COLUMN}' completada usando la moda: '{country_mode}'.")
    return df

def _impute_children(df: pd.DataFrame) -> pd.DataFrame:
    """Asume ausencia (0) en los registros nulos de la columna 'children'."""
    if df["children"].isna().any():
        n = df["children"].isna().sum()
        logger.info(f"Imputando {n} nulos en 'children' con 0")
        df["children"] = df["children"].fillna(0).astype(int)
    return df

def _binarize_agent_company(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma las columnas 'agent' y 'company' en flags categóricos binarizados (0/1)."""
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    df = df.drop(columns=["agent", "company"])
    logger.info("Columnas 'agent' y 'company' binarizadas como 'has_agent' y 'has_company' y columnas originales eliminadas")
    return df


def _remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """ Elimina filas inválidas identificadas en el EDA:
        - adr negativos (precio negativo no tiene sentido de negocio)
        - market_segment == 'Undefined' (categoría sin significado)
    """
    n_before = len(df)

    mask_adr = df["adr"] < 0
    mask_undefined = df["market_segment"] == MARKET_SEGMENT_UNDEFINED
    mask_invalid = mask_adr | mask_undefined

    df = df[~mask_invalid].copy()
    n_removed = n_before - len(df)
    logger.info(f"Filas eliminadas (adr<0 o market_segment=Undefined): {n_removed}")
    return df


def _collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa registros idénticos añadiendo un contador de volumen."""
    n_before = len(df)
    
    # Colapsamos duplicados manteniendo los nulos para no perder registros
    df = df.value_counts(dropna=False).reset_index(name="room_count")
    
    n_after = len(df)
    logger.info(
        f"Registros colapsados por duplicidad: {n_before:,} → {n_after:,} "
        f"(Se creó la columna 'room_count')"
    )
    return df


def _validate_output(df: pd.DataFrame) -> None:
    """Asegura la calidad e integridad mínima del DataFrame procesado antes de su salida."""
    if df.empty:
        raise ValueError("El DataFrame resultante está vacío tras la limpieza.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET_COL}' no está presente.")

    if df[TARGET_COL].isnull().any():
        raise ValueError(f"La columna objetivo '{TARGET_COL}' contiene nulos.")

    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError(
            f"'{TARGET_COL}' contiene valores distintos de 0/1: {df[TARGET_COL].unique()}"
        )

    # Avisar (no lanzar) si quedan nulos en otras columnas
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        logger.warning(f"Columnas con nulos tras limpieza: {null_cols}")

def _add_cancellation_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """ Ratio de cancelaciones históricas del cliente"""
    df["cancellation_ratio"] = (
        df["previous_cancellations"] /
        (df["previous_cancellations"] + df["previous_bookings_not_canceled"] + 1)
    )
    return df

def _add_total_nights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features derivadas de duración y composición del grupo.
    """
    df["total_nights"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    )
    return df