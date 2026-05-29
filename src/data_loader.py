#Función para cargar y transformar los datos.  
"""
data_loader.py
--------------
Responsabilidad única: cargar el CSV y devolver un DataFrame limpio y consistente,
aplicando todas las decisiones de limpieza tomadas durante el EDA.

 
Que hace este módulo:
    - Carga y valida el CSV
    - Elimina columnas con leakage o baja varianza (decisión fija del EDA)
    - Imputa nulos conocidos
    - Binariza 'agent' → 'has_agent'
    - Agrupa países minoritarios → OTHER
    - Winsoriza 'adr'
    - Elimina filas inválidas (adr negativos, market_segment Undefined)
    - Garantiza que is_canceled es numérica (0/1)
 
Lo que NO hace este módulo:
    - Encoding (One-Hot, Label) — responsabilidad de preprocessor.py
    - Scaling — responsabilidad de preprocessor.py
    - Split train/test — responsabilidad de preprocessor.py
"""

import logging
from pathlib import Path
 
import pandas as pd
 
from config import (
    ADR_PERCENTILE,
    COLS_TO_DROP,
    COUNTRY_OTHER_LABEL,
    MARKET_SEGMENT_UNDEFINED,
    RAW_DATA_PATH,
    TARGET_COL,
    TOP_COUNTRIES,
)
 
logger = logging.getLogger(__name__)

def load_clean_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carga el CSV y la limpieza descrita en el EDA.

    Parameters
    ----------
    path : Path  Ruta al CSV original. Por defecto usa RAW_DATA_PATH de config.

    Returns
    -------
    pd.DataFrame limpio listo para pasar a preprocessor.py.

    Raises
    ------
    FileNotFoundError
        Si el CSV no existe en la ruta indicada.
    ValueError
        Si el DataFrame resultante está vacío o le faltan columnas esperadas.
    """

    logger.info(f"Cargando datos desde: {path}")
    df = _read_csv(path)
    logger.info(f"Registros cargados: {len(df):,}")

    df = _drop_leakage_columns(df)
    # df = _fix_target(df)
    df = _impute_children(df)
    df = _binarize_agent(df)
    df = _group_countries(df)
    df = _remove_invalid_rows(df)
    df = _remove_sup99_adr(df)

    _validate_output(df)
    logger.info(f"Registros tras limpieza: {len(df):,}")
    return df


# Funciones privadas
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset en: {path}")
    return pd.read_csv(path)

def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas identificadas en el EDA a eliminar. Se traen de config."""
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    dropped = [c for c in COLS_TO_DROP if c not in df.columns]
    if dropped:
        logger.warning(f"Columnas a eliminar no encontradas (ya ausentes): {dropped}")
    return df.drop(columns=cols_present)

def _impute_children(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa nulos en 'children' con 0 (decisión EDA: ausencia = 0 niños)."""
    if df["children"].isna().any():
        n = df["children"].isna().sum()
        logger.info(f"Imputando {n} nulos en 'children' con 0")
        df["children"] = df["children"].fillna(0).astype(int)
    return df

def _binarize_agent(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte 'agent' (ID numérico o nulo) en 'has_agent' (0/1)."""
    df["has_agent"] = df["agent"].notna().astype(int)
    df = df.drop(columns=["agent"])
    logger.info("'agent' binarizada como 'has_agent' y columna original eliminada")
    return df

def _group_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa países minoritarios bajo la etiqueta OTHER."""
    original_unique = df["country"].nunique()
    df["country"] = df["country"].apply(
        lambda x: x if x in TOP_COUNTRIES else COUNTRY_OTHER_LABEL
    )
    logger.info(
        f"'country' agrupada: {original_unique} valores únicos → "
        f"{df['country'].nunique()} (top + OTHER)"
    )
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

def _remove_sup99_adr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina los valores superiores al percentil P99 en 'adr'.
    Valores por encima del umbral se recortan al umbral.
    """
    cap = df["adr"].quantile(ADR_PERCENTILE)
    n_capped = (df["adr"] > cap).sum()
    df["adr"] = df["adr"].clip(upper=cap)
    logger.info(
        f"'adr' winsorizado en P{int(ADR_PERCENTILE*100)}: "
        f"cap={cap:.2f}, registros afectados={n_capped}"
    )
    return df

def _validate_output(df: pd.DataFrame) -> None:
    """
    Validaciones mínimas sobre el DataFrame limpio antes de devolverlo.
    Lanza ValueError si algo crítico falla.
    """
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