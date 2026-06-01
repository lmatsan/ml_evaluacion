import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.data_loader import load_clean_data
from src.data_loader import split_features_and_target

from src.config import (
    ADR_PERCENTILE,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    COUNTRY_OTHER_LABEL,
    TARGET_COL,
    COUNTRY_TOP_N,
    COUNTRY_COLUMN, 
    ADR_COLUMN,
)


def prepare_training_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    dict,
    ]:
    """Prepara los conjuntos finales gestionando los splits y las reglas anti-leakage."""
    df_base = load_clean_data()

    df_train, df_validation = train_test_split(
        df_base,
        test_size=0.2,
        random_state=42,
        stratify=df_base[TARGET_COL],
    )
    # Ajustar reglas dinámicas sobre Train (Winsorización estricta de ML)
    preprocessing_rules = _fit_train_rules(df_train)
    
    # Aplicar reglas calculadas a ambos conjuntos
    df_train_proc = apply_train_rules(df_train, preprocessing_rules)
    df_validation_proc = apply_train_rules(df_validation, preprocessing_rules)
    
    # 5. Separación en Features (X) y Target (y)
    X_train, y_train = split_features_and_target(df_train_proc, TARGET_COL)
    X_validation, y_validation = split_features_and_target(df_validation_proc, TARGET_COL)
    
    return X_train, X_validation, y_train, y_validation, preprocessing_rules

# Prepara el tratamiento de columnas que necesita cada tipo de modelo.
def build_preprocessor(model_key: str) -> ColumnTransformer:
    if model_key == "logistic_regression":
        numerical_transformer = StandardScaler()
    elif model_key == "neural_network":
        numerical_transformer = MinMaxScaler()
    elif model_key in {"decision_tree", "random_forest", "catboost"}:
        numerical_transformer = "passthrough"
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, NUMERIC_COLS),
            ("categorical", categorical_transformer, CATEGORICAL_COLS),
        ]
    )


# =====================================================================
# Funciones Privadas (Reglas de Entrenamiento anti Data Leakage)
# =====================================================================

# Aprende las reglas de limpieza que dependen solo de los datos de entrenamiento.
def _fit_train_rules(df_train: pd.DataFrame) -> dict:
    """Aprende reglas estadísticas exclusivamente del conjunto de entrenamiento."""
    _ensure_columns_exist(
        df_train,
        [COUNTRY_COLUMN, ADR_COLUMN],
    )

    # Regla del ADR
    adr_upper_bound = df_train[ADR_COLUMN].quantile(ADR_PERCENTILE)
    
    # Regla de Países: Calculamos el TOP N dinámicamente usando SOLO Train
    top_countries = (
        df_train[COUNTRY_COLUMN]
        .value_counts()
        .head(COUNTRY_TOP_N) 
        .index.tolist()
    )
    
    return {
        "adr_upper_bound": float(adr_upper_bound),
        "top_countries": top_countries,  # Guardamos la lista de países top aprendida
    }


def apply_train_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Aplica las reglas estadísticas calculadas en el set de Train."""
    df_out = df.copy()
    
    # Aplicar el recorte de precio
    df_out[ADR_COLUMN] = df_out[ADR_COLUMN].clip(upper=rules["adr_upper_bound"])
    
    # Aplicar la reducción de países basada en la lista que aprendió Train
    df_out[COUNTRY_COLUMN] = df_out[COUNTRY_COLUMN].where(
        df_out[COUNTRY_COLUMN].isin(rules["top_countries"]),
        COUNTRY_OTHER_LABEL  # El resto se vuelve 'OTHER'
    )
    
    return df_out


def _ensure_columns_exist(
    dataframe: pd.DataFrame, required_columns: list[str]
) -> None:
    '''Comprueba que existan las columnas necesarias antes de seguir.'''
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in dataframe.columns
    ]

    if missing_columns:
        missing_columns_text = ", ".join(missing_columns)
        raise ValueError(
            "dataframe missing"
            f"{missing_columns_text}"
        )
