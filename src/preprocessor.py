import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from src import config
from src.data_loader import load_dataset
from src.data_loader import split_features_and_target
from src.data_loader import validate_dataset_structure

MONTH_COLUMN = "arrival_date_month"
COUNTRY_COLUMN = "country"
ADR_COLUMN = "adr"
AGENT_COLUMN = "agent"
COMPANY_COLUMN = "company"
CHILDREN_COLUMN = "children"
MARKET_SEGMENT_COLUMN = "market_segment"
HAS_AGENT_COLUMN = "has_agent"
HAS_COMPANY_COLUMN = "has_company"


# Comprueba que existan las columnas necesarias antes de seguir.
def _ensure_columns_exist(
    dataframe: pd.DataFrame, required_columns: list[str]
) -> None:
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


# Revisa que los meses del archivo tengan valores validos.
def _validate_month_values(dataframe: pd.DataFrame) -> None:
    month_values = dataframe[MONTH_COLUMN].astype(str).str.strip()
    invalid_values = sorted(
        value for value in month_values.unique() if value not in config.MONTH_MAPPING
    )

    if invalid_values:
        invalid_values_text = ", ".join(invalid_values)
        raise ValueError(
            f"unexpected values in '{MONTH_COLUMN}': "
            f"{invalid_values_text}"
        )


# Hace la limpieza base del dataframe antes del resto de transformaciones.
def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns_exist(df_raw, config.RAW_DATASET_COLUMNS)

    df_cleaned = df_raw.copy()
    df_cleaned = df_cleaned.drop_duplicates().copy()
    df_cleaned = df_cleaned[
        df_cleaned[MARKET_SEGMENT_COLUMN]
        != config.UNDEFINED_MARKET_SEGMENT_VALUE
    ].copy()
    df_cleaned = df_cleaned[
        df_cleaned[ADR_COLUMN] >= 0
    ].copy()

    _validate_month_values(df_cleaned)

    df_cleaned = df_cleaned.drop(
        columns=config.LEAKAGE_COLUMNS + config.REDUNDANT_COLUMNS,
        errors="ignore",
    )
    return df_cleaned


# Aplica las transformaciones fijas que siempre son iguales, como en el notebook.
def apply_fixed_preprocessing(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        MONTH_COLUMN,
        CHILDREN_COLUMN,
        AGENT_COLUMN,
        COMPANY_COLUMN,
        *config.BASE_TYPE_CASTS.keys(),
    ]
    _ensure_columns_exist(df_cleaned, required_columns)

    df_preprocessed = df_cleaned.copy()
    df_preprocessed = df_preprocessed.astype(config.BASE_TYPE_CASTS)
    df_preprocessed[CHILDREN_COLUMN] = df_preprocessed[
        CHILDREN_COLUMN
    ].fillna(config.FIXED_FILL_VALUES[CHILDREN_COLUMN])
    df_preprocessed[AGENT_COLUMN] = df_preprocessed[
        AGENT_COLUMN
    ].fillna(config.FIXED_FILL_VALUES[AGENT_COLUMN])
    df_preprocessed[COMPANY_COLUMN] = df_preprocessed[
        COMPANY_COLUMN
    ].fillna(config.FIXED_FILL_VALUES[COMPANY_COLUMN])
    df_preprocessed[MONTH_COLUMN] = (
        df_preprocessed[MONTH_COLUMN]
        .str.strip()
        .map(config.MONTH_MAPPING)
    )
    return df_preprocessed


# Divide el dataframe en una parte para entrenar y otra para validar.
def split_train_validation(
    df_preprocessed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df_preprocessed,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df_preprocessed[config.TARGET_COLUMN],
    )


# Aprende las reglas de limpieza que dependen solo de los datos de entrenamiento.
def fit_preprocessing_rules(df_train: pd.DataFrame) -> dict:
    _ensure_columns_exist(
        df_train,
        [COUNTRY_COLUMN, ADR_COLUMN],
    )

    country_mode = df_train[COUNTRY_COLUMN].mode(dropna=True)
    if country_mode.empty:
        raise ValueError(
            f"invalid values '{COUNTRY_COLUMN}'"
        )

    country_fill_value = country_mode.iloc[0]
    country_series = df_train[COUNTRY_COLUMN].fillna(country_fill_value)
    top_countries = (
        country_series.value_counts().head(config.COUNTRY_TOP_N).index.tolist()
    )
    adr_upper_bound = df_train[ADR_COLUMN].quantile(config.ADR_UPPER_QUANTILE)

    return {
        "country_fill_value": country_fill_value,
        "top_countries": top_countries,
        "adr_upper_bound": float(adr_upper_bound),
    }


# Aplica al dataframe las reglas aprendidas con el conjunto de entrenamiento.
def apply_preprocessing_rules(
    df_preprocessed: pd.DataFrame,
    preprocessing_rules: dict,
) -> pd.DataFrame:
    required_columns = [
        config.TARGET_COLUMN,
        COUNTRY_COLUMN,
        ADR_COLUMN,
        AGENT_COLUMN,
        COMPANY_COLUMN,
    ]
    _ensure_columns_exist(df_preprocessed, required_columns)

    df_model_input = df_preprocessed.copy()
    df_model_input[COUNTRY_COLUMN] = df_model_input[
        COUNTRY_COLUMN
    ].fillna(preprocessing_rules["country_fill_value"])
    df_model_input = df_model_input[
        df_model_input[ADR_COLUMN]
        <= preprocessing_rules["adr_upper_bound"]
    ].copy()
    df_model_input[COUNTRY_COLUMN] = df_model_input[
        COUNTRY_COLUMN
    ].where(
        df_model_input[COUNTRY_COLUMN].isin(preprocessing_rules["top_countries"]),
        config.COUNTRY_OTHER_LABEL,
    )
    df_model_input[HAS_AGENT_COLUMN] = (
        df_model_input[AGENT_COLUMN] != config.AGENT_MISSING_VALUE
    ).astype(int)
    df_model_input[HAS_COMPANY_COLUMN] = (
        df_model_input[COMPANY_COLUMN] != config.COMPANY_MISSING_VALUE
    ).astype(int)
    df_model_input = df_model_input.astype(config.FINAL_TYPE_CASTS)
    df_model_input = df_model_input.drop(
        columns=[AGENT_COLUMN, COMPANY_COLUMN],
        errors="ignore",
    )
    df_model_input = df_model_input[config.FINAL_DATAFRAME_COLUMNS].copy()
    return df_model_input


# Prepara en orden los datos finales que usaran los modelos para entrenar y validar.
def prepare_training_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    dict,
]:
    df_raw = load_dataset()
    validate_dataset_structure(df_raw)
    df_cleaned = clean_dataset(df_raw)
    df_preprocessed = apply_fixed_preprocessing(df_cleaned)
    df_train, df_validation = split_train_validation(df_preprocessed)
    preprocessing_rules = fit_preprocessing_rules(df_train)
    df_train_preprocessed = apply_preprocessing_rules(
        df_train,
        preprocessing_rules,
    )
    df_validation_preprocessed = apply_preprocessing_rules(
        df_validation,
        preprocessing_rules,
    )
    X_train, y_train = split_features_and_target(
        df_train_preprocessed,
        config.TARGET_COLUMN,
    )
    X_validation, y_validation = split_features_and_target(
        df_validation_preprocessed,
        config.TARGET_COLUMN,
    )
    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
        preprocessing_rules,
    )


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
            ("numerical", numerical_transformer, config.NUMERICAL_FEATURES),
            ("categorical", categorical_transformer, config.CATEGORICAL_FEATURES),
        ]
    )
