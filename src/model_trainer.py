import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from src import config
from src.data_loader import load_features_and_target
from src.evaluator import build_model_comparison
from src.evaluator import evaluate_classification_model
from src.evaluator import export_comparative_roc_curve
from src.evaluator import export_confusion_matrix_figure
from src.evaluator import export_random_forest_feature_importance
from src.evaluator import export_training_report

MONTH_COLUMN = "arrival_date_month"
COUNTRY_COLUMN = "country"
ADR_COLUMN = "adr"
AGENT_COLUMN = "agent"
COMPANY_COLUMN = "company"
CHILDREN_COLUMN = "children"
MARKET_SEGMENT_COLUMN = "market_segment"
HAS_AGENT_COLUMN = "has_agent"
HAS_COMPANY_COLUMN = "has_company"


def get_model_display_name(model_key: str) -> str:
    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "catboost": "CatBoost",
        "neural_network": "Neural Network",
    }
    return display_names[model_key]


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
            "The dataframe is missing required columns: "
            f"{missing_columns_text}"
        )


def _validate_month_values(dataframe: pd.DataFrame) -> None:
    month_values = dataframe[MONTH_COLUMN].astype(str).str.strip()
    invalid_values = sorted(
        value for value in month_values.unique() if value not in config.MONTH_MAPPING
    )

    if invalid_values:
        invalid_values_text = ", ".join(invalid_values)
        raise ValueError(
            f"Unexpected values found in '{MONTH_COLUMN}': "
            f"{invalid_values_text}"
        )


def _ensure_output_directories() -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _remove_file_if_exists(file_path) -> None:
    if file_path.exists():
        file_path.unlink()


def clean_dataset(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    required_columns = [
        MONTH_COLUMN,
        MARKET_SEGMENT_COLUMN,
        ADR_COLUMN,
        CHILDREN_COLUMN,
        AGENT_COLUMN,
        COMPANY_COLUMN,
        *config.LEAKAGE_COLUMNS,
        *config.REDUNDANT_COLUMNS,
    ]
    _ensure_columns_exist(X, required_columns)

    combined_dataframe = X.copy()
    combined_dataframe[y.name] = y.copy()
    combined_dataframe = combined_dataframe.drop_duplicates().copy()
    combined_dataframe = combined_dataframe[
        combined_dataframe[MARKET_SEGMENT_COLUMN]
        != config.UNDEFINED_MARKET_SEGMENT_VALUE
    ].copy()
    combined_dataframe = combined_dataframe[
        combined_dataframe[ADR_COLUMN] >= 0
    ].copy()

    _validate_month_values(combined_dataframe)

    combined_dataframe[MONTH_COLUMN] = (
        combined_dataframe[MONTH_COLUMN]
        .astype(str)
        .str.strip()
        .map(config.MONTH_MAPPING)
    )
    combined_dataframe[CHILDREN_COLUMN] = combined_dataframe[
        CHILDREN_COLUMN
    ].fillna(config.CHILDREN_IMPUTATION_VALUE)
    combined_dataframe[AGENT_COLUMN] = combined_dataframe[
        AGENT_COLUMN
    ].fillna(config.AGENT_MISSING_VALUE)
    combined_dataframe[COMPANY_COLUMN] = combined_dataframe[
        COMPANY_COLUMN
    ].fillna(config.COMPANY_MISSING_VALUE)
    combined_dataframe = combined_dataframe.drop(
        columns=config.LEAKAGE_COLUMNS + config.REDUNDANT_COLUMNS,
        errors="ignore",
    )

    y = combined_dataframe.pop(y.name)
    X = combined_dataframe
    return X, y


def split_train_validation(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def fit_preprocessing_rules(X_train: pd.DataFrame) -> dict:
    _ensure_columns_exist(
        X_train,
        [COUNTRY_COLUMN, ADR_COLUMN],
    )

    country_mode = X_train[COUNTRY_COLUMN].mode(dropna=True)
    if country_mode.empty:
        raise ValueError(
            f"Column '{COUNTRY_COLUMN}' does not contain valid values."
        )

    country_fill_value = country_mode.iloc[0]
    country_series = X_train[COUNTRY_COLUMN].fillna(country_fill_value)
    top_countries = (
        country_series.value_counts().head(config.COUNTRY_TOP_N).index.tolist()
    )
    adr_upper_bound = X_train[ADR_COLUMN].quantile(config.ADR_UPPER_QUANTILE)

    return {
        "country_fill_value": country_fill_value,
        "top_countries": top_countries,
        "adr_upper_bound": float(adr_upper_bound),
    }


def apply_preprocessing_rules(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessing_rules: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    _ensure_columns_exist(
        X,
        [
            COUNTRY_COLUMN,
            ADR_COLUMN,
            AGENT_COLUMN,
            COMPANY_COLUMN,
        ],
    )

    X = X.copy()
    y = y.copy()
    X[COUNTRY_COLUMN] = X[
        COUNTRY_COLUMN
    ].fillna(preprocessing_rules["country_fill_value"])
    X = X[
        X[ADR_COLUMN]
        <= preprocessing_rules["adr_upper_bound"]
    ].copy()
    y = y.loc[X.index].copy()
    X[COUNTRY_COLUMN] = X[
        COUNTRY_COLUMN
    ].where(
        X[COUNTRY_COLUMN].isin(preprocessing_rules["top_countries"]),
        config.COUNTRY_OTHER_LABEL,
    )
    X[HAS_AGENT_COLUMN] = (
        X[AGENT_COLUMN] != config.AGENT_MISSING_VALUE
    ).astype(int)
    X[HAS_COMPANY_COLUMN] = (
        X[COMPANY_COLUMN] != config.COMPANY_MISSING_VALUE
    ).astype(int)
    X = X.drop(
        columns=[AGENT_COLUMN, COMPANY_COLUMN],
        errors="ignore",
    )
    X[config.CATEGORICAL_FEATURES] = X[
        config.CATEGORICAL_FEATURES
    ].astype(str)
    X = X[config.FINAL_FEATURE_COLUMNS].copy()
    return X, y


def prepare_training_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    dict,
]:
    X, y = load_features_and_target()
    X, y = clean_dataset(X, y)
    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = split_train_validation(X, y)
    preprocessing_rules = fit_preprocessing_rules(X_train)
    X_train, y_train = (
        apply_preprocessing_rules(
            X_train,
            y_train,
            preprocessing_rules,
        )
    )
    X_validation, y_validation = (
        apply_preprocessing_rules(
            X_validation,
            y_validation,
            preprocessing_rules,
        )
    )
    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
        preprocessing_rules,
    )


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
        ],
        remainder="drop",
    )


def build_classical_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=config.LOGISTIC_MAX_ITER,
            random_state=config.RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=config.DECISION_TREE_MAX_DEPTH,
            random_state=config.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config.RANDOM_FOREST_N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=config.CATBOOST_ITERATIONS,
            depth=config.CATBOOST_DEPTH,
            learning_rate=config.CATBOOST_LEARNING_RATE,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=0,
            random_seed=config.RANDOM_STATE,
        ),
    }


def build_neural_network(input_dim: int) -> tf.keras.Model:
    model = Sequential()
    model.add(
        Dense(
            config.NEURAL_NETWORK_HIDDEN_UNITS[0],
            activation="relu",
            input_shape=(input_dim,),
        )
    )
    model.add(
        Dense(config.NEURAL_NETWORK_HIDDEN_UNITS[1], activation="relu")
    )
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_classical_model(
    model_key: str,
    model,
    X_train,
    y_train,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(model_key)),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_neural_network_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
) -> dict:
    preprocessor = build_preprocessor("neural_network")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)
    X_train_transformed = np.asarray(
        X_train_transformed,
        dtype=np.float32,
    )
    X_validation_transformed = np.asarray(
        X_validation_transformed,
        dtype=np.float32,
    )
    neural_network = build_neural_network(X_train_transformed.shape[1])
    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    neural_network.fit(
        X_train_transformed,
        y_train.to_numpy(),
        validation_data=(
            X_validation_transformed,
            y_validation.to_numpy(),
        ),
        epochs=config.NEURAL_NETWORK_EPOCHS,
        batch_size=config.NEURAL_NETWORK_BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0,
    )
    return {
        "model": neural_network,
        "preprocessor": preprocessor,
    }


def train_all_models() -> dict:
    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        preprocessing_rules,
    ) = prepare_training_datasets()
    model_results = {}

    classical_models = build_classical_models()
    for model_key, model in classical_models.items():
        best_model = train_classical_model(
            model_key,
            model,
            X_train,
            y_train,
        )
        evaluation_result = evaluate_classification_model(
            get_model_display_name(model_key),
            best_model,
            X_validation,
            y_validation,
            "classical",
        )
        model_results[model_key] = {
            **evaluation_result,
            "model_object": best_model,
        }

    best_model = train_neural_network_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    evaluation_result = evaluate_classification_model(
        get_model_display_name("neural_network"),
        best_model,
        X_validation,
        y_validation,
        "neural_network",
    )
    model_results["neural_network"] = {
        **evaluation_result,
        "model_object": best_model,
    }
    return {
        "model_results": model_results,
        "preprocessing_rules": preprocessing_rules,
    }


def select_best_model(model_results: dict) -> dict:
    return max(
        model_results.values(),
        key=lambda result: result[config.MAIN_METRIC],
    )


def save_best_artifact(
    best_result: dict,
    preprocessing_rules: dict,
) -> None:
    _ensure_output_directories()
    joblib.dump(
        preprocessing_rules,
        config.BEST_PREPROCESSING_RULES_PATH,
    )

    if best_result["model_type"] == "classical":
        joblib.dump(
            best_result["model_object"],
            config.BEST_MODEL_PIPELINE_PATH,
        )
        _remove_file_if_exists(config.BEST_NEURAL_NETWORK_MODEL_PATH)
        _remove_file_if_exists(config.BEST_NEURAL_PREPROCESSOR_PATH)
    elif best_result["model_type"] == "neural_network":
        best_result["model_object"]["model"].save(
            config.BEST_NEURAL_NETWORK_MODEL_PATH
        )
        joblib.dump(
            best_result["model_object"]["preprocessor"],
            config.BEST_NEURAL_PREPROCESSOR_PATH,
        )
        _remove_file_if_exists(config.BEST_MODEL_PIPELINE_PATH)
    else:
        raise ValueError(
            f"Unsupported model type: {best_result['model_type']}"
        )


def run_training_pipeline() -> dict:
    _ensure_output_directories()
    training_output = train_all_models()
    model_results = training_output["model_results"]
    preprocessing_rules = training_output["preprocessing_rules"]
    model_result_list = list(model_results.values())
    comparison_dataframe = build_model_comparison(model_result_list)
    comparative_roc_curve_path = export_comparative_roc_curve(
        model_result_list,
        config.COMPARATIVE_ROC_CURVE_PATH,
    )
    best_result = select_best_model(model_results)
    best_confusion_matrix_path = export_confusion_matrix_figure(
        best_result["model_name"],
        np.array(best_result["confusion_matrix"]),
        config.BEST_MODEL_CONFUSION_MATRIX_PATH,
    )
    random_forest_result = model_results["random_forest"]
    random_forest_feature_importance_path = (
        export_random_forest_feature_importance(
            random_forest_result["model_object"],
            config.RANDOM_FOREST_IMPORTANCE_PLOT_PATH,
        )
    )
    export_training_report(
        comparison_dataframe,
        config.TRAINING_REPORT_PATH,
        best_result,
        best_confusion_matrix_path,
        random_forest_feature_importance_path,
    )
    save_best_artifact(best_result, preprocessing_rules)
    return {
        "comparison_dataframe": comparison_dataframe,
        "best_result": best_result,
        "report_path": config.TRAINING_REPORT_PATH,
        "comparative_roc_curve_path": comparative_roc_curve_path,
        "best_confusion_matrix_path": best_confusion_matrix_path,
        "random_forest_feature_importance_path": (
            random_forest_feature_importance_path
        ),
    }


def main() -> None:
    result = run_training_pipeline()
    print(result["report_path"])
    print(result["comparative_roc_curve_path"])
    print(result["best_confusion_matrix_path"])
    print(result["random_forest_feature_importance_path"])


if __name__ == "__main__":
    main()
