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

COMPARATIVE_ROC_CURVE_PATH = config.OUTPUTS_DIR / "comparative_roc_curve.png"
BEST_MODEL_CONFUSION_MATRIX_PATH = (
    config.OUTPUTS_DIR / "best_model_confusion_matrix.png"
)
RANDOM_FOREST_IMPORTANCE_PLOT_PATH = (
    config.OUTPUTS_DIR / "random_forest_feature_importances.png"
)
BEST_MODEL_PIPELINE_PATH = config.OUTPUTS_DIR / "best_model_pipeline.joblib"
BEST_NEURAL_NETWORK_MODEL_PATH = config.OUTPUTS_DIR / "best_model.keras"
BEST_NEURAL_PREPROCESSOR_PATH = (
    config.OUTPUTS_DIR / "best_model_preprocessor.joblib"
)
TRAINING_REPORT_PATH = config.OUTPUTS_DIR / "training_report.md"
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


def clean_dataset(
    features: pd.DataFrame, target: pd.Series
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
    _ensure_columns_exist(features, required_columns)

    combined_dataframe = features.copy()
    combined_dataframe[target.name] = target.copy()
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

    cleaned_target = combined_dataframe.pop(target.name)
    cleaned_features = combined_dataframe
    return cleaned_features, cleaned_target


def split_train_validation(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=target,
    )


def fit_preprocessing_rules(training_features: pd.DataFrame) -> dict:
    _ensure_columns_exist(
        training_features,
        [COUNTRY_COLUMN, ADR_COLUMN],
    )

    country_mode = training_features[COUNTRY_COLUMN].mode(dropna=True)
    if country_mode.empty:
        raise ValueError(
            f"Column '{COUNTRY_COLUMN}' does not contain valid values."
        )

    country_fill_value = country_mode.iloc[0]
    country_series = training_features[COUNTRY_COLUMN].fillna(country_fill_value)
    top_countries = (
        country_series.value_counts().head(config.COUNTRY_TOP_N).index.tolist()
    )
    adr_upper_bound = training_features[ADR_COLUMN].quantile(config.ADR_UPPER_QUANTILE)

    return {
        "country_fill_value": country_fill_value,
        "top_countries": top_countries,
        "adr_upper_bound": float(adr_upper_bound),
    }


def apply_preprocessing_rules(
    features: pd.DataFrame,
    target: pd.Series,
    preprocessing_rules: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    _ensure_columns_exist(
        features,
        [
            COUNTRY_COLUMN,
            ADR_COLUMN,
            AGENT_COLUMN,
            COMPANY_COLUMN,
        ],
    )

    processed_features = features.copy()
    processed_target = target.copy()
    processed_features[COUNTRY_COLUMN] = processed_features[
        COUNTRY_COLUMN
    ].fillna(preprocessing_rules["country_fill_value"])
    processed_features = processed_features[
        processed_features[ADR_COLUMN]
        <= preprocessing_rules["adr_upper_bound"]
    ].copy()
    processed_target = processed_target.loc[processed_features.index].copy()
    processed_features[COUNTRY_COLUMN] = processed_features[
        COUNTRY_COLUMN
    ].where(
        processed_features[COUNTRY_COLUMN].isin(preprocessing_rules["top_countries"]),
        config.COUNTRY_OTHER_LABEL,
    )
    processed_features[HAS_AGENT_COLUMN] = (
        processed_features[AGENT_COLUMN] != config.AGENT_MISSING_VALUE
    ).astype(int)
    processed_features[HAS_COMPANY_COLUMN] = (
        processed_features[COMPANY_COLUMN] != config.COMPANY_MISSING_VALUE
    ).astype(int)
    processed_features = processed_features.drop(
        columns=[AGENT_COLUMN, COMPANY_COLUMN],
        errors="ignore",
    )
    processed_features[config.CATEGORICAL_FEATURES] = processed_features[
        config.CATEGORICAL_FEATURES
    ].astype(str)
    processed_features = processed_features[config.FINAL_FEATURE_COLUMNS].copy()
    return processed_features, processed_target


def prepare_training_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    features, target = load_features_and_target()
    cleaned_features, cleaned_target = clean_dataset(features, target)
    (
        train_features,
        validation_features,
        train_target,
        validation_target,
    ) = split_train_validation(cleaned_features, cleaned_target)
    preprocessing_rules = fit_preprocessing_rules(train_features)
    processed_train_features, processed_train_target = (
        apply_preprocessing_rules(
            train_features,
            train_target,
            preprocessing_rules,
        )
    )
    processed_validation_features, processed_validation_target = (
        apply_preprocessing_rules(
            validation_features,
            validation_target,
            preprocessing_rules,
        )
    )
    return (
        processed_train_features,
        processed_validation_features,
        processed_train_target,
        processed_validation_target,
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
    train_features,
    train_target,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(model_key)),
            ("model", model),
        ]
    )
    pipeline.fit(train_features, train_target)
    return pipeline


def train_neural_network_model(
    train_features,
    train_target,
    validation_features,
    validation_target,
) -> dict:
    preprocessor = build_preprocessor("neural_network")
    transformed_train_features = preprocessor.fit_transform(train_features)
    transformed_validation_features = preprocessor.transform(validation_features)
    transformed_train_features = np.asarray(
        transformed_train_features,
        dtype=np.float32,
    )
    transformed_validation_features = np.asarray(
        transformed_validation_features,
        dtype=np.float32,
    )
    neural_network = build_neural_network(
        transformed_train_features.shape[1]
    )
    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    neural_network.fit(
        transformed_train_features,
        train_target.to_numpy(),
        validation_data=(
            transformed_validation_features,
            validation_target.to_numpy(),
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
        train_features,
        validation_features,
        train_target,
        validation_target,
    ) = prepare_training_datasets()
    results = {}

    classical_models = build_classical_models()
    for model_key, model in classical_models.items():
        trained_model = train_classical_model(
            model_key,
            model,
            train_features,
            train_target,
        )
        evaluation = evaluate_classification_model(
            get_model_display_name(model_key),
            trained_model,
            validation_features,
            validation_target,
            "classical",
        )
        results[model_key] = {
            **evaluation,
            "model_object": trained_model,
        }

    trained_neural_network = train_neural_network_model(
        train_features,
        train_target,
        validation_features,
        validation_target,
    )
    neural_network_evaluation = evaluate_classification_model(
        get_model_display_name("neural_network"),
        trained_neural_network,
        validation_features,
        validation_target,
        "neural_network",
    )
    results["neural_network"] = {
        **neural_network_evaluation,
        "model_object": trained_neural_network,
    }
    return results


def select_best_model(results: dict) -> dict:
    return max(
        results.values(),
        key=lambda result: result[config.MAIN_METRIC],
    )


def save_best_artifact(best_result: dict) -> None:
    _ensure_output_directories()

    if best_result["model_type"] == "classical":
        joblib.dump(best_result["model_object"], BEST_MODEL_PIPELINE_PATH)
    elif best_result["model_type"] == "neural_network":
        best_result["model_object"]["model"].save(BEST_NEURAL_NETWORK_MODEL_PATH)
        joblib.dump(
            best_result["model_object"]["preprocessor"],
            BEST_NEURAL_PREPROCESSOR_PATH,
        )
    else:
        raise ValueError(
            f"Unsupported model type: {best_result['model_type']}"
        )


def run_training_pipeline() -> dict:
    _ensure_output_directories()
    results = train_all_models()
    result_list = list(results.values())
    comparison_dataframe = build_model_comparison(result_list)
    comparative_roc_curve_path = export_comparative_roc_curve(
        result_list,
        COMPARATIVE_ROC_CURVE_PATH,
    )
    best_result = select_best_model(results)
    best_confusion_matrix_path = export_confusion_matrix_figure(
        best_result["model_name"],
        np.array(best_result["confusion_matrix"]),
        BEST_MODEL_CONFUSION_MATRIX_PATH,
    )
    random_forest_result = results["random_forest"]
    random_forest_feature_importance_path = (
        export_random_forest_feature_importance(
            random_forest_result["model_object"],
            RANDOM_FOREST_IMPORTANCE_PLOT_PATH,
        )
    )
    export_training_report(
        comparison_dataframe,
        TRAINING_REPORT_PATH,
        best_result,
        best_confusion_matrix_path,
        random_forest_feature_importance_path,
    )
    save_best_artifact(best_result)
    return {
        "comparison_dataframe": comparison_dataframe,
        "best_result": best_result,
        "report_path": TRAINING_REPORT_PATH,
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
