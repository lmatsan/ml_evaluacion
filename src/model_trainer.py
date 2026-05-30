import joblib
import numpy as np
import tensorflow as tf
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from src import config
from src.evaluator import build_model_comparison
from src.evaluator import evaluate_classification_model
from src.evaluator import export_comparative_roc_curve
from src.evaluator import export_confusion_matrix_figure
from src.evaluator import export_random_forest_feature_importance
from src.evaluator import export_training_report
from src.preprocessor import build_preprocessor
from src.preprocessor import prepare_training_datasets


# Traduce el nombre interno de cada modelo a un nombre mas legible.
def get_model_display_name(model_key: str) -> str:
    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "catboost": "CatBoost",
        "neural_network": "Neural Network",
    }
    return display_names[model_key]


# Crea la carpeta de salida si todavia no existe.
def _ensure_output_directories() -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# Borra un archivo antiguo si ya no hace falta conservarlo.
def _remove_file_if_exists(file_path) -> None:
    if file_path.exists():
        file_path.unlink()


# Reune los modelos clasicos que se van a comparar.
def build_classical_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            C=config.LOGISTIC_C,
            max_iter=config.LOGISTIC_MAX_ITER,
            random_state=config.RANDOM_STATE,
            solver=config.LOGISTIC_SOLVER,
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


# Construye la red neuronal que se usara como uno de los modelos.
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


# Entrena uno de los modelos clasicos con los datos preparados.
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


# Entrena la red neuronal usando entrenamiento y validacion.
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


# Entrena todos los modelos y guarda sus resultados para compararlos.
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
        trained_model = train_classical_model(
            model_key,
            model,
            X_train,
            y_train,
        )
        evaluation_result = evaluate_classification_model(
            get_model_display_name(model_key),
            trained_model,
            X_validation,
            y_validation,
            "classical",
        )
        model_results[model_key] = {
            **evaluation_result,
            "model_object": trained_model,
        }

    trained_model = train_neural_network_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    evaluation_result = evaluate_classification_model(
        get_model_display_name("neural_network"),
        trained_model,
        X_validation,
        y_validation,
        "neural_network",
    )
    model_results["neural_network"] = {
        **evaluation_result,
        "model_object": trained_model,
    }
    return {
        "model_results": model_results,
        "preprocessing_rules": preprocessing_rules,
    }


# Elige el modelo que mejor resultado obtuvo segun la metrica principal.
def select_best_model(model_results: dict) -> dict:
    return max(
        model_results.values(),
        key=lambda result: result[config.MAIN_METRIC],
    )


# Guarda el mejor modelo y las reglas necesarias para usarlo despues.
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


# Ejecuta de principio a fin todo el proceso de entrenamiento y guardado.
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


# Lanza el entrenamiento principal cuando se ejecuta este archivo.
def main() -> None:
    result = run_training_pipeline()
    print(result["report_path"])
    print(result["comparative_roc_curve_path"])
    print(result["best_confusion_matrix_path"])
    print(result["random_forest_feature_importance_path"])


if __name__ == "__main__":
    main()
