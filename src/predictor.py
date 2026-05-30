import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src import config
from src.model_trainer import apply_preprocessing_rules
from src.model_trainer import clean_dataset

def payload_to_dataframe(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([payload])


def load_best_model_artifact() -> dict:
    has_classical_model = config.BEST_MODEL_PIPELINE_PATH.exists()
    has_neural_model = config.BEST_NEURAL_NETWORK_MODEL_PATH.exists()
    has_neural_preprocessor = config.BEST_NEURAL_PREPROCESSOR_PATH.exists()
    has_preprocessing_rules = config.BEST_PREPROCESSING_RULES_PATH.exists()

    if not has_preprocessing_rules:
        raise FileNotFoundError("no preprocessing rules found")

    if has_classical_model and (has_neural_model or has_neural_preprocessor):
        raise ValueError(
            "only one trained model type"
        )

    if has_neural_model != has_neural_preprocessor:
        raise ValueError(
            "no neural network artifacts found"
        )

    if has_classical_model:
        return {
            "model_type": "classical",
            "model_object": joblib.load(config.BEST_MODEL_PIPELINE_PATH),
            "preprocessing_rules": joblib.load(
                config.BEST_PREPROCESSING_RULES_PATH
            ),
        }

    if has_neural_model and has_neural_preprocessor:
        return {
            "model_type": "neural_network",
            "model_object": tf.keras.models.load_model(
                config.BEST_NEURAL_NETWORK_MODEL_PATH
            ),
            "preprocessor": joblib.load(config.BEST_NEURAL_PREPROCESSOR_PATH),
            "preprocessing_rules": joblib.load(
                config.BEST_PREPROCESSING_RULES_PATH
            ),
        }

    raise FileNotFoundError(
        "no trained model"

    )


def prepare_features(
    X: pd.DataFrame,
    preprocessing_rules: dict,
) -> pd.DataFrame:
    y = pd.Series(
        [0],
        index=X.index,
        name=config.TARGET_COLUMN,
    )
    X, y = clean_dataset(
        X,
        y,
    )
    X, _ = apply_preprocessing_rules(
        X,
        y,
        preprocessing_rules,
    )
    if X.empty:
        raise ValueError("payload removed by preprocessing rules")
    return X


def predict_booking_from_payload(
    payload: dict,
) -> dict:
    X = payload_to_dataframe(payload)
    best_model_artifact = load_best_model_artifact()
    X = prepare_features(
        X,
        best_model_artifact["preprocessing_rules"],
    )

    if best_model_artifact["model_type"] == "classical":
        y_proba = best_model_artifact["model_object"].predict_proba(
            X
        )[:, 1]
    else:
        X_transformed = best_model_artifact["preprocessor"].transform(
            X
        )
        y_proba = best_model_artifact["model_object"].predict(
            X_transformed,
            verbose=0,
        ).reshape(-1)

    predicted_probability = float(np.asarray(y_proba)[0])
    predicted_label = int(predicted_probability >= 0.5)
    predicted_class_name = config.CLASS_LABELS[predicted_label]

    return {
        "predicted_label": predicted_label,
        "predicted_class_name": predicted_class_name,
        "predicted_probability": predicted_probability,
        "model_type": best_model_artifact["model_type"],
    }
