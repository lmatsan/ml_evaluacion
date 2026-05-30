import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src import config
from src.data_loader import load_features_and_target
from src.model_trainer import apply_preprocessing_rules
from src.model_trainer import clean_dataset
from src.model_trainer import fit_preprocessing_rules

BEST_MODEL_PIPELINE_PATH = config.OUTPUTS_DIR / "best_model_pipeline.joblib"
BEST_NEURAL_NETWORK_MODEL_PATH = config.OUTPUTS_DIR / "best_model.keras"
BEST_NEURAL_PREPROCESSOR_PATH = (
    config.OUTPUTS_DIR / "best_model_preprocessor.joblib"
)

def payload_to_dataframe(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([payload])


def load_best_model_artifact() -> dict:
    has_classical_model = BEST_MODEL_PIPELINE_PATH.exists()
    has_neural_model = BEST_NEURAL_NETWORK_MODEL_PATH.exists()
    has_neural_preprocessor = BEST_NEURAL_PREPROCESSOR_PATH.exists()

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
            "model_object": joblib.load(BEST_MODEL_PIPELINE_PATH),
        }

    if has_neural_model and has_neural_preprocessor:
        return {
            "model_type": "neural_network",
            "model_object": tf.keras.models.load_model(
                BEST_NEURAL_NETWORK_MODEL_PATH
            ),
            "preprocessor": joblib.load(BEST_NEURAL_PREPROCESSOR_PATH),
        }

    raise FileNotFoundError(
        "no trained model"

    )


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    payload_features = features.copy()
    features, target = load_features_and_target()
    cleaned_features, cleaned_target = clean_dataset(features, target)
    preprocessing_rules = fit_preprocessing_rules(cleaned_features)
    payload_target = pd.Series(
        [0],
        index=payload_features.index,
        name=cleaned_target.name,
    )
    cleaned_payload_features, cleaned_payload_target = clean_dataset(
        payload_features,
        payload_target,
    )
    processed_features, _ = apply_preprocessing_rules(
        cleaned_payload_features,
        cleaned_payload_target,
        preprocessing_rules,
    )
    return processed_features


def predict_booking_from_payload(
    payload: dict,
) -> dict:
    payload_features = payload_to_dataframe(payload)
    best_model_artifact = load_best_model_artifact()
    processed_features = prepare_features(payload_features)

    if best_model_artifact["model_type"] == "classical":
        probability = best_model_artifact["model_object"].predict_proba(
            processed_features
        )[:, 1]
    else:
        transformed_features = best_model_artifact["preprocessor"].transform(
            processed_features
        )
        probability = best_model_artifact["model_object"].predict(
            transformed_features,
            verbose=0,
        ).reshape(-1)

    probability = float(np.asarray(probability)[0])
    predicted_label = int(probability >= 0.5)
    predicted_class_name = config.CLASS_LABELS[predicted_label]

    return {
        "predicted_label": predicted_label,
        "predicted_class_name": predicted_class_name,
        "predicted_probability": probability,
        "model_type": best_model_artifact["model_type"],
    }
