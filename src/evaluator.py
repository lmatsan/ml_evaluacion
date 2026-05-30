from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from src import config

COMPARATIVE_ROC_CURVE_PATH = config.OUTPUTS_DIR / "comparative_roc_curve.png"


def _extract_probabilities(model_object, validation_features, model_type: str):
    if model_type == "classical":
        return model_object.predict_proba(validation_features)[:, 1]

    if model_type == "neural_network":
        transformed_features = model_object["preprocessor"].transform(
            validation_features
        )
        predictions = model_object["model"].predict(
            transformed_features,
            verbose=0,
        )
        return predictions.reshape(-1)

    raise ValueError(f"Unsupported model: {model_type}")


def _build_predictions(probabilities):
    return (probabilities >= 0.5).astype(int)


def evaluate_classification_model(
    model_name,
    model_object,
    validation_features,
    validation_target,
    model_type,
) -> dict:
    probabilities = _extract_probabilities(
        model_object,
        validation_features,
        model_type,
    )
    predicted_labels = _build_predictions(probabilities)
    confusion_matrix_values = confusion_matrix(validation_target, predicted_labels)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        validation_target,
        probabilities,
    )
    roc_auc = roc_auc_score(validation_target, probabilities)

    return {
        "model_name": model_name,
        "model_type": model_type,
        "accuracy": accuracy_score(validation_target, predicted_labels),
        "precision": precision_score(
            validation_target,
            predicted_labels,
            zero_division=0,
        ),
        "recall": recall_score(
            validation_target,
            predicted_labels,
            zero_division=0,
        ),
        "f1": f1_score(
            validation_target,
            predicted_labels,
            zero_division=0,
        ),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix_values.tolist(),
        "roc_curve": {
            "false_positive_rate": false_positive_rate.tolist(),
            "true_positive_rate": true_positive_rate.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }


def build_model_comparison(results: list[dict]) -> pd.DataFrame:
    comparison_rows = []

    for result in results:
        comparison_rows.append(
            {
                "model_name": result["model_name"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
            }
        )

    comparison_dataframe = pd.DataFrame(comparison_rows)
    comparison_dataframe = comparison_dataframe.sort_values(
        by=config.MAIN_METRIC,
        ascending=False,
    ).reset_index(drop=True)
    return comparison_dataframe


def export_comparative_roc_curve(results: list[dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))

    for result in results:
        axis.plot(
            result["roc_curve"]["false_positive_rate"],
            result["roc_curve"]["true_positive_rate"],
            label=f"{result['model_name']} (AUC={result['roc_auc']:.4f})",
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_title("Comparative ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def export_confusion_matrix_figure(
    model_name: str,
    confusion_matrix_values,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    image = axis.imshow(confusion_matrix_values, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title(f"Confusion Matrix - {model_name}")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks([0, 1], config.CLASS_LABELS)
    axis.set_yticks([0, 1], config.CLASS_LABELS)

    for row_index in range(confusion_matrix_values.shape[0]):
        for column_index in range(confusion_matrix_values.shape[1]):
            axis.text(
                column_index,
                row_index,
                int(confusion_matrix_values[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def export_random_forest_feature_importance(
    model_object,
    output_path: Path,
    top_n: int = config.RANDOM_FOREST_IMPORTANCE_TOP_N,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor = model_object.named_steps["preprocessor"]
    model = model_object.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    feature_importances = model.feature_importances_

    if len(feature_names) != len(feature_importances):
        raise ValueError(
            "transformed features does not match feature importances"
        )

    feature_importance_dataframe = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": feature_importances,
        }
    ).sort_values(by="importance", ascending=False)

    top_feature_importances = feature_importance_dataframe.head(top_n).copy()
    top_feature_importances = top_feature_importances.iloc[::-1]

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(
        top_feature_importances["feature_name"],
        top_feature_importances["importance"],
        color="steelblue",
    )
    axis.set_title("Random Forest feature_importances_")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _build_markdown_table(comparison_dataframe: pd.DataFrame) -> str:
    table_columns = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    header = "| " + " | ".join(table_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table_columns)) + " |"
    rows = [header, separator]

    for _, row in comparison_dataframe.iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['roc_auc']:.4f}",
                ]
            )
            + " |"
        )

    return "\n".join(rows)


def export_training_report(
    comparison_df: pd.DataFrame,
    output_path: Path,
    best_result: dict,
    best_confusion_matrix_path: Path,
    random_forest_feature_importance_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Training Report",
        "",
        f"- Main metric: `{config.MAIN_METRIC}`",
        f"- Best model: `{best_result['model_name']}`",
        f"- Comparative ROC curve: `{COMPARATIVE_ROC_CURVE_PATH}`",
        "- Best model confusion matrix: "
        f"`{best_confusion_matrix_path}`",
        "- Random Forest feature importances: "
        f"`{random_forest_feature_importance_path}`",
        "",
        "## Model Comparison",
        "",
        _build_markdown_table(comparison_df),
        "",
    ]
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
