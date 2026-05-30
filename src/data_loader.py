from pathlib import Path

import pandas as pd

from src import config


def _resolve_dataset_path(dataset_path: str | Path | None = None) -> Path:
    if dataset_path is None:
        return config.DATASET_PATH
    return Path(dataset_path)


def _find_missing_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column_name
        for column_name in config.RAW_DATASET_COLUMNS
        if column_name not in dataframe.columns
    ]


def _find_duplicated_columns(dataframe: pd.DataFrame) -> list[str]:
    duplicated_mask = dataframe.columns.duplicated()
    return dataframe.columns[duplicated_mask].tolist()


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    resolved_dataset_path = _resolve_dataset_path(dataset_path)

    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"file not found: {resolved_dataset_path}")

    return pd.read_csv(resolved_dataset_path)


def validate_dataset_structure(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        raise ValueError("dataset empty")

    if config.TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"column '{config.TARGET_COLUMN}' missing"
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


def split_features_and_target(
    dataframe: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in dataframe.columns:
        raise ValueError(
            f"column '{target_column}' missing"
        )

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]
    return features, target


def load_features_and_target(
    dataset_path: str | Path | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    dataframe = load_dataset(dataset_path)
    validate_dataset_structure(dataframe)
    return split_features_and_target(dataframe, config.TARGET_COLUMN)
