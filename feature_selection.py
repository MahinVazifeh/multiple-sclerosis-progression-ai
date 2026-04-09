from __future__ import annotations

from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import (
    MODEL_INPUT_FILE,
    FEATURE_IMPORTANCE_OUTPUT_FILE,
    SELECTED_FEATURES_OUTPUT_FILE,
    AUTOGLUON_MODEL_DIR,
    PATIENT_NUMBER_COLUMN,
    OBSERVATION_TIME_COLUMN,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    TIME_LIMIT,
    EVAL_METRIC,
    PROBLEM_TYPE,
    PRESETS,
    TOP_K_FEATURES,
)
from data_preprocessing import preprocess_data


META_COLUMNS = [
    PATIENT_NUMBER_COLUMN,
    OBSERVATION_TIME_COLUMN,
    TARGET_COLUMN,
]


def load_data(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(file_path)


def split_by_patient(
    df: pd.DataFrame,
    patient_column: str = PATIENT_NUMBER_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by patient ID so that the same patient does not appear
    in both train and test sets.
    """
    patient_ids = df[patient_column].unique().tolist()

    train_patients, test_patients = train_test_split(
        patient_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_df = df[df[patient_column].isin(train_patients)].copy()
    test_df = df[df[patient_column].isin(test_patients)].copy()

    return train_df, test_df


def normalize_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize train and test data with MinMaxScaler, then restore
    metadata/target columns unchanged.
    """
    if meta_columns is None:
        meta_columns = META_COLUMNS

    scaler = MinMaxScaler()

    train_meta = train_df[meta_columns].reset_index(drop=True)
    test_meta = test_df[meta_columns].reset_index(drop=True)

    normalized_train = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns,
    )
    normalized_test = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns,
    )

    normalized_train[meta_columns] = train_meta
    normalized_test[meta_columns] = test_meta

    columns_to_drop = [PATIENT_NUMBER_COLUMN, OBSERVATION_TIME_COLUMN]
    normalized_train = normalized_train.drop(columns=columns_to_drop, errors="ignore")
    normalized_test = normalized_test.drop(columns=columns_to_drop, errors="ignore")

    return normalized_train, normalized_test


def train_automl_model(train_df: pd.DataFrame) -> TabularPredictor:
    """
    Train an AutoGluon model for multiclass classification.
    """
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        eval_metric=EVAL_METRIC,
        problem_type=PROBLEM_TYPE,
        path=str(AUTOGLUON_MODEL_DIR),
    )

    predictor.fit(
        train_data=train_df,
        time_limit=TIME_LIMIT,
        presets=PRESETS,
    )

    return predictor


def get_important_features(
    predictor: TabularPredictor,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute feature importance and return:
    - full importance DataFrame
    - selected top features with positive importance
    """
    importance_df = predictor.feature_importance(test_df)
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    selected_features = importance_df[importance_df["importance"] > 0].index.tolist()
    selected_features = selected_features[:TOP_K_FEATURES]

    return importance_df, selected_features

def save_feature_importance(importance_df: pd.DataFrame, file_path: Path) -> None:
    """Save feature importance table."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(file_path, index=True)


def save_selected_features(selected_features: list[str], file_path: Path) -> None:
    """Save selected feature names to a text file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        for feature in selected_features:
            file.write(f"{feature}\n")


def run_feature_selection(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], TabularPredictor]:
    """
    Full feature selection pipeline:
    - split by patient
    - normalize train/test
    - train AutoML
    - compute important features
    """
    train_df, test_df = split_by_patient(df)
    normalized_train, normalized_test = normalize_train_test(train_df, test_df)

    predictor = train_automl_model(normalized_train)
    importance_df, selected_features = get_important_features(predictor, normalized_test)

    return importance_df, selected_features, predictor


def main() -> None:
    data = load_data(MODEL_INPUT_FILE)
    data = preprocess_data(data)

    importance_df, selected_features, _ = run_feature_selection(data)

    save_feature_importance(importance_df, FEATURE_IMPORTANCE_OUTPUT_FILE)
    save_selected_features(selected_features, SELECTED_FEATURES_OUTPUT_FILE)

    print("Feature importance saved to:", FEATURE_IMPORTANCE_OUTPUT_FILE)
    print("Selected features saved to:", SELECTED_FEATURES_OUTPUT_FILE)
    print("\nTop selected features:")
    print(selected_features)
    print("\nTop feature importance:")
    print(importance_df.head())


if __name__ == "__main__":
    main()