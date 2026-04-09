from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from config import (
    MODEL_INPUT_FILE,
    RESULTS_PICKLE_FILE,
    DISTRIBUTION_FILE,
    TRAINING_LOSS_PLOT_FILE,
    PATIENT_NUMBER_COLUMN,
    OBSERVATION_TIME_COLUMN,
    TARGET_COLUMN,
)
from data_preprocessing import load_data, preprocess_data
from evaluation import (
    build_distribution_table,
    save_distribution_table,
    save_training_loss_plot,
)
from training import split_train_val_test, run_cross_validation_training
from feature_selection import run_feature_selection

def save_pickle(obj: Any, file_path: Path) -> None:
    """Save an object as a pickle file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle)


def main() -> None:
    data = load_data(MODEL_INPUT_FILE)
    data = preprocess_data(data)

    print("Prepared data shape:", data.shape)
    print("Unique patients:", data[PATIENT_NUMBER_COLUMN].nunique())

    train_val_data, test_data = split_train_val_test(data)

    print("Train/validation patients:", train_val_data[PATIENT_NUMBER_COLUMN].nunique())
    print("Test patients:", test_data[PATIENT_NUMBER_COLUMN].nunique())

    importance_df, selected_features, predictor = run_feature_selection(train_val_data)

    feature_columns = selected_features + [
        PATIENT_NUMBER_COLUMN,
        OBSERVATION_TIME_COLUMN,
        TARGET_COLUMN,
    ]

    results = run_cross_validation_training(
        train_val_data=train_val_data,
        test_data=test_data,
        feature_columns=feature_columns,
    )

    results["feature_importance"] = importance_df
    results["important_feature_columns"] = selected_features

    save_pickle(results, RESULTS_PICKLE_FILE)

    distribution_df = build_distribution_table(data)
    save_distribution_table(distribution_df, DISTRIBUTION_FILE)

    save_training_loss_plot(
        history_by_model_and_seq=results["history_by_model_and_seq"],
        output_file=TRAINING_LOSS_PLOT_FILE,
        model_name="GRU",
        sequence_lengths=[2, 3, 4],
    )

    print("Results saved to:", RESULTS_PICKLE_FILE)
    print("Distribution file saved to:", DISTRIBUTION_FILE)
    print("Training loss plot saved to:", TRAINING_LOSS_PLOT_FILE)


if __name__ == "__main__":
    main()