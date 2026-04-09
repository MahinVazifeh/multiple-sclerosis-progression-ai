from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler

from config import (
    RANDOM_STATE,
    N_SPLITS,
    MODEL_NAMES,
    SEQUENCE_LENGTHS,
    BATCH_SIZE,
    EPOCHS,
    PATIENT_NUMBER_COLUMN,
    OBSERVATION_TIME_COLUMN,
    TARGET_COLUMN,
)
from evaluation import compute_metrics
from model_builder import build_model
from sequence_builder import create_sequences


META_COLUMNS = [
    PATIENT_NUMBER_COLUMN,
    OBSERVATION_TIME_COLUMN,
    TARGET_COLUMN,
]


def normalize_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize train, validation, and test sets using a scaler fitted on train only.
    Metadata and target columns are restored unchanged after scaling.
    """
    if meta_columns is None:
        meta_columns = META_COLUMNS

    scaler = MinMaxScaler()

    train_meta = train_df[meta_columns].reset_index(drop=True)
    val_meta = val_df[meta_columns].reset_index(drop=True)
    test_meta = test_df[meta_columns].reset_index(drop=True)

    normalized_train = pd.DataFrame(
        scaler.fit_transform(train_df),
        columns=train_df.columns,
        index=train_df.index,
    )
    normalized_val = pd.DataFrame(
        scaler.transform(val_df),
        columns=val_df.columns,
        index=val_df.index,
    )
    normalized_test = pd.DataFrame(
        scaler.transform(test_df),
        columns=test_df.columns,
        index=test_df.index,
    )

    normalized_train[meta_columns] = train_meta.values
    normalized_val[meta_columns] = val_meta.values
    normalized_test[meta_columns] = test_meta.values

    return normalized_train, normalized_val, normalized_test


def split_train_val_test(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create one outer split:
    - train/validation set
    - held-out test set
    """
    outer_cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_val_index, test_index = next(
        outer_cv.split(data, data[TARGET_COLUMN], data[PATIENT_NUMBER_COLUMN])
    )

    train_val_data = data.iloc[train_val_index].reset_index(drop=True)
    test_data = data.iloc[test_index].reset_index(drop=True)

    return train_val_data, test_data


def to_fixed_one_hot(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """
    Convert labels {1, 2, 3} to fixed-width one-hot encoded labels with shape (n, 3).
    """
    y = np.asarray(y).astype(np.int32)
    y_zero_based = y - 1

    if y_zero_based.size == 0:
        return np.empty((0, num_classes), dtype=np.float32)

    if np.any(y_zero_based < 0) or np.any(y_zero_based >= num_classes):
        raise ValueError(
            f"Labels must be in 1..{num_classes}. Got labels: {np.unique(y)}"
        )

    y_one_hot = np.zeros((y_zero_based.size, num_classes), dtype=np.float32)
    y_one_hot[np.arange(y_zero_based.size), y_zero_based] = 1.0
    return y_one_hot


def run_cross_validation_training(
    train_val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """
    Run inner cross-validation training for all model types and sequence lengths.
    """
    missing_train_columns = [col for col in feature_columns if col not in train_val_data.columns]
    if missing_train_columns:
        raise ValueError(
            f"Missing feature columns in train/validation data: {missing_train_columns}"
        )

    missing_test_columns = [col for col in feature_columns if col not in test_data.columns]
    if missing_test_columns:
        raise ValueError(
            f"Missing feature columns in test data: {missing_test_columns}"
        )

    inner_cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    recall_scores = defaultdict(list)
    precision_scores = defaultdict(list)
    accuracy_scores = defaultdict(list)
    auc_scores = defaultdict(list)

    y_pred_dict = defaultdict(list)
    y_pred_mean_dict = {}
    y_pred_std_dict = {}
    y_prediction_mean_dict = {}
    test_y_dict = {}

    models = defaultdict(list)
    trains = defaultdict(list)
    tests = defaultdict(list)
    histories = defaultdict(list)
    history_by_model_and_seq = defaultdict(list)

    counter = 0

    test_data = test_data[feature_columns].copy()

    for model_name in MODEL_NAMES:
        for seq_len in SEQUENCE_LENGTHS:
            fold_number = 0
            current_test_y = None

            for train_index, val_index in inner_cv.split(
                train_val_data,
                train_val_data[TARGET_COLUMN],
                train_val_data[PATIENT_NUMBER_COLUMN],
            ):
                train_data = train_val_data.iloc[train_index].reset_index(drop=True)
                val_data = train_val_data.iloc[val_index].reset_index(drop=True)

                train_data = train_data[feature_columns].copy()
                val_data = val_data[feature_columns].copy()

                normalized_train, normalized_val, normalized_test = normalize_datasets(
                    train_data,
                    val_data,
                    test_data,
                )

                train_x, train_y_raw = create_sequences(
                    normalized_train,
                    sequence_length=seq_len,
                )
                val_x, val_y_raw = create_sequences(
                    normalized_val,
                    sequence_length=seq_len,
                )
                test_x, test_y_raw = create_sequences(
                    normalized_test,
                    sequence_length=seq_len,
                )

                if len(train_x) == 0 or len(train_y_raw) == 0:
                    print(
                        f"Skipping model={model_name}, seq_len={seq_len}, fold={fold_number} "
                        f"because training sequences are empty."
                    )
                    fold_number += 1
                    continue

                if len(val_x) == 0 or len(val_y_raw) == 0:
                    print(
                        f"Skipping model={model_name}, seq_len={seq_len}, fold={fold_number} "
                        f"because validation sequences are empty."
                    )
                    fold_number += 1
                    continue

                if len(test_x) == 0 or len(test_y_raw) == 0:
                    print(
                        f"Skipping model={model_name}, seq_len={seq_len}, fold={fold_number} "
                        f"because test sequences are empty."
                    )
                    fold_number += 1
                    continue

                train_x = train_x.astype(np.float32)
                val_x = val_x.astype(np.float32)
                test_x = test_x.astype(np.float32)

                train_y = train_y_raw.astype(np.int32) - 1
                val_y = val_y_raw.astype(np.int32) - 1
                test_y = test_y_raw.astype(np.int32) - 1

                train_y_onehot = to_fixed_one_hot(train_y_raw, num_classes=3)
                val_y_onehot = to_fixed_one_hot(val_y_raw, num_classes=3)

                model = build_model(
                    sequence_length=seq_len,
                    feature_size=train_x.shape[2],
                    model_name=model_name,
                )

                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                )

                counter += 1
                print(
                    f"Training model={model_name}, seq_len={seq_len}, "
                    f"fold={fold_number}, counter={counter}"
                )

                history = model.fit(
                    train_x,
                    train_y_onehot,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(val_x, val_y_onehot),
                    verbose=0,
                    callbacks=[early_stop],
                )

                y_pred = model.predict(test_x, verbose=0)
                metrics = compute_metrics(test_y, y_pred)

                recall_scores[(model_name, seq_len)].append(metrics["recall"])
                precision_scores[(model_name, seq_len)].append(metrics["precision"])
                accuracy_scores[(model_name, seq_len)].append(metrics["accuracy"])
                auc_scores[(model_name, seq_len)].append(metrics["auc"])

                y_pred_dict[(model_name, seq_len)].append(y_pred)
                current_test_y = test_y
                test_y_dict[str(seq_len)] = test_y

                models[(model_name, seq_len, fold_number)].append(model)
                trains[(model_name, seq_len, fold_number)].append(train_x)
                tests[(model_name, seq_len, fold_number)].append(test_x)
                histories[(model_name, seq_len, fold_number)].append(history)
                history_by_model_and_seq[(model_name, seq_len)].append(history.history)

                print(model_name, seq_len, fold_number, metrics)

                fold_number += 1

            if not y_pred_dict[(model_name, seq_len)]:
                print(f"No successful folds for model={model_name}, seq_len={seq_len}")
                continue

            y_pred_mean = np.mean(y_pred_dict[(model_name, seq_len)], axis=0)
            y_pred_std = np.std(y_pred_dict[(model_name, seq_len)], axis=0)

            y_prediction_mean_dict[(model_name, seq_len)] = y_pred_mean
            y_pred_std_dict[(model_name, seq_len)] = y_pred_std

            if current_test_y is None:
                raise ValueError(
                    f"Test targets were not generated for model={model_name}, seq_len={seq_len}."
                )

            y_pred_mean_dict[(model_name, seq_len)] = compute_metrics(
                current_test_y,
                y_pred_mean,
            )

    return {
        "summary_metrics": y_pred_mean_dict,
        "raw_predictions": y_pred_dict,
        "recall_scores": recall_scores,
        "precision_scores": precision_scores,
        "accuracy_scores": accuracy_scores,
        "auc_scores": auc_scores,
        "prediction_std": y_pred_std_dict,
        "test_targets": test_y_dict,
        "models": models,
        "train_data": trains,
        "test_data": tests,
        "histories": histories,
        "mean_predictions": y_prediction_mean_dict,
        "history_by_model_and_seq": history_by_model_and_seq,
    }