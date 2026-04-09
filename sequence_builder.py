import numpy as np
import pandas as pd

from config import (
    PATIENT_NUMBER_COLUMN,
    OBSERVATION_TIME_COLUMN,
    TARGET_COLUMN,
)


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create patient-level sequential samples to predict the next observation class.
    """
    patient_ids = df[PATIENT_NUMBER_COLUMN].unique().tolist()

    x_sequences = []
    y_labels = []

    for patient_id in patient_ids:
        patient_df = df[df[PATIENT_NUMBER_COLUMN] == patient_id]
        patient_df = patient_df.sort_values(
            by=[OBSERVATION_TIME_COLUMN]
        ).reset_index(drop=True)

        patient_length = len(patient_df)

        if patient_length > sequence_length:
            y_values = patient_df[TARGET_COLUMN].values
            patient_features = patient_df.drop(
                columns=[TARGET_COLUMN, PATIENT_NUMBER_COLUMN],
                errors="ignore",
            )

            for i in range(patient_length - sequence_length):
                x_sequences.append(
                    patient_features.iloc[i:i + sequence_length].values
                )
                y_labels.append(y_values[i + sequence_length])

        else:
            padded_df = pd.concat(
                [patient_df, patient_df.tail(sequence_length - patient_length + 1)],
                ignore_index=True,
            )
            padded_df = padded_df.iloc[: sequence_length + 1]

            y_values = padded_df[TARGET_COLUMN].values
            patient_features = padded_df.drop(
                columns=[TARGET_COLUMN, PATIENT_NUMBER_COLUMN],
                errors="ignore",
            )

            if len(patient_features) > sequence_length:
                x_sequences.append(patient_features.iloc[:sequence_length].values)
                y_labels.append(y_values[sequence_length])

    return np.array(x_sequences), np.array(y_labels)


def one_hot_encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Convert labels {1, 2, 3} to one-hot encoded zero-based labels.
    """
    y_zero_based = y.astype(np.int32) - 1
    y_one_hot = np.zeros((y_zero_based.size, y_zero_based.max() + 1), dtype=int)
    y_one_hot[np.arange(y_zero_based.size), y_zero_based] = 1
    return y_one_hot