from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from config import (
    MSSS_CLASSIFIED_COLUMN,
    PATIENT_NUMBER_COLUMN,
)


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict[str, float]:
    """
    Compute classification metrics from true labels and predicted probabilities.
    """
    y_pred = np.argmax(y_pred_prob, axis=1)

    return {
        "recall": round(recall_score(y_true, y_pred, average="macro"), 3),
        "precision": round(precision_score(y_true, y_pred, average="macro"), 3),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "auc": round(
            roc_auc_score(y_true, y_pred_prob, average="macro", multi_class="ovr"),
            3,
        ),
    }


def build_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table showing patient count and observation count per MSSS decile.
    """
    distribution = defaultdict(list)

    for decile in range(1, 11):
        subset = df[df[MSSS_CLASSIFIED_COLUMN] == decile]
        patient_count = subset[PATIENT_NUMBER_COLUMN].nunique()
        observation_count = len(subset)
        distribution[decile].append((patient_count, observation_count))

    return pd.DataFrame.from_dict(
        distribution,
        orient="index",
        columns=["(Patient Count, Observation Count)"],
    )


def save_distribution_table(df: pd.DataFrame, output_file: Path) -> None:
    """
    Save the MSSS distribution table to CSV.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=True)


def save_training_loss_plot(
    history_by_model_and_seq: dict,
    output_file: Path,
    model_name: str = "GRU",
    sequence_lengths: list[int] | None = None,
) -> None:
    """
    Save training and validation loss plots for the selected model.
    """
    if sequence_lengths is None:
        sequence_lengths = [2, 3, 4]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(sequence_lengths), figsize=(15, 5), sharey=True)

    if len(sequence_lengths) == 1:
        axes = [axes]

    for i, seq_len in enumerate(sequence_lengths):
        ax = axes[i]
        histories = history_by_model_and_seq.get((model_name, seq_len), [])

        if histories:
            history = histories[0]
            ax.plot(history["loss"], label="Train", color="#17c3b2")
            ax.plot(history["val_loss"], label="Validation", color="#FF9A9A")

        ax.set_title(f"{model_name} - {seq_len} Observations")
        ax.set_xlabel("Epochs")
        if i == 0:
            ax.set_ylabel("Loss")
        ax.grid(True, linestyle="-", color="black", alpha=0.2)

        if histories:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)