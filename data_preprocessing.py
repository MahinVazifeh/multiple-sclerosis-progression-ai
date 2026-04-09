from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    PATIENT_ID_COLUMN,
    PATIENT_NUMBER_COLUMN,
    OBSERVATION_TIME_COLUMN,
    NUM_OBS_COLUMN,
    MSSS_COLUMN,
    MSSS_CLASSIFIED_COLUMN,
    TARGET_COLUMN,
    EDSS_COLUMN,
    DISEASE_DURATION_YEARS_COLUMN,
    MAX_OBSERVATIONS_PER_PATIENT,
    DROP_COLUMN_INDEX_RANGES,
)


def load_data(file_path: Path) -> pd.DataFrame:
    """Load modeling dataset from CSV."""
    return pd.read_csv(file_path)


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with too many observations or missing MSSS class."""
    df = df.copy()
    df = df[df[NUM_OBS_COLUMN] < MAX_OBSERVATIONS_PER_PATIENT].copy()
    df = df[df[MSSS_CLASSIFIED_COLUMN].notna()].copy()
    return df


def add_patient_number(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric patient identifiers."""
    df = df.copy()
    df[PATIENT_NUMBER_COLUMN] = df[PATIENT_ID_COLUMN].factorize()[0] + 1
    return df


def add_observation_time(df: pd.DataFrame) -> pd.DataFrame:
    """Create sequential observation order for each patient."""
    df = df.copy()
    df[OBSERVATION_TIME_COLUMN] = df.groupby(PATIENT_NUMBER_COLUMN).cumcount() + 1
    return df


def add_target_group(df: pd.DataFrame) -> pd.DataFrame:
    """Map MSSS deciles into 3 grouped classes."""
    df = df.copy()
    df[TARGET_COLUMN] = df[MSSS_CLASSIFIED_COLUMN].apply(
        lambda x: 1 if x == 1 else 2 if x in [2, 3] else 3
    )
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed for sequence modeling."""
    df = df.copy()

    columns_to_drop = [
        MSSS_COLUMN,
        DISEASE_DURATION_YEARS_COLUMN,
        EDSS_COLUMN,
    ]
    existing_columns = [column for column in columns_to_drop if column in df.columns]

    return df.drop(columns=existing_columns, errors="ignore")


def drop_column_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Drop feature ranges based on configured column index ranges."""
    df = df.copy()

    idx_parts = [np.r_[start:end] for start, end in DROP_COLUMN_INDEX_RANGES]
    idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)
    valid_idx = [i for i in idx if i < len(df.columns)]

    return df.drop(df.columns[valid_idx], axis=1)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline."""
    df = filter_valid_rows(df)
    df = add_patient_number(df)
    df = add_observation_time(df)
    df = add_target_group(df)
    df = drop_unused_columns(df)
    df = drop_column_ranges(df)
    return df