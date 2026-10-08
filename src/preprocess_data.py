"""
Phase 2 - Data Preprocessing + Time-Series Sequence Creation
Project: Electricity Consumption Prediction Using LSTM

This script prepares the Phase 1 dataset for the LSTM:
    1. Load and check the CSV
    2. Sort chronologically
    3. Split into train / validation / test (70% / 15% / 15%, NO shuffling)
    4. Fit a MinMaxScaler on the TRAINING data only (avoids data leakage)
    5. Scale all three splits with that scaler
    6. Create sequences: previous 24 hours -> next hour
    7. Save the NumPy arrays and the scaler

Run from the project root folder:
    python src/preprocess_data.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
SEQUENCE_LENGTH = 24          # how many past hours the LSTM looks at (configurable)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15              # the remaining 15% is the test set

# Input features (the order matters! Phase 3+ must use the same order)
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "temperature",
    "electricity_consumption",
]
TARGET_COLUMN = "electricity_consumption"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "electricity_consumption.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


# ----------------------------------------------------------------------
# STEP 1: LOAD + CHECK
# ----------------------------------------------------------------------
def load_dataset(csv_path):
    """Load the CSV, convert timestamp to datetime and sort chronologically."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def check_dataset(df):
    """Check missing values and duplicate timestamps. Handle them if found."""
    missing_values = int(df.isnull().sum().sum())
    duplicate_timestamps = int(df["timestamp"].duplicated().sum())

    print(f"Total records: {len(df)}\n")
    print(f"Missing values: {missing_values}")
    print(f"Duplicate timestamps: {duplicate_timestamps}\n")

    # Phase 1 data is clean, but we handle problems anyway so the script is reusable.
    if duplicate_timestamps > 0:
        df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
        print("Duplicate timestamps removed.")

    if missing_values > 0:
        # Time-series friendly fix: fill gaps using neighbouring hours
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].interpolate(method="linear").bfill().ffill()
        print("Missing values filled using linear interpolation.")

    return df


# ----------------------------------------------------------------------
# STEP 2: CHRONOLOGICAL SPLIT
# ----------------------------------------------------------------------
def split_chronologically(data, train_ratio, val_ratio):
    """
    Split the rows into train / validation / test WITHOUT shuffling.

    Example with 8,760 rows:
        train = first 70%  (oldest data)
        val   = next 15%
        test  = last 15%   (newest data)
    """
    total_rows = len(data)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    return train_data, val_data, test_data


# ----------------------------------------------------------------------
# STEP 3: SCALING (no data leakage)
# ----------------------------------------------------------------------
def fit_scaler(train_data):
    """Fit a MinMaxScaler on TRAINING data only (values become 0 to 1)."""
    scaler = MinMaxScaler()
    scaler.fit(train_data[FEATURE_COLUMNS])
    return scaler


def scale_data(data, scaler):
    """Scale a split using the scaler that was fitted on the training data."""
    return scaler.transform(data[FEATURE_COLUMNS])


# ----------------------------------------------------------------------
# STEP 4: SEQUENCES
# ----------------------------------------------------------------------
def create_sequences(data, sequence_length, target_index):
    """
    Turn a 2D array of scaled data into LSTM sequences.

    data          : array with shape (rows, features)
    sequence_length: number of past hours used as input (e.g. 24)
    target_index  : column number of the value we want to predict

    Returns:
        X : shape (samples, sequence_length, features)  -> previous hours
        y : shape (samples,)                            -> the next hour's value

    Example (sequence_length = 24):
        X[0] = rows 0..23   ->  y[0] = row 24's target
        X[1] = rows 1..24   ->  y[1] = row 25's target
        ...
    """
    X, y = [], []
    for start in range(len(data) - sequence_length):
        end = start + sequence_length
        X.append(data[start:end])                  # the 24 previous hours
        y.append(data[end, target_index])          # the hour right after them
    return np.array(X), np.array(y)


def inverse_transform_target(scaled_values, scaler):
    """
    Convert scaled electricity consumption (0-1) back to real kWh.

    Use this in later phases on the model's predictions, e.g.:
        scaler = joblib.load("models/scaler.pkl")
        predictions_kwh = inverse_transform_target(predictions, scaler)
    """
    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)
    data_min = scaler.data_min_[target_index]
    data_range = scaler.data_range_[target_index]
    return np.asarray(scaled_values) * data_range + data_min


# ----------------------------------------------------------------------
# STEP 5: SAVE + VERIFY
# ----------------------------------------------------------------------
def save_outputs(arrays, scaler):
    """Save the NumPy arrays and the fitted scaler."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, array in arrays.items():
        np.save(PROCESSED_DIR / f"{name}.npy", array)

    joblib.dump(scaler, SCALER_PATH)


def verify_outputs(arrays, scaler, raw_df):
    """Run simple sanity checks on the final arrays."""
    n_features = len(FEATURE_COLUMNS)
    checks = []

    for split in ["train", "val", "test"]:
        X, y = arrays[f"X_{split}"], arrays[f"y_{split}"]
        checks.append((f"X_{split} shape is (samples, {SEQUENCE_LENGTH}, {n_features})",
                       X.ndim == 3 and X.shape[1:] == (SEQUENCE_LENGTH, n_features)))
        checks.append((f"X_{split} and y_{split} have the same number of samples",
                       len(X) == len(y)))
        checks.append((f"No NaN values in {split} arrays",
                       not np.isnan(X).any() and not np.isnan(y).any()))

    # The scaler was fitted on training data, so training values must be exactly 0 to 1.
    checks.append(("Training values are within 0-1",
                   arrays["X_train"].min() >= 0 and arrays["X_train"].max() <= 1))

    # Inverse transform check: the first test target must match the real kWh value.
    split_start = int(len(raw_df) * (TRAIN_RATIO + VAL_RATIO))
    expected_kwh = raw_df[TARGET_COLUMN].iloc[split_start]
    recovered_kwh = inverse_transform_target(arrays["y_test"][0], scaler)
    checks.append(("Inverse transform recovers the original kWh value",
                   bool(np.isclose(expected_kwh, recovered_kwh))))

    print("Verification:")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if not all(passed for _, passed in checks):
        raise ValueError("Verification failed - see the FAIL lines above.")
    print()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    print("Loading dataset...\n")
    df = load_dataset(RAW_CSV_PATH)
    df = check_dataset(df)

    # 1) Split chronologically (before scaling, so the scaler never sees val/test)
    train_data, val_data, test_data = split_chronologically(
        df[FEATURE_COLUMNS], TRAIN_RATIO, VAL_RATIO
    )

    # 2) Fit scaler on training data ONLY, then scale all three splits
    scaler = fit_scaler(train_data)
    train_scaled = scale_data(train_data, scaler)
    val_scaled = scale_data(val_data, scaler)
    test_scaled = scale_data(test_data, scaler)

    # 3) Create sequences
    #    Validation/test sequences need the 24 hours BEFORE their first target.
    #    We borrow those hours from the end of the previous split. This only
    #    uses past data as input (no leakage) and means every validation/test
    #    hour gets a prediction target.
    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)
    val_with_history = np.vstack([train_scaled[-SEQUENCE_LENGTH:], val_scaled])
    test_with_history = np.vstack([val_scaled[-SEQUENCE_LENGTH:], test_scaled])

    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH, target_index)
    X_val, y_val = create_sequences(val_with_history, SEQUENCE_LENGTH, target_index)
    X_test, y_test = create_sequences(test_with_history, SEQUENCE_LENGTH, target_index)

    arrays = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }

    # 4) Print shapes
    print(f"Sequence length: {SEQUENCE_LENGTH} hours")
    print(f"Features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}\n")
    print(f"Rows per split -> train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}\n")
    print("Training sequences:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}\n")
    print("Validation sequences:")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}\n")
    print("Testing sequences:")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}\n")

    # 5) Verify, then save
    verify_outputs(arrays, scaler, df)
    save_outputs(arrays, scaler)

    print(f"Processed arrays saved to: {PROCESSED_DIR}")
    print(f"Scaler saved to:           {SCALER_PATH}\n")
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()