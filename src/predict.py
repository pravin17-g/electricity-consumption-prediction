"""
Phase 5 - Prediction System
Electricity Consumption Prediction Using LSTM

A reusable module that LOADS the trained LSTM and predicts:
  * predict_next_hour()      -> next hour's consumption (kWh)
  * predict_next_24_hours()  -> next 24 hours (kWh), recursive multi-step

It never trains anything. It only loads:
  models/lstm_model.keras
  models/scaler.pkl
  data/processed/X_test.npy, y_test.npy   (source of the latest 24 hours)

Run from the project root:
    python src/predict.py
"""

from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save the graph to a file; no display needed
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FORECAST_PLOT_PATH = RESULTS_DIR / "future_24_hour_prediction.png"

# --------------------------------------------------------------------------
# Settings (must match how Phase 2 prepared the data)
# --------------------------------------------------------------------------
FORECAST_HOURS = 24

# Column index of the electricity-consumption (kWh) feature inside each
# timestep. Same value as TARGET_COLUMN_INDEX in train.py.
TARGET_COLUMN_INDEX = 0

# How to fill the NON-consumption features (hour of day, weather, etc.) for a
# future hour, because the model needs a full feature row at every step:
#   "yesterday" -> reuse the row from the same hour 24 h earlier (default;
#                  daily patterns such as hour-of-day repeat correctly)
#   "last"      -> repeat the most recent row
FEATURE_FILL_STRATEGY = "yesterday"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model():
    """Load the trained LSTM (cached, so it is only read from disk once)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run the training script first:  python src/train.py"
        )
    from tensorflow import keras  # imported here so import errors are clear

    return keras.models.load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def load_scaler():
    """Load the fitted scaler saved during preprocessing (cached)."""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}\n"
            "It is created by the preprocessing step (Phase 2). If it was saved "
            "under another name/location, copy it to models/scaler.pkl."
        )
    return joblib.load(SCALER_PATH)


def _load_processed(name):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}\nRun the preprocessing step first."
        )
    return np.load(path)


# --------------------------------------------------------------------------
# Scaling helpers
# --------------------------------------------------------------------------
def inverse_scale_target(values, scaler=None):
    """Convert scaled consumption values (0-1 range) back to kWh.

    Works whether the scaler was fitted on the target only, or on all columns
    (then TARGET_COLUMN_INDEX selects the consumption column).
    """
    scaler = scaler if scaler is not None else load_scaler()
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    n_cols = int(getattr(scaler, "n_features_in_", 1))

    if n_cols == 1:
        return scaler.inverse_transform(values).ravel()

    if not 0 <= TARGET_COLUMN_INDEX < n_cols:
        raise ValueError(
            f"TARGET_COLUMN_INDEX={TARGET_COLUMN_INDEX} is invalid for a scaler "
            f"with {n_cols} columns."
        )
    dummy = np.zeros((len(values), n_cols))
    dummy[:, TARGET_COLUMN_INDEX] = values[:, 0]
    return scaler.inverse_transform(dummy)[:, TARGET_COLUMN_INDEX]


# --------------------------------------------------------------------------
# Sequence preparation
# --------------------------------------------------------------------------
def _build_next_row(window, predicted_scaled):
    """Create the feature row for the next hour.

    The consumption column gets the (scaled) value; the other features are
    filled according to FEATURE_FILL_STRATEGY.
    """
    if FEATURE_FILL_STRATEGY == "yesterday":
        row = window[0].copy()      # same hour, 24 h earlier
    elif FEATURE_FILL_STRATEGY == "last":
        row = window[-1].copy()
    else:
        raise ValueError(
            f"Unknown FEATURE_FILL_STRATEGY: {FEATURE_FILL_STRATEGY!r}"
        )
    row[TARGET_COLUMN_INDEX] = predicted_scaled
    return row


def _slide_window(window, predicted_scaled):
    """Drop the oldest hour, append the new one. Shape stays (24, F)."""
    next_row = _build_next_row(window, predicted_scaled)
    return np.vstack([window[1:], next_row])


def prepare_latest_sequence():
    """Return the latest 24-hour window, scaled, shape (24, num_features).

    Source: the last sequence of the processed test data (X_test[-1]). That
    window ends one hour before the final known reading, so the final known
    actual value (y_test[-1]) is slid in to make the window end at the most
    recent hour of data.
    """
    X_test = _load_processed("X_test.npy")
    y_test = _load_processed("y_test.npy")

    if X_test.ndim != 3 or len(X_test) == 0:
        raise ValueError(f"X_test.npy has an unexpected shape: {X_test.shape}")

    window = X_test[-1].astype(np.float32)
    last_actual_scaled = float(np.asarray(y_test).reshape(-1)[-1])
    return _slide_window(window, last_actual_scaled)


def _check_sequence(sequence, model):
    """Make sure the sequence matches the model's expected input (24, F)."""
    sequence = np.asarray(sequence, dtype=np.float32)
    expected = tuple(model.input_shape[1:])
    if sequence.shape != expected:
        raise ValueError(
            f"Sequence shape {sequence.shape} does not match the model input "
            f"shape {expected}."
        )
    return sequence


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _predict_scaled(model, sequence):
    """One forward pass. Input (24, F) -> scaled prediction (float)."""
    batch = sequence.reshape(1, *sequence.shape)  # (1, 24, F)
    output = model(batch, training=False).numpy()
    return float(output[0, 0])


def predict_next_hour(sequence=None):
    """Predict the next hour's electricity consumption in kWh."""
    model = load_model()
    scaler = load_scaler()
    if sequence is None:
        sequence = prepare_latest_sequence()
    sequence = _check_sequence(sequence, model)

    scaled = _predict_scaled(model, sequence)
    kwh = inverse_scale_target([scaled], scaler)[0]
    return float(max(kwh, 0.0))  # consumption cannot be negative


def predict_next_24_hours(sequence=None, hours=FORECAST_HOURS):
    """Recursive multi-step forecast. Returns an array of `hours` kWh values."""
    model = load_model()
    scaler = load_scaler()
    if sequence is None:
        sequence = prepare_latest_sequence()
    window = _check_sequence(sequence, model).copy()

    scaled_predictions = []
    for _ in range(hours):
        scaled = _predict_scaled(model, window)       # predict next hour
        scaled_predictions.append(scaled)
        window = _slide_window(window, scaled)         # feed it back in

    kwh = inverse_scale_target(scaled_predictions, scaler)
    return np.maximum(kwh, 0.0)


# --------------------------------------------------------------------------
# Graph + display
# --------------------------------------------------------------------------
def plot_forecast(predictions, path=FORECAST_PLOT_PATH):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hours = np.arange(1, len(predictions) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(hours, predictions, marker="o", linewidth=2)
    plt.xlabel("Future Hour")
    plt.ylabel("Predicted Electricity Consumption (kWh)")
    plt.title("Electricity Consumption Forecast - Next 24 Hours")
    plt.xticks(hours)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    try:
        sequence = prepare_latest_sequence()
        next_hour = predict_next_hour(sequence)
        forecast = predict_next_24_hours(sequence)
        plot_path = plot_forecast(forecast)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)

    print("Electricity Consumption Forecast\n")
    print("Next Hour:")
    print(f"{next_hour:.2f} kWh\n")
    print("Next 24 Hours:\n")
    for hour, value in enumerate(forecast, start=1):
        print(f"Hour {hour}: {value:.2f} kWh")
    print(f"\nGraph saved to: {plot_path}")


if __name__ == "__main__":
    main()