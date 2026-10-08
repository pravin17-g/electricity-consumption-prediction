"""
Phase 4 - Model Training and Evaluation
Electricity Consumption Prediction Using LSTM

This script:
  1. Loads the processed train / validation / test arrays
  2. Builds the Phase 3 LSTM architecture (imported from build_model.py)
  3. Trains with EarlyStopping + ModelCheckpoint
  4. Saves the best model to models/lstm_model.keras
  5. Saves the training history and the training-loss graph
  6. Reloads the best model and predicts on the test set
  7. Converts predictions and targets back to kWh
  8. Computes MAE, RMSE, MAPE and saves them
  9. Saves the actual-vs-predicted graph

Run from the project root:
    python src/train.py

The original dataset and the processed arrays are only READ, never modified.
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed; we only save images
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reuse the exact architecture and compile settings from Phase 3
from build_model import build_lstm_model, compile_model

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODELS_DIR / "lstm_model.keras"
HISTORY_JSON_PATH = RESULTS_DIR / "training_history.json"
LOSS_PLOT_PATH = RESULTS_DIR / "training_loss.png"
PREDICTION_PLOT_PATH = RESULTS_DIR / "actual_vs_predicted.png"
METRICS_PATH = RESULTS_DIR / "metrics.txt"

# --------------------------------------------------------------------------
# Configurable training settings
# --------------------------------------------------------------------------
RANDOM_SEED = 42
EPOCHS = 30
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 5   # epochs without val_loss improvement before stopping
PLOT_HOURS = 168              # actual-vs-predicted graph shows 168 h = 1 week

# --------------------------------------------------------------------------
# Inverse-scaling settings (must match how Phase 2 scaled the data)
# --------------------------------------------------------------------------
# Leave as None to auto-detect a saved scaler in models/ or data/processed/.
# Or set it explicitly, e.g. PROJECT_ROOT / "models" / "scaler.pkl"
SCALER_PATH = None

# If the scaler was fitted on ALL columns (not just the target), this is the
# column index of the electricity-consumption (kWh) column in that table.
TARGET_COLUMN_INDEX = 0

# MAPE is ignored for hours where actual consumption is below this value (kWh)
MAPE_MIN_ACTUAL = 1e-3


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_array(name):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the earlier phases first.")
    return np.load(path)


def load_all_data():
    X_train, y_train = load_array("X_train.npy"), load_array("y_train.npy")
    X_val, y_val = load_array("X_val.npy"), load_array("y_val.npy")
    X_test, y_test = load_array("X_test.npy"), load_array("y_test.npy")
    return X_train, y_train, X_val, y_val, X_test, y_test


# --------------------------------------------------------------------------
# Scaler handling
# --------------------------------------------------------------------------
def find_scaler_path():
    """Locate the scaler saved during Phase 2."""
    if SCALER_PATH is not None:
        path = Path(SCALER_PATH)
        if not path.exists():
            raise FileNotFoundError(f"SCALER_PATH does not exist: {path}")
        return path

    candidates = []
    for folder in (MODELS_DIR, PROCESSED_DIR):
        if folder.exists():
            for pattern in ("*.pkl", "*.joblib", "*.save"):
                candidates.extend(sorted(folder.glob(pattern)))

    if not candidates:
        raise FileNotFoundError(
            "No scaler file (.pkl/.joblib) found in models/ or data/processed/. "
            "Set SCALER_PATH at the top of train.py."
        )

    # Prefer a scaler dedicated to the target if one exists
    for path in candidates:
        if "target" in path.name.lower():
            return path
    for path in candidates:
        if "scaler" in path.name.lower():
            return path
    return candidates[0]


def inverse_scale_target(values, scaler):
    """Convert scaled consumption values back to kWh.

    Works for both cases:
      * scaler fitted on the target only (1 feature)
      * scaler fitted on all columns -> uses TARGET_COLUMN_INDEX
    """
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
# Metrics
# --------------------------------------------------------------------------
def safe_mape(y_true, y_pred, min_actual=MAPE_MIN_ACTUAL):
    """MAPE in %, ignoring points where actual is ~0 (avoids division by zero)."""
    mask = np.abs(y_true) > min_actual
    skipped = int((~mask).sum())
    if mask.sum() == 0:
        return float("nan"), skipped
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return float(mape), skipped


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape, skipped = safe_mape(y_true, y_pred)
    return mae, rmse, mape, skipped


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_training_loss(history):
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["loss"], label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE, scaled data)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred):
    n = min(PLOT_HOURS, len(y_true))
    hours = np.arange(n)
    plt.figure(figsize=(12, 5))
    plt.plot(hours, y_true[:n], label="Actual Electricity Consumption", linewidth=2)
    plt.plot(
        hours,
        y_pred[:n],
        label="Predicted Electricity Consumption",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel(f"Hour (first {n} hours of the test set)")
    plt.ylabel("Electricity Consumption (kWh)")
    plt.title("Actual vs Predicted Electricity Consumption (Test Data)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    # Reproducibility (seeds Python, NumPy and TensorFlow)
    keras.utils.set_random_seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("Loading processed data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_data()
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    sequence_length, num_features = X_train.shape[1], X_train.shape[2]
    print(f"Sequence length: {sequence_length}   Features: {num_features}")

    # 2. Build + compile (Adam, MSE, MAE - from Phase 3)
    model = build_lstm_model(sequence_length, num_features)
    compile_model(model)
    model.summary()

    # 3. Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # 4. Train
    print(f"\nTraining: up to {EPOCHS} epochs, batch size {BATCH_SIZE}")
    history_obj = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
    )
    history = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}

    # 5. Save history + loss graph
    with open(HISTORY_JSON_PATH, "w") as f:
        json.dump(history, f, indent=4)
    plot_training_loss(history)
    print(f"\nTraining history saved to: {HISTORY_JSON_PATH}")
    print(f"Training loss graph saved to: {LOSS_PLOT_PATH}")

    # 6. Confirm the best model was saved, then load it
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model was NOT saved: {MODEL_PATH} does not exist.")
    best_model = keras.models.load_model(MODEL_PATH)
    print(f"Best model loaded from: {MODEL_PATH}")

    # 7. Predict on the test set (still in scaled space)
    y_pred_scaled = best_model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

    # 8. Convert back to the original kWh scale
    scaler_path = find_scaler_path()
    print(f"Using scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)

    y_pred = inverse_scale_target(y_pred_scaled, scaler)
    y_true = inverse_scale_target(y_test, scaler)
    print(
        f"Test actual range (kWh): {y_true.min():.3f} to {y_true.max():.3f}  |  "
        f"predicted range: {y_pred.min():.3f} to {y_pred.max():.3f}"
    )

    # 9. Metrics
    mae, rmse, mape, skipped = compute_metrics(y_true, y_pred)
    if skipped:
        print(f"Note: {skipped} near-zero actual values were excluded from MAPE.")

    metrics_text = (
        "Electricity Consumption Prediction - LSTM\n\n"
        "MAE:\n"
        f"{mae:.4f} kWh\n\n"
        "RMSE:\n"
        f"{rmse:.4f} kWh\n\n"
        "MAPE:\n"
        f"{mape:.2f} %\n"
    )
    with open(METRICS_PATH, "w") as f:
        f.write(metrics_text)

    print("\n" + metrics_text)
    print(f"Metrics saved to: {METRICS_PATH}")

    # 10. Actual vs predicted graph
    plot_actual_vs_predicted(y_true, y_pred)
    print(f"Actual vs predicted graph saved to: {PREDICTION_PLOT_PATH}")

    # 11. Final status
    stopped_epoch = len(history["loss"])
    print(f"\nTraining ran for {stopped_epoch} epoch(s).")
    print(f"SUCCESS: trained model saved at {MODEL_PATH}")


if __name__ == "__main__":
    main()