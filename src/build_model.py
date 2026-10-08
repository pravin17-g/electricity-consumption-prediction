"""
Phase 3 - LSTM Model Development
Electricity Consumption Prediction Using LSTM

This script:
  1. Loads the processed training and validation arrays
  2. Detects the input shape from the data
  3. Builds the LSTM architecture
  4. Prints the model summary
  5. Compiles the model (Adam, MSE, MAE)
  6. Saves the model configuration
  7. Runs a SMALL sanity-check training (NOT a real training run)

Run from the project root:
    python src/build_model.py
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = MODELS_DIR / "model_config.json"

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
RANDOM_SEED = 42
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
DENSE_UNITS = 16
LEARNING_RATE = 0.001

# Sanity-check run only (kept deliberately tiny)
SANITY_EPOCHS = 2
SANITY_BATCH_SIZE = 32
SANITY_TRAIN_SAMPLES = 2000
SANITY_VAL_SAMPLES = 500


def load_data():
    """Load the processed train/validation arrays."""
    files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    for name in files:
        if not (PROCESSED_DIR / name).exists():
            raise FileNotFoundError(
                f"Missing {PROCESSED_DIR / name}. Run Phase 2 preprocessing first."
            )

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    # Sanity checks on the data
    assert X_train.ndim == 3, f"X_train must be 3D, got shape {X_train.shape}"
    assert X_val.ndim == 3, f"X_val must be 3D, got shape {X_val.shape}"
    assert X_train.shape[1:] == X_val.shape[1:], "Train/val shapes do not match"
    assert len(X_train) == len(y_train), "X_train / y_train length mismatch"
    assert len(X_val) == len(y_val), "X_val / y_val length mismatch"

    return X_train, y_train, X_val, y_val


def build_lstm_model(sequence_length, num_features):
    """Build the LSTM model: LSTM(64) -> Dropout -> LSTM(32) -> Dropout -> Dense(16) -> Dense(1)."""
    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length, num_features), name="input_sequence"),
            layers.LSTM(LSTM_UNITS_1, return_sequences=True, name="lstm_1"),
            layers.Dropout(DROPOUT_RATE, name="dropout_1"),
            layers.LSTM(LSTM_UNITS_2, return_sequences=False, name="lstm_2"),
            layers.Dropout(DROPOUT_RATE, name="dropout_2"),
            layers.Dense(DENSE_UNITS, activation="relu", name="dense_hidden"),
            layers.Dense(1, name="output_next_hour"),
        ],
        name="electricity_lstm",
    )
    return model


def compile_model(model):
    """Compile with Adam, MSE loss and MAE metric."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def save_config(sequence_length, num_features, model):
    """Save a small JSON description of the model for later phases."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "Next-hour electricity consumption prediction",
        "sequence_length": int(sequence_length),
        "num_features": int(num_features),
        "input_shape": [int(sequence_length), int(num_features)],
        "architecture": [
            {"layer": "LSTM", "units": LSTM_UNITS_1, "return_sequences": True},
            {"layer": "Dropout", "rate": DROPOUT_RATE},
            {"layer": "LSTM", "units": LSTM_UNITS_2, "return_sequences": False},
            {"layer": "Dropout", "rate": DROPOUT_RATE},
            {"layer": "Dense", "units": DENSE_UNITS, "activation": "relu"},
            {"layer": "Dense", "units": 1, "activation": "linear"},
        ],
        "compile": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "mean_squared_error",
            "metrics": ["mae"],
        },
        "total_parameters": int(model.count_params()),
        "status": "architecture built - sanity check only, NOT fully trained",
        "final_model_path": "models/lstm_model.keras",
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Model configuration saved to: {CONFIG_PATH}")


def sanity_check_training(model, X_train, y_train, X_val, y_val):
    """Tiny training run to confirm data -> model -> loss works end to end."""
    n_train = min(SANITY_TRAIN_SAMPLES, len(X_train))
    n_val = min(SANITY_VAL_SAMPLES, len(X_val))

    print(
        f"\nSanity check: {SANITY_EPOCHS} epochs on {n_train} train / {n_val} val samples"
    )
    print("(This is only a pipeline test, NOT real training.)\n")

    history = model.fit(
        X_train[:n_train],
        y_train[:n_train],
        validation_data=(X_val[:n_val], y_val[:n_val]),
        epochs=SANITY_EPOCHS,
        batch_size=SANITY_BATCH_SIZE,
        verbose=1,
    )

    final_loss = history.history["loss"][-1]
    if not np.isfinite(final_loss):
        raise RuntimeError("Sanity check failed: loss is NaN/inf. Check the data.")

    # Confirm output shape is (batch, 1)
    preds = model.predict(X_val[:5], verbose=0)
    print(f"\nPrediction output shape for 5 samples: {preds.shape}  (expected (5, 1))")
    print("Sanity check passed: the model trains and produces predictions.")


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # 1 & 2. Load data
    print("Loading processed data...")
    X_train, y_train, X_val, y_val = load_data()
    print(f"X_train: {X_train.shape}   y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}   y_val:   {y_val.shape}")

    # 3. Determine input shape from the data (no hard-coding)
    sequence_length, num_features = X_train.shape[1], X_train.shape[2]
    print(f"\nDetected sequence length : {sequence_length}")
    print(f"Detected number of features: {num_features}")

    # 4. Build the model and print summary
    model = build_lstm_model(sequence_length, num_features)
    print("\nModel summary:")
    model.summary()

    # 5. Compile
    compile_model(model)
    print("\nModel compiled (Adam, MSE loss, MAE metric).")

    # 6. Save configuration
    save_config(sequence_length, num_features, model)

    # 7. Small sanity-check training run
    sanity_check_training(model, X_train, y_train, X_val, y_val)

    print(
        "\nPhase 3 complete. The model is NOT fully trained; "
        "full training (Phase 4) will produce models/lstm_model.keras."
    )


if __name__ == "__main__":
    main()