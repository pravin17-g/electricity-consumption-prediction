"""
Electricity Consumption Prediction - Streamlit App (Phase 6)

Run with:  streamlit run app/app.py

Only the prediction / scaling logic is "fixed" here. The model, scaler,
dataset and preprocessing files are loaded as-is and never modified.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore", category=UserWarning)  # sklearn feature-name warnings

# ----------------------------------------------------------------------
# Paths (resolved relative to the project root, so the app works no matter
# which folder you launch streamlit from)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_model.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "electricity_consumption.csv"
X_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "X_test.npy"
Y_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "y_test.npy"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.txt"

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
# The scaler was fitted on ALL 6 columns in exactly this order, and the LSTM
# was trained on the same 6 columns. Order matters: MinMaxScaler stores one
# min/max per column POSITION, so a different order would scale the wrong data.
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "temperature",
    "electricity_consumption",
]
N_FEATURES = len(FEATURE_COLUMNS)  # 6
TARGET_COLUMN = "electricity_consumption"
TARGET_INDEX = FEATURE_COLUMNS.index(TARGET_COLUMN)  # 5
SEQ_LEN = 24  # the model looks at the previous 24 hours
FORECAST_HOURS = 24

TIMESTAMP_CANDIDATES = ["timestamp", "datetime", "date_time", "date", "time"]


# ----------------------------------------------------------------------
# Helper: inverse transform ONLY the target column
# ----------------------------------------------------------------------
def inverse_transform_target(values, scaler, target_index=TARGET_INDEX):
    """
    Convert scaled electricity_consumption values (0-1) back to kWh.

    Why this is needed: the scaler was fitted on 6 columns, so
    scaler.inverse_transform() demands an array of shape (n, 6). The model
    only outputs 1 column (n, 1), which caused:
        "non-broadcastable output operand with shape (300,1) doesn't match
         the broadcast shape (300,6)"
    We place the values in the target column of a temporary (n, 6) array,
    inverse-transform that, and read the target column back out.

    Returns a 1D array of original kWh values.
    """
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    dummy = np.zeros((len(values), scaler.n_features_in_))
    dummy[:, target_index] = values[:, 0]
    restored = scaler.inverse_transform(dummy)
    return restored[:, target_index]


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data(show_spinner=False)
def load_history():
    """Load the raw dataset, sort by timestamp, and make sure all 6 feature columns exist."""
    df = pd.read_csv(RAW_DATA_PATH)

    ts_col = next((c for c in TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise ValueError(
            f"No timestamp column found. Expected one of {TIMESTAMP_CANDIDATES}, "
            f"got {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df[ts_col])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Derive calendar features from the timestamp only if the CSV lacks them.
    # (Monday=0 ... Sunday=6, weekend = Sat/Sun.)
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "month" not in df.columns:
        df["month"] = df["timestamp"].dt.month

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


@st.cache_data(show_spinner=False)
def load_test_data():
    return np.load(X_TEST_PATH), np.load(Y_TEST_PATH)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_metrics(y_actual, y_pred):
    """MAE / RMSE / MAPE on ORIGINAL kWh values (never on 0-1 scaled values)."""
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_actual - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nonzero = y_actual != 0  # avoid division by zero in MAPE
    mape = float(np.mean(np.abs(err[nonzero] / y_actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return mae, rmse, mape


def get_test_predictions(model, scaler):
    """Predict on X_test and return (y_actual_kwh, y_pred_kwh), both 1D arrays of shape (N,)."""
    X_test, y_test = load_test_data()
    assert X_test.ndim == 3 and X_test.shape[1:] == (SEQ_LEN, N_FEATURES), (
        f"X_test has shape {X_test.shape}, expected (N, {SEQ_LEN}, {N_FEATURES})"
    )

    y_pred_scaled = model.predict(X_test, verbose=0)          # (N, 1), scaled 0-1
    y_pred = inverse_transform_target(y_pred_scaled, scaler)   # (N,), kWh
    y_actual = inverse_transform_target(y_test, scaler)        # (N,), kWh (handles (N,) or (N,1))

    assert y_pred.shape == y_actual.shape, f"Shape mismatch: {y_pred.shape} vs {y_actual.shape}"
    return y_actual, y_pred


# ----------------------------------------------------------------------
# 24-hour recursive forecast
# ----------------------------------------------------------------------
def forecast_next_24_hours(model, scaler, history):
    """
    Recursively forecast the next 24 hours.

    The LSTM input is ALWAYS (1, 24, 6): 24 time steps x 6 features. The
    scaler also expects 6 columns, so every row we scale/append must contain
    all six features, with the predicted consumption in the last column.
    """
    # Latest 24 rows (history is already sorted by timestamp), in exact feature order.
    recent_raw = history[FEATURE_COLUMNS].tail(SEQ_LEN).to_numpy(dtype=float)
    assert recent_raw.shape == (SEQ_LEN, N_FEATURES), f"Not enough history: {recent_raw.shape}"

    # Scale all 6 columns together, exactly as during training.
    recent_scaled = scaler.transform(recent_raw)
    assert recent_scaled.shape == (SEQ_LEN, N_FEATURES)

    # Deterministic temperature estimate: mean temperature per (month, hour),
    # falling back to the overall mean. No randomness, so results are reproducible.
    temp_lookup = history.groupby(["month", "hour"])["temperature"].mean()
    overall_temp = float(history["temperature"].mean())

    last_ts = history["timestamp"].iloc[-1]
    sequence = recent_scaled.copy()
    rows = []

    for step in range(1, FORECAST_HOURS + 1):
        future_ts = last_ts + pd.Timedelta(hours=step)

        hour = future_ts.hour
        day_of_week = future_ts.dayofweek
        is_weekend = int(day_of_week >= 5)
        month = future_ts.month
        temperature = float(temp_lookup.get((month, hour), overall_temp))

        # Model input: (1, 24, 6)
        model_input = sequence.reshape(1, SEQ_LEN, N_FEATURES)
        assert model_input.shape == (1, SEQ_LEN, N_FEATURES)

        # The model returns ONE scaled consumption value.
        pred_scaled = float(np.asarray(model.predict(model_input, verbose=0)).reshape(-1)[0])

        # Convert to kWh for display (and to build a full raw row).
        pred_kwh = float(inverse_transform_target([pred_scaled], scaler)[0])

        # Build the complete 6-feature raw row, in training order, and scale it
        # with the full scaler. (Scaling a 1-feature array caused:
        # "X has 1 features, but MinMaxScaler is expecting 6 features".)
        future_row_raw = np.array(
            [[hour, day_of_week, is_weekend, month, temperature, pred_kwh]], dtype=float
        )
        future_row_scaled = scaler.transform(future_row_raw)
        assert future_row_scaled.shape == (1, N_FEATURES)

        # Slide the window: drop oldest row, append the new one.
        sequence = np.vstack([sequence[1:], future_row_scaled])
        assert sequence.shape == (SEQ_LEN, N_FEATURES)

        rows.append(
            {
                "forecast_hour": step,
                "timestamp": future_ts,
                "predicted_consumption_kwh": round(pred_kwh, 4),
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.set_page_config(page_title="Electricity Consumption Prediction", page_icon="⚡", layout="wide")
st.title("⚡ Electricity Consumption Prediction")
st.caption("LSTM model • previous 24 hours → next hour")

# Load everything up front
try:
    model, scaler = load_artifacts()
    history = load_history()
except Exception as e:
    st.error(f"Could not load model / scaler / dataset: {e}")
    st.stop()

# Sanity check: model and scaler agree on 6 features
if scaler.n_features_in_ != N_FEATURES:
    st.warning(f"Scaler expects {scaler.n_features_in_} features, app is configured for {N_FEATURES}.")

# Compute test predictions once (used by metrics + Actual vs Predicted)
y_actual, y_pred, test_error = None, None, None
try:
    y_actual, y_pred = get_test_predictions(model, scaler)
except Exception as e:
    test_error = str(e)

if "forecast_df" not in st.session_state:
    st.session_state["forecast_df"] = None

# ---------------- Top metrics row ----------------
st.subheader("Model Metrics")
if y_actual is not None:
    mae, rmse, mape = compute_metrics(y_actual, y_pred)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (kWh)", f"{mae:.4f}")
    c2.metric("RMSE (kWh)", f"{rmse:.4f}")
    c3.metric("MAPE (%)", f"{mape:.2f}")
    forecast_df_state = st.session_state["forecast_df"]
    c4.metric(
        "Predicted Next Hour (kWh)",
        "--" if forecast_df_state is None else f"{forecast_df_state['predicted_consumption_kwh'].iloc[0]:.3f}",
    )
else:
    st.error(f"Could not generate test predictions: {test_error}")
    forecast_df_state = st.session_state["forecast_df"]
    st.metric(
        "Predicted Next Hour (kWh)",
        "--" if forecast_df_state is None else f"{forecast_df_state['predicted_consumption_kwh'].iloc[0]:.3f}",
    )

if METRICS_PATH.exists():
    with st.expander("Saved training metrics (results/metrics.txt)"):
        st.text(METRICS_PATH.read_text())

st.divider()

# ---------------- Historical consumption ----------------
st.subheader("Historical Consumption")
days = st.slider("Days to display", min_value=1, max_value=60, value=7)
hist_view = history.tail(days * 24)
fig_hist = go.Figure()
fig_hist.add_trace(
    go.Scatter(x=hist_view["timestamp"], y=hist_view[TARGET_COLUMN], mode="lines", name="Consumption")
)
fig_hist.update_layout(
    title="Historical Electricity Consumption",
    xaxis_title="Timestamp",
    yaxis_title="Consumption (kWh)",
)
st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ---------------- Actual vs Predicted ----------------
st.subheader("Actual vs Predicted")
if y_actual is not None:
    n_points = st.slider(
        "Test points to display",
        min_value=50,
        max_value=int(min(1000, len(y_actual))),
        value=int(min(300, len(y_actual))),
    )
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(y=y_actual[-n_points:], mode="lines", name="Actual"))
    fig_avp.add_trace(go.Scatter(y=y_pred[-n_points:], mode="lines", name="Predicted"))
    fig_avp.update_layout(
        title="Actual vs Predicted Electricity Consumption",
        xaxis_title="Test Sample",
        yaxis_title="Consumption (kWh)",
    )
    st.plotly_chart(fig_avp, use_container_width=True)
else:
    st.info("Actual vs Predicted is unavailable until test predictions succeed.")

st.divider()

# ---------------- 24-hour forecast ----------------
st.subheader("Next 24-Hour Electricity Forecast")
if st.button("Generate 24-Hour Forecast"):
    try:
        with st.spinner("Generating forecast..."):
            st.session_state["forecast_df"] = forecast_next_24_hours(model, scaler, history)
        st.rerun()  # refresh so the "Predicted Next Hour" metric above updates
    except Exception as e:
        st.error(f"Forecast failed: {e}")

forecast_df = st.session_state["forecast_df"]
if forecast_df is not None:
    st.dataframe(forecast_df, use_container_width=True)

    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["predicted_consumption_kwh"],
            mode="lines+markers",
            name="Forecast",
        )
    )
    fig_fc.update_layout(
        title="Next 24 Hours Electricity Consumption Forecast",
        xaxis_title="Timestamp",
        yaxis_title="Predicted Consumption (kWh)",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.download_button(
        "Download Forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="electricity_forecast_24h.csv",
        mime="text/csv",
    )