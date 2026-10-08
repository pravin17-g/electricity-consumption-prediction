# Electricity Consumption Prediction Using LSTM

## Phase 1: Synthetic Dataset Generation

### Objective

This phase creates a **synthetic hourly electricity consumption dataset** that behaves like a real household/building. The dataset will be used in later phases for **time-series forecasting with an LSTM (Long Short-Term Memory)** deep learning model.

No model is trained in this phase. The goal is only to build realistic data and prepare the project structure.

### Dataset

* **1 full year** of data (2025-01-01 00:00 to 2025-12-31 23:00)
* **8,760 hourly records** (365 days x 24 hours)
* 7 columns:

| Column | Description |
| --- | --- |
| `timestamp` | Hourly date and time |
| `hour` | Hour of the day (0-23) |
| `day_of_week` | 0 = Monday ... 6 = Sunday |
| `is_weekend` | 0 = weekday, 1 = weekend |
| `month` | Month number (1-12) |
| `temperature` | Synthetic temperature in Celsius (18-38) |
| `electricity_consumption` | Electricity used in kWh (target variable) |

### Synthetic Data Generation

Consumption is not random. It is built from a formula with these parts:

* **Base consumption** of about 1.6 kWh (appliances that are always on)
* **Daily pattern**: low at night (00:00-05:00), morning peak (07:00-09:00), moderate daytime use, strong evening peak (18:00-22:00), smooth drop after 23:00 (built with smooth Gaussian curves, so there are no sudden jumps)
* **Weekend variation**: a different daily shape (later and softer morning peak, higher daytime usage, slightly later evening peak)
* **Seasonal variation**: smooth yearly cycle, with higher usage in the hotter months and lower usage in the cooler months
* **Temperature influence**: higher temperature increases consumption (cooling load), and the effect grows faster above 30 C
* **Random noise**: small Gaussian noise so the data does not look artificially perfect

Temperature itself also follows a seasonal pattern, a daily pattern (warmest around 14:00), and small random weather variation.

A fixed random seed (`42`) makes the dataset **reproducible**.

The script also runs automatic data quality checks (record count, missing values, duplicate timestamps, chronological order, positive values, temperature range, daily/weekly/seasonal patterns, temperature influence) and stops with an error if any check fails.

### Project Structure

```text
electricity-consumption-prediction/
|
|-- data/
|   |-- raw/
|   |   |-- electricity_consumption.csv
|   |   `-- initial_consumption_plot.png
|   |
|   `-- processed/
|       |-- X_train.npy
|       |-- y_train.npy
|       |-- X_val.npy
|       |-- y_val.npy
|       |-- X_test.npy
|       `-- y_test.npy
|
|-- models/
|   `-- scaler.pkl
|
|-- src/
|   |-- generate_dataset.py
|   `-- preprocess_data.py
|
|-- requirements.txt
`-- README.md
```

### How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run the generator from the project root folder:

```bash
python src/generate_dataset.py
```

### Output

After running the script:

* The dataset is saved to `data/raw/electricity_consumption.csv`
* A graph of the first 7 days is saved to `data/raw/initial_consumption_plot.png`
* A summary (record count, date range, statistics, first 5 rows) is printed in the terminal

---

## Phase 2: Data Preprocessing + Sequence Creation

### Objective

Prepare the Phase 1 dataset for the LSTM. No model is trained in this phase.

### What the script does

1. Loads `data/raw/electricity_consumption.csv`, converts `timestamp` to datetime and sorts it chronologically
2. Checks missing values and duplicate timestamps
3. Splits the data **chronologically**: 70% training, 15% validation, 15% testing (no shuffling)
4. Fits a `MinMaxScaler` on the **training data only** (prevents data leakage) and uses it to scale validation and test data
5. Creates sequences with `create_sequences()`: the previous **24 hours** are the input, the **next hour's electricity consumption** is the target
6. Saves the arrays to `data/processed/` and the fitted scaler to `models/scaler.pkl`

### Input features (in this order)

`hour`, `day_of_week`, `is_weekend`, `month`, `temperature`, `electricity_consumption`

### Output shapes

| Array | Shape | Meaning |
| --- | --- | --- |
| `X_train` | (6108, 24, 6) | 6,108 samples, 24 hours each, 6 features |
| `y_train` | (6108,) | next-hour consumption (scaled 0-1) |
| `X_val` / `y_val` | (1314, 24, 6) / (1314,) | validation set |
| `X_test` / `y_test` | (1314, 24, 6) / (1314,) | test set |

### Converting predictions back to kWh

```python
import joblib
from src.preprocess_data import inverse_transform_target

scaler = joblib.load("models/scaler.pkl")
predictions_kwh = inverse_transform_target(scaled_predictions, scaler)
```

### How to Run

```bash
pip install -r requirements.txt
python src/preprocess_data.py
```

(Run `python src/generate_dataset.py` first if `data/raw/electricity_consumption.csv` does not exist yet.)