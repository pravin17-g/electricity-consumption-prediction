"""
Phase 1 - Synthetic Dataset Generation
Project: Electricity Consumption Prediction Using LSTM

This script creates one full year of hourly electricity consumption data
(365 days x 24 hours = 8,760 records) using a formula-based approach.

Consumption is built from these parts:
    base level + daily pattern (morning/evening peaks) + weekend changes
    + seasonal effect + temperature effect + small random noise

Run from the project root folder:
    python src/generate_dataset.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# SETTINGS (change these numbers to tune the dataset)
# ----------------------------------------------------------------------
RANDOM_SEED = 42                      # fixed seed -> same dataset every run
START_DATE = "2025-01-01 00:00:00"
NUM_DAYS = 365
TOTAL_RECORDS = NUM_DAYS * 24         # 8,760 hourly records

BASE_CONSUMPTION = 1.6                # kWh, the "always on" level (fridge, standby...)
NOISE_LEVEL = 0.15                    # standard deviation of consumption noise (kWh)

MEAN_TEMPERATURE = 27.0               # yearly average temperature (Celsius)
SEASONAL_TEMP_SWING = 3.0             # summer/winter difference around the mean
DAILY_TEMP_SWING = 5.0                # day/night difference around the mean
HOTTEST_DAY_OF_YEAR = 120             # around the end of April (hottest time)
MIN_TEMPERATURE = 18.0
MAX_TEMPERATURE = 38.0

# Paths (pathlib makes this work on Windows, Mac and Linux)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PATH = RAW_DATA_DIR / "electricity_consumption.csv"
PLOT_PATH = RAW_DATA_DIR / "initial_consumption_plot.png"

EXPECTED_COLUMNS = [
    "timestamp",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "temperature",
    "electricity_consumption",
]


# ----------------------------------------------------------------------
# STEP 1: TIME FEATURES
# ----------------------------------------------------------------------
def create_time_features():
    """Create the hourly timestamps and the time-related columns."""
    timestamps = pd.date_range(start=START_DATE, periods=TOTAL_RECORDS, freq="h")

    df = pd.DataFrame({"timestamp": timestamps})
    df["hour"] = df["timestamp"].dt.hour                 # 0-23
    df["day_of_week"] = df["timestamp"].dt.dayofweek     # 0 = Monday ... 6 = Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month               # 1-12
    return df


# ----------------------------------------------------------------------
# HELPER: smooth "bump" used to build the daily pattern
# ----------------------------------------------------------------------
def gaussian_peak(hour, center, width, height):
    """
    A smooth bell-shaped bump centred on `center` o'clock.

    The distance is measured around the 24-hour clock, so the curve
    flows smoothly from 23:00 into 00:00 (no sudden jump at midnight).
    """
    distance = np.abs(hour - center)
    distance = np.minimum(distance, 24 - distance)       # wrap around midnight
    return height * np.exp(-(distance ** 2) / (2 * width ** 2))


# ----------------------------------------------------------------------
# STEP 2: TEMPERATURE
# ----------------------------------------------------------------------
def generate_temperature(df):
    """
    Temperature (Celsius) with:
      - a seasonal pattern (hottest around April/May, coolest around Dec/Jan)
      - a daily pattern (warmest at 14:00, coolest at night)
      - a random "weather" offset for each day + a little hourly noise
    """
    hour = df["hour"].to_numpy()
    day_of_year = df["timestamp"].dt.dayofyear.to_numpy()

    seasonal = SEASONAL_TEMP_SWING * np.cos(
        2 * np.pi * (day_of_year - HOTTEST_DAY_OF_YEAR) / 365
    )
    daily = DAILY_TEMP_SWING * np.cos(2 * np.pi * (hour - 14) / 24)

    # One random value per day (hot day / cooler day), repeated for its 24 hours
    daily_weather = np.repeat(np.random.normal(0, 1.2, NUM_DAYS), 24)
    hourly_noise = np.random.normal(0, 0.4, len(df))

    temperature = MEAN_TEMPERATURE + seasonal + daily + daily_weather + hourly_noise
    temperature = np.clip(temperature, MIN_TEMPERATURE, MAX_TEMPERATURE)
    return temperature.round(2)


# ----------------------------------------------------------------------
# STEP 3: ELECTRICITY CONSUMPTION
# ----------------------------------------------------------------------
def weekday_pattern(hour):
    """Extra kWh (above the base level) for each hour on a weekday."""
    night_dip = gaussian_peak(hour, center=3, width=2.5, height=-0.35)    # sleeping hours
    morning_peak = gaussian_peak(hour, center=8, width=1.3, height=1.2)   # 07:00-09:00
    midday = gaussian_peak(hour, center=13, width=3.0, height=0.35)       # moderate daytime
    evening_peak = gaussian_peak(hour, center=20, width=2.0, height=2.0)  # 18:00-22:00
    return night_dip + morning_peak + midday + evening_peak


def weekend_pattern(hour):
    """
    Extra kWh for each hour on a weekend. Different shape from weekdays:
    later and smaller morning peak, higher daytime use, slightly later
    and stronger evening peak.
    """
    night_dip = gaussian_peak(hour, center=4, width=2.8, height=-0.35)    # sleep in later
    morning_peak = gaussian_peak(hour, center=9.5, width=1.6, height=0.9) # later, softer
    midday = gaussian_peak(hour, center=14, width=3.5, height=0.8)        # people are home
    evening_peak = gaussian_peak(hour, center=20.5, width=2.0, height=2.2)
    return night_dip + morning_peak + midday + evening_peak


def generate_consumption(df):
    """Combine all components into electricity consumption (kWh)."""
    hour = df["hour"].to_numpy()
    is_weekend = df["is_weekend"].to_numpy()
    temperature = df["temperature"].to_numpy()
    day_of_year = df["timestamp"].dt.dayofyear.to_numpy()

    # 1) Daily pattern - pick the weekday or the weekend shape
    daily_pattern = np.where(is_weekend == 1, weekend_pattern(hour), weekday_pattern(hour))

    # 2) Seasonal effect - smooth factor, about +10% in the hottest season
    #    and -10% in the coolest season
    seasonal_factor = 1 + 0.10 * np.cos(
        2 * np.pi * (day_of_year - HOTTEST_DAY_OF_YEAR) / 365
    )

    # 3) Temperature effect - usage rises with temperature (fans / air conditioner),
    #    and rises faster once it is hotter than 30 C
    temperature_effect = 0.05 * (temperature - 27) + 0.08 * np.maximum(temperature - 30, 0)

    # 4) Random noise
    noise = np.random.normal(0, NOISE_LEVEL, len(df))

    consumption = (BASE_CONSUMPTION + daily_pattern) * seasonal_factor + temperature_effect + noise

    # Electricity use can never be negative - keep a small positive minimum
    consumption = np.maximum(consumption, 0.2)
    return consumption.round(3)


# ----------------------------------------------------------------------
# STEP 4: VALIDATION
# ----------------------------------------------------------------------
def validate_dataset(df):
    """Run automatic data-quality checks. Raises an error if any check fails."""
    results = []  # list of (check name, passed?)

    # --- Basic structure ---
    results.append(("Exactly 8,760 records", len(df) == TOTAL_RECORDS))
    results.append(("Correct columns", list(df.columns) == EXPECTED_COLUMNS))
    results.append(("No missing values", int(df.isnull().sum().sum()) == 0))
    results.append(("No duplicate timestamps", int(df["timestamp"].duplicated().sum()) == 0))
    results.append(("Timestamps in chronological order", df["timestamp"].is_monotonic_increasing))
    results.append((
        "Exactly 1 hour between records",
        bool((df["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()),
    ))

    # --- Value ranges ---
    results.append(("Consumption always positive", bool((df["electricity_consumption"] > 0).all())))
    results.append((
        "Temperature within 18-38 C",
        bool(df["temperature"].between(MIN_TEMPERATURE, MAX_TEMPERATURE).all()),
    ))

    # --- Daily pattern (weekdays only) ---
    weekday_avg = df[df["is_weekend"] == 0].groupby("hour")["electricity_consumption"].mean()
    morning_peak_hour = weekday_avg.loc[6:10].idxmax()
    evening_peak_hour = weekday_avg.idxmax()
    night_low_hour = weekday_avg.idxmin()
    results.append(("Morning peak between 07:00 and 09:00", 7 <= morning_peak_hour <= 9))
    results.append(("Evening peak between 18:00 and 22:00", 18 <= evening_peak_hour <= 22))
    results.append(("Lowest usage during 00:00-05:00", 0 <= night_low_hour <= 5))
    results.append((
        "Evening peak larger than morning peak",
        weekday_avg.loc[evening_peak_hour] > weekday_avg.loc[morning_peak_hour],
    ))

    # --- Weekly pattern ---
    weekday_midday = df[(df["is_weekend"] == 0) & df["hour"].between(10, 16)]["electricity_consumption"].mean()
    weekend_midday = df[(df["is_weekend"] == 1) & df["hour"].between(10, 16)]["electricity_consumption"].mean()
    results.append(("Weekend daytime usage higher than weekday", weekend_midday > weekday_midday))

    # --- Seasonal pattern ---
    monthly_temperature = df.groupby("month")["temperature"].mean()
    monthly_consumption = df.groupby("month")["electricity_consumption"].mean()
    hottest_month = monthly_temperature.idxmax()
    coolest_month = monthly_temperature.idxmin()
    results.append((
        "Hottest month uses more electricity than coolest month",
        monthly_consumption[hottest_month] > monthly_consumption[coolest_month],
    ))

    # --- Temperature influence ---
    # Compare each hour with the average for the same hour of day, so the daily
    # pattern does not hide the temperature effect.
    temp_difference = df["temperature"] - df.groupby("hour")["temperature"].transform("mean")
    consumption_difference = (
        df["electricity_consumption"]
        - df.groupby("hour")["electricity_consumption"].transform("mean")
    )
    temperature_correlation = temp_difference.corr(consumption_difference)
    results.append(("Higher temperature -> higher consumption", temperature_correlation > 0.3))

    # --- Print the report ---
    print("Data quality checks:")
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"  (temperature vs consumption correlation: {temperature_correlation:.2f})")

    if not all(passed for _, passed in results):
        raise ValueError("Dataset validation failed - see the FAIL lines above.")
    print()


# ----------------------------------------------------------------------
# STEP 5: SAVE, PLOT, SUMMARY
# ----------------------------------------------------------------------
def save_dataset(df):
    """Save the dataset as a CSV file."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)


def plot_first_week(df):
    """Plot the first 7 days (168 hours) of electricity consumption."""
    first_week = df.iloc[: 7 * 24]

    plt.figure(figsize=(12, 5))
    plt.plot(first_week["timestamp"], first_week["electricity_consumption"], color="tab:blue")
    plt.xlabel("Time")
    plt.ylabel("Electricity Consumption (kWh)")
    plt.title("Synthetic Electricity Consumption - First 7 Days")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def print_summary(df):
    """Print useful information about the generated dataset."""
    consumption = df["electricity_consumption"]
    print("Dataset generated successfully!\n")
    print(f"Number of records: {len(df)}")
    print(f"Number of columns: {df.shape[1]}\n")
    print("Date range:")
    print(df["timestamp"].min())
    print("to")
    print(df["timestamp"].max())
    print()
    print(f"Missing values: {int(df.isnull().sum().sum())}")
    print(f"Duplicate timestamps: {int(df['timestamp'].duplicated().sum())}\n")
    print("Electricity Consumption Statistics:")
    print(f"Mean: {consumption.mean():.3f} kWh")
    print(f"Minimum: {consumption.min():.3f} kWh")
    print(f"Maximum: {consumption.max():.3f} kWh")
    print(f"Standard deviation: {consumption.std():.3f} kWh\n")
    print("Temperature Statistics:")
    print(f"Mean: {df['temperature'].mean():.2f} C")
    print(f"Minimum: {df['temperature'].min():.2f} C")
    print(f"Maximum: {df['temperature'].max():.2f} C\n")
    print("First 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\nCSV saved to:  {CSV_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    np.random.seed(RANDOM_SEED)  # makes the "random" numbers repeatable

    df = create_time_features()
    df["temperature"] = generate_temperature(df)
    df["electricity_consumption"] = generate_consumption(df)

    validate_dataset(df)
    save_dataset(df)
    plot_first_week(df)
    print_summary(df)


if __name__ == "__main__":
    main()