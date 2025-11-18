import sys
import os

# make project root importable
sys.path.append(os.path.abspath("."))

from src.data.merge_stations import merge_prsa_data, add_beijing_coordinates
from src.data.load_and_clean import clean_data, save_clean_data

from src.features.feature_engineering import (
    add_datetime_features,
    add_rolling_features,
    build_station_feature_table,
    station_feature_matrix,
)

from src.models.greedy_placement import greedy_diverse_selection
from src.models.kmeans_placement import kmeans_selection
from src.baselines.random_baseline import random_sensor_selection, top_polluted_stations

from src.models.evaluation import evaluate_sensor_selection  # <-- NEW IMPORT

from src.viz.plots import plot_timeseries, plot_distribution, plot_locations

import matplotlib.pyplot as plt


# -------------------------------------------------
# PATHS
# -------------------------------------------------
raw_folder = "data/raw"
clean_path = "data/processed/prsa_beijing_cleaned.csv"


# -------------------------------------------------
# LOAD + MERGE RAW DATA
# -------------------------------------------------
print("Merging multi-station PRSA dataset from:", raw_folder)
df_raw = merge_prsa_data(raw_folder)
df_raw = add_beijing_coordinates(df_raw)

print("Raw merged shape:", df_raw.shape)
print("Stations found:", df_raw["station"].unique())


# -------------------------------------------------
# CLEANING
# -------------------------------------------------
print("\nCleaning data...")
df_clean = clean_data(df_raw)

print("Saving cleaned data...")
save_clean_data(df_clean, clean_path)


# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
print("\nAdding datetime features...")
df_fe = add_datetime_features(df_clean)

print("Adding rolling features for PM2.5...")
df_fe = add_rolling_features(df_fe, pollutant="pm2.5", windows=(3, 6, 24))

print("Building station-level feature table...")
station_features = build_station_feature_table(df_fe)

if station_features is None:
    print("⚠ Could not build station features. Skipping optimization steps.")
    exit()

X, station_list = station_feature_matrix(station_features)

print("\n=== Optimization: Sensor Placement ===")
k = 5

print(f"\nGreedy diverse selection (k={k}):")
greedy_selected = greedy_diverse_selection(X, station_list, k=k)
print(greedy_selected)

print(f"\nKMeans-based selection (k={k}):")
kmeans_selected = kmeans_selection(X, station_list, k=k)
print(kmeans_selected)


# -------------------------------------------------
# EVALUATION: RMSE vs Number of Sensors
# -------------------------------------------------
print("\nEvaluating optimization performance...")

k_values = range(2, 11)
rmse_greedy = []
rmse_kmeans = []
rmse_random = []

for k in k_values:
    rmse_greedy.append(evaluate_sensor_selection(
        df_fe, station_list, k,
        lambda k: greedy_diverse_selection(X, station_list, k)
    ))

    rmse_kmeans.append(evaluate_sensor_selection(
        df_fe, station_list, k,
        lambda k: kmeans_selection(X, station_list, k)
    ))

    rmse_random.append(evaluate_sensor_selection(
        df_fe, station_list, k,
        lambda k: random_sensor_selection(station_list, k)
    ))

# RMSE Plot
plt.figure(figsize=(8, 5))
plt.plot(k_values, rmse_greedy, marker='o', label="Greedy")
plt.plot(k_values, rmse_kmeans, marker='o', label="KMeans")
plt.plot(k_values, rmse_random, marker='o', label="Random Baseline")
plt.xlabel("Number of Sensors (k)")
plt.ylabel("RMSE")
plt.title("Sensor Selection Performance")
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------------------------------
# BASELINES FOR REFERENCE
# -------------------------------------------------
print("\nRandom baseline sensors:")
print(random_sensor_selection(station_list, k=5))
print(top_polluted_stations(df_clean, pollutant="pm2.5", k=5))


print("\nTop polluted stations:")
print(top_polluted_stations(df_clean, pollutant="pm2.5", k=5))


# -------------------------------------------------
# PLOTS
# -------------------------------------------------
print("\nGenerating time-series plot (city average)...")
plot_timeseries(df_clean, pollutant="pm2.5")

print("\nGenerating distribution plot...")
plot_distribution(df_clean, pollutant="pm2.5")

print("\nGenerating location plot...")
plot_locations(df_clean)

