import sys
import os
from src.viz.map_viz import create_geospatial_map

# make project root importable
sys.path.append(os.path.abspath("."))

from src.data.merge_stations import merge_prsa_data, add_beijing_coordinates
from src.data.load_and_clean import clean_data, save_clean_data
from src.models.geospatial_optimization import geospatial_optimized_selection
from src.models.prediction_model import evaluate_sensor_selection_ml, get_rf_predictions_for_selection

from src.viz.plots import plot_timeseries, plot_distribution, plot_locations, plot_geospatial_selection

from src.features.feature_engineering import (
    add_datetime_features,
    add_rolling_features,
    build_station_feature_table,
    station_feature_matrix,
)

from src.models.greedy_placement import greedy_diverse_selection
from src.models.kmeans_placement import kmeans_selection
from src.baselines.random_baseline import random_sensor_selection, top_polluted_stations

from src.models.geospatial_optimization import geospatial_optimized_selection
from src.models.prediction_model import evaluate_sensor_selection_ml

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

# make sure coordinates are present (they are, from your screenshot)
print("df_clean columns:", df_clean.columns)


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

# -------------------------------------------------
# ADD LAT/LON TO A COPY FOR GEOSPATIAL OPTIMIZATION
# -------------------------------------------------

# 1) Build lookup table: station -> (latitude, longitude) from df_clean
station_geo = (
    df_clean[["station", "latitude", "longitude"]]
    .drop_duplicates("station")
    .set_index("station")
)

# 2) Make sure station_features has station as index
if "station" in station_features.columns:
    station_features_indexed = station_features.set_index("station")
else:
    # if it's already indexed by station, just use it
    station_features_indexed = station_features

# 3) Join coordinates to create station_features_geo
station_features_geo = station_features_indexed.join(station_geo, how="left")

print("station_features_geo columns:", station_features_geo.columns)

# -------------------------------------------------
# FEATURE MATRIX (for greedy / kmeans)
# -------------------------------------------------
X, station_list = station_feature_matrix(station_features)


print("\n=== Optimization: Sensor Placement ===")
k = 5

print(f"\nGreedy diverse selection (k={k}):")
greedy_selected = greedy_diverse_selection(X, station_list, k=k)
print(greedy_selected)

print(f"\nKMeans-based selection (k={k}):")
kmeans_selected = kmeans_selection(X, station_list, k=k)
print(kmeans_selected)

print(f"\nGeospatial (distance + pollution diversity) selection (k={k}):")
geo_selected = geospatial_optimized_selection(
    station_features_geo,      # ✅ USE GEO VERSION
    k=k,
    lat_col="latitude",        # ✅ correct names
    lon_col="longitude",
    alpha=0.5,
    beta=0.5,
)
print(geo_selected)

# -------------------------------------------------
# REALISTIC INTERACTIVE MAP
# -------------------------------------------------
print("\nCreating interactive geospatial map with OpenStreetMap background...")
create_geospatial_map(
    df_clean,
    geo_selected,
    station_col="station",
    lat_col="latitude",
    lon_col="longitude",
    map_filename="geospatial_selection_k5.html",
)

print("\nPlotting geospatial sensor placement for k=5...")
plot_geospatial_selection(
    df_clean,
    geo_selected,
    station_col="station",
    lat_col="latitude",
    lon_col="longitude",
    title="Geospatial Sensor Placement (k=5)",
)
plt.show()

# -------------------------------------------------
# EVALUATION: ML RMSE vs Number of Sensors
# -------------------------------------------------
print("\nEvaluating optimization performance with Random Forest model...")

k_values = range(2, 11)
rmse_greedy = []
rmse_kmeans = []
rmse_geo = []
rmse_random = []

for k in k_values:
    print(f"  Evaluating k = {k} ...")

    rmse_greedy.append(
        evaluate_sensor_selection_ml(
            df_fe,
            station_list,
            k,
            lambda k: greedy_diverse_selection(X, station_list, k),
        )
    )

    rmse_kmeans.append(
        evaluate_sensor_selection_ml(
            df_fe,
            station_list,
            k,
            lambda k: kmeans_selection(X, station_list, k),
        )
    )

    rmse_geo.append(
        evaluate_sensor_selection_ml(
            df_fe,
            station_list,
            k,
            lambda k: geospatial_optimized_selection(
                station_features_geo,     # ✅ USE GEO VERSION
                k=k,
                lat_col="latitude",
                lon_col="longitude",
                alpha=0.5,
                beta=0.5,
            ),
        )
    )

    rmse_random.append(
        evaluate_sensor_selection_ml(
            df_fe,
            station_list,
            k,
            lambda k: random_sensor_selection(station_list, k),
        )
    )

print("\nPlotting Random Forest prediction vs actual for geospatial selection (k=5)...")

time_test, y_test, y_pred = get_rf_predictions_for_selection(
    df_fe,
    geo_selected,         # use the geospatial sensors for k=5
    pollutant="pm2.5",
    station_col="station",
    datetime_col="datetime",
    test_size=0.2,
)

plt.figure(figsize=(10, 4))
plt.plot(time_test, y_test, label="Actual city-wide PM2.5")
plt.plot(time_test, y_pred, label="Predicted PM2.5 (Geo + RF)")
plt.xlabel("Time")
plt.ylabel("PM2.5")
plt.title("Random Forest Prediction Using Geospatial Sensor Set (k=5)")
plt.legend()
plt.tight_layout()
plt.show()


# RMSE Plot (ML-based)
plt.figure(figsize=(8, 5))
plt.plot(k_values, rmse_greedy, marker="o", label="Greedy (features only)")
plt.plot(k_values, rmse_kmeans, marker="o", label="KMeans (clustering)")
plt.plot(k_values, rmse_geo, marker="o", label="Geospatial (dist + pollution)")
plt.plot(k_values, rmse_random, marker="o", label="Random Baseline")
plt.xlabel("Number of Sensors (k)")
plt.ylabel("RMSE (Random Forest)")
plt.title("Sensor Selection Performance with ML Prediction")
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------------------------------
# BASELINES FOR REFERENCE
# -------------------------------------------------
print("\nRandom baseline sensors:")
print(random_sensor_selection(station_list, k=5))

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
