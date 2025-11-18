import sys
import os
sys.path.append(os.path.abspath("."))

from src.data.load_and_clean import load_raw_data, clean_data, save_clean_data
from src.baselines.random_baseline import random_sensor_selection, top_polluted_stations
from src.viz.plots import plot_timeseries, plot_distribution, plot_locations

# paths
raw_path = r"C:\Users\Meghana\OneDrive\Desktop\meghna\.vs studio\air-quality-sensor-placement\data\raw\PRSA_Data_Aotizhongxin_20130301-20170228.csv.zip"
clean_path = "data/processed/beijing_cleaned.csv"

print("Loading raw data...")
df_raw = load_raw_data(raw_path)

print("Cleaning data...")
df_clean = clean_data(df_raw)

print("Saving cleaned data...")
save_clean_data(df_clean, clean_path)

# -------------------------------------------------
# BASELINES
# -------------------------------------------------
if "station" not in df_clean.columns:
    print("⚠ station column missing in dataset. Cannot run baselines.")
else:
    stations = df_clean["station"].unique()
    
    print("\nRandom baseline sensors:")
    print(random_sensor_selection(stations, k=5))

    print("\nTop polluted stations:")
    print(top_polluted_stations(df_clean, pollutant="pm2.5", k=5))

# -------------------------------------------------
# PLOTS
# -------------------------------------------------
print("\nGenerating time-series plot...")
plot_timeseries(df_clean, pollutant="pm2.5")

print("\nGenerating distribution plot...")
plot_distribution(df_clean, pollutant="pm2.5")

print("\nGenerating location plot...")
plot_locations(df_clean)

print("\n🎉 Meghana's module completed successfully!")
