import random
import pandas as pd

def random_sensor_selection(stations, k=5):
    """Select random stations safely."""
    
    stations = list(stations)

    if len(stations) < k:
        print(f"⚠ Only {len(stations)} stations found. Returning all.")
        return stations

    return random.sample(stations, k)

def top_polluted_stations(df, pollutant="pm2.5", k=5):
    """Select top polluted stations."""
    
    if "station" not in df.columns:
        print("⚠ station column missing!")
        return []

    if pollutant not in df.columns:
        print(f"⚠ pollutant {pollutant} missing!")
        return []

    station_mean = df.groupby("station")[pollutant].mean().sort_values(ascending=False)

    return list(station_mean.head(k).index)
