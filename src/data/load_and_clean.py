import pandas as pd
import numpy as np
import os

def load_raw_data(path):
    """Load raw CSV file"""
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """Cleaning steps"""

    # clean column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # convert datetime if present
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # unify station column name
    possible_station_cols = ["station", "site", "station_id", "area", "location"]
    for col in possible_station_cols:
        if col in df.columns:
            df.rename(columns={col: "station"}, inplace=True)
            break

    # drop duplicates
    df = df.drop_duplicates()

    # pollutant list
    pollutants = [c for c in ["pm2.5", "pm10", "no2", "so2", "co", "o3"] if c in df.columns]

    if len(pollutants) == 0:
        print("⚠ No pollutant columns found!")
        return df

    # drop rows where all pollutants missing
    df = df.dropna(subset=pollutants, how="all")

    # interpolate missing pollutant values
    for col in pollutants:
        df[col] = df[col].interpolate(method='linear')

    return df

def save_clean_data(df, save_path):
    df.to_csv(save_path, index=False)
    print(f"Cleaned dataset saved at: {save_path}")
