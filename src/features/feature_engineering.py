import pandas as pd
import numpy as np


def detect_pollutants(df):
    possible = ["pm2.5", "pm10", "no2", "so2", "co", "o3"]
    return [c for c in possible if c in df.columns]


def add_datetime_features(df):
    """Add hour, dayofweek, month, is_weekend if datetime exists."""
    if "datetime" not in df.columns:
        print("⚠ 'datetime' column not found. Skipping datetime features.")
        return df

    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def add_rolling_features(df, pollutant="pm2.5", windows=(3, 6, 24)):
    """Rolling mean features for a pollutant (per station)."""
    if pollutant not in df.columns:
        print(f"⚠ '{pollutant}' not found. Skipping rolling features.")
        return df

    df = df.copy()

    sort_cols = [c for c in ["station", "datetime"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    for w in windows:
        col_name = f"{pollutant}_rollmean_{w}"
        if "station" in df.columns:
            df[col_name] = (
                df.groupby("station")[pollutant]
                .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
            )
        else:
            df[col_name] = df[pollutant].rolling(window=w, min_periods=1).mean()

    return df


def build_station_feature_table(df):
    """Per-station aggregated statistics of pollutants."""
    if "station" not in df.columns:
        print("⚠ 'station' column missing. Cannot build station-level features.")
        return None

    df = df.copy()
    pollutants = detect_pollutants(df)
    if not pollutants:
        print("⚠ No pollutant columns found. Station feature table will be empty.")
        return None

    agg_dict = {}
    for p in pollutants:
        agg_dict[p] = ["mean", "std", "min", "max"]

    grouped = df.groupby("station").agg(agg_dict)
    grouped.columns = [f"{p}_{stat}" for p, stat in grouped.columns]
    grouped["n_samples"] = df.groupby("station")[pollutants[0]].count()
    grouped = grouped.reset_index()

    return grouped


def station_feature_matrix(station_features, drop_cols=("station",)):
    """Convert station feature table → (X matrix, station list)."""
    if station_features is None or station_features.empty:
        print("⚠ Empty station feature table.")
        return None, None

    sf = station_features.copy()
    stations = sf["station"].tolist()
    X = sf.drop(columns=[c for c in drop_cols if c in sf.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    return X.values, stations
