# src/models/geospatial_optimization.py

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def _get_pollution_columns(station_features, pollution_cols=None):
    """
    Try to automatically detect pollution-related columns if none are given.
    """
    if pollution_cols is not None:
        return pollution_cols

    candidates = [
        c
        for c in station_features.columns
        if "pm2.5" in c.lower()
        or "pm25" in c.lower()
        or "pollution" in c.lower()
        or c.lower().startswith("pm")
    ]

    if not candidates:
        raise ValueError(
            "Could not automatically find pollution columns. "
            "Pass pollution_cols=['col1', 'col2', ...] explicitly."
        )
    return candidates


def compute_distance_matrix(station_features, lat_col="latitude", lon_col="longitude"):
    """
    Compute pairwise distance matrix between stations based on latitude/longitude.
    Uses simple Euclidean distance in degree space (good enough inside 1 city).
    """
    coords = station_features[[lat_col, lon_col]].to_numpy()
    dist_matrix = pairwise_distances(coords, metric="euclidean")
    return dist_matrix


def geospatial_diversity_score(selected_indices, dist_matrix):
    """
    Average pairwise distance between selected sensors.
    """
    n = len(selected_indices)
    if n <= 1:
        return 0.0

    sub = dist_matrix[np.ix_(selected_indices, selected_indices)]
    triu = sub[np.triu_indices(n, k=1)]
    return float(np.nanmean(triu)) if triu.size > 0 else 0.0


def pollution_diversity_score(station_features, selected_indices, pollution_cols):
    """
    Measures how diverse pollution statistics are across the selected stations.
    Uses variance between stations as the diversity metric.
    """
    sub = station_features.iloc[selected_indices][pollution_cols]
    return float(sub.var(axis=0, ddof=1).mean())


def geospatial_optimized_selection(
    station_features,
    k,
    lat_col="latitude",
    lon_col="longitude",
    pollution_cols=None,
    alpha=0.5,
    beta=0.5,
):
    """
    Greedy selection that maximizes:
        score = alpha * (spatial diversity) + beta * (pollution diversity)

    Parameters
    ----------
    station_features : pd.DataFrame
        One row per station. Must contain station name and lat/lon columns.
        Either:
          - a 'station' column, OR
          - index named 'station'.
    k : int
        Number of sensors to select.
    """

    if not isinstance(station_features, pd.DataFrame):
        raise ValueError("station_features must be a pandas DataFrame")

    df = station_features.copy()

    # Ensure we have a 'station' column
    if "station" not in df.columns:
        if df.index.name == "station":
            df = df.reset_index()
        else:
            raise ValueError(
                "station_features must have a 'station' column or index named 'station'."
            )

    # Drop stations without valid coordinates
    df = df.dropna(subset=[lat_col, lon_col])

    if df.empty:
        raise ValueError("No stations with valid latitude/longitude found.")

    # Detect pollution columns
    pollution_cols = _get_pollution_columns(df, pollution_cols)

    # If k is larger than remaining valid stations, cap it
    if k > len(df):
        raise ValueError(
            f"k={k} is larger than the number of stations with valid coordinates ({len(df)})."
        )

    # Distance matrix for valid stations
    dist_matrix = compute_distance_matrix(df, lat_col=lat_col, lon_col=lon_col)

    selected_indices = []
    remaining_indices = list(range(len(df)))

    # Greedy forward selection
      # Greedy forward selection
    while len(selected_indices) < k and remaining_indices:
        best_score = -np.inf
        best_idx = None

        for idx in remaining_indices:
            trial_selected = selected_indices + [idx]

            geo_score = geospatial_diversity_score(trial_selected, dist_matrix)
            pol_score = pollution_diversity_score(df, trial_selected, pollution_cols)
            score = alpha * geo_score + beta * pol_score

            # Skip NaN scores
            if np.isnan(score):
                continue

            if score > best_score:
                best_score = score
                best_idx = idx

        # If we couldn't find any valid candidate, stop early
        if best_idx is None:
            break

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    # 🔒 SAFETY: if we stopped early and have fewer than k,
    # fill the remaining slots randomly from what's left
    if len(selected_indices) < k and remaining_indices:
        need = k - len(selected_indices)
        # if fewer remaining than needed, just take all of them
        extra = np.random.choice(
            remaining_indices,
            size=min(need, len(remaining_indices)),
            replace=False,
        )
        selected_indices.extend(list(extra))

    # Final sanity check: cap at k unique indices
    selected_indices = list(dict.fromkeys(selected_indices))[:k]

    selected_stations = df["station"].iloc[selected_indices].tolist()
    return selected_stations
