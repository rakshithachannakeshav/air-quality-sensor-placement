# src/models/prediction_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def build_dataset_for_selected_sensors(
    df,
    selected_stations,
    pollutant="pm2.5",
    station_col="station",
    datetime_col="datetime",
):
    """
    Build a supervised learning dataset:

    X = PM2.5 at selected stations (wide pivot: one column per station)
    y = city-wide average PM2.5 at each timestamp
    """
    # features: only selected stations
    df_sel = df[df[station_col].isin(selected_stations)].copy()

    X_wide = df_sel.pivot_table(
        index=datetime_col,
        columns=station_col,
        values=pollutant,
    )

    # target: city-wide average over *all* stations
    city_mean = (
        df.groupby(datetime_col)[pollutant]
        .mean()
        .reindex(X_wide.index)
    )

    # align and drop missing timestamps
    dataset = pd.concat([X_wide, city_mean.rename("target")], axis=1).dropna()

    X = dataset.drop(columns=["target"]).to_numpy()
    y = dataset["target"].to_numpy()

    return X, y


def evaluate_sensor_selection_ml(
    df,
    station_list,
    k,
    selection_fn,
    pollutant="pm2.5",
    station_col="station",
    datetime_col="datetime",
    test_size=0.2,
    random_state=42,
):
    """
    Evaluate a sensor placement strategy using a Random Forest model.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature/enriched dataframe (df_fe).
    station_list : list[str]
        List of all station names (unused inside, kept for consistency).
    k : int
        Number of sensors.
    selection_fn : callable
        Function that receives k and returns a list of selected station names.
    pollutant : str
        Pollutant column to predict.

    Returns
    -------
    rmse : float
        RMSE of Random Forest prediction of city-wide average PM2.5.
    """
    selected = selection_fn(k)
    if len(selected) != k:
        raise ValueError("selection_fn must return exactly k stations")

    X, y = build_dataset_for_selected_sensors(
        df,
        selected,
        pollutant=pollutant,
        station_col=station_col,
        datetime_col=datetime_col,
    )

    if X.shape[0] < 10:
        raise ValueError("Not enough samples to train/test the model.")

    # time-series style split: respect chronological order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return rmse

def get_rf_predictions_for_selection(
    df,
    selected_stations,
    pollutant="pm2.5",
    station_col="station",
    datetime_col="datetime",
    test_size=0.2,
    random_state=42,
):
    """
    Train a Random Forest for a given set of selected stations and
    return time index, true values, and predictions (for plotting).

    Returns
    -------
    time_test : np.ndarray
        Timestamps for the test set.
    y_test : np.ndarray
        True city-wide average PM2.5.
    y_pred : np.ndarray
        Predicted PM2.5 by the Random Forest.
    """
    # --- build dataset (same logic as build_dataset_for_selected_sensors) ---
    df_sel = df[df[station_col].isin(selected_stations)].copy()

    X_wide = df_sel.pivot_table(
        index=datetime_col,
        columns=station_col,
        values=pollutant,
    )

    city_mean = (
        df.groupby(datetime_col)[pollutant]
        .mean()
        .reindex(X_wide.index)
    )

    dataset = pd.concat([X_wide, city_mean.rename("target")], axis=1).dropna()

    X = dataset.drop(columns=["target"]).to_numpy()
    y = dataset["target"].to_numpy()
    time_index = dataset.index.to_numpy()

    if X.shape[0] < 10:
        raise ValueError("Not enough samples to train/test the model.")

    # --- manual chronological split ---
    n = len(X)
    split_idx = int((1.0 - test_size) * n)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    time_train, time_test = time_index[:split_idx], time_index[split_idx:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return time_test, y_test, y_pred
