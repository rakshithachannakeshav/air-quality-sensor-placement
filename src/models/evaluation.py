import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_sensor_selection(df, station_list, k, selector_fn):
    selected = selector_fn(k)
    df_sel = df[df["station"].isin(selected)]

    pred = df_sel.groupby("datetime")["pm2.5"].mean()
    truth = df.groupby("datetime")["pm2.5"].mean()

    pred, truth = pred.align(truth, join="inner")
    rmse = np.sqrt(mean_squared_error(truth, pred))
    return rmse
