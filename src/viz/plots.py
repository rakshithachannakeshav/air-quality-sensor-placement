import matplotlib.pyplot as plt

def plot_timeseries(df, pollutant="pm2.5"):
    if "datetime" not in df.columns or pollutant not in df.columns:
        print("⚠ datetime or pollutant missing")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(df["datetime"], df[pollutant])
    plt.title(f"{pollutant.upper()} Over Time")
    plt.xlabel("Time")
    plt.ylabel(pollutant.upper())
    plt.tight_layout()
    plt.show()

def plot_distribution(df, pollutant="pm2.5"):
    if pollutant not in df.columns:
        print(f"⚠ {pollutant} missing")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(df[pollutant].dropna(), bins=40)
    plt.title(f"{pollutant.upper()} Distribution")
    plt.xlabel(pollutant.upper())
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_locations(df):
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("⚠ latitude/longitude missing")
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(df["longitude"], df["latitude"], alpha=0.5)
    plt.title("Sensor Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt


def plot_geospatial_selection(
    df,
    selected_stations,
    station_col="station",
    lat_col="latitude",
    lon_col="longitude",
    title="Geospatial Sensor Placement",
):
    """
    Plot all station locations and highlight the selected sensors.
    """
    # unique station locations
    stations = (
        df[[station_col, lat_col, lon_col]]
        .drop_duplicates(station_col)
        .dropna(subset=[lat_col, lon_col])
    )

    selected_mask = stations[station_col].isin(selected_stations)
    selected = stations[selected_mask]
    others = stations[~selected_mask]

    plt.figure(figsize=(6, 6))

    # all stations (light markers)
    plt.scatter(
        others[lon_col],
        others[lat_col],
        s=30,
        alpha=0.6,
        label="Other stations",
    )

    # selected stations (bigger markers)
    plt.scatter(
        selected[lon_col],
        selected[lat_col],
        s=80,
        marker="^",
        label="Selected sensors",
    )

    for _, row in selected.iterrows():
        plt.text(
            row[lon_col],
            row[lat_col],
            str(row[station_col]),
            fontsize=8,
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
