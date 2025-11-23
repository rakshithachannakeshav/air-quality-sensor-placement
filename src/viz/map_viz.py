# src/viz/map_viz.py

import os
import folium
import pandas as pd


def create_geospatial_map(
    df,
    selected_stations,
    station_col="station",
    lat_col="latitude",
    lon_col="longitude",
    map_filename="geospatial_selection.html",
    zoom_start=10,
):
    """
    Create an interactive map (Leaflet/OSM) showing all stations and 
    highlighting the selected sensors.

    Saves an HTML file that you can open in your browser.
    """

    # unique station locations
    stations = (
        df[[station_col, lat_col, lon_col]]
        .drop_duplicates(station_col)
        .dropna(subset=[lat_col, lon_col])
    )

    # center of the map = mean lat/lon
    center_lat = stations[lat_col].mean()
    center_lon = stations[lon_col].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Add all stations (base layer)
    for _, row in stations.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4,
            popup=f"Station: {row[station_col]}",
            fill=True,
            fill_opacity=0.5,
            opacity=0.7,
        ).add_to(m)

    # Add selected stations (highlighted)
    selected = stations[stations[station_col].isin(selected_stations)]

    for _, row in selected.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=7,
            popup=f"SELECTED: {row[station_col]}",
            color="red",
            fill=True,
            fill_opacity=0.9,
            opacity=1.0,
        ).add_to(m)

    # ensure output folder exists
    os.makedirs("maps", exist_ok=True)
    output_path = os.path.join("maps", map_filename)
    m.save(output_path)

    print(f"✅ Interactive map saved to: {output_path}")
    print("👉 Open this file in your browser to see the real map view.")
