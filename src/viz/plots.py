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
