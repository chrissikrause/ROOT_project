import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df_path = '/dss/dsshome1/01/di97rov/ROOT_project/scaling/di_ts_6steps_1.feather'

df = pd.read_feather(df_path)
print(df.head())
print(f"Shape: {df.shape}")

# Eindeutige Pixel IDs
unique_pixels = df["pixel_id"].unique()
unique_pixels_count = df["pixel_id"].nunique()
print(f"\nAnzahl eindeutiger Pixel: {unique_pixels_count}")

unique_time = df["time"].nunique()
print(f"\nAnzahl eindeutiger Zeitschritte DI: {unique_time}")
print(f"\nEindeutigeZeitschritte DI: {df["time"].unique()}")


unique_timestep = df["timestep"].nunique()
print(f"\nAnzahl eindeutiger Zeitschritte FCC: {unique_timestep}")


# Erstes Pixel auswählen
first_pixel_id = unique_pixels[5]
print(f"\nErstes Pixel ID: {first_pixel_id}")

# Zeitreihe für dieses Pixel extrahieren
pixel_ts = df[df["pixel_id"] == first_pixel_id].sort_values("time")
print(f"\nZeitreihe für Pixel {first_pixel_id}:")
print(pixel_ts[["time", "di", "timestep"]])

# Optional: Plot der Zeitreihe
plt.figure(figsize=(10, 5))
plt.plot(pixel_ts["time"], pixel_ts["di"], marker='o')
plt.xlabel("Time")
plt.ylabel("DI")
plt.title(f"Zeitreihe für Pixel {first_pixel_id}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"/dss/dsshome1/01/di97rov/ROOT_project/scaling/pixel_{first_pixel_id}_timeseries.png")
print(f"\nPlot gespeichert als: pixel_{first_pixel_id}_timeseries.png")