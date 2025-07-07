import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import os
import dask
from shapely.geometry import Point


def extract_time_series_from_polygons(nc_path, filepath, crs_epsg=3035):
    polygons_gdf = gpd.read_file(filepath).to_crs(epsg=crs_epsg)
    polygons_gdf = polygons_gdf.rename(columns={"id": "polygon_id"})
    if "Flight_Dat" in polygons_gdf.columns and "Flight_Date" not in polygons_gdf.columns:
        polygons_gdf = polygons_gdf.rename(columns={"Flight_Dat": "Flight_Date"})

    ds = xr.open_dataset(nc_path, chunks={})
    ds = ds.rio.write_crs(f"EPSG:{crs_epsg}")

    all_time_series = []

    for idx, row in polygons_gdf.iterrows():
        polygon = row.geometry
        polygon_id = row['polygon_id']
        class_label = row.get('class', 'unknown')
        flight_date = pd.to_datetime(row.get('Flight_Date', pd.NaT), errors='coerce')

        if pd.isna(flight_date):
            print(f"Kein gültiges Flight_Date für Polygon {polygon_id}. Übersprungen.")
            continue

        try:
            clipped = ds.rio.clip([polygon], crs=f"EPSG:{crs_epsg}", drop=True, invert=False)
        except Exception as e:
            print(f"Fehler beim Clippen für Polygon {polygon_id}: {e}")
            continue

        try:
            di_data = clipped['di']
            ts_df = di_data.to_series().reset_index(name='di')

            # Filter: ±6 Monate um Flight_Date
            start_date = flight_date - pd.DateOffset(months=6)
            end_date = flight_date + pd.DateOffset(months=6)
            ts_df = ts_df[(ts_df['time'] >= start_date) & (ts_df['time'] <= end_date)]

            if ts_df.empty:
                print(f"Keine Daten im deefininerten Zeitraum Monate für Polygon {polygon_id}")
                continue

            ts_df['pixel_id'] = 'x' + ts_df['x'].astype(str) + '_y' + ts_df['y'].astype(str)
            ts_df['polygon_id'] = polygon_id
            ts_df['class'] = class_label
            ts_df['Flight_Date'] = flight_date

            ts_df['rel_week'] = ((ts_df['time'] - flight_date).dt.days // 7)
            ts_df['time_str'] = ts_df['rel_week'].apply(lambda x: f"di_t{int(x):+}")

            # Erzeuge Point-Geometrien aus Pixelkoordinaten
            ts_df['point'] = ts_df.apply(lambda row: Point(row['x'], row['y']), axis=1)

            # Filter nur die Punkte, die im Polygon liegen
            ts_df = ts_df[ts_df['point'].apply(lambda p: polygon.contains(p))]

            all_time_series.append(ts_df)

        except Exception as e:
            print(f"Fehler beim Extrahieren für Polygon {polygon_id}: {e}")
            continue

    return all_time_series

def extract_time_series_from_polygon_folder(nc_path, polygon_folder, output_folder, crs_epsg=3035):
    import os
    import pandas as pd

    os.makedirs(output_folder, exist_ok=True)
    all_time_series = []

    for filename in os.listdir(polygon_folder):
        if filename.endswith((".gpkg", ".shp", ".geojson")):
            filepath = os.path.join(polygon_folder, filename)
            print(f"\n🔄 Verarbeite: {filename}")
            ts_list = extract_time_series_from_polygons(
                nc_path=nc_path,
                filepath=filepath,
                crs_epsg=crs_epsg
            )
            all_time_series.extend(ts_list)

    if not all_time_series:
        print("⚠️ Keine gültigen Zeitreihen aus allen Dateien gefunden.")
        return

    # Long Format
    final_df = pd.concat(all_time_series, ignore_index=True)
    long_csv = os.path.join(output_folder, "all_polygons_time_series_long.csv")
    final_df.to_csv(long_csv, index=False)

    # Wide Format
    wide_df = final_df.pivot_table(
        index=['polygon_id', 'pixel_id', 'x', 'y', 'class'],
        columns='time_str',
        values='di'
    ).reset_index()

    # Flight_Date hinzufügen
    date_info = final_df[['polygon_id', 'Flight_Date']].drop_duplicates()
    wide_df = wide_df.merge(date_info, on='polygon_id', how='left')

    wide_csv = os.path.join(output_folder, "all_polygons_time_series_wide.csv")
    wide_df.to_csv(wide_csv, index=False)

    print(f"✅ Gesamtverarbeitung abgeschlossen. {len(wide_df)} Pixel extrahiert.")


extract_time_series_from_polygon_folder(
    nc_path="data/DI_timeseries/di_diff_biweek_42_35k.nc",
    polygon_folder="data/polygons",
    output_folder="data/extracted_DI_polygons",
    crs_epsg=3035
)

