import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray  # wichtig für `.rio`-Funktionen
import os

# Change working directory
os.chdir('/Users/christinakrause/HIWI_DLR_Forest/')

def extract_time_series_from_polygons(nc_path, filepath, output_folder, crs_epsg=3035):
    # Lade Polygone und transformiere CRS
    polygons_gdf = gpd.read_file(filepath).to_crs(epsg=crs_epsg)
    polygons_gdf = polygons_gdf.rename(columns={"id": "polygon_id"})
    if "Flight_Dat" in polygons_gdf.columns and "Flight_Date" not in polygons_gdf.columns:
        polygons_gdf = polygons_gdf.rename(columns={"Flight_Dat": "Flight_Date"})

    # Lade NetCDF und schreibe CRS
    ds = xr.open_dataset(nc_path)
    ds = ds.rio.write_crs(f"EPSG:{crs_epsg}")

    # Container für alle Zeitreihen
    all_time_series = []

    for idx, row in polygons_gdf.iterrows():
        polygon = row.geometry
        polygon_id = row['polygon_id']
        class_label = row.get('class', 'unknown')
        flight_date = pd.to_datetime(row.get('Flight_Date', pd.NaT), errors='coerce')

        # Clip Dataset mit Polygon
        try:
            clipped = ds.rio.clip([polygon], crs=f"EPSG:{crs_epsg}", drop=True, invert=False)
        except Exception as e:
            print(f"❌ Fehler beim Clippen für Polygon {polygon_id}: {e}")
            continue

        try:
            di_data = clipped['di']

            # In DataFrame umwandeln
            ts_df = di_data.to_series().reset_index(name='di')
            ts_df['pixel_id'] = 'x' + ts_df['x'].astype(str) + '_y' + ts_df['y'].astype(str)

            # Metadaten hinzufügen
            ts_df['polygon_id'] = polygon_id
            ts_df['class'] = class_label
            ts_df['Flight_Date'] = flight_date

            # Relative Woche zur Flight_Date
            if pd.notna(flight_date):
                ts_df['rel_week'] = ((ts_df['time'] - flight_date).dt.days // 7)
                ts_df['time_str'] = ts_df['rel_week'].apply(lambda x: f"di_t{int(x):+}")
            else:
                ts_df['rel_week'] = np.nan
                ts_df['time_str'] = None

            all_time_series.append(ts_df)

        except Exception as e:
            print(f"❌ Fehler beim Extrahieren für Polygon {polygon_id}: {e}")
            continue

    if not all_time_series:
        print("⚠️ Keine gültigen Zeitreihen gefunden.")
        return

    # Long Format speichern
    final_df = pd.concat(all_time_series, ignore_index=True)
    basename = os.path.splitext(os.path.basename(filepath))[0]

    long_csv = os.path.join(output_folder, f"{basename}_time_series_long.csv")
    final_df.to_csv(long_csv, index=False)

    # Wide Format (pro Pixel)
    wide_df = final_df.pivot_table(
        index=['polygon_id', 'pixel_id', 'x', 'y', 'class'],
        columns='time_str',
        values='di'
    ).reset_index()

    # Flight_Date hinzufügen
    date_info = final_df[['polygon_id', 'Flight_Date']].drop_duplicates()
    wide_df = wide_df.merge(date_info, on='polygon_id', how='left')

    wide_csv = os.path.join(output_folder, f"{basename}_time_series_wide.csv")
    wide_df.to_csv(wide_csv, index=False)

    print(f"✅ Fertig: {basename} ({len(wide_df)} Pixel extrahiert)")

def extract_time_series_from_polygon_folder(nc_path, polygon_folder, output_folder, crs_epsg=3035):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(polygon_folder):
        if filename.endswith((".gpkg", ".shp", ".geojson")):
            filepath = os.path.join(polygon_folder, filename)
            print(f"\n🔄 Verarbeite: {filename}")
            extract_time_series_from_polygons(
                nc_path=nc_path,
                filepath=filepath,
                output_folder=output_folder,
                crs_epsg=crs_epsg
            )

extract_time_series_from_polygon_folder(
    nc_path="Datenpaket_Beginn/di_diff_biweek_42_35k.nc",
    polygon_folder="Data_Collection/polygons",
    output_folder="Data_Collection/DI_pixel_timeseries",
    crs_epsg=3035
)

