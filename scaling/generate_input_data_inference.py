#!/usr/bin/env python3
"""
Extracts DI time series for all pixels in a tile,
only for those pixels that have an FCC disturbance timestep.
Each pixel gets ±WINDOW_SIZE timesteps around its disturbance time,
with one row per time step and constant pixel_id + timestep.
"""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from dask.distributed import Client, LocalCluster
import os
from shapely.geometry import box

WINDOW_SIZE = 6


def extract_all_ts_tile(tilenr):
    print(f"\nStarting extraction for tile {tilenr}")

    # --- Dask setup ---
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        memory_limit="16GB"
    )
    client = Client(cluster)
    print(f"Dask dashboard: {client.dashboard_link}")

    # --- Load DI time series ---
    di_ts_path = '/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/ROOT_ts'
    di_ts_file = os.path.join(di_ts_path, f"di_diff_biweek_{tilenr}_35k.nc")

    print(f"Loading DI time series from {di_ts_file}")
    di_ts = xr.open_dataset(di_ts_file, chunks={'time': -1, 'x': 1000, 'y': 1000}).di
    di_ts = di_ts.rio.write_crs("EPSG:3035")
    di_ts = di_ts.where(di_ts > -99999, np.nan)

    # --- Load FCC layer ---
    fccl_path = "/dss/dsshome1/01/di97rov/ROOT_project/data/fcc_15122024_bav.tiff"
    print(f"Loading FCC layer from {fccl_path}")
    fccl = rioxarray.open_rasterio(fccl_path, chunks={"x": 1000, "y": 1000}).squeeze()
    fccl = fccl.where(fccl > 0, np.nan)

    # --- Clip FCC to DI extent ---
    bbox = box(float(di_ts.x.min().compute()),
               float(di_ts.y.min().compute()),
               float(di_ts.x.max().compute()),
               float(di_ts.y.max().compute()))
    fccl = fccl.rio.clip([bbox], di_ts.rio.crs)

    # --- Determine unique timesteps in FCC ---
    fccl_values = fccl.values
    if hasattr(fccl_values, "compute"):
        fccl_values = fccl_values.compute()
    unique_timesteps = np.unique(fccl_values)
    unique_timesteps = unique_timesteps[~np.isnan(unique_timesteps)]
    unique_timesteps = unique_timesteps[
        (unique_timesteps > WINDOW_SIZE) & (unique_timesteps < len(di_ts.time))
    ].astype(int)

    print(f"Found {len(unique_timesteps)} FCC timesteps in this tile: {unique_timesteps[:10]}{'...' if len(unique_timesteps) > 10 else ''}")

    # (optional) limit for test runs
    unique_timesteps = unique_timesteps[:2]

    df_list = []

    for timestep in unique_timesteps:
        print(f"\n⏳ Processing FCC timestep {timestep} ...")

        # --- Find pixels that have this FCC timestep ---
        mask = (fccl == timestep)
        ys, xs = np.where(mask)
        n_pixels = len(xs)
        if n_pixels == 0:
            print(f"No pixels for timestep {timestep}, skipping.")
            continue
        print(f"Found {n_pixels} disturbed pixels at timestep {timestep}")

        # --- Create unique pixel IDs from coordinates ---
        pixel_x = di_ts.x.values[xs]
        pixel_y = di_ts.y.values[ys]
        pixel_ids = [f"{int(x)}_{int(y)}" for x, y in zip(pixel_x, pixel_y)]

        # --- Define DI time window (±WINDOW_SIZE around FCC timestep) ---
        center_idx = timestep - 1
        start_idx = max(0, center_idx - WINDOW_SIZE)
        end_idx = min(len(di_ts.time), center_idx + WINDOW_SIZE + 1)
        da_window = di_ts.isel(time=slice(start_idx, end_idx))
        time_values = da_window.time.values

        print(f"Extracting {len(time_values)} timesteps ({str(time_values[0])[:10]} → {str(time_values[-1])[:10]})")

        # --- Extract DI values for all relevant pixels ---
        da_pixels = da_window.isel(
            y=xr.DataArray(ys, dims="pixel"),
            x=xr.DataArray(xs, dims="pixel")
        )

        # --- Compute DataFrame (time major order) ---
        da_pixels = da_pixels.transpose("pixel", "time")  # WICHTIG!
        df = da_pixels.compute().to_dataframe(name="di").reset_index()

        # --- Richtige time Zuordnung ---
        df["pixel_id"] = [f"{int(pixel_x[i])}_{int(pixel_y[i])}" for i in df["pixel"]]
        df["timestep"] = timestep

        # --- Drop Hilfsindex ---
        df = df.drop(columns=["x", "y", "pixel"])

        # --- Keep order ---
        df = df[["pixel_id", "timestep", "time", "di"]]

        df_list.append(df)
        print(f"Extracted {len(df)} rows for timestep {timestep}")

    # --- Combine results ---
    if df_list:
        all_df = pd.concat(df_list, ignore_index=True)
        print(f"\nCombined {len(all_df)} rows for {all_df['pixel_id'].nunique()} unique pixels total")

        out_dir = "/dss/dsshome1/01/di97rov/ROOT_project/scaling" # "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/di97rov/input_data_scaling_ROOT"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"di_ts_{WINDOW_SIZE}steps_{tilenr}.feather")
        all_df.to_feather(out_file)
        print(f"Saved result to {out_file}")
    else:
        print("No valid data extracted for this tile.")

    client.close()
    cluster.close()
    print("Finished successfully!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tilenr", type=int, default=None, help="Tile number")
    args = parser.parse_args()

    if args.tilenr is not None:
        extract_all_ts_tile(args.tilenr)
    else:
        di_ts_path = '/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/ROOT_ts'
        nc_files = sorted([f for f in os.listdir(di_ts_path) if f.endswith('.nc')])
        print(f"Gefundene Dateien: {len(nc_files)}")
        for f in nc_files:
            try:
                tilenr = int(f.split('_')[3])
            except Exception:
                print(f"Could not extract tile number from {f}, skipping.")
                continue
            extract_all_ts_tile(tilenr)
