import rioxarray, xarray as xr
import numpy as np

fccl = rioxarray.open_rasterio("/dss/dsshome1/01/di97rov/ROOT_project/data/fcc_15122024_bav.tiff").squeeze()
ds = xr.open_dataset("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/ROOT_ts/di_diff_biweek_42_35k.nc")

print("FCC unique values:", np.unique(fccl)[0:10])
print("DI first 10 timesteps:", ds.time.values[:10])
print("DI number of timesteps:", len(ds.time))

print("FCC CRS:", fccl.rio.crs)
print("DI CRS:", ds.rio.crs)