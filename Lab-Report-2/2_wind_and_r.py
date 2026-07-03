from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# Load the data.
path = Path("May2000-uvt.nc")

ds = xr.open_dataset(path, engine='scipy')

level = 1000
u = ds['u'].sel(level=level)
v = ds['v'].sel(level=level)

corr = xr.corr(u, v, dim=['time'])
# try with np as well

lons, lats = np.meshgrid(u.longitude, u.latitude)
u_mean = u.mean(dim='time')
v_mean = v.mean(dim='time')


# Plotting.
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

ax.coastlines()

ax.set_title(f"Global {level} hPa wind vectors plotted over Correlation Coefficient.")

gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.3, linewidth=0.7)
gl.top_labels = False
gl.right_labels = False

corr_plot = ax.pcolormesh(
    ds.longitude,
    ds.latitude,
    corr,
    transform=ccrs.PlateCarree(),
    vmin=-1,
    vmax=1,
    cmap='RdBu_r'
)

skip = 4

wind_plot = ax.quiver(
    lons[::skip, ::skip],
    lats[::skip, ::skip],
    u_mean[::skip, ::skip],
    v_mean[::skip, ::skip],
    transform=ccrs.PlateCarree(),

    scale=400,
    width=0.0015,
    pivot='middle'
)

cbar = fig.colorbar(
    corr_plot,
    ax=ax,
    pad=0.02,
    aspect=40,
    fraction=0.012
)
cbar.set_label(r'Correrlation correlation', fontsize=16)


plt.show()