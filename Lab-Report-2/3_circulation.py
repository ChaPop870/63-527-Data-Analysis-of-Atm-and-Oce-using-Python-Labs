from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

path1 = Path("may2000-surf.nc")
path2 = Path("May2000-uvt.nc")

ds1 = xr.open_dataset(path1, engine='scipy')
ds2 = xr.open_dataset(path2, engine='scipy')

# Load grid.
x = ds1['longitude']
y = ds1['latitude']
X, Y = np.meshgrid(x, y)

# Load surface wind data.
u10 = ds1["u10"].load().sel(time="2000-05").mean(dim='time')
v10 = ds1["v10"].load().sel(time="2000-05").mean(dim='time')
sfc_wind_speed = np.sqrt(u10**2 + v10**2)

# Load upper-air wind data.
u = ds2["u"].load().sel(time="2000-05")
v = ds2["v"].load().sel(time="2000-05")

# Load tropopause wind data
u250 = u.sel(level=250)
v250 = v.sel(level=250)
u250_mean = u250.mean(dim='time')
v250_mean = v250.mean(dim='time')

trop_wind_speed = np.sqrt(u250**2 + v250**2)
mean_trop_wind_speed = trop_wind_speed.mean(dim='time')

# Level wind speed.
wspeed = np.sqrt(u**2 + v**2)

# Assign level wind speed
ds2["wspeed"] = wspeed
ds2["wspeed"].attrs = {
    "units": "m s**-1",
    "long_name": "Wind Speed at given level",
    "standard_name": "Wind speed"
}
print(f"wspeed variable and attributes added to {path2}.")

output_filename = "May2000-uvt-wspeed.nc"
output_path = Path(output_filename)
# ds2.to_netcdf("output_path")
print(f"Created NetCDF file at: {output_path.resolve()}")


# Plotting.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), subplot_kw=dict(projection=ccrs.PlateCarree()))

for ax in [ax1, ax2]:
    ax.coastlines(linewidth=0.8)

    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

# Surface plot.
sfc_wind_plot = ax1.pcolormesh(
    X,
    Y,
    sfc_wind_speed,
    transform=ccrs.PlateCarree(),
    cmap='turbo',
    vmin=0,
    vmax=12,
    shading='auto'
)

fig.colorbar(sfc_wind_plot, ax=ax1, label=r"Wind speed / m s$^{-1}$", extend='max',  orientation='horizontal', shrink=0.7, fraction=0.05, aspect=30)

skip = 4

q = ax1.quiver(
    X[::skip,::skip],
    Y[::skip,::skip],
    u10[::skip,::skip],
    v10[::skip,::skip],
    transform=ccrs.PlateCarree(),

    scale=400,      # controls arrow length.
    width=0.0015,
    pivot='middle'
)
ax1.quiverkey(
    q,
    X=0.92,
    Y=-0.12,
    U=10,
    label=r'10 m s$^{-1}$',
    labelpos='E',
    coordinates='axes'
)

ax1.set_title("Mean global surface wind speed plot for May 2000")


# Tropopause plot.
trop_wind_plot = ax2.pcolormesh(
    X,
    Y,
    mean_trop_wind_speed,
    transform=ccrs.PlateCarree(),
    cmap='turbo',
    vmin=0,
    vmax=40,
    shading='auto'
)

fig.colorbar(trop_wind_plot, ax=ax2, label=r"Wind speed / m s$^{-1}$", extend='max',  orientation='horizontal', shrink=0.7, fraction=0.05, aspect=30)

skip = 4

q = ax2.quiver(
    X[::skip,::skip],
    Y[::skip,::skip],
    u250_mean[::skip,::skip],
    v250_mean[::skip,::skip],
    transform=ccrs.PlateCarree(),

    scale=1500,      # controls arrow length.
    width=0.0015,
    pivot='middle'
)
ax2.quiverkey(
    q,
    X=0.92,
    Y=-0.12,
    U=30,
    label=r'30 m s$^{-1}$',
    labelpos='E',
    coordinates='axes'
)
ax2.set_title("Mean global tropopause wind speed plot for May 2000")


plt.show()