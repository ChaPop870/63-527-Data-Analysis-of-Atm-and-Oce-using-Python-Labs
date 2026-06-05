from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


path = Path("May2000-uvt.nc")

ds = xr.open_dataset(path, engine='scipy')


# Select wind components.
u = ds['u'].sel(level=1000, method='nearest')
v = ds['v'].sel(level=1000, method='nearest')

# Lats and Lons.
x = ds['longitude']
y = ds['latitude']
X, Y = np.meshgrid(x, y)

# Compute wind speed.
wind_speed = np.sqrt(u**2 + v**2)

# Storms threshold.
storms = wind_speed.where(wind_speed >= 20)

# Global color scale.
vmin = 20
vmax = float(storms.max())

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cmap = mpl.cm.Reds


# Plotting
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection=ccrs.PlateCarree()))

ax.coastlines()

gl = ax.gridlines(draw_labels=True, lw=0.5, ls=':', alpha=0.7)
gl.top_labels = False
gl.right_labels = False

for t in wind_speed.time:

    u_t = u.sel(time=t)
    v_t = v.sel(time=t)
    ws = wind_speed.sel(time=t)

    mask = ws >= 20

    ax.scatter(
        X[mask],
        Y[mask],
        c=ws.values[mask.values],
        s=5,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        cmap='Reds'
    )

    ax.quiver(
        X[mask.values],
        Y[mask.values],
        u_t.values[mask.values],
        v_t.values[mask.values],
        transform=ccrs.PlateCarree(),
        scale=1200,
        width=0.001
    )


# Colorbar.
sm = mpl.cm.ScalarMappable(
    norm=norm,
    cmap=cmap
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation='horizontal',
    shrink=0.7,
    fraction=0.05,
    aspect=30,
    pad=0.05
)

cbar.set_label('Wind speed (m s$^{-1}$)')

plt.tight_layout()
plt.show()