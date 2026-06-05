from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


path = Path('2000monthly-surft-prec.nc')

ds = xr.open_dataset(path, engine='scipy')


# Coordinates of Sauteurs, Grenada.
lon_center, lat_center = 298.3, 12.3
lon = ds['longitude'].sel(longitude=lon_center, method='nearest')
lat = ds['latitude'].sel(latitude=lat_center, method='nearest')

# Find the year
year = ds.time.mean().dt.year.values

# Sauteurs temperature and LSP data.
sauteurs_temps = ds['t2m'].sel(longitude=lon_center, latitude=lat_center, method='nearest') - 273.15
sauteurs_rain = ds['lsp'].sel(longitude=lon_center, latitude=lat_center, method='nearest') * 1_000

# Calculate the correlation between temperature and LSP.
corr = np.corrcoef(sauteurs_temps, sauteurs_rain)[0, 1]


# Plotting.
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(sauteurs_rain, sauteurs_temps, color='red')
ax.set_ylabel(r"Temperature / $^\circ C$", fontsize=14)
ax.set_xlabel("Large-scale precipitation / mm", fontsize=14)
ax.set_title(textwrap.fill(f"Scatterplot of Monthly Temperature vs Large-scale precipitation at Sauteurs during {year}", width=50), fontsize=16)

ax.text(
    x=0.66,
    y=25.6,
    s=f"r: {corr:.3f}",
    fontsize=14,
    bbox=dict(facecolor='white', edgecolor='black')
)


plt.show()