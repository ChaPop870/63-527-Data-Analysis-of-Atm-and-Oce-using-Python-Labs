from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


path = Path('2000monthly-surft-prec.nc')

ds = xr.open_dataset(path, engine='scipy')


# Coordinates of Sauteurs, Grenada.
lon_center, lat_center = 298.3, 12.3
lon = ds['longitude'].sel(longitude=lon_center, method='nearest')
lat = ds['latitude'].sel(latitude=lat_center, method='nearest')

# Sauteurs temperature and LSP data.
sauteurs_temps = ds['t2m'].sel(longitude=lon_center, latitude=lat_center, method='nearest') - 273.15
sauteurs_rain = ds['lsp'].sel(longitude=lon_center, latitude=lat_center, method='nearest') * 1_000

# Calculate the correlation between temperature and LSP.
corr = np.corrcoef(sauteurs_temps, sauteurs_rain)[0, 1]


# Plotting.
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(sauteurs_rain, sauteurs_temps, color='red')
ax.set_ylabel(r"Temperature / $^\circ C$")
ax.set_xlabel("Large-scale precipitation / mm")
ax.set_title("Scatter plot of Monthly Temperature vs Large-scale precipitation at Sauteurs")

ax.text(
    x=0.6,
    y=25.7,
    s=f"r: {corr:.3f}"
)

plt.show()