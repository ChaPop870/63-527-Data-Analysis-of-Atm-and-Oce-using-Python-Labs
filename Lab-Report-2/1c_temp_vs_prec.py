from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

path = Path('2000monthly-surft-prec.nc')

ds = xr.open_dataset(path, engine='scipy')

# Coordinates of Sauteurs, Grenada.
lon_center, lat_center = 298.3, 12.3
lon = ds['longitude'].sel(longitude=lon_center, method='nearest')
lat = ds['latitude'].sel(latitude=lat_center, method='nearest')

sauteurs_temps = ds['t2m'].sel(longitude=lon_center, latitude=lat_center, method='nearest') - 273.15

sauteurs_rain = ds['lsp'].sel(longitude=lon_center, latitude=lat_center, method='nearest') * 1_000

fig, ax1 = plt.subplots(figsize=(8, 8))
ax1.scatter(ds.time, sauteurs_temps, color='red')
ax1.set_ylabel(r"Temperature / $^\circ C$")
ax1.set_xlabel("Time")
ax1.set_ylim(25, 29)

ax2 = ax1.twinx()
ax2.scatter(ds.time, sauteurs_rain)
ax2.set_ylabel("Large-scale precipitation / mm")

fig.suptitle(f"Temperature vs precipitation at Sauteurs Grenada ({lon_center-236.5:.1f} W, {lat_center} N), in 2000", y=0.93)

plt.show()