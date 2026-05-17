from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

path = Path('2000monthly-surft-prec.nc')

ds = xr.open_dataset(path, engine='scipy')

# Coordinates of Sauteurs, Grenada.
lon_center, lat_center = 298.3, 12.3
lon = ds['longitude'].sel(longitude=lon_center, method='nearest')
lat = ds['latitude'].sel(latitude=lat_center, method='nearest')

# Create 25 x 25 degree box around city.
longitudes: slice = slice(lon-12.5, lon+12.5)
latitudes: slice = slice(lat+12.5, lat-12.5)

# Select region from dataset.
rain = ds['lsp'].sel(latitude=latitudes, longitude=longitudes) * 1000

# Large-scale precipitation.
ax = rain.plot(x="longitude", y="latitude", col='time', col_wrap=4, cmap='YlGn',
               subplot_kws={'projection': ccrs.PlateCarree()},
               transform=ccrs.PlateCarree(),
               robust=True)
ax.map(lambda: plt.gca().coastlines())

plt.show()