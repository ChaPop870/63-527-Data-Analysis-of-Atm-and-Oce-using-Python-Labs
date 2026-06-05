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
longitudes: slice = slice(lon - 12.5, lon + 12.5)
latitudes: slice = slice(lat + 12.5, lat - 12.5)
# lons, lats = np.meshgrid(lon, lat)

rain = ds['lsp'].sel(latitude=latitudes, longitude=longitudes) * 1000
temps = ds['t2m'].sel(latitude=latitudes, longitude=longitudes)

rain_var = rain.max(dim='time') - rain.min(dim='time')
temps_var = temps.max(dim='time') - temps.min(dim='time')

year = rain.time.dt.year.values[0]
titles = [f"Annual Precipitation Variation in {year}", f"Annual Temperature Variation in {year}"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})

pc_rain = ax1.pcolormesh(
    rain['longitude'],
    rain['latitude'],
    rain_var,
    cmap='GnBu',
    transform=ccrs.PlateCarree(),
    vmin=0,
    vmax=8
)
cbar_1 = fig.colorbar(pc_rain, ax=ax1, orientation='horizontal', shrink=1, pad=0.05, extend='max')
cbar_1.set_label("Large-Scale Precipitation Variation / mm")

pc_temp = ax2.pcolormesh(
    rain['longitude'],
    rain['latitude'],
    temps_var,
    cmap='YlOrRd',
    transform=ccrs.PlateCarree(),
    vmin=0,
    vmax=5
)
cbar_2 = fig.colorbar(pc_temp, ax=ax2, orientation='horizontal', shrink=1, pad=0.05, extend='max')
cbar_2.set_label("2-meter Temperature Variation / K")

for ax, title in zip([ax1, ax2], titles):
    ax.coastlines('10m')
    ax.set_title(title)

    gl = ax.gridlines(draw_labels=True, lw=0.5, ls=':', alpha=0.7)

    gl.top_labels = False
    gl.right_labels = False


plt.show()