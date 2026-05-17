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
tmax = ds['t2m'].sel(latitude=latitudes, longitude=longitudes).max(dim='time') - 273.15
tmin = ds['t2m'].sel(latitude=latitudes, longitude=longitudes).min(dim='time') - 273.15
tvar = tmax - tmin

pmax = ds['lsp'].sel(latitude=latitudes, longitude=longitudes).max(dim='time') * 1000
pmin = ds['lsp'].sel(latitude=latitudes, longitude=longitudes).min(dim='time') * 1000
pvar = pmax - pmin

x = tvar['longitude']
y = tvar['latitude']

# Plotting.
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(14, 6),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# Temperature range.
temp_plot = ax1.pcolormesh(
    x,
    y,
    tvar,
    cmap='magma',
    shading='auto',
    transform=ccrs.PlateCarree()
)

ax1.coastlines()
ax1.set_title('Temperature Range (°C)')

fig.colorbar(temp_plot, ax=ax1)

# Precip range.
precip_plot = ax2.pcolormesh(
    x,
    y,
    pvar,
    cmap='Blues',
    shading='auto',
    transform=ccrs.PlateCarree()
)

ax2.coastlines()
ax2.set_title('Precipitation Range (mm)')

fig.colorbar(precip_plot, ax=ax2)

plt.show()