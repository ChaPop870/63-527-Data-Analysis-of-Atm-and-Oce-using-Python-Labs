from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


path = Path('2000monthly-surft-prec.nc')

ds = xr.open_dataset(path, engine='scipy')

# Coordinates of Sauteurs, Grenada.
lon_center, lat_center = 298.3, 12.3
lon = ds['longitude'].sel(longitude=lon_center, method='nearest')
lat = ds['latitude'].sel(latitude=lat_center, method='nearest')

# Create 25 × 25 degree box around city.
longitudes: slice = slice(lon-12.5, lon+12.5)
latitudes: slice = slice(lat+12.5, lat-12.5)
lons, lats = np.meshgrid(longitudes, latitudes)

# Select mean lsp and driest month across the region.
lsp = ds['lsp'].sel(latitude=latitudes, longitude=longitudes) * 1000
lsp_mean = lsp.mean(dim=['latitude', 'longitude'])

driest_month_idx = lsp_mean.argmin()
driest_month_da = lsp_mean['time'].isel(time=driest_month_idx)
driest_month = driest_month_da.dt.strftime('%B').values

# Select t2m and warmest month across the region.
t2m = ds['t2m'].sel(latitude=latitudes, longitude=longitudes) - 273.15
t2m_mean = t2m.mean(dim=['latitude', 'longitude'])

warmest_month_idx = t2m_mean.argmax()
warmest_month_da = t2m_mean['time'].isel(time=warmest_month_idx)
warmest_month = warmest_month_da.dt.strftime('%B').values


# Plotting.
fig, ax1 = plt.subplots(figsize=(10, 8))

ax1.plot(
    ds['time'],
    t2m_mean,
    color='red',
    linewidth=2,
    label="2 m Temperature"
)

ax1.plot(
    warmest_month_da,
    t2m_mean[warmest_month_idx],
    marker='x',
    color='red',
    markersize=10,
    zorder=5
)

ax1.annotate(
    f"Warmest month:\n{warmest_month_da.dt.strftime('%B').item()}",
    xy=(warmest_month_da, t2m_mean[warmest_month_idx]),
    xytext=(15, 20),  # offset in points
    textcoords='offset points',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', lw=1),
    arrowprops=dict(arrowstyle='->', color='red'),
)

ax1.set_xlabel("Time")
ax1.set_ylabel("Temperature / °C")
ax1.set_title(
    (
        "Region-averaged Temperature and Precipitation in 2000\n"
        f"between longitudes:{longitudes.start.values-240} and {longitudes.stop.values-240} W and latitudes:{latitudes.stop.values} and {latitudes.start.values} N"
    )
)
ax1.set_xlim(ds['time'].min(), ds['time'].max())
ax1.set_ylim(24, 27)

ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=np.arange(1, 13)))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.tick_params(axis='x', which='major', rotation=45, bottom=True)

ax1.grid(False)


ax2 = ax1.twinx()

ax2.plot(
    ds['time'],
    lsp_mean,
    color='blue',
    linewidth=2,
    label="LSP Mean"
)

ax2.plot(
    driest_month_da,
    lsp_mean[driest_month_idx],
    marker='o',
    color='blue',
    markersize=8,
    zorder=5
)

ax2.annotate(
    f"Driest month:\n{driest_month_da.dt.strftime('%B').item()}",
    xy=(driest_month_da, lsp_mean[driest_month_idx]),
    xytext=(20, -22),  # offset in points
    textcoords='offset points',
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='blue', lw=1),
    arrowprops=dict(arrowstyle='->', color='blue'),
)

ax2.set_ylabel("Large-scale Precipitation / mm")
ax2.set_ylim(0.35, 1.0)
ax2.grid(False)


# Build legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')


plt.show()