from pathlib import Path

import matplotlib.dates as mdates
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

storm_count = (
    (wind_speed > 20)
    .astype(int)
    .sum(dim=['latitude', 'longitude'])
    .resample(time='1D')
    .sum()
)


fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(storm_count.time, storm_count, label='Daily storm counts')

ax.set_ylabel("Number of Storms", fontsize=16)
ax.set_xlabel("Day", fontsize=16)
ax.set_title(f"Global Storm Counts per day in {ds.time.dt.strftime('%B, %Y').values[0]}", fontsize=20)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

ax.tick_params(axis='x', which='minor', rotation=15, labelsize=14)
ax.tick_params(axis='y', which='major', labelsize=14)

ax.set_ylim(0, 60)
ax.set_xlim(storm_count.time.min(), storm_count.time.max())

ax.legend(fontsize=16)

plt.tight_layout()
plt.show()