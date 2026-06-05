from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


path = Path("2000monthly-surft-prec.nc")

ds = xr.open_dataset(path, engine='scipy')

year = 2000

time_bins = [
    datetime(year - 1, 12, 1),
    datetime(year, 2, 1),
    datetime(year, 5, 1),
    datetime(year, 8, 1),
    datetime(year, 11, 1),
    datetime(year + 1, 2, 1),
]

bin_labels = ['JF', 'MAM', 'JJA', 'SON', 'D']

seasonal_t2m = ds.t2m.groupby_bins(
    group=ds.time,
    bins=time_bins,
    labels=bin_labels
)

zonal_seasonal_t2m_max = seasonal_t2m.max(dim=['time', 'longitude'])
zonal_seasonal_t2m_min = seasonal_t2m.min(dim=['time', 'longitude'])

interseasonal_variability_t2m = zonal_seasonal_t2m_max - zonal_seasonal_t2m_min


seasonal_lsp = ds.lsp.groupby_bins(
    group=ds.time,
    bins=time_bins,
    labels=bin_labels
)

zonal_seasonal_lsp_max = seasonal_lsp.max(dim=['time', 'longitude'])
zonal_seasonal_lsp_min = seasonal_lsp.min(dim=['time', 'longitude'])

interseasonal_variability_lsp = zonal_seasonal_lsp_max - zonal_seasonal_lsp_min


# Plotting.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

for season in zonal_seasonal_t2m.time_bins.values:
    ax1.plot(
        zonal_seasonal_t2m.latitude,
        interseasonal_variability_t2m.sel(time_bins=season),
        label=season
    )

ax1.set_ylabel("Seasonal variation in 2 m Temperature / °C")

for season in zonal_seasonal_lsp.time_bins.values:
    ax2.plot(
        zonal_seasonal_lsp.latitude,
        interseasonal_variability_lsp.sel(time_bins=season),
        label=season
    )

ax2.set_ylabel("Seasonal variation in LSP / mm")


for ax in [ax1, ax2]:
    ax.set_xlabel("Latitude")
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, None)
    ax.legend()

plt.show()