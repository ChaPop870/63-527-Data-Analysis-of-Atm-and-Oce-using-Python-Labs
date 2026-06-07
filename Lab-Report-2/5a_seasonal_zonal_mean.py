from datetime import datetime
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import xarray as xr


path = Path("2000monthly-surft-prec.nc")

ds = xr.open_dataset(path, engine='scipy')

year = ds.time.mean().dt.year.values

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
zonal_seasonal_t2m = seasonal_t2m.mean(dim=['time', 'longitude']) - 273.15


seasonal_lsp = ds.lsp.groupby_bins(
    group=ds.time,
    bins=time_bins,
    labels=bin_labels
)
zonal_seasonal_lsp = seasonal_lsp.mean(dim=['time', 'longitude']) * 1000


# Plotting.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

for season in zonal_seasonal_t2m.time_bins.values:
    ax1.plot(
        zonal_seasonal_t2m.latitude,
        zonal_seasonal_t2m.sel(time_bins=season),
        label=season
    )

ax1.set_ylabel("2 m Temperature / °C", fontsize=14)

for season in zonal_seasonal_lsp.time_bins.values:
    ax2.plot(
        zonal_seasonal_lsp.latitude,
        zonal_seasonal_lsp.sel(time_bins=season),
        label=season
    )

ax2.set_ylabel("Large-scale precipitation / mm", fontsize=14)
ax2.set_xlabel("Latitude", fontsize=14)

for ax in [ax1, ax2]:
    ax.set_xlim(-90, 90)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14)

fig.suptitle(textwrap.fill("Seasonal zonally-averaged temperature and large-scale precipitation", 40), fontsize=20)

plt.tight_layout()
plt.show()