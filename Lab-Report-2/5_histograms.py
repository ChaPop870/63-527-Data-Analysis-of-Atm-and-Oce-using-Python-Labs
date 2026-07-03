from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import xarray as xr


# Load the data.
path = Path("May2000-uvt.nc")

ds = xr.open_dataset(path, engine='scipy')


# Select the data.
lat_n = 45
lat_s = - lat_n
level = 500

u_500_n = ds['u'].sel(latitude=lat_n, level=level, method='nearest').values.ravel()
u_500_n_mean = u_500_n.mean()
u_500_n_std = u_500_n.std()

u_500_s = ds['u'].sel(latitude=lat_s, level=level, method='nearest').values.ravel()
u_500_s_mean = u_500_s.mean()
u_500_s_std = u_500_s.std()

# Student t-test
t, p = ttest_ind(u_500_n, u_500_s)


# Plotting
fig, ax = plt.subplots(figsize=(9, 7))

sns.histplot(
    data=u_500_n,
    stat='probability',
    binwidth=1,
    label="45°N",
    ax=ax
)

sns.histplot(
    data=u_500_s,
    stat='probability',
    binwidth=1,
    label="45°S",
    ax=ax
)

text = (
    f"Stats at 45°N\n"
    f"Mean: {u_500_n_mean:.1f} m/s\n"
    f"Std: {u_500_n_std:.1f} m/s\n\n"
    f"Stats at 45°S\n"
    f"Mean: {u_500_s_mean:.1f} m/s\n"
    f"Std: {u_500_s_std:.1f} m/s"
)

ax.text(
    x=0.76,
    y=0.83,
    s=text,
    transform=ax.transAxes,
    ha='left',
    va='top',
    bbox=dict(boxstyle='round', fc='w', ec='k', lw=0.5),
    fontsize=14,
)

ax.set_xlabel(r"Zonal wind (u) / m s$^{-1}$")
ax.set_ylabel("Probability")
ax.set_title(f"Histograms of zonal wind at {level} hPa for 45°N and 45°S in May 2000")

ax.set_xticks(np.arange(-30, 56, 10))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(
    fontsize=14,
    fancybox=True,
    edgecolor='black'
)
ax.grid(False)


plt.show()

print(f"T-score: {t:.3g}.")
print(f"P-value: {p:.3g}.")