from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr


# Load the data.
path = Path("May2000-uvt.nc")

ds = xr.open_dataset(path, engine='scipy')

level = 500
ds_500 = ds['u'].sel(level=level)

# Convert to DataFrame for box plots on ax1.
df = ds_500.to_dataframe()

# Compute variance for line plot on ax2.
u_var = ds_500.var(dim=['longitude', 'time'])


# Plotting
fig, (ax1, ax2) = plt.subplots(2, figsize=(14, 12.5))

# Box plots on ax1.
sns.boxplot(data=df, x='latitude', y='u', ax=ax1)

ax1.set_ylabel(r'u / m s$^{-1}$', fontsize=16)
ax1.set_xlabel('Latitude / °N', fontsize=16)
ax1.set_title(rf'Box plots of {level} hPa zonal wind $u$ versus latitude', fontsize=18)

ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_yticks(np.arange(-30, 65, 10))

for idx, label in enumerate(ax1.get_xticklabels()):
    label.set_visible(idx % 8 == 0)


# Variance plot on ax2.
u_var.plot(ax=ax2, color='k')

ax2.set_ylabel(r'Variance of $u$ / m$^2$ s$^{-2}$', fontsize=16)
ax2.set_xlabel('Latitude / °N', fontsize=16)
ax2.set_title(rF'Variance of {level} hPa zonal wind $u$ versus latitude', fontsize=18)

ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.set_xticks(np.arange(-90, 91, 20))
ax2.set_xlim(-90, 90)


plt.show()