from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import xarray as xr


# Load the data.
path = Path("May2000-uvt.nc")

ds = xr.open_dataset(path, engine='scipy')


# Select u wind at 250 hPa at a longitude and average it over latitudes 15N and 15S.
lon_1 = 120
u_trop_1 = ds['u'].sel(longitude=lon_1, latitude=slice(15, -15), level=250).mean(dim='latitude')

lon_2 = 105
u_trop_2 = ds['u'].sel(longitude=lon_2, latitude=slice(15, -15), level=250).mean(dim='latitude')

t = u_trop_1.time.values.astype("datetime64[s]").astype(float)
t = (t - t[0]) / 86_400


# Define sine model for curve-fitting with first guess.
def sine_model(t, a, b, w, c):
    return a + b * np.sin(w * t + c)

# guess = [
#     u_trop.mean(dim='time'),
#     0.5 * (u_trop.max(dim='time') - u_trop.min(dim='time')),
#     2*np.pi / 15,
#     0
# ]

guess = [
    -7.5,
    10,
    2*np.pi / 20,
    2
]

coefs_1, _ = curve_fit(
    sine_model,
    t,
    u_trop_1.values,
    p0=guess,
    bounds=((-10, 4, -1, -30),
            (0, 20, 1, 30))
)
a, b, w, c = coefs_1

u_trop_1_pred = sine_model(t, a, b, w, c)


# Define sine model with same frequency for another latitude.
u_trop_2_pred = sine_model(t, -6.45, 5.5, w, 1.8)


# Compute the model period
model_period = 2*np.pi / w


# Compute R² for both cases
r2 = r2_score(u_trop_1, u_trop_1_pred)
r2_2 = r2_score(u_trop_2, u_trop_2_pred)


# Plotting.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(u_trop_1.time, u_trop_1, color='black', label=f'Tropopause at {lon_1} E')

ax1.plot(u_trop_1.time, u_trop_1_pred, color='red', label=f"Model (R² = {r2:.3f})")

ax1.set_xlabel("Time", fontsize=16)
ax1.set_ylabel(r"Mean zonal wind $\bar{u}(t)$ / m s$^{-1}$", fontsize=16)
ax1.set_title(f"Mean 250 hPa zonal wind for May 2000 at longitude {lon_1} E.")

ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=2))
ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

ax1.tick_params(axis='x', which='major', pad=12)
ax1.tick_params(axis='x', which='minor', rotation=15)
ax1.set_yticks(np.arange(-15, 6, 5))

ax1.set_xlim(u_trop_1.time.min(), u_trop_1.time.max())
ax1.set_ylim(-15, 5)

ax1.text(
    0.98, 0.02,
    s=f"Model Period: {model_period:.1f} days.",
    transform=ax1.transAxes,
    ha='right',
    va='bottom',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.5)
)

ax1.legend(fontsize=14)


ax2.plot(u_trop_2.time, u_trop_2, color='black', label=f'Tropopause at {lon_2} E')

ax2.plot(u_trop_2.time, u_trop_2_pred, color='red', label=f"Model (R² = {r2_2:.3f})")

ax2.set_xlabel("Time", fontsize=16)
ax2.set_ylabel(r"Mean zonal wind $\bar{u}(t)$ / m s$^{-1}$", fontsize=16)
ax2.set_title(f"Mean 250 hPa zonal wind for May 2000 at longitude {lon_2} E.")

ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=2))
ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

ax2.tick_params(axis='x', which='major', pad=12)
ax2.tick_params(axis='x', which='minor', rotation=15)
ax2.set_yticks(np.arange(-15, 6, 5))

ax2.set_xlim(u_trop_2.time.min(), u_trop_2.time.max())
ax2.set_ylim(-15, 5)

ax2.text(
    0.98, 0.02,
    f"Model Period: {model_period:.1f} days",
    transform=ax2.transAxes,
    ha='right',
    va='bottom',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        edgecolor='black',
        alpha=0.5
    )
)

ax2.legend(fontsize=14)


plt.show()