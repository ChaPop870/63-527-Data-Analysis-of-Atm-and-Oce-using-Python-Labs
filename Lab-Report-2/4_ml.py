from pathlib import Path
import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import xarray as xr


def select_predictors_and_observations(ds, lon, lat, level):
    """Select the surface temperature predictors and wind observations for a given coordinate."""
    # Load observed u.
    u_obs = ds['u'].sel(longitude=lon, latitude=lat, level=level, method='nearest')

    # Get index of the longitude.
    i = ds.get_index("longitude").get_loc(float(u_obs.longitude))

    # Select temperature west and east of longitude i.
    T_west = ds['t'].isel(longitude=(i-1)).sel(latitude=lat, level=level, method='nearest')
    T_east = ds['t'].isel(longitude=(i+1)).sel(latitude=lat, level=level, method='nearest')
    T_i = ds['t'].isel(longitude=i).sel(latitude=lat, level=level, method='nearest')

    return T_east, T_i, T_west, u_obs


def u_pred(u_obs, *predictors):
    """Predict zonal wind using temperature predictors."""
    # Create matrix for predictors T_west and T_east
    X = np.column_stack(predictors)

    # Fit regression model.
    regr = linear_model.LinearRegression()
    regr.fit(X, u_obs)
    u_pred = regr.predict(X)

    # Compute R² using the T_west and T_east.
    r2 = r2_score(u_obs, u_pred)

    # Compute MSE.
    mse = mean_squared_error(u_obs, u_pred)

    return u_pred, r2, mse


# Load the dataset.
path = Path("May2000-uvt.nc")

ds = xr.open_dataset(path, engine='scipy')


# Select points on map.
lon1, lat1 = 298.3, 12.3
level = 1000

lon2, lat2 = 30, 0

times = ds.time.values


# Load the data for each coordinate.
T_east_1, T_i_1, T_west_1, u_obs_1 = select_predictors_and_observations(ds, lon1, lat1, level)
T_east_2, T_i_2, T_west_2, u_obs_2 = select_predictors_and_observations(ds, lon2, lat2, level)


# Compute predicted zonal wind and R².
u_pred_a1, r2_a1, mse_a1 = u_pred(u_obs_1, T_east_1, T_west_1)
u_pred_a2, r2_a2, mse_a2 = u_pred(u_obs_1, T_east_1, T_i_1, T_west_1)

u_pred_b1, r2_b1, mse_b1 = u_pred(u_obs_2, T_east_2, T_west_2)
u_pred_b2, r2_b2, mse_b2 = u_pred(u_obs_2, T_east_2, T_i_1, T_west_2)


# Create dictionary for the data.
data = {
    "point A": {
        'u_obs': u_obs_1,
        'u_pred': u_pred_a1,
        'u_pred2': u_pred_a2,
        'r2': float(r2_a1),
        'r2_2': float(r2_a2),
        'mse': float(mse_a1),
        'mse2': float(mse_a2)
    },
    "point B": {
        'u_obs': u_obs_2,
        'u_pred': u_pred_b1,
        'u_pred2': u_pred_b2,
        'r2': r2_b1,
        'r2_2': r2_b2,
        'mse': float(mse_b1),
        'mse2': float(mse_b2)
    }
}


# Plotting.
fig, axes = plt.subplots(2, 1, figsize=(10, 15))

for idx, (_, point) in enumerate(data.items()):
    u_obs  = point['u_obs']
    u_pred = point['u_pred']
    u_pred2 = point['u_pred2']
    r2 = point['r2']
    r2_2 = point['r2_2']
    mse = point['mse']
    mse2 = point['mse2']

    ax = axes[idx]

    ax.plot(times, u_obs, label=r"Observed u")
    ax.plot(times, u_pred, label=rf"Pred $u$ with E, W. R²={r2:.3f}, MSE={mse:.3f}")
    ax.plot(times, u_pred2, label=rf"Pred $u$ with center. R²={r2_2:.3f}, MSE={mse2:.3f}")

    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel(r"Zonal wind (u) / m s$^{-1}$", fontsize=14)
    ax.set_title(textwrap.fill(f"Observed zonal wind and predicted zonal wind using the temperature at {float(u_obs.latitude)}°N, {(float(u_obs.longitude)):.1f}°E", max_lines=2), fontsize=16)

    ax.set_ylim(u_obs.min() - 1, u_obs.max() + 1)
    ax.invert_yaxis()
    ax.set_xlim(u_obs.time.min(), u_obs.time.max())

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

    ax.tick_params(axis='x', which='major', pad=12)
    ax.tick_params(axis='x', which='minor', rotation=15)

    ax.legend(fontsize=14)

plt.show()