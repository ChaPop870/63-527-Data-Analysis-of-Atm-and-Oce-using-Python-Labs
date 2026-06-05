from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# Create functions to calculate data arrays.
def u(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute zonally wind on a fantasy planet given the latitude
    and longitude."""
    return -10 * np.sin(2*np.pi*y / 180) * np.cos(np.pi*x / 180)**2


def v(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute meridional wind on a fantasy planet given the latitude
     and longitude."""
    return 10 * np.cos(np.pi*y / 180)**2 * np.sin(2*np.pi*x / 180)


def kinetic_energy(
        u: np.ndarray,
        v: np.ndarray
)\
        -> np.ndarray:
    """Compute the kinetic energy of the wind on the fantasy given
    the zonal and meridional wind."""
    return 0.5 * (u**2 + v**2)


# Create lat / lon grid.
latitudes = np.arange(-90, 90, 2)
longitudes = np.arange(-180, 180, 2)
X, Y = np.meshgrid(longitudes, latitudes)

# Create data variables.
zonal_wind = u(X, Y)
meridional_wind = v(X, Y)
kinetic_energy = kinetic_energy(zonal_wind, meridional_wind)


# Create data arrays.
zonal_wind_da = xr.DataArray(
    zonal_wind,
    coords=[latitudes, longitudes],
    dims=('latitude', 'longitude' ),
    name='Zonal Wind',
    attrs={
        'long_name': 'Zonal wind speed on fantasy planet',
        'standard_name': 'zonal_wind_speed',
        'units': 'meter per second'
    }
)

meridional_wind_da = xr.DataArray(
    meridional_wind,
    coords=[latitudes, longitudes],
    dims=('latitude', 'longitude'),
    name='Meridional Wind',
    attrs={
        'long_name': 'Meridional wind speed on fantasy planet',
        'standard_name': 'Meridional wind speed',
        'units': 'meter per second'
    }
)

kinetic_energy_da = xr.DataArray(
    kinetic_energy,
    coords=[latitudes, longitudes],
    dims=('latitude', 'longitude'),
    name='Kinetic Energy',
    attrs={
        'long_name': 'Kinetic energy from zonal and meridional wind',
        'standard_name': 'Kinetic Energy',
        'units': 'meter**2 / second**2',
        "description": "0.5*(u**2 + v**2)",
        'cell_methods': 'Kinetic energy from zonal and meridional wind'
    }
)

kinetic_energy_density_da = kinetic_energy_da.mean(dim='longitude')

kinetic_energy_density_da.name = "Kinetic Energy Density"

kinetic_energy_density_da.attrs = dict(
    long_name='Zonally-averaged Kinetic Energy Density',
    standard_name='Specific Kinetic Energy',
    units='meter**2 / second**2',
    description="0.5*(u^2 + v^2) averaged over longitude",
    cell_methods="longitude: mean"
)


# Create xarray dataset.
ds = xr.Dataset(
    data_vars=dict(
        u=zonal_wind_da,
        v=meridional_wind_da,
        kinetic_energy=kinetic_energy_da,
        kinetic_energy_density=kinetic_energy_density_da
    ),
    attrs=dict(
        Author="Chavez Pope",
        Affiliation="Max Planck Institute for Meteorology",
        Discription="Wind and Kinetic Energy data on a fantasy planet",
        Date_Created=f"{datetime.now()}",
        Purpose="Lab Report 2"
    )
)

# ds.to_netcdf("fantasy_planet_wind_data.nc")
ds