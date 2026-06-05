from pathlib import Path

import xarray as xr


# Open the folder containing data.
directory = Path("../Lab-Report-2")

# Create a list of the monthly-surft-prec files.
files = list(directory.glob("*monthly-surft-prec.nc"))

ds = xr.open_mfdataset(files, engine='scipy')

# Save the combined dataset.
ds.to_netcdf("1990-2000monthly-surft-prec.nc")

print("1990-2000monthly-surft-prec.nc was created.")