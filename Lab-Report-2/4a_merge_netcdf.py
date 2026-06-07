from pathlib import Path

import xarray as xr


# Open the folder containing data.
directory = Path("../Lab-Report-2")

# Create a list of the monthly-surft-prec files.
files = list(directory.glob("*monthly-surft-prec.nc"))

ds = xr.open_mfdataset(files, engine='scipy')

# Save the combined dataset.
file_name = "1990-2000monthly-surft-prec_.nc"
file_path = Path(file_name)
# ds.to_netcdf(file_path)

print(f"Merged dataset saved to: {file_path.resolve()}.")