from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr


# Open the folder containing data.
directory = Path("../Lab-Report-2")

# Create a list of the monthly-surft-prec files.
files = list(directory.glob("*monthly-surft-prec.nc"))

ds = xr.open_mfdataset(files, engine='scipy')

# Compute means
temp = ds.t2m.mean(dim=['latitude', 'longitude']) - 273.15
prec = ds.lsp.mean(dim=['latitude', 'longitude']) * 1000

start_year = int(ds.time.dt.year.min())
end_year = int(ds.time.dt.year.max())


# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

ax1.plot(ds.time, temp, color='red', label='2 m Temperature')
ax1.set_ylabel("2 m Temperature / °C")
ax1.set_ylim(0, float(temp.max()) + 1)
ax1.set_title(f"Globally averaged 2 m Temperature and Large-scale Precipitation ({start_year}-{end_year})", pad=10)


ax2.plot(ds.time, prec, color='blue', label='Large-scale precipitation')
ax2.set_ylabel("Large-scale Precipitation / mm")
ax2.set_xlabel("Time")


plt.show()