import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmax

# Load data from the .dat files.
x = np.load('kdvx.dat')
t = np.load('kdvt.dat')
y = np.load('kdvsol.dat')

# Set up the meshgrid for plotting.
X, T = np.meshgrid(x, t)

# Computing wavelength by crest tracking.
j = len(t) // 2
yx_slice = y[j, :]
x_peaks_idx = argrelmax(yx_slice, order=2)[0]
x_peaks = x[x_peaks_idx]
dx = np.diff(x_peaks)
wavelength = np.mean(dx)

# Computing period by crest tracking.
i = len(x) // 2
yt_slice = y[:, i]
t_peaks_idx = argrelmax(yt_slice, order=2)[0]
t_peaks = t[t_peaks_idx]
dt = np.diff(t_peaks)
period = np.mean(dt)

# Computing wave speed.
wave_speed = wavelength / period
print(f"Wave speed: {wave_speed:.4f} m/s")

# Plotting.
fig, ax = plt.subplots(figsize=(8, 8))

data = ax.pcolormesh(X, T, y)
ax.scatter(x_peaks, t[j] * np.ones_like(x_peaks), color='red', label=f"Crests along t={t[j]:.1f}")
ax.scatter(x[i] * np.ones_like(t_peaks), t_peaks, color='blue', label=f"Crests along x={x[i]:.1f}")
ax.set_title("Korteveg de Vries on a Periodic Domain", fontsize=16)
ax.set_xlabel('x / m', fontsize=14)
ax.set_ylabel('t / s', fontsize=14)
ax.legend(loc='lower right', fontsize=12)

text = (
    f"Period = {period:.4f} s\n"
    f"Wavelength = {wavelength:.4f} m\n"
    rf"Wave Speed = {wave_speed:.4f} m s$^{{-1}}$"
)
ax.text(0.02, 0.98,
        text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, boxstyle='round,pad=0.4'))

cbar = fig.colorbar(ax=ax,
                    mappable=data,
                    orientation='vertical'
)
cbar.set_label('Wave Amplitude / m', fontsize=14)

plt.show()