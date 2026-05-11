import matplotlib.pyplot as plt
import numpy as np


def u_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the zonal wind field for a given longitude and latitude"""
    return -10 * np.sin(2 * np.pi * y / 180) * (np.cos(np.pi * x / 180))**2


def v_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the meridional wind field for a given longitude and latitude"""
    return 10 * (np.cos(np.pi * y / 180))**2 * np.sin(2 * np.pi * x / 180)


def windspeed(u: float | np.ndarray, v: float | np.ndarray) -> float | np.ndarray:
    """Return the windspeed given the zonal and meridional wind field"""
    return np.sqrt(u**2 + v**2)


# Set up lat / lon grid
x = np.linspace(-180, 180, 1_000)
y = np.linspace(-90, 90, 1_000)
X, Y = np.meshgrid(x, y)

# Compute fields
U = u_field(X, Y)
V = v_field(X, Y)
speed = windspeed(U, V)

zonal_mean = np.average(speed, axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(y, zonal_mean, color='red')
ax.set_xlabel("Latitude")
ax.set_ylabel("Zonally averaged wind speed")
ax.set_title("Zonally averaged wind speed vs latitude")
ax.set_xlim(-90, 90)
ax.set_ylim(0, None)

plt.show()