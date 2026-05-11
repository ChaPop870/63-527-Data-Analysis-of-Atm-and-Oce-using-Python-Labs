import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

R = 2.0
s = np.linspace(0, 2*np.pi, 300)
t = np.linspace(0, 2*np.pi, 300)
S, T = np.meshgrid(s, t)

x = R * np.cos(T) * np.cos(S)
y = R * np.cos(T) * np.sin(S)
z = R * np.sin(T)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='3d'))
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
ax.set_title("Sphere: $x^2 + y^2 + z^2 = 4$", fontsize=16)
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
ax.set_zlabel("z", fontsize=14)
ax.set_box_aspect([1, 1, 1])
ax.grid()

plt.show()