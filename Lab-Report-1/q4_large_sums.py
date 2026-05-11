from time import time
from numba import jit

import matplotlib.pyplot as plt
import numpy as np

def s1(n: int) -> float:
    """Computes the sum using Python cycles"""
    sum = 0
    for i in range(1, n):
        sum += (-1)**(i + 1) * np.exp(-i * np.log(2)) / i
    return sum

def s2(n: int) -> np.ndarray:
    """Computes the sum using Python vectorization"""
    x = np.arange(1, n, dtype=float)
    terms = (-1)**(x + 1) * np.exp(-x * np.log(2)) / x
    return terms

@jit
def s3(n: int) -> float:
    """Computes the sum using Python cycles and Jit"""
    sum = 0
    for i in range(1, n):
        sum += (-1)**(i + 1) * np.exp(-i * np.log(2)) / i
    return sum

# Testing integer
n: int = 50_000_000

# Compute sum using cycles
s1_start = time()
s1 = s1(n)
s1_end = time()

s1_time = s1_end - s1_start

# Compute sum using vectorization
s2_start = time()
s2 = s2(n)
s2_sum = float(s2.sum())
s2_end = time()

s2_time = s2_end - s2_start

# Compute sum numba
s3_start = time()
s3 = s3(n)
s3_end = time()

s3_time = s3_end - s3_start

# Print statistics
print(f"Sum using cycles: {s1:.3f}.")
print(f"Cycle computation time: {s1_time:.3f} seconds.\n")

print(f"Sum using vectorization: {s2_sum:.3f}.")
print(f"Vectorized computation time: {s2_time:.3f} seconds\n")

print(f"Sum using numba: {s3:.3f}.")
print(f"Numba computation time: {s3_time:.3f} seconds\n")

# Plotting
x1 = np.arange(1, n)
s2_cumsum = s2.cumsum()

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x1,
        s2_cumsum,
        color='black',
        linewidth=0.7,
        label=r'$s(n) = \sum_{k=1}^n \dfrac{(-1)^{k+1}}{k\cdot 2^k}$'
)
ax.axhline(y=np.log(1.5), linestyle=':', color='red', label='log(1.5)', alpha=0.5)
ax.set_xlabel('n')
ax.set_ylabel('s(n)')
ax.set_title(r"Graph of the cumulative sum $s(n)$")
ax.set_xscale('log')
ax.set_xlim(1, None)
ax.set_ylim(0.35, 0.51)

plt.show()