import math
import numpy as np
import time
from numba import jit

# -----------------------------
# 1. Pure Python loop (slow)
def slow_sin(arr):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = math.sin(arr[i])
    return result

# -----------------------------
# 2. Numba JIT-compiled version
@jit(nopython=True)
def fast_sin(arr):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        result[i] = math.sin(arr[i])
    return result

# -----------------------------
# 3. NumPy vectorized version
def numpy_sin(arr):
    return np.sin(arr)

# -----------------------------
# Run benchmark
arr = np.linspace(0, 2 * np.pi, 100_000_000)

# Warm-up JIT
fast_sin(arr[:10])

# Timing pure Python
start = time.perf_counter()
out1 = slow_sin(arr)
end = time.perf_counter()
print(f"Pure Python loop: {end - start:.4f} seconds")

# Timing JIT version
start = time.perf_counter()
out2 = fast_sin(arr)
end = time.perf_counter()
print(f"Numba JIT version: {end - start:.4f} seconds")

# Timing NumPy version
start = time.perf_counter()
out3 = numpy_sin(arr)
end = time.perf_counter()
print(f"NumPy vectorized: {end - start:.4f} seconds")

# Optional: check if outputs are equal
print("All results equal:", np.allclose(out1, out2) and np.allclose(out2, out3))
