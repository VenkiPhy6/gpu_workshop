import numpy as np
import cupy as cp
import time

# NumPy benchmark
x_np = np.random.rand(1_000_000_000)
start = time.time()
np.sin(x_np)
end = time.time()
print(f"NumPy sin: {end - start:.6f} seconds")

# CuPy benchmark
x_cp = cp.asarray(x_np)
cp.sin(x_cp)  # warm-up to reduce overhead
cp.cuda.Device(0).synchronize()

start = time.time()
cp.sin(x_cp)
cp.cuda.Device(0).synchronize()  # wait for syncing
end = time.time()
print(f"CuPy sin: {end - start:.6f} seconds")
