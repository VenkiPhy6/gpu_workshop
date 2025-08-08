from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = x[i] + y[i]

n = 1024
x = np.arange(n).astype(np.float32)
y = np.ones(n, dtype=np.float32)
out = np.zeros_like(x)

# Move to device
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array_like(x)

# Launch
threads_per_block = 128
blocks = (n + threads_per_block - 1) // threads_per_block
add_kernel[blocks, threads_per_block](d_x, d_y, d_out)

# Copy back
d_out.copy_to_host(out)
print(out[:10])
