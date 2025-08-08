import numpy as np
#import matplotlib.pyplot as plt
from numba import cuda, float32
import math
import time

# Parameters
N = 256
TOLERANCE = 1e-4
MAX_ITERS = 100000
dx = 1.0 / N
BLOCK_SIZE = 16  # For CUDA grid

@cuda.jit
def jacobi_kernel(p, f, p_new, dx2):
    i, j = cuda.grid(2)
    ################################
    ##### Write your code here #####
    ################################


@cuda.jit
def compute_diff_kernel(p_new, p, diff_array):
    i, j = cuda.grid(2)
    if i < p.shape[0] and j < p.shape[1]:
        diff = p_new[i, j] - p[i, j]
        cuda.atomic.add(diff_array, 0, diff * diff)

# Host-side function
def jacobi_gpu(N, tolerance, max_iters):
    dx2 = dx * dx

    # Allocate arrays on host
    p_host = np.zeros((N, N), dtype=np.float32)
    f_host = np.zeros((N, N), dtype=np.float32)
    f_host[N//2, N//2] = 10000

    # Allocate on device
    p = cuda.to_device(p_host)
    f = cuda.to_device(f_host)
    p_new = cuda.device_array_like(p)

    # CUDA block/grid config
    threadsperblock = (BLOCK_SIZE, BLOCK_SIZE)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    for iteration in range(max_iters):
        # Jacobi update
        jacobi_kernel[blockspergrid, threadsperblock](p, f, p_new, dx2)

        # Compute L2 norm of difference
        diff_array = cuda.device_array(1, dtype=np.float32)
        diff_array[0] = 0.0
        compute_diff_kernel[blockspergrid, threadsperblock](p_new, p, diff_array)
        diff = math.sqrt(diff_array.copy_to_host()[0])

        if diff < tolerance:
            print(f"Converged after {iteration} iterations")
            break

        # Swap buffers
        p, p_new = p_new, p

    return p.copy_to_host()

# Run
start = time.time()
result = jacobi_gpu(N, TOLERANCE, MAX_ITERS)
end = time.time()
print("Heat source at equilibrium: ", result[N//2, N//2])
print("Elapsed time: ", end - start, " s")

# Plot
#plt.imshow(result, cmap='hot')
#plt.colorbar()
#plt.title("Numba CUDA Jacobi Heat Diffusion")
#plt.show()
