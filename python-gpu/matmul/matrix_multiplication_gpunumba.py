import numpy as np
import time
import sys
from numba import cuda

# GPU kernel for matrix multiplication
@cuda.jit
def matmul_gpu(A, B, C):
    i, j = cuda.grid(2)
    N = A.shape[0]

    if i < N and j < N:
        tmp = 0.0
        for k in range(N):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

# ----------------------------
# Command-line arg for N
# ----------------------------
if len(sys.argv) < 2:
    print("Usage: python matrix_multiplication_gpu.py <N>")
    sys.exit(1)

N = int(sys.argv[1])

# ----------------------------
# Create input matrices
# ----------------------------
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Transfer to GPU
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.device_array((N, N), dtype=np.float32)

# ----------------------------
# Configure GPU threads
# ----------------------------
TPB = 16  # Threads per block (can tune this)
blocks_per_grid = ( (N + TPB - 1) // TPB, (N + TPB - 1) // TPB )

# ----------------------------
# Warm-up kernel (compile)
# ----------------------------
matmul_gpu[blocks_per_grid, (TPB, TPB)](d_A, d_B, d_C)
cuda.synchronize()

# ----------------------------
# Benchmark
# ----------------------------
start = time.time()
matmul_gpu[blocks_per_grid, (TPB, TPB)](d_A, d_B, d_C)
cuda.synchronize()
end = time.time()

# Copy back to host if needed
d_C.copy_to_host(C)

print(f"Numba CUDA GPU matmul for N={N} took {end - start:.4f} seconds")
