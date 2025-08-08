import numpy as np
import time
import sys
from numba import cuda, float32

# ----------------------------
# Shared memory optimized kernel
# ----------------------------
TPB = 16  # Threads per block (tile width)

@cuda.jit
def matmul_shared(A, B, C):
    sA = cuda.shared.array((TPB, TPB), dtype=float32)
    sB = cuda.shared.array((TPB, TPB), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    x = bx * TPB + tx
    y = by * TPB + ty

    tmp = 0.0
    for m in range((A.shape[1] + TPB - 1) // TPB):
        if y < A.shape[0] and m * TPB + tx < A.shape[1]:
            sA[ty, tx] = A[y, m * TPB + tx]
        else:
            sA[ty, tx] = 0.0

        if x < B.shape[1] and m * TPB + ty < B.shape[0]:
            sB[ty, tx] = B[m * TPB + ty, x]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads()

        for k in range(TPB):
            tmp += sA[ty, k] * sB[k, tx]

        cuda.syncthreads()

    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp

# ----------------------------
# Parse command-line argument
# ----------------------------
if len(sys.argv) < 2:
    print("Usage: python matrix_multiplication_gpu.py <N>")
    sys.exit(1)

N = int(sys.argv[1])

# ----------------------------
# Create matrices
# ----------------------------
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# Transfer to GPU
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.device_array((N, N), dtype=np.float32)

# Configure grid and blocks
blocks_per_grid = ((N + TPB - 1) // TPB, (N + TPB - 1) // TPB)
threads_per_block = (TPB, TPB)

# ----------------------------
# Warm-up
# ----------------------------
matmul_shared[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
cuda.synchronize()

# ----------------------------
# Benchmark
# ----------------------------
start = time.time()
matmul_shared[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
cuda.synchronize()
end = time.time()

d_C.copy_to_host(C)
print(f"Optimized shared-memory GPU matmul for N={N} took {end - start:.4f} seconds")
