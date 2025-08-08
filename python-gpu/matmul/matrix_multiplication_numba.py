import numpy as np
import time
from numba import njit, prange
import sys

# Optimized matrix multiplication using Numba
@njit(parallel=True, fastmath=True)
def matmul(A, B):
    N = A.shape[0]
    C = np.empty((N, N), dtype=np.float32)
    for i in prange(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp += A[i, k] * B[k, j]
            C[i, j] = tmp
    return C

# ----------------------------
# Input argument
# ----------------------------
if len(sys.argv) < 2:
    print("Usage: python matrix_multiplication_numba.py <N>")
    sys.exit(1)

N = int(sys.argv[1])

# ----------------------------
# Create input matrices
# ----------------------------
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Warm-up for JIT compilation
matmul(A[:10, :10], B[:10, :10])

# ----------------------------
# Benchmark
# ----------------------------
start = time.time()
C = matmul(A, B)
end = time.time()

print(f"Numba (parallel) matrix multiplication for N={N} took {end - start:.4f} seconds")
