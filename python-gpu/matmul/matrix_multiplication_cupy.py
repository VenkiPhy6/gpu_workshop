import cupy as cp
import time
import sys
# Parse N from command line
if len(sys.argv) < 2:
    print("Usage: python matrix_multiplication_cupy.py <N>")
    sys.exit(1)

N = int(sys.argv[1])

# Create random matrices on the GPU
A = cp.random.rand(N, N, dtype=cp.float32)
B = cp.random.rand(N, N, dtype=cp.float32)

# Warm-up
_ = A @ B
cp.cuda.Device().synchronize()

# Time the multiplication
start = time.time()
C = A @ B
cp.cuda.Device().synchronize()  # Ensure GPU computation is done
end = time.time()

elapsed = end - start
print(f"N={N}, CuPy matrix multiplication took {elapsed:.4f} seconds")

# Optionally write to a shared output file
with open("cupy_times_summary.txt", "a") as f:
    f.write(f"N={N}, time={elapsed:.4f} seconds\n")
