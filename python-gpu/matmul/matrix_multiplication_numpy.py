# matrix_multiplication_numpy.py
import numpy as np
import time
import sys

# Parse N from command line
if len(sys.argv) < 2:
    print("Usage: python matrix_multiplication_numpy.py <N>")
    sys.exit(1)

N = int(sys.argv[1])

# Create random matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Time the multiplication
start = time.time()
C = A @ B
end = time.time()

elapsed = end - start
print(f"N={N}, NumPy matrix multiplication took {elapsed:.4f} seconds")

# Optionally write to a shared output file
with open("numpy_times_summary.txt", "a") as f:
    f.write(f"N={N}, time={elapsed:.4f} seconds\n")
